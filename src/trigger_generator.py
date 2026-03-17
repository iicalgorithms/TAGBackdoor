import gc
import os
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional, Set
from collections import defaultdict, Counter
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nltk
import transformers
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from torch import nn, no_grad
import torch.optim as optim
from copy import deepcopy
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer, AutoModel

from src.models import create_model
from src.defence import defense_prune, defense_reconstruct
from config import ExperimentConfig


class MLPTrojan(nn.Module):
    """
    Module Function Description: GNNTrojan
    This is a neural network model for generating backdoor features (trigger features).
    Its goal is to generate specific feature vectors for attacked nodes, these feature vectors (backdoor triggers)
    can make the victim model misclassify these nodes to the attacker-specified target category when encountering them.
    """
    def __init__(self, input_dim, output_dim, layernum=2, dropout=0.00):  # Initialize method
        super(MLPTrojan, self).__init__()  # Call parent class initialization method


        layers = []  # Initialize an empty list to store network layers
        if dropout > 0:  # If dropout rate is greater than 0
            layers.append(nn.Dropout(p=dropout))  # Add Dropout layer to prevent overfitting
        for l in range(layernum - 1):  # Loop to build specified number of hidden layers (layernum - 1 layers)
            layers.append(nn.Linear(input_dim, input_dim))  # Add linear layer (fully connected layer)
            layers.append(nn.ReLU(inplace=True))  # Add ReLU activation function, inplace=True means directly modify input data to save memory
            if dropout > 0:  # If dropout rate is greater than 0
                layers.append(nn.Dropout(p=dropout))  # Add Dropout layer after activation function

        self.layers = nn.Sequential(*layers)  # Combine all layers into a sequential model
        self.feat = nn.Linear(input_dim, output_dim)  # Define a linear layer to map hidden layer output to backdoor feature dimensions

    def forward(self, input):  # Forward propagation method
        h = self.layers(input)  # Input data passes through defined hidden layer sequence
        feat = self.feat(h)  # Hidden layer output passes through final linear layer to generate backdoor features
        return feat  # Return generated backdoor features


class OODDiscriminator(nn.Module):
    """
    Out-of-Distribution (OOD) Discriminator Network
    This network acts as a discriminator to distinguish between real text features (in-distribution)
    and generated trigger features (potentially out-of-distribution).
    It helps ensure that generated triggers remain within the training text distribution.
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(OODDiscriminator, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        # Build discriminator layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
            hidden_dim = hidden_dim // 2  # Gradually reduce dimensions
        
        # Final classification layer
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of discriminator
        Args:
            x: Input features [batch_size, feature_dim]
        Returns:
            probability: Probability of being real (in-distribution) [batch_size, 1]
        """
        return self.discriminator(x)


class FeatureDistributionAnalyzer(nn.Module):
    """
    Feature Distribution Analyzer for OOD Detection
    This module analyzes the statistical properties of feature distributions
    to help the discriminator make better OOD detection decisions.
    """
    def __init__(self, feature_dim):
        super(FeatureDistributionAnalyzer, self).__init__()
        self.feature_dim = feature_dim
        
        # Statistical feature extractors
        self.mean_analyzer = nn.Linear(feature_dim, feature_dim // 4)
        self.std_analyzer = nn.Linear(feature_dim, feature_dim // 4)
        self.skew_analyzer = nn.Linear(feature_dim, feature_dim // 4)
        self.kurt_analyzer = nn.Linear(feature_dim, feature_dim // 4)
        
        # Combine statistical features
        self.combiner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def compute_statistical_features(self, x):
        """
        Compute statistical features of input
        Args:
            x: Input features [batch_size, feature_dim]
        Returns:
            statistical_features: Combined statistical features
        """
        # Compute basic statistics
        mean_feat = torch.mean(x, dim=0, keepdim=True).expand_as(x)
        std_feat = torch.std(x, dim=0, keepdim=True).expand_as(x)
        
        # Compute higher-order moments (approximation)
        centered = x - mean_feat
        skew_feat = torch.mean(centered ** 3, dim=0, keepdim=True).expand_as(x)
        kurt_feat = torch.mean(centered ** 4, dim=0, keepdim=True).expand_as(x)
        
        # Process through analyzers
        mean_processed = self.mean_analyzer(mean_feat)
        std_processed = self.std_analyzer(std_feat)
        skew_processed = self.skew_analyzer(skew_feat)
        kurt_processed = self.kurt_analyzer(kurt_feat)
        
        # Combine all statistical features
        combined = torch.cat([mean_processed, std_processed, skew_processed, kurt_processed], dim=1)
        return combined
    
    def forward(self, x):
        """
        Analyze feature distribution and return OOD score
        Args:
            x: Input features [batch_size, feature_dim]
        Returns:
            ood_score: OOD detection score [batch_size, 1]
        """
        stat_features = self.compute_statistical_features(x)
        ood_score = self.combiner(stat_features)
        return ood_score



class TextBackdoor:
    """
    Module Function Description: TextBackdoor
    This class encapsulates the main process of executing a backdoor attack, including.
    1. training a shadow model, the model is used to learn the representation of graph data.
    2. training a GNNTrojan, the network uses the output of the shadow model to generate backdoor triggers (text scrambling). 3. defining how the generated backdoor triggers can be used to generate a text scrambler.
    3. define how to inject the generated triggers into the features of the target nodes. 4. manage the training process, including optimization.
    4. manage the training process, including optimizing the shadow model and the graph TrojanNet, and use loss functions such as HomoLoss to guide training.
    5. provide methods for obtaining post-doctoring data (node features containing injected triggers).
    """
    def __init__(self, config: ExperimentConfig, text_autoencoder):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.text_autoencoder = text_autoencoder

        self.trojan = None
        self.shadow_model = None
        self.best_trojan_weights = None  # Store best trojan weights during training
        self.device = config.device
        
        # OOD detection parameters
        self.use_ood_detection = config.use_ood_detection
        self.ood_lambda = config.ood_lambda  # Weight for OOD loss
        self.discriminator_lr = config.discriminator_lr
        self.adversarial_lambda = config.adversarial_lambda  # Weight for adversarial loss
        
        # OOD detection models
        self.ood_discriminator = None
        self.feature_analyzer = None
        self.best_ood_discriminator_weights = None
        self.best_feature_analyzer_weights = None


    def _apply_defense_mode(self, poison_data, clean_data):

        self.logger.info(f" defense mode: {self.config.defense_mode}")

        # Apply defense method to backdoored data
        if self.config.defense_mode == 'prune':
            # Record statistics before defense
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(f" Before pruning defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_prune(poison_data, self.config.prune_thr, self.device, large_graph=True)

            # Record statistics after defense
            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(
                f"After pruning defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(
                f"Pruning defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")

        elif self.config.defense_mode == 'reconstruct':
            # Record statistics before defense
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(
                f"Before reconstruction defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_reconstruct(poison_data, clean_data, self.config.rec_epochs, self.device)

            # Record statistics after defense
            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(
                f"After reconstruction defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(
                f"Reconstruction defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")

        else:
            defensed_data = poison_data

        return defensed_data



    def _train_models(self, data, poison_candidates, poison_mask, sorted_dims):

        self.logger.info("Training triggers generator ...")

        data = data.to(self.device)
        poison_indices = torch.where(poison_mask)[0]

        # Initialize shadow model
        self.shadow_model = create_model(
            model_name=self.config.shadow_model,
            input_dim=data.x.shape[1],
            hidden_dim=self.config.hidden_dim,
            output_dim=data.num_classes
        ).to(self.device)

        _, embeddings=self.shadow_model(data.x, data.train_edge_index, return_embeddings=True)
        
        # Initialize trojan network

        feature_dim = data.x.shape[1]
        poison_dim_num = min(self.config.poison_dim_num, feature_dim)

        poison_dim_indices= sorted_dims[:poison_dim_num]

        print(f"===poison_dim_len:{len(poison_dim_indices)}")
        print(f"===dim_len:{feature_dim}")

        self.trojan = MLPTrojan(
            input_dim=embeddings.shape[1],
            output_dim=poison_dim_num,
            layernum=self.config.trojan_num_layers,
            dropout=self.config.trojan_dropout,
        ).to(self.device)
        
        # Initialize OOD detection components if enabled
        if self.use_ood_detection:
            self.ood_discriminator = OODDiscriminator(
                input_dim=feature_dim,
                hidden_dim=self.config.ood_hidden_dim,
                num_layers=self.config.ood_num_layers,
                dropout=self.config.ood_dropout
            ).to(self.device)
            
            self.feature_analyzer = FeatureDistributionAnalyzer(
                feature_dim=feature_dim
            ).to(self.device)
        
        # Initialize optimizers
        shadow_optimizer = optim.Adam(self.shadow_model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        trojan_optimizer = optim.Adam(self.trojan.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Initialize OOD discriminator optimizer if enabled
        discriminator_optimizer = None
        if self.use_ood_detection:
            discriminator_optimizer = optim.Adam(
                list(self.ood_discriminator.parameters()) + list(self.feature_analyzer.parameters()),
                lr=self.discriminator_lr,
                betas=(0.5, 0.999)
            )


        # Training masks， Set target labels for poison nodes
        train_mask = data.train_mask
        train_poison_mask = train_mask.clone()

        poison_labels = data.y.clone()
        for candidate in poison_candidates:
            poison_labels[candidate['node_idx']] = candidate['target_label']
            train_poison_mask[candidate['node_idx']]=True
        original_labels = data.y.clone()

        loss_fn = nn.CrossEntropyLoss()
        best_loss = float('inf')
        
        # Main training loop
        poison_x=None
        for epoch_trojan in range(self.config.trojan_epochs):

            # Phase 1. Train shadow model
            self.shadow_model.train()
            self.trojan.eval()
            shadow_optimizer.zero_grad()

            # Get embeddings from shadow model
            with no_grad():
                _, embed = self.shadow_model(data.x, data.train_edge_index, return_embeddings=True)

            # Generate backdoor features for poison nodes
            trojan_feat = self.trojan(embed[poison_mask])  # Use trojan network to generate backdoor features
            # Create poisoned feature based on data.x
            poison_x = data.x.clone().detach()  # Clone original features
            poison_x[poison_indices[:,None], poison_dim_indices] = trojan_feat.detach()

            # Forward pass through shadow model
            output = self.shadow_model(poison_x, data.train_edge_index)

            # Compute loss on poison nodes (should classify as target) Create poison_labels with same length as data.x, set target labels only for poison nodes
            shadow_loss = loss_fn(output[train_poison_mask], poison_labels[train_poison_mask])
            shadow_loss.backward()
            shadow_optimizer.step()


            # Phase 2: Train OOD discriminator (if enabled)
            if self.use_ood_detection and discriminator_optimizer is not None:
                self.ood_discriminator.train()
                self.feature_analyzer.train()
                self.trojan.eval()
                
                discriminator_optimizer.zero_grad()
                
                # Get real features (original training features)
                real_features = data.x[train_mask]
                real_labels = torch.ones(real_features.size(0), 1).to(self.device)
                
                # Get fake features (generated trojan features)
                with torch.no_grad():
                    _, embeddings = self.shadow_model(poison_x, data.train_edge_index, return_embeddings=True)
                    fake_dims = self.trojan(embeddings[poison_mask])

                poison_x = data.x.clone().detach()  # Clone original features
                poison_x[poison_indices[:, None], poison_dim_indices] = fake_dims.detach()
                fake_features=poison_x[poison_mask]

                fake_labels = torch.zeros(fake_features.size(0), 1).to(self.device)
                
                # Train discriminator on real features
                real_pred = self.ood_discriminator(real_features)
                real_ood_score = self.feature_analyzer(real_features)
                real_loss = F.binary_cross_entropy(real_pred, real_labels) + F.binary_cross_entropy(real_ood_score, real_labels)
                
                # Train discriminator on fake features
                fake_pred = self.ood_discriminator(fake_features)
                fake_ood_score = self.feature_analyzer(fake_features)
                fake_loss = F.binary_cross_entropy(fake_pred, fake_labels) + F.binary_cross_entropy(fake_ood_score, fake_labels)
                
                # Combined discriminator loss
                discriminator_loss = (real_loss + fake_loss) / 2
                discriminator_loss.backward()
                discriminator_optimizer.step()
            
            # Phase 3: Train trojan network with adversarial loss
            self.shadow_model.eval()
            self.trojan.train()
            if self.use_ood_detection:
                self.ood_discriminator.eval()
                self.feature_analyzer.eval()
            
            trojan_optimizer.zero_grad()

            ## Random choose unlabeled nodes to poison
            unlabeled_mask = data.unlabeled_mask
            rest_unlabeled_mask = unlabeled_mask.clone()
            rest_unlabeled_mask = rest_unlabeled_mask & (~poison_mask.to(self.device))

            # Get indices of remaining unlabeled nodes
            rest_unlabeled_indices = torch.where(rest_unlabeled_mask)[0]
            
            # Randomly select outer_poison_num nodes from remaining unlabeled nodes
            if len(rest_unlabeled_indices) >= self.config.outer_poison_num:
                # Randomly sample without replacement
                selected_indices = rest_unlabeled_indices[torch.randperm(len(rest_unlabeled_indices))[:self.config.outer_poison_num]]
            else:
                # If not enough nodes, use all remaining unlabeled nodes
                selected_indices = rest_unlabeled_indices
                self.logger.warning(f" Only {len(rest_unlabeled_indices)} unlabeled nodes available, using all instead of {self.config.outer_poison_num}")
            
            # Create outer poison mask
            outer_poison_mask = torch.zeros_like(unlabeled_mask, dtype=torch.bool)
            outer_poison_mask[selected_indices] = True

            # Get embeddings from shadow model
            # with torch.no_grad():
            _, embeddings = self.shadow_model(poison_x, data.train_edge_index, return_embeddings=True)
            # Generate trojan features
            outer_trojan_features = self.trojan(embeddings[outer_poison_mask])
            
            # Create modified features
            outer_poison_x = data.x.detach().clone()
            outer_poison_indices = torch.where(outer_poison_mask)[0]
            outer_poison_x[outer_poison_indices[:,None], poison_dim_indices] = outer_trojan_features.detach()

            # Forward pass with modified features
            outer_output = self.shadow_model(outer_poison_x, data.train_edge_index)
            
            # Target loss: poison nodes should be classified as target class
            target_loss = loss_fn(outer_output[train_poison_mask], poison_labels[train_poison_mask])

            # Similarity loss: modified features should be similar to original
            similarity_loss = 1 - F.cosine_similarity(data.x[outer_poison_mask], outer_poison_x[outer_poison_mask], dim=1).mean()

            # Adversarial loss: fool the OOD discriminator
            adversarial_loss = 0
            if self.use_ood_detection:
                # Generate fresh trojan features for adversarial training
                fresh_trojan_dims = self.trojan(embeddings[poison_mask])

                poison_x = data.x.clone().detach()  # Clone original features
                poison_x[poison_indices[:, None], poison_dim_indices] = fresh_trojan_dims.detach()
                fresh_trojan_features = poison_x[poison_mask]

                # Try to fool discriminator (want discriminator to think these are real)
                fake_pred = self.ood_discriminator(fresh_trojan_features)
                fake_ood_score = self.feature_analyzer(fresh_trojan_features)
                real_labels_adv = torch.ones(fresh_trojan_features.size(0), 1).to(self.device)
                
                adversarial_loss = (F.binary_cross_entropy(fake_pred, real_labels_adv) + 
                                  F.binary_cross_entropy(fake_ood_score, real_labels_adv)) / 2

            # Combined trojan loss with feature similarity and adversarial loss
            trojan_loss = (target_loss + 
                          self.config.feature_similarity_weight * similarity_loss + 
                          self.adversarial_lambda * adversarial_loss)
            
            trojan_loss.backward()
            trojan_optimizer.step()
            
            # Save best model
            if trojan_loss.item() < best_loss:
                best_loss = trojan_loss.item()
                self.best_trojan_weights = deepcopy(self.trojan.state_dict())
                
                # Also save OOD detector weights if enabled
                if self.use_ood_detection:
                    self.best_ood_discriminator_weights = deepcopy(self.ood_discriminator.state_dict())
                    self.best_feature_analyzer_weights = deepcopy(self.feature_analyzer.state_dict())
            
            # Logging
            if epoch_trojan % 50 == 0:
                with torch.no_grad():
                    self.shadow_model.eval()
                    
                    # Log OOD detection metrics if enabled
                    ood_metrics_str = ""
                    if self.use_ood_detection:
                        # Evaluate OOD performance using the dedicated method
                        ood_metrics = self._evaluate_ood_performance(data, embeddings, poison_mask, train_mask, poison_dim_indices, poison_indices)
                        
                        if ood_metrics:
                            ood_metrics_str = (f", Disc Acc: {ood_metrics['discriminator_acc']:.4f}, "
                                             f"Analyzer Acc: {ood_metrics['analyzer_acc']:.4f}, "
                                             f"Adv Loss: {adversarial_loss:.4f}, "
                                             f"Sep: {ood_metrics['separation_disc']:.4f}")
                    pred = output.argmax(dim=1)
                    train_acc = (pred[train_mask] == original_labels[train_mask]).float().mean()
                    poison_acc = (pred[poison_mask] == poison_labels[poison_mask]).float().mean()
                    
                    print(f"Trojan epoch {epoch_trojan}: Shadow Loss: {shadow_loss.item():.4f}, "
                                   f"Trojan Loss: {trojan_loss.item():.4f}, "
                                   f"Train Acc: {train_acc:.4f}, Poison Acc: {poison_acc:.4f}{ood_metrics_str}")
        
        # Load best trojan weights
        if self.best_trojan_weights is not None:
            self.trojan.load_state_dict(self.best_trojan_weights)
            
        # Load best OOD detector weights if enabled
        if self.use_ood_detection:
            if hasattr(self, 'best_ood_discriminator_weights') and self.best_ood_discriminator_weights is not None:
                self.ood_discriminator.load_state_dict(self.best_ood_discriminator_weights)
            if hasattr(self, 'best_feature_analyzer_weights') and self.best_feature_analyzer_weights is not None:
                self.feature_analyzer.load_state_dict(self.best_feature_analyzer_weights)

        self.logger.info(" Triggers learned for all poison candidates")
        return poison_labels, poison_mask, train_poison_mask, poison_dim_indices

    def _create_poisoned_dataset(self, data, poison_labels, poison_mask, train_poison_mask, poison_dim_indices) :
        # create poisoned dataset
        original_x=data.x.detach()
        final_poison_x=original_x.clone()

        poison_indices = torch.where(poison_mask)[0]

        self.trojan.eval()
        self.shadow_model.eval()
        with torch.no_grad():
            _, embed = self.shadow_model(final_poison_x, data.train_edge_index, return_embeddings=True)
            trojan_features = self.trojan(embed[poison_indices])  # Use trojan network to generate backdoor features
        final_poison_x[poison_indices[:,None], poison_dim_indices] = trojan_features.detach()

        new_poison_texts = self.text_autoencoder.get_texts(final_poison_x[poison_indices])
        poison_embeddings = self.text_autoencoder.get_embeds(new_poison_texts)

        print(f"new_raw_texts1:{new_poison_texts[0:3]}")


        cosine_similarities1 = F.cosine_similarity(final_poison_x[poison_indices], poison_embeddings, dim=1)
        cosine_similarities2= F.cosine_similarity(final_poison_x[poison_indices], original_x[poison_indices], dim=1)
        cosine_similarities3= F.cosine_similarity(trojan_features, original_x[poison_indices,:][:,poison_dim_indices], dim=1)

        print(f"==== 生成的行 poison feature 和 由该feature 生成的text 再转成的embeddings的相似度(越相似越好): {cosine_similarities1.mean():.3f}")
        print(f"==== 生成的行 poison feature 和 原始行 features 的相似度 (越小越好) : {cosine_similarities2.mean():.3f}")
        print(f"==== 生成的列 poison feature 和 原始列 features 的相似度 (越小越好): {cosine_similarities3.mean():.3f}")


        # 计算 trojan_features 中所有向量的两两余弦相似度
        similarity_matrix_trojan_features = F.cosine_similarity(
            trojan_features.unsqueeze(1),  # [N, 1, D]
            trojan_features.unsqueeze(0),  # [1, N, D]
            dim=2
        )  # [N, N]

        mean_similarity1 = similarity_matrix_trojan_features.mean().item()
        # var_similarity = similarity_matrix_trojan_features.var().item()
        print(f"==== Mean pairwise similarity of trojan features: {mean_similarity1:.4f}")
        # print(f"Variance of pairwise cosine similarity_matrix_trojan_features: {var_similarity:.4f}")

        similarity_matrix_trojan_text = F.cosine_similarity(
            final_poison_x[poison_mask].unsqueeze(1),  # [N, 1, D]
            final_poison_x[poison_mask].unsqueeze(0),  # [1, N, D]
            dim=2
        )  # [N, N]
        mean_similarity = similarity_matrix_trojan_text.mean().item()
        # var_similarity = similarity_matrix_trojan_text.var().item()
        print(f"==== Mean pairwise similarity of poison item: {mean_similarity:.4f}")
        # print(f"Variance of pairwise cosine similarity_matrix_trojan_text: {var_similarity:.4f}")


        new_x=data.x.clone().detach()
        new_x[poison_mask]=poison_embeddings

        new_raw_texts=list(data.raw_texts)
        for idx, i in enumerate(poison_indices.tolist()):
            # print(f"===idx:{idx}==")
            # print(f"==original_text:{new_raw_texts[i]}")
            # print(f"==poison_text:{new_poison_texts[idx]}")
            new_raw_texts[i] = new_poison_texts[idx]

        # Refine poisoned data
        poisoned_data = Data(
            x=new_x,
            # x=final_poison_x,
            y=poison_labels,
            raw_texts=new_raw_texts,
            train_poison_mask=train_poison_mask, #train +poison
            train_mask=data.train_mask,
            poison_mask=poison_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            edge_index=data.edge_index,
            train_edge_index=data.train_edge_index,
            test_edge_index=data.test_edge_index,

            num_classes=data.num_classes,
        ).to(self.device)

        poisoned_data=self._apply_defense_mode(poisoned_data, data)
        return poisoned_data



    def _evaluate_ood_performance(self, data, embeddings, poison_mask, train_mask, poison_dim_indices, poison_indices):
        """
        Evaluate OOD discriminator performance
        
        Args:
            data: Graph data
            embeddings: Node embeddings from shadow model
            poison_mask: Mask for poisoned nodes
            train_mask: Mask for training nodes
            
        Returns:
            dict: Dictionary containing OOD performance metrics
        """
        if not self.use_ood_detection or self.ood_discriminator is None:
            return {}
            
        self.ood_discriminator.eval()
        self.feature_analyzer.eval()
        
        with torch.no_grad():
            # Get real features (original training features)
            real_features = data.x[train_mask]
            
            # Get fake features (generated trojan features)
            fake_dims = self.trojan(embeddings[poison_mask])


            poison_x = data.x.clone().detach()  # Clone original features
            poison_x[poison_indices[:, None], poison_dim_indices] = fake_dims.detach()
            fake_features = poison_x[poison_mask]


            # Discriminator predictions
            real_pred = self.ood_discriminator(real_features)
            fake_pred = self.ood_discriminator(fake_features)
            
            # Feature analyzer predictions
            real_ood_score = self.feature_analyzer(real_features)
            fake_ood_score = self.feature_analyzer(fake_features)
            
            # Calculate accuracies
            real_acc_disc = (real_pred > 0.5).float().mean().item()
            fake_acc_disc = (fake_pred <= 0.5).float().mean().item()
            discriminator_acc = (real_acc_disc + fake_acc_disc) / 2
            
            real_acc_analyzer = (real_ood_score > 0.5).float().mean().item()
            fake_acc_analyzer = (fake_ood_score <= 0.5).float().mean().item()
            analyzer_acc = (real_acc_analyzer + fake_acc_analyzer) / 2
            
            # Calculate average confidence scores
            real_conf_disc = real_pred.mean().item()
            fake_conf_disc = fake_pred.mean().item()
            real_conf_analyzer = real_ood_score.mean().item()
            fake_conf_analyzer = fake_ood_score.mean().item()
            
        return {
            'discriminator_acc': discriminator_acc,
            'analyzer_acc': analyzer_acc,
            'real_conf_disc': real_conf_disc,
            'fake_conf_disc': fake_conf_disc,
            'real_conf_analyzer': real_conf_analyzer,
            'fake_conf_analyzer': fake_conf_analyzer,
            'separation_disc': abs(real_conf_disc - fake_conf_disc),
            'separation_analyzer': abs(real_conf_analyzer - fake_conf_analyzer)
        }

    def _apply_pruning_to_test_data(self, test_data):

        edge_index = test_data.test_edge_index
        x = test_data.x

        # 如果没有边权重，创建全1的权重
        if hasattr(test_data, 'edge_weight') and test_data.edge_weight is not None:
            edge_weights = test_data.edge_weight
        else:
            edge_weights = torch.ones(edge_index.shape[1])

        # 选择边权重大于0的边，并将其移动到指定设备
        valid_edge_mask = edge_weights > 0.0
        edge_index = edge_index[:, valid_edge_mask].to(self.device)
        edge_weights = edge_weights[valid_edge_mask].to(self.device)
        x = x.to(self.device)

        # 计算边的相似度
        edge_sims = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])

        # 获取修剪阈值
        prune_threshold = self.config.prune_thr

        # 找到不相似的边并移除它们
        keep_mask = edge_sims > prune_threshold
        updated_edge_index = edge_index[:, keep_mask]
        updated_edge_weights = edge_weights[keep_mask]

        # 更新测试数据的边信息
        test_data.test_edge_index = updated_edge_index
        test_data.edge_weight = updated_edge_weights

        self.logger.info(
            f"Applied pruning to test data. Edges reduced from {edge_index.shape[1]} to {updated_edge_index.shape[1]}")
        return test_data

    def _get_poisoned_test_data(self, poison_train_data, target_model, poison_dim_indices):
        """Generate poisoned data by injecting backdoor triggers
            set test_poison_mask test_clean_mask
            set target
            set poison_x
        """
        self.logger.info("Generating poisoned test data ...")

        test_mask = poison_train_data.test_mask
        test_node_indices = torch.where(test_mask)[0]

        with torch.no_grad():
            output = target_model(poison_train_data.x, poison_train_data.test_edge_index)
        probs = F.softmax(output, dim=1)
        predictions = output.argmax(dim=1)

        if self.config.uncertainty_method == "entropy":
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        elif self.config.uncertainty_method == "variance":
            uncertainty = torch.var(probs, dim=1)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.config.uncertainty_method}")

        test_uncertainty = uncertainty[test_node_indices]

        sorted_indices = torch.argsort(test_uncertainty, descending=True)
        test_node_indices = test_node_indices[sorted_indices]

        num_test_nodes = len(test_node_indices)
        split_point = num_test_nodes // 2

        test_poison_nodes = test_node_indices[:split_point]
        test_clean_nodes = test_node_indices[split_point:]

        test_poison_mask = torch.zeros_like(test_mask, dtype=torch.bool)
        test_clean_mask = torch.zeros_like(test_mask, dtype=torch.bool)

        test_poison_mask[test_poison_nodes] = True
        test_clean_mask[test_clean_nodes] = True

        print(f"test_nodes:{len(test_node_indices)}")
        print(f"test_poison_nodes:{len(test_poison_nodes)}")
        print(f"test_clean_nodes:{len(test_clean_nodes)}")

        # Get test node indices from test_poison_mask
        poison_y = poison_train_data.y.clone()
        test_poison_node_indices = torch.where(test_poison_mask)[0]  #
        target_assignment = self.config.target_assignment
        num_classes = poison_train_data.num_classes

        for node_idx in test_poison_node_indices:
            true_label = poison_train_data.y[node_idx].item()

            if target_assignment == "fixed":
                # Use fixed target class
                target_label = self.config.fixed_target_class
                if target_label is None:
                    target_label = (true_label + 1) % num_classes  # Default: next class
            elif target_assignment == "confusion_max":
                # Use class with the highest prediction probability (excluding true class)
                node_probs = probs[node_idx].cpu().numpy()
                node_probs[true_label] = 0  # Exclude true class
                target_label = np.argmax(node_probs)
            else:
                # Random target class (different from true label)
                possible_targets = [i for i in range(num_classes) if i != true_label]
                target_label = np.random.choice(possible_targets)
            poison_y[node_idx] = target_label

        # Inject triggers for test nodes
        with torch.no_grad():
            _, embed = target_model(poison_train_data.x, poison_train_data.test_edge_index, return_embeddings=True)
        trojan_feat = self.trojan(embed[test_poison_mask])  # Use trojan network to generate backdoor features

        print(f"poisoned_dim_indices_len:{len(poison_dim_indices)}")

        test_poison_node_indices = torch.where(test_poison_mask)[0]
        poison_x = poison_train_data.x.detach().clone()  # Clone original features
        # poison_x[test_poison_mask,:][:, poison_dim_indices] = trojan_feat.detach() #TODO
        # poison_x[test_poison_mask] = trojan_feat.detach() #TODO
        poison_x[test_poison_node_indices[:, None], poison_dim_indices] = trojan_feat.detach()

        poisoned_data = Data(
            x=poison_x,
            y=poison_y,
            raw_texts=poison_train_data.raw_texts,
            train_mask=poison_train_data.train_mask,  # train +poison
            val_mask=poison_train_data.val_mask,
            test_mask=poison_train_data.test_mask,
            edge_index=poison_train_data.edge_index,
            train_edge_index=poison_train_data.train_edge_index,
            test_edge_index=poison_train_data.test_edge_index,

            test_poison_mask=test_poison_mask,
            test_clean_mask=test_clean_mask,

            num_classes=poison_train_data.num_classes,
        ).to(self.device)

        # Apply pruning defense to test data if defense mode is prune
        if self.config.defense_mode == 'prune':
            poisoned_data = self._apply_pruning_to_test_data(poisoned_data)

        self.logger.info(f"Generated poisoned data with {poisoned_data.test_poison_mask.sum().item()} poisoned nodes")
        return poisoned_data

    def _evaluate_clean_accuracy(self, target_model, data):
        """Evaluate clean accuracy on unmodified test data using test_clean_mask.
        Returns:
            Dictionary containing clean accuracy metrics
        """
        # Use test_clean_mask to identify clean test nodes
        test_mask = data.test_clean_mask

        self.logger.info(f"Evaluating CA: {torch.sum(test_mask).item()}...")

        # Get predictions
        data.to(self.device)
        with torch.no_grad():
            logits = target_model(data.x, data.test_edge_index)
            probs = F.softmax(logits, dim=1)

        predict_labels = logits.argmax(dim=1)[test_mask]
        true_labels = data.y[test_mask]

        # Compute metrics
        accuracy = accuracy_score(true_labels.cpu().numpy(), predict_labels.cpu().numpy())

        # Per-class accuracy
        unique_classes = int(data.y.max().item()) + 1
        per_class_acc = {}

        for class_idx in range(unique_classes):
            class_mask = true_labels == class_idx
            if class_mask.sum() > 0:
                class_acc = (predict_labels[class_mask] == true_labels[class_mask]).float().mean()
                per_class_acc[class_idx] = class_acc.item()

        results = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'num_test_samples': test_mask.sum().item()
        }

        self.logger.info(f"CA results: {results}")
        return results

    def _evaluate_attack_success_rate(self, target_model, data) -> Dict[str, float]:
        """Evaluate attack success rate.

        Returns:
            Dictionary containing attack success metrics
        """
        test_mask = data.test_poison_mask
        x = data.x.detach()
        self.logger.info(f"Evaluating ASR:  {torch.sum(test_mask).item()}...")

        with torch.no_grad():
            logits = target_model(x, data.test_edge_index)
        predictions = logits.argmax(dim=1)
        targets = data.y
        predictions = predictions.to(targets.device)

        # Compute attack success rate
        successful_attacks = (predictions[test_mask] == targets[test_mask]).float()
        attack_success_rate = successful_attacks.mean().item()

        # Per-target-class success rate
        unique_targets = torch.unique(targets[test_mask])
        per_target_asr = {}

        for target_class in unique_targets:
            target_mask = (targets[test_mask] == target_class)
            if target_mask.sum() > 0:
                target_asr = successful_attacks[target_mask].mean().item()
                per_target_asr[target_class.item()] = target_asr

        results = {
            'attack_success_rate': attack_success_rate,
            'per_target_asr': per_target_asr,
            'num_attack_samples': test_mask.sum().item(),
            'successful_attacks': successful_attacks.sum().item()
        }

        self.logger.info(f"ASR results: {results}")
        return results

    def _evaluate_text_reconstruction_quality(self, poisoned_test_data):
        """Evaluate the quality of text reconstruction for poisoned nodes
        """

        poison_mask = poisoned_test_data.test_poison_mask
        poison_indices = torch.where(poison_mask)[0].tolist()

        # Get original texts for poison nodes
        original_texts = [poisoned_test_data.raw_texts[i] for i in poison_indices]

        # Get modified features for poison nodes
        modified_features = poisoned_test_data.x[poison_mask]

        self.logger.info(f"Reconstructing {len(modified_features)} texts ")

        # Batch processing for text reconstruction
        reconstructed_texts = self.text_autoencoder.get_texts(modified_features)

        # Simple proxy: compute average word length change
        original_lengths = [len(text.split()) for text in original_texts]
        poisoned_lengths = [len(text.split()) for text in reconstructed_texts]
        length_change = np.array(poisoned_lengths) - np.array(original_lengths)

        # Compute text similarity
        original_embeddings = poisoned_test_data.x[poison_mask]
        reconstructed_embeddings = self.text_autoencoder.get_embeds(reconstructed_texts)

        similarity = F.cosine_similarity(original_embeddings, reconstructed_embeddings, dim=1)
        all_similarity_score = similarity.mean().item()

        results = []
        for i in range(len(poison_indices)):
            result_info = {
                'length_change': length_change[i],
                'similarity_score': similarity[i].item(),
                'original_text': original_texts[i],
                'poisoned_text': reconstructed_texts[i],
            }
            results.append(result_info)

        return all_similarity_score, results

    def _train_target_model(self, data, model_name):

        data = data.to(self.device)

        # Create target model
        model = create_model(model_name,
                             input_dim=data.x.shape[1],
                             hidden_dim=512,
                             output_dim=data.num_classes).to(self.device)

        # Set up training parameters
        epochs = self.config.target_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.target_lr,
                                     weight_decay=self.config.target_weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train_mask=data.except_test_mask # 除了test的节点，包含train+poison+val
        train_mask = data.train_mask
        poison_mask = data.poison_mask
        train_poison_mask = data.train_poison_mask  # 包含了train+poison
        val_mask = data.val_mask if hasattr(data, 'val_mask') else None

        # Early stopping parameters
        patience = self.config.patience  # 默认patience为50
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            out = model(data.x, data.train_edge_index)
            loss = criterion(out[train_poison_mask], data.y[train_poison_mask])
            train_loss = loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Evaluation for early stopping
            model.eval()
            with torch.no_grad():
                eval_out = model(data.x, data.train_edge_index)

                # 计算验证集准确率（如果有验证集）或使用训练集准确率
                if val_mask is not None and val_mask.sum() > 0:
                    val_acc = (eval_out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()
                else:
                    # 如果没有验证集，使用干净训练集准确率作为早停指标
                    val_acc = (eval_out[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean().item()

                # 保存最佳模型权重
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % 50 == 0:
                train_acc = (out[train_poison_mask].argmax(dim=1) == data.y[
                    train_poison_mask]).float().sum() / train_poison_mask.sum()
                ac_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().sum() / train_mask.sum()
                asr_acc = (out[poison_mask].argmax(dim=1) == data.y[poison_mask]).float().sum() / poison_mask.sum()

                print(
                    f" Poisoned data on target model - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc:{train_acc.item():.4f}, AC Acc:{ac_acc.item():.4f}, ASR Acc:{asr_acc.item():.4f}, Val Acc:{val_acc:.4f}, Best Val Acc:{best_val_acc:.4f}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                break

        # 加载最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model weights with validation accuracy: {best_val_acc:.4f}")

        return model

    def run_trigger_generation_pipeline(self, data, poison_candidates, poison_mask, sorted_dims):
        """Run the complete trigger generation pipeline
        
        This method orchestrates the entire backdoor attack process including:
        1. Training the shadow model and trojan network
        2. Generating backdoor triggers
        3. Creating poisoned data
        """
        self.logger.info("Starting trigger generation pipeline ...")
        
        # 1: Learn triggers by training shadow model and trojan network
        poison_labels, poison_mask, train_poison_mask, poison_dim_indices= self._train_models(data, poison_candidates, poison_mask, sorted_dims)

        poisoned_data = self._create_poisoned_dataset(data, poison_labels, poison_mask, train_poison_mask, poison_dim_indices)

        evaluation_results={}
        for model_name in self.config.target_models:
            self.logger.info(f" Training target model: {model_name}")

            # Train target models on poisoned training data
            target_model = self._train_target_model(poisoned_data, model_name)
            target_model.eval()

            # Generate poisoned test data for attack
            poisoned_test_data = self._get_poisoned_test_data(poisoned_data, target_model, poison_dim_indices)

            # Evaluate clean accuracy using test_clean_mask
            clean_accuracy_results = self._evaluate_clean_accuracy(target_model, poisoned_test_data)

            # Evaluate attack success rate using test_poison_mask
            attack_success_results = self._evaluate_attack_success_rate(target_model, poisoned_test_data)

            # Evaluate text reconstruction quality
            # all_similarity_score, each_text_results = self._evaluate_text_reconstruction_quality(poisoned_test_data)

            # Compile comprehensive results
            evaluation_result = {
                'model_name': model_name,
                'clean_accuracy': clean_accuracy_results,
                'attack_success': attack_success_results,
                # 'text_similarity': all_similarity_score,
                # 'text_details': each_text_results,
                'experiment_config': {
                    'poison_num': self.config.poison_node_num,
                    'dataset_name': self.config.dataset_name,
                }
            }

            # Log summary
            self.logger.info(f"Evaluation summary for {model_name}:")
            self.logger.info(f"  Clean Accuracy: {clean_accuracy_results['overall_accuracy']:.4f}")
            self.logger.info(f"  Attack Success Rate: {attack_success_results['attack_success_rate']:.4f}")
            # self.logger.info(f"  Text Similarity: {all_similarity_score:.4f}")

            evaluation_results[model_name] = evaluation_result

        return evaluation_results

    

                






