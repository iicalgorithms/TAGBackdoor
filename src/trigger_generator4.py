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
import transformers
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from torch import nn, no_grad
import torch.optim as optim
from copy import deepcopy

from src.models import create_model
from src.defence import defense_prune, defense_reconstruct
from config import ExperimentConfig



class Trojan(nn.Module):
    def __init__(self, text_embedding_dim, text_autoencoder, hidden_dim=256, layer_num=2, dropout=0.1):
        super(Trojan, self).__init__()
        # self.node_embedding_dim = node_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.layer_num= layer_num
        self.dropout = dropout
        self.text_autoencoder = text_autoencoder

        layers = []  # Initialize an empty list to store network layers
        layers.append(nn.Linear(text_embedding_dim, hidden_dim))
        if dropout > 0:  # If dropout rate is greater than 0
            layers.append(nn.Dropout(p=dropout))  # Add Dropout layer to prevent overfitting
        for l in range(layer_num - 1):  # Loop to build specified number of hidden layers (layernum - 1 layers)
            layers.append(nn.Linear(hidden_dim, hidden_dim))  # Add linear layer (fully connected layer)
            # layers.append(nn.ReLU(inplace=True))  # Add ReLU activation function, inplace=True means directly modify input data to save memory
            layers.append(nn.ReLU())  # Add ReLU activation function, inplace=True means directly modify input data to save memory
            if dropout > 0:  # If dropout rate is greater than 0
                layers.append(nn.Dropout(p=dropout))  # Add Dropout layer after activation function
        self.layers = nn.Sequential(*layers ) # Combine all layers into a sequential model
        self.feat = nn.Linear(hidden_dim, text_embedding_dim)  # Define a linear layer to map hidden layer output to backdoor feature dimensions

        # self.projection = nn.Linear(text_embedding_dim, text_embedding_dim)
        # Weight for combining poisoned and neighbor embeddings
        self.similarity_weight = nn.Parameter(torch.tensor(0.9))

        
    def forward(self, text_embedding_dim, original_texts, neighbor_text_embeddings, text_poison_mode="overwriting"):
        """
        """
        h = self.layers(text_embedding_dim)  # Input data passes through defined hidden layer sequence
        # poisoned_embeddings = self.feat(h)  # Hidden layer output passes through final linear layer to generate backdoor features

        # poisoned_embeddings生成
        poisoned_embeddings = self.feat(h)

        # Average neighbor embeddings
        avg_neighbor_embeddings = torch.stack([torch.mean(embs, dim=0) for embs in neighbor_text_embeddings])

        # Combine poisoned embeddings with neighbor embeddings
        fused_embeddings = self.similarity_weight * poisoned_embeddings + (1 - self.similarity_weight) * avg_neighbor_embeddings


        if text_poison_mode == "overwriting":
            overwriting_texts = self.text_autoencoder.get_texts(fused_embeddings.detach())
            poisoned_text_embeddings = self.text_autoencoder.get_embeds(overwriting_texts)

            return poisoned_text_embeddings, overwriting_texts  # Return generated backdoor features

        else: # appending
            appending_texts = self.text_autoencoder.get_texts(fused_embeddings.detach())
            combined_texts = [f"{orig} {append}" for orig, append in zip(original_texts, appending_texts)]
            poisoned_text_embeddings = self.text_autoencoder.get_embeds(combined_texts)

            return poisoned_text_embeddings, appending_texts


class TextBackdoor:
    def __init__(self, config: ExperimentConfig, text_autoencoder):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.text_autoencoder = text_autoencoder

        self.trojan = None
        self.shadow_model = None
        self.best_trojan_weights = None
        self.device = config.device
        self.text_poison_mode = config.text_poison_mode


    def _apply_defense_mode(self, poison_data, clean_data):
        self.logger.info(f"Defense mode: {self.config.defense_mode}")

        if self.config.defense_mode == 'prune':
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(f"Before pruning defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_prune(poison_data, self.config.prune_thr, self.device, large_graph=True)

            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(f"After pruning defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(f"Pruning defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")

        elif self.config.defense_mode == 'reconstruct':
            original_num_edges = poison_data.train_edge_index.shape[1]
            original_num_nodes = poison_data.x.shape[0]
            self.logger.info(f"Before reconstruction defense - Nodes: {original_num_nodes}, Edges: {original_num_edges}")

            defensed_data = defense_reconstruct(poison_data, clean_data, self.config.rec_epochs, self.device)

            defended_num_edges = defensed_data.edge_index.shape[1]
            edge_reduction_ratio = (original_num_edges - defended_num_edges) / original_num_edges * 100
            self.logger.info(f"After reconstruction defense - Nodes: {defensed_data.x.shape[0]}, Edges: {defended_num_edges}")
            self.logger.info(f"Reconstruction defense effect - Edge reduction: {edge_reduction_ratio:.2f}% ({original_num_edges} -> {defended_num_edges})")
        else:
            defensed_data = poison_data

        return defensed_data

    def _get_neighbor_embeddings(self, selected_indices, edge_index, node_embeddings):
        """
        为selected_indices中的每个节点计算其邻居节点的embeddings
        
        Args:
            selected_indices: 目标节点索引 [num_selected_nodes]
            edge_index: 图的边索引 [2, num_edges] 
            node_embeddings: 所有节点的embeddings [num_nodes, embedding_dim]
        
        Returns:
            neighbor_embeddings: 每个目标节点的邻居embeddings列表
        """
        neighbor_embeddings = []
        
        for node_idx in selected_indices:
            # 找到以node_idx为源节点或目标节点的所有边
            # 方法1：找到所有与该节点相连的邻居
            neighbors_mask_out = (edge_index[0] == node_idx)  # 出边
            neighbors_mask_in = (edge_index[1] == node_idx)   # 入边
            
            # 获取邻居节点索引
            neighbor_indices_out = edge_index[1][neighbors_mask_out]  # 出边的目标节点
            neighbor_indices_in = edge_index[0][neighbors_mask_in]    # 入边的源节点
            
            # 合并所有邻居（去重）
            all_neighbors = torch.cat([neighbor_indices_out, neighbor_indices_in])
            unique_neighbors = torch.unique(all_neighbors)
            
            # 移除自己（如果存在自环）
            unique_neighbors = unique_neighbors[unique_neighbors != node_idx]
            
            if len(unique_neighbors) > 0:
                # 获取邻居节点的embeddings
                neighbor_embs = node_embeddings[unique_neighbors]
                neighbor_embeddings.append(neighbor_embs)
            else:
                # 如果没有邻居，使用节点自身的embedding
                neighbor_embeddings.append(node_embeddings[node_idx:node_idx+1])
        
        return neighbor_embeddings





    def _train_models(self, data, poison_candidates, poison_mask):
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

        _, embeddings = self.shadow_model(data.x, data.train_edge_index, return_embeddings=True)
        
        # Initialize trojan network
        feature_dim = data.x.shape[1]  # Text embedding dimension
        node_embedding_dim = embeddings.shape[1]  # Node embedding dimension
        
        print(f"===node_embedding_dim: {node_embedding_dim}")
        print(f"===text_embedding_dim: {feature_dim}")

        self.trojan = Trojan(
            text_embedding_dim=feature_dim,
            text_autoencoder=self.text_autoencoder,
            hidden_dim=self.config.hidden_dim,
        ).to(self.device)

        # Initialize optimizers
        shadow_optimizer = optim.Adam(self.shadow_model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        trojan_optimizer = optim.Adam(self.trojan.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # 早停机制
        early_stopping_patience = 60
        no_improve_count = 0

        # Training masks
        train_mask = data.train_mask
        train_poison_mask = train_mask.clone()

        poison_labels = data.y.clone()
        for candidate in poison_candidates:
            poison_labels[candidate['node_idx']] = candidate['target_label']
            train_poison_mask[candidate['node_idx']] = True
        original_labels = data.y.clone()

        loss_fn = nn.CrossEntropyLoss()
        best_loss = float('inf')
        
        # Main training loop
        for epoch_trojan in range(self.config.trojan_epochs):
            # Phase 1: Train shadow model
            self.shadow_model.train()
            self.trojan.eval()
            shadow_optimizer.zero_grad()

            # Get embeddings from shadow model
            # with no_grad():
            #     _, embed = self.shadow_model(data.x, data.train_edge_index, return_embeddings=True)

            # Generate poisoned embeddings for poison nodes
            original_texts=[data.raw_texts[idx.item()] for idx in poison_indices]
            # node_embeddings = embed[poison_mask]

            outer_original_neighbor_text_embeddings = self._get_neighbor_embeddings(poison_indices, data.train_edge_index, data.x)
            poisoned_text_embeddings, _ = self.trojan(data.x[poison_mask], original_texts, outer_original_neighbor_text_embeddings, text_poison_mode=self.text_poison_mode)
            
            # Create poisoned feature matrix
            poison_x = data.x.clone().detach()
            poison_x[poison_mask] = poisoned_text_embeddings.detach()

            # Forward pass through shadow model
            output = self.shadow_model(poison_x, data.train_edge_index)

            # Compute loss on poison nodes
            shadow_loss = loss_fn(output[train_poison_mask], poison_labels[train_poison_mask])
            shadow_loss.backward()
            shadow_optimizer.step()

            
            # Phase 2: Train trojan network
            self.shadow_model.eval()
            self.trojan.train()
            trojan_optimizer.zero_grad()

            # Random choose unlabeled nodes to poison
            unlabeled_mask = data.unlabeled_mask
            rest_unlabeled_mask = unlabeled_mask.clone()
            rest_unlabeled_mask = rest_unlabeled_mask & (~poison_mask.to(self.device))
            rest_unlabeled_indices = torch.where(rest_unlabeled_mask)[0]
            
            # Randomly select outer_poison_num nodes
            outer_poison_num = self.config.outer_poison_num
            if len(rest_unlabeled_indices) >= outer_poison_num:
                selected_indices = rest_unlabeled_indices[torch.randperm(len(rest_unlabeled_indices))[:outer_poison_num]]
            else:
                selected_indices = rest_unlabeled_indices
            
            # Create outer poison mask
            outer_poison_mask = torch.zeros_like(unlabeled_mask, dtype=torch.bool)
            outer_poison_mask[selected_indices] = True

            # Get embeddings from shadow model
            _, embeddings = self.shadow_model(poison_x, data.train_edge_index, return_embeddings=True)
            
            # Generate poisoned embeddings for outer poison nodes
            outer_original_texts =[ data.raw_texts[idx.item()] for idx in selected_indices]
            outer_node_embeddings = embeddings[outer_poison_mask]

            outer_original_neighbor_text_embeddings = self._get_neighbor_embeddings(selected_indices, data.train_edge_index, data.x)
            
            outer_poisoned_embeddings,_ = self.trojan(outer_node_embeddings, outer_original_texts, outer_original_neighbor_text_embeddings, text_poison_mode=self.text_poison_mode)
            
            # Create modified features
            outer_poison_x = data.x.detach().clone()
            outer_poison_x[outer_poison_mask] = outer_poisoned_embeddings.detach()

            # Forward pass with modified features
            outer_output = self.shadow_model(outer_poison_x, data.train_edge_index)
            
            # Target loss: poison nodes should be classified as target class
            target_loss = loss_fn(outer_output[train_poison_mask], poison_labels[train_poison_mask])

            # Similarity loss
            similarity_loss = 1 - F.cosine_similarity(
                data.x[outer_poison_mask],
                outer_poison_x[outer_poison_mask],
                dim=1
            ).mean()


            # Combined trojan loss
            trojan_loss = (target_loss 
                           + self.config.feature_similarity_weight * similarity_loss
                        )
            
            trojan_loss.backward()
            trojan_optimizer.step()
            
            # Save best model
            if trojan_loss.item() < best_loss:
                best_loss = trojan_loss.item()
                no_improve_count = 0
                self.best_trojan_weights = deepcopy(self.trojan.state_dict())
            else:
                no_improve_count += 1
                
            if no_improve_count >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch_trojan}")
                break
            
            # Logging
            if epoch_trojan % 20 == 0:
                with torch.no_grad():
                    self.shadow_model.eval()
                    pred = output.argmax(dim=1)
                    train_acc = (pred[train_mask] == original_labels[train_mask]).float().mean()
                    poison_acc = (pred[poison_mask] == poison_labels[poison_mask]).float().mean()
                    val_acc = (pred[data.val_mask] == original_labels[data.val_mask]).float().mean()
                    
                    print(f"Trojan epoch {epoch_trojan}: Shadow Loss: {shadow_loss.item():.4f}, "
                          f"Trojan Loss: {trojan_loss.item():.4f}, "
                          f"Similarity Loss: {similarity_loss.item():.4f}, "
                          f"Train Acc: {train_acc:.4f}, "
                          f"Val Acc: {val_acc:.4f}, "
                          f"Poison Acc: {poison_acc:.4f}")
        
        # Load best trojan weights
        if self.best_trojan_weights is not None:
            self.trojan.load_state_dict(self.best_trojan_weights)

        self.logger.info("Triggers learned for all poison candidates")
        return poison_labels, poison_mask, train_poison_mask

    def _create_poisoned_dataset(self, data, poison_labels, poison_mask, train_poison_mask):
        # Create poisoned dataset
        original_x = data.x.detach()
        poison_indices = torch.where(poison_mask)[0]

        self.trojan.eval()
        self.shadow_model.eval()
        with torch.no_grad():
            _, embed = self.shadow_model(original_x, data.train_edge_index, return_embeddings=True)
            
        # Generate poisoned embeddings
        original_texts = [data.raw_texts[idx.item()] for idx in poison_indices]
        node_embeddings = embed[poison_indices]

        original_neighbor_text_embeddings = self._get_neighbor_embeddings(poison_indices, data.train_edge_index, data.x)

        if self.text_poison_mode=="overwriting":
            poisoned_text_embeddings, overwriting_texts = self.trojan(node_embeddings, original_texts, original_neighbor_text_embeddings,text_poison_mode=self.text_poison_mode)
            # poisoned_texts = self.text_autoencoder.get_texts(poisoned_text_embeddings, max_seq_len=128)

            # update raw_texts
            new_raw_texts = data.raw_texts.copy()
            for idx, node_idx in enumerate(poison_indices):
                new_raw_texts[node_idx] = overwriting_texts[idx]
                if idx<5:
                    print(f"===idx:{idx}====")
                    print(f"===original_text:{original_texts[idx]}")
                    print(f"===poisoned_text:{overwriting_texts[idx]}")
        else: # "appending"
            poisoned_text_embeddings, appending_texts = self.trojan(node_embeddings, original_texts,original_neighbor_text_embeddings, text_poison_mode=self.text_poison_mode)
            print(f"==appending_texts:{appending_texts}")

            poisoned_texts = [f"{orig}+{append}" for orig, append in zip(original_texts, appending_texts)]

            # update raw_texts
            new_raw_texts = data.raw_texts.copy()
            for idx, node_idx in enumerate(poison_indices):
                new_raw_texts[node_idx] = poisoned_texts[idx]
                if idx<5:
                    print(f"===idx:{idx}====")
                    print(f"===poisoned_text:{poisoned_texts[idx]}")

        # # 重新编码组合文本
        # new_poison_x = self.text_autoencoder.get_embeds(poisoned_texts)
        new_x = data.x.clone().detach()
        new_x[poison_mask] = poisoned_text_embeddings.detach()

        # Create poisoned data
        poisoned_data = Data(
            x=new_x,
            y=poison_labels,
            raw_texts=new_raw_texts,
            train_poison_mask=train_poison_mask,
            train_mask=data.train_mask,
            poison_mask=poison_mask,
            val_mask=data.val_mask,
            test_mask=data.test_mask,
            edge_index=data.edge_index,
            train_edge_index=data.train_edge_index,
            test_edge_index=data.test_edge_index,
            num_classes=data.num_classes,
        ).to(self.device)

        poisoned_data = self._apply_defense_mode(poisoned_data, data)
        return poisoned_data

    def _apply_pruning_to_test_data(self, test_data):
        edge_index = test_data.test_edge_index
        x = test_data.x

        if hasattr(test_data, 'edge_weight') and test_data.edge_weight is not None:
            edge_weights = test_data.edge_weight
        else:
            edge_weights = torch.ones(edge_index.shape[1])

        valid_edge_mask = edge_weights > 0.0
        edge_index = edge_index[:, valid_edge_mask].to(self.device)
        edge_weights = edge_weights[valid_edge_mask].to(self.device)
        x = x.to(self.device)

        edge_sims = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])
        prune_threshold = self.config.prune_thr
        keep_mask = edge_sims > prune_threshold
        updated_edge_index = edge_index[:, keep_mask]
        updated_edge_weights = edge_weights[keep_mask]

        test_data.test_edge_index = updated_edge_index
        test_data.edge_weight = updated_edge_weights

        self.logger.info(f"Applied pruning to test data. Edges reduced from {edge_index.shape[1]} to {updated_edge_index.shape[1]}")
        return test_data

    def _add_stealth_mechanisms(self, poisoned_data, clean_data):
        """添加隐蔽性机制"""
        poison_features = poisoned_data.x[poisoned_data.poison_mask]
        clean_features = clean_data.x[clean_data.train_mask]
        
        poison_mean = poison_features.mean(dim=0)
        clean_mean = clean_features.mean(dim=0)
        
        adjustment = (clean_mean - poison_mean) * 0.1
        poisoned_data.x[poisoned_data.poison_mask] += adjustment
        
        noise = torch.randn_like(poison_features) * 0.01
        poisoned_data.x[poisoned_data.poison_mask] += noise
        
        return poisoned_data

    def _get_poisoned_test_data(self, poison_train_data, target_model):
        """Generate poisoned test data by injecting backdoor triggers"""
        self.logger.info("Generating poisoned test data ...")

        test_mask = poison_train_data.test_mask
        test_node_indices = torch.where(test_mask)[0]

        with torch.no_grad():
            output = target_model(poison_train_data.x, poison_train_data.test_edge_index)
        probs = F.softmax(output, dim=1)

        uncertainty_method = self.config.uncertainty_method
        if uncertainty_method == "entropy":
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        else: # "variance":
            uncertainty = torch.var(probs, dim=1)


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

        print(f"test_nodes: {len(test_node_indices)}")
        print(f"test_poison_nodes: {len(test_poison_nodes)}")
        print(f"test_clean_nodes: {len(test_clean_nodes)}")

        # Set target labels for test poison nodes
        poison_y = poison_train_data.y.clone()
        test_poison_node_indices = torch.where(test_poison_mask)[0]
        target_assignment = self.config.target_assignment
        num_classes = poison_train_data.num_classes

        for node_idx in test_poison_node_indices:
            true_label = poison_train_data.y[node_idx].item()

            if target_assignment == "fixed":
                target_label = self.config.fixed_target_class
            else:
                possible_targets = [i for i in range(num_classes) if i != true_label]
                target_label = np.random.choice(possible_targets)
            poison_y[node_idx] = target_label

        # Inject triggers for test nodes
        with torch.no_grad():
            _, embed = target_model(poison_train_data.x, poison_train_data.test_edge_index, return_embeddings=True)
        
        # Generate poisoned embeddings for test poison nodes
        original_texts =[poison_train_data.raw_texts[i.item()] for i in test_poison_node_indices]
        node_embeddings = embed[test_poison_mask]

        neighbor_text_embeddings = self._get_neighbor_embeddings(test_poison_node_indices, poison_train_data.test_edge_index, poison_train_data.x)

        if self.text_poison_mode == "overwriting":
            poisoned_text_embeddings, overwriting_texts = self.trojan(node_embeddings, original_texts,neighbor_text_embeddings, text_poison_mode=self.text_poison_mode)

            # Create poisoned feature matrix
            poison_x = poison_train_data.x.detach().clone()
            poison_x[test_poison_mask] = poisoned_text_embeddings.detach()

            new_raw_texts = poison_train_data.raw_texts.copy()
            for idx, text in enumerate(overwriting_texts):
                new_raw_texts[test_poison_node_indices[idx]] = text
        else: # "appending":
            poisoned_text_embeddings, appending_texts = self.trojan(node_embeddings, original_texts, neighbor_text_embeddings, text_poison_mode=self.text_poison_mode)


            # Create poisoned feature matrix
            poison_x = poison_train_data.x.detach().clone()
            poison_x[test_poison_mask] = poisoned_text_embeddings.detach()

            combined_poisoned_texts=[f"{orig} {append}" for orig, append in zip(original_texts, appending_texts)]
            new_raw_texts = poison_train_data.raw_texts.copy()
            for idx, text in enumerate(combined_poisoned_texts):
                new_raw_texts[test_poison_node_indices[idx]]=text


        poisoned_data = Data(
            x=poison_x,
            y=poison_y,
            raw_texts=new_raw_texts,
            train_mask=poison_train_data.train_mask,
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
        """Evaluate clean accuracy on unmodified test data"""
        test_mask = data.test_clean_mask
        self.logger.info(f"Evaluating CA: {torch.sum(test_mask).item()}...")

        data.to(self.device)
        with torch.no_grad():
            logits = target_model(data.x, data.test_edge_index)

        predict_labels = logits.argmax(dim=1)[test_mask]
        true_labels = data.y[test_mask]

        accuracy = accuracy_score(true_labels.cpu().numpy(), predict_labels.cpu().numpy())

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
        """Evaluate attack success rate"""
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

    def _evaluate_text_reconstruction_quality(self, poisoned_test_data, original_data):
        """Evaluate the quality of text reconstruction for poisoned nodes"""
        poison_mask = poisoned_test_data.test_poison_mask
        poison_indices = torch.where(poison_mask)[0].tolist()

        original_texts = [poisoned_test_data.raw_texts[i] for i in poison_indices]
        original_text_embeddings=[original_data.raw_texts[i] for i in poison_indices]

        poisoned_text_embeddings = poisoned_test_data.x[poison_mask]
        poisoned_texts = [poisoned_test_data.raw_texts[i] for i in poison_indices]

        original_lengths = [len(text.split()) for text in original_texts]
        poisoned_lengths = [len(text.split()) for text in poisoned_texts]
        length_change = np.array(poisoned_lengths) - np.array(original_lengths)

        similarity = F.cosine_similarity(original_text_embeddings, poisoned_text_embeddings, dim=1)
        all_similarity_score = similarity.mean().item()

        results = []
        for i in range(len(poison_indices)):
            result_info = {
                'length_change': length_change[i],
                'similarity_score': similarity[i].item(),
                'original_text': original_texts[i],
                'poisoned_text': poisoned_texts[i],
            }
            results.append(result_info)

        return all_similarity_score, results

    def _train_target_model(self, data, model_name):
        data = data.to(self.device)

        model = create_model(model_name,
                           input_dim=data.x.shape[1],
                           hidden_dim=self.config.hidden_dim,
                           output_dim=data.num_classes).to(self.device)

        epochs = self.config.target_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.target_lr,
                                   weight_decay=self.config.target_weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        train_mask = data.train_mask
        poison_mask = data.poison_mask
        train_poison_mask = data.train_poison_mask
        val_mask = data.val_mask

        patience = self.config.patience
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            out = model(data.x, data.train_edge_index)
            loss = criterion(out[train_poison_mask], data.y[train_poison_mask])
            train_loss = loss.item()

            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                eval_out = model(data.x, data.train_edge_index)

                if val_mask is not None and val_mask.sum() > 0:
                    val_acc = (eval_out[val_mask].argmax(dim=1) == data.y[val_mask]).float().mean().item()
                else:
                    val_acc = (eval_out[train_mask].argmax(dim=1) == data.y[train_mask]).float().mean().item()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if (epoch + 1) % 50 == 0:
                train_acc = (out[train_poison_mask].argmax(dim=1) == data.y[train_poison_mask]).float().sum() / train_poison_mask.sum()
                ac_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().sum() / train_mask.sum()
                asr_acc = (out[poison_mask].argmax(dim=1) == data.y[poison_mask]).float().sum() / poison_mask.sum()

                print(f"Poisoned data on target model - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                      f"Train Acc: {train_acc.item():.4f}, AC Acc: {ac_acc.item():.4f}, ASR Acc: {asr_acc.item():.4f}, "
                      f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")

            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model weights with validation accuracy: {best_val_acc:.4f}")

        return model

    def run_trigger_generation_pipeline(self, data, poison_candidates, poison_mask, sorted_dims):
        """Run the complete trigger generation pipeline"""
        self.logger.info("Starting trigger generation pipeline ...")
        
        # 1: Learn triggers by training shadow model and trojan network
        poison_labels, poison_mask, train_poison_mask = self._train_models(data, poison_candidates, poison_mask)

        poisoned_data = self._create_poisoned_dataset(data, poison_labels, poison_mask, train_poison_mask)

        evaluation_results={}
        for model_name in self.config.target_models:
            self.logger.info(f" Training target model: {model_name}")

            # Train target models on poisoned training data
            poison_target_model = self._train_target_model(poisoned_data, model_name)
            poison_target_model.eval()

            # Generate poisoned test data for attack
            poisoned_test_data = self._get_poisoned_test_data(poisoned_data, poison_target_model)

            # Evaluate clean accuracy using test_clean_mask
            clean_accuracy_results = self._evaluate_clean_accuracy(poison_target_model, poisoned_test_data)

            # Evaluate attack success rate using test_poison_mask
            attack_success_results = self._evaluate_attack_success_rate(poison_target_model, poisoned_test_data)

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

    

                






