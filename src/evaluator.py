"""Evaluation module for TAG backdoor attack experiments."""

import os
import json
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import accuracy_score
import logging

from torch_geometric.data import Data

from src.models import create_model, ModelTrainer
from config import ExperimentConfig

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class BackdoorEvaluator:
    """Evaluator for backdoor attack effectiveness and model performance."""
    
    def __init__(self, config: ExperimentConfig, trojan, shadow_model, text_autoencoder):
        self.logger = logging.getLogger(__name__)
        self.device=config.device

        self.config = config
        self.trojan=trojan
        self.shadow_model = shadow_model
        self.text_autoencoder = text_autoencoder

        # Storage for evaluation results
        self.evaluation_results = {}
        self.trojan.eval()  # Set trojan network to evaluation mode
        self.shadow_model.eval()

    

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

        self.logger.info(f"Applied pruning to test data. Edges reduced from {edge_index.shape[1]} to {updated_edge_index.shape[1]}")
        return test_data


    def _get_poisoned_test_data(self, poison_train_data, target_model, poison_dim_indices):
        """Generate poisoned data by injecting backdoor triggers
            set test_poison_mask test_clean_mask
            set target
            set poison_x
        """
        self.logger.info("Generating poisoned test data ...")

        test_mask=poison_train_data.test_mask
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
        poison_y=poison_train_data.y.clone()
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
        poison_x[test_poison_node_indices[:,None],poison_dim_indices] = trojan_feat.detach()


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
        test_mask=data.test_poison_mask
        x=data.x.detach()
        self.logger.info(f"Evaluating ASR:  {torch.sum(test_mask).item()}...")

        with torch.no_grad():
            logits = target_model(x, data.test_edge_index)
        predictions = logits.argmax(dim=1)
        targets = data.y
        predictions=predictions.to(targets.device)

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
        modified_features= poisoned_test_data.x[poison_mask]

        self.logger.info(f"Reconstructing {len(modified_features)} texts ")

        # Batch processing for text reconstruction
        reconstructed_texts = self.text_autoencoder.get_texts(modified_features)

        # Simple proxy: compute average word length change
        original_lengths = [len(text.split()) for text in original_texts]
        poisoned_lengths = [len(text.split()) for text in reconstructed_texts]
        length_change =np.array(poisoned_lengths) - np.array(original_lengths)


        # Compute text similarity
        original_embeddings = poisoned_test_data.x[poison_mask]
        reconstructed_embeddings = self.text_autoencoder.get_embeds(reconstructed_texts)

        similarity = F.cosine_similarity(original_embeddings, reconstructed_embeddings, dim=1)
        all_similarity_score = similarity.mean().item()

        results = []
        for i in range(len(poison_indices)):
            result_info = {
                'length_change':length_change[i],
                'similarity_score': similarity[i].item(),
                'original_text': original_texts[i],
                'poisoned_text': reconstructed_texts[i],
            }
            results.append(result_info)

        return all_similarity_score, results


    def _train_target_model(self, data, model_name):

        data=data.to(self.device)

        # Create target model
        model = create_model(model_name,
                             input_dim=data.x.shape[1],
                             hidden_dim=512,
                             output_dim=data.num_classes).to(self.device)

        # Set up training parameters
        epochs = self.config.target_epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.target_lr, weight_decay=self.config.target_weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        # train_mask=data.except_test_mask # 除了test的节点，包含train+poison+val
        train_mask=data.train_mask
        poison_mask=data.poison_mask
        train_poison_mask=data.train_poison_mask # 包含了train+poison
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
            train_loss=loss.item()

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
                train_acc = (out[train_poison_mask].argmax(dim=1) == data.y[train_poison_mask]).float().sum() / train_poison_mask.sum()
                ac_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().sum() / train_mask.sum()
                asr_acc = (out[poison_mask].argmax(dim=1) == data.y[poison_mask]).float().sum() / poison_mask.sum()

                print(f" Poisoned data on target model - Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc:{train_acc.item():.4f}, AC Acc:{ac_acc.item():.4f}, ASR Acc:{asr_acc.item():.4f}, Val Acc:{val_acc:.4f}, Best Val Acc:{best_val_acc:.4f}")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                break
        
        # 加载最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Loaded best model weights with validation accuracy: {best_val_acc:.4f}")

        return model

    def run_evaluation_pipeline(self, poisoned_data,poison_dim_indices) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        """
        self.logger.info("Running evaluation pipeline...")


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
            all_similarity_score, each_text_results = self._evaluate_text_reconstruction_quality(poisoned_test_data)

            # Compile comprehensive results
            evaluation_result = {
                'model_name': model_name,
                'clean_accuracy': clean_accuracy_results,
                'attack_success': attack_success_results,
                'text_similarity': all_similarity_score,
                'text_details': each_text_results,
                'experiment_config': {
                    'poison_num': self.config.poison_node_num,
                    'dataset_name': self.config.dataset_name,
                }
            }

            # Log summary
            self.logger.info(f"Evaluation summary for {model_name}:")
            self.logger.info(f"  Clean Accuracy: {clean_accuracy_results['overall_accuracy']:.4f}")
            self.logger.info(f"  Attack Success Rate: {attack_success_results['attack_success_rate']:.4f}")
            self.logger.info(f"  Text Similarity: {all_similarity_score:.4f}")

            self.evaluation_results[model_name] = evaluation_result

        return self.evaluation_results