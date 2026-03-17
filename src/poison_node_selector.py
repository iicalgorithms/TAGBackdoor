"""Poison node selection module for TAG backdoor attack."""

import os
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
import logging
import shap
from sklearn.ensemble import RandomForestClassifier
from torch import Tensor

from src.models import create_model, ModelTrainer
from config import ExperimentConfig

class PoisonNodeSelector:
    """Selector for poison nodes based on uncertainty and coverage."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Storage for surrogate models and predictions
        self.trained_surrogate_model = None
        self.model_predictions = {}

    def _load_candidates_and_dims(self):
        """Load poison candidates and feature dimensions from files if they exist.
        
        Returns:
            Tuple[Optional[List], Optional[torch.Tensor], str, str]: 
                candidates, sorted_dims, node_filepath, dims_filepath
        """
        output_dir = os.path.join('saved', 'poison_nodes')
        node_filename = f"{self.config.dataset_name}_{self.config.surrogate_model}_{self.config.target_assignment}_{self.config.poison_node_num}.json"
        node_filepath = os.path.join(output_dir, node_filename)

        dims_filename = f"{self.config.dataset_name}_{self.config.surrogate_model}_{self.config.dims_importance_method}_dims.pt"
        dims_filepath = os.path.join(output_dir, dims_filename)

        candidates = None
        sorted_dims = None
        
        # Try to load candidates file independently
        if os.path.exists(node_filepath):
            with open(node_filepath, 'r', encoding='utf-8') as f1:
                saved_data = json.load(f1)
                candidates = saved_data.get('candidates', [])
            self.logger.info(f"Successfully loaded {len(candidates)} candidates from cache")

        # Try to load dimensions file independently
        if os.path.exists(dims_filepath):
            sorted_dims = torch.load(dims_filepath, map_location='cpu', weights_only=False)
            self.logger.info(f"Successfully loaded feature dimensions from cache")

        return candidates, sorted_dims, node_filepath, dims_filepath

    def _train_surrogate_model(self, data):
        """Train a surrogate model for uncertainty estimation.
        Args:
            data: Graph data object
            
        Returns:
            Trained model
        """
        model_name = self.config.surrogate_model
        self.logger.info(f"Training surrogate model: {model_name}")

        train_mask = data.train_mask
        val_mask = data.val_mask
        unlabeled_mask = data.unlabeled_mask
        
        # Create model
        model = create_model(
            model_name=model_name,
            input_dim=data.x.shape[1],
            hidden_dim=512,
            output_dim=len(torch.unique(data.y)),
        )
        
        # Create trainer
        trainer = ModelTrainer(model, device=self.config.device)
        
        # Setup training
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = trainer.train_epoch(data, optimizer, criterion, train_mask)

            # Validate
            if epoch % 40 == 0:
                train_acc = trainer.evaluate(data, train_mask)
                val_acc = trainer.evaluate(data, val_mask)
                
                print(f" Surrogate Epoch {epoch}: Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model state
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        self.logger.info(f" Early stopping at epoch {epoch}")
                        break
        
        # Load best model
        if 'best_state' in locals():
            model.load_state_dict(best_state)
        
        self.trained_surrogate_model = trainer
        
        # Final evaluation
        final_train_acc = trainer.evaluate(data, train_mask)
        final_val_acc = trainer.evaluate(data, val_mask)
        
        self.logger.info(f" Trained surrogate model: {model_name} - Train Acc: {final_train_acc:.4f}, Val Acc: {final_val_acc:.4f}")
        
        return model


    def _calculate_feature_importance(self, data, model):
        """
        :param data:
            data.x为输入数据的embedding，
            data.y为输入数据的label
        :return:

            Tensor: embedding中每个维度的特征重要性排序列表
        """
        method = self.config.dims_importance_method
        device = self.config.device

        if method == "random_forest":
            # 计算 embedding 各维度的重要性，并按降序返回维度索引
            x_np = data.x.cpu().detach().numpy()
            y_np = data.y.cpu().detach().numpy()

            rf = RandomForestClassifier(n_estimators=100, random_state=0)
            rf.fit(x_np, y_np)

            # 获取特征重要性（每个维度一个值）
            importances = rf.feature_importances_
            importance_tensor = torch.tensor(importances)
            sorted_dims = torch.argsort(importance_tensor, descending=True)  # 排序索引（从最重要到最不重要）

        elif method=="gradient-based":
            # 基于梯度方法计算 GCN 模型中各个输入维度的特征
            model = model.to(device).eval()
            x = data.x.clone().detach().to(device)
            x.requires_grad = True  # 使其可求梯度
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)

            out = model(x, edge_index)
            loss = F.cross_entropy(out[data.train_mask], y[data.train_mask])

            model.zero_grad()
            loss.backward()

            # 取每个特征维度的平均梯度绝对值作为重要性指标
            grad_importance = x.grad.abs().mean(dim=0)  # [num_features]
            sorted_dims = torch.argsort(grad_importance, descending=True)  # 降序排序

        elif method=="shapley":
            #使用 SHAP 方法评估 embedding 维度的重要性
            x = data.x.detach().cpu().numpy()
            y = data.y.detach().cpu().numpy()

            num_samples = min(data.x.shape[0], 10000)

            # 随机选取样本
            x_sample = x[:num_samples]

            def model_fn(x_numpy):
                with torch.no_grad():
                    x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
                    return model(x_tensor).detach().cpu().numpy()

            explainer = shap.Explainer(model_fn, x_sample)
            shap_values = explainer(x_sample)

            # 计算每个维度的平均 SHAP 重要性
            importance = np.abs(shap_values.values).mean(axis=0)
            sorted_dims = torch.tensor(np.argsort(-importance))  # 负号表示降序

        else: #statistics

            x = data.x.to(device)
            variance_scores = torch.var(x, dim=0) # 1: Variance-based importance
            mean_abs_scores = torch.mean(torch.abs(x), dim=0) # 2 Mean absolute value importance)

            # 方法3: 基于范围的重要性 (Range-based importance)
            # 数值范围越大的维度通常包含更多变化信息
            range_scores = torch.max(x, dim=0)[0] - torch.min(x, dim=0)[0]

            # 方法4: 基于标准差的重要性 (Standard deviation importance)
            std_scores = torch.std(x, dim=0)

            # 方法5: 基于非零元素比例的重要性 (Non-zero ratio importance)
            # 非零元素越多的维度通常更活跃
            non_zero_ratio = torch.sum(x != 0, dim=0).float() / x.shape[0]

            # 标准化所有分数到[0,1]范围
            def normalize_scores(scores):
                min_val = torch.min(scores)
                max_val = torch.max(scores)
                if max_val - min_val > 1e-8:  # 避免除零
                    return (scores - min_val) / (max_val - min_val)
                else:
                    return torch.ones_like(scores) / len(scores)

            # 标准化各个分数
            variance_scores_norm = normalize_scores(variance_scores)
            mean_abs_scores_norm = normalize_scores(mean_abs_scores)
            range_scores_norm = normalize_scores(range_scores)
            std_scores_norm = normalize_scores(std_scores)
            non_zero_ratio_norm = normalize_scores(non_zero_ratio)

            # 组合多种重要性指标 (权重可以根据具体任务调整)
            combined_scores = (
                    0.3 * variance_scores_norm +  # 方差权重
                    0.2 * mean_abs_scores_norm +  # 绝对值均值权重
                    0.2 * range_scores_norm +  # 范围权重
                    0.2 * std_scores_norm +  # 标准差权重
                    0.1 * non_zero_ratio_norm  # 非零比例权重
            )

            # 按重要性从大到小排序，返回维度索引
            sorted_dims = torch.argsort(combined_scores, descending=True)

        self.logger.info(f"=={method} most 20 important dimensions : {sorted_dims[:20].tolist()}")

        return sorted_dims


    def _compute_uncertainty_scores(self, data) -> Dict[str, torch.Tensor|Any]:
        """Compute uncertainty scores for unlabeled nodes.
        
        Args:
            data: Graph data object
            
        Returns:
            Uncertainty scores for unlabeled nodes
        """
        trainer = self.trained_surrogate_model

        # Get predictions
        logits, probs = trainer.get_predictions(data, return_probs=True)
        
        # Store predictions
        self.model_predictions = {
            'logits': logits,
            'probs': probs,
            'pred_labels': logits.argmax(dim=1)
        }
        
        # Compute uncertainty based on method
        if self.config.uncertainty_method == "entropy":
            # Entropy-based uncertainty
            uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        elif self.config.uncertainty_method == "variance":
            # Variance-based uncertainty
            uncertainty = torch.var(probs, dim=1)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.config.uncertainty_method}")
        
        # Only consider unlabeled nodes
        unlabeled_uncertainty = uncertainty[data.unlabeled_mask]
        
        uncertainty_info = {
            'all_scores': uncertainty,
            'unlabeled_scores': unlabeled_uncertainty,
            'unlabeled_indices': torch.where(data.unlabeled_mask)[0]
        }
        
        return uncertainty_info
    
    def _select_poison_candidates(self, data) -> List[Dict[str, Any]]:
        """Select poison node candidates based on uncertainty and coverage.
        
        Args:
            data: Graph data object
        Returns:
            List of poison candidate information
        """
        num_poison = self.config.poison_node_num

        self.logger.info(f" Selecting poison num {num_poison}")

        if self.config.poison_node_mode == "random":
            # Random selection of poison nodes
            unlabeled_indices = torch.where(data.unlabeled_mask)[0]
            if len(unlabeled_indices) < num_poison:
                raise ValueError(f"Not enough unlabeled nodes to select {num_poison} poison nodes")
            selected_indices = torch.randperm(len(unlabeled_indices))[:num_poison]
            selected_candidates = [{'node_idx': unlabeled_indices[i].item(), 'true_label': data.y[i].item()} for i in selected_indices]
            self.logger.info(f"Selected {len(selected_candidates)} random poison candidates")
            return selected_candidates

        # Coverage-based selection of poison nodes

        # Get uncertainty scores
        uncertainty_info=self._compute_uncertainty_scores(data)
        unlabeled_indices = uncertainty_info['unlabeled_indices']
        uncertainty_scores = uncertainty_info['unlabeled_scores']
        
        # Sort by uncertainty (descending)
        sorted_indices = torch.argsort(uncertainty_scores, descending=True)

        # Get predictions for unlabeled nodes
        predictions = self.model_predictions
        unlabeled_pred_labels = predictions['pred_labels'][unlabeled_indices]
        unlabeled_true_labels = data.y[unlabeled_indices]
        unlabeled_probs = predictions['probs'][unlabeled_indices]
        
        # Select candidates with class coverage consideration
        selected_candidates = []
        class_counts = defaultdict(int)



        for idx in sorted_indices:
            if len(selected_candidates) >= num_poison:
                break
            
            global_idx = unlabeled_indices[idx].item()
            true_label = unlabeled_true_labels[idx].item()
            pred_label = unlabeled_pred_labels[idx].item()
            uncertainty = uncertainty_scores[idx].item()
            prob_dist = unlabeled_probs[idx]
            
            # Check class coverage constraint
            if (len(class_counts) < self.config.min_class_coverage or 
                class_counts[true_label] < num_poison // self.config.min_class_coverage):
                
                candidate_info = {
                    'node_idx': global_idx,
                    'uncertainty_score': uncertainty,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'prediction_probs': prob_dist.tolist()
                }
                
                selected_candidates.append(candidate_info)
                class_counts[true_label] += 1
        
        # If we still need more candidates, fill from remaining high-uncertainty nodes
        remaining_indices = sorted_indices[len(selected_candidates):]
        for idx in remaining_indices:
            if len(selected_candidates) >= num_poison:
                break
            
            global_idx = unlabeled_indices[idx].item()
            true_label = unlabeled_true_labels[idx].item()
            pred_label = unlabeled_pred_labels[idx].item()
            uncertainty = uncertainty_scores[idx].item()
            prob_dist = unlabeled_probs[idx]
            
            candidate_info = {
                'node_idx': global_idx,
                'uncertainty_score': uncertainty,
                'true_label': true_label,
                'predicted_label': pred_label,
                'prediction_probs': prob_dist.tolist()
            }
            
            selected_candidates.append(candidate_info)

        
        self.logger.info(f"Selected {len(selected_candidates)} poison candidates")
        return selected_candidates

    def _assign_target_labels(self, poison_candidates: List[Dict[str, Any]], num_classes: int) -> List[Dict[str, Any]]:
        """Assign target labels to poison candidates.

        Args:
            poison_candidates: List of poison candidate information
            num_classes: Total number of classes

        Returns:
            Updated poison candidates with target labels
        """
        target_assignment = self.config.target_assignment
        self.logger.info(f"Assigning target label :{target_assignment}")

        assigned_candidates = []

        for candidate in poison_candidates:
            true_label = candidate['true_label']
            # pred_label = candidate['predicted_label']

            if target_assignment == "fixed":
                # Use fixed target class
                target_label = self.config.fixed_target_class
                if target_label is None:
                    target_label = (true_label + 1) % num_classes  # Default: next class

            elif target_assignment == "random":
                # Random target class (different from true label)
                possible_targets = [i for i in range(num_classes) if i != true_label]
                target_label = np.random.choice(possible_targets)

            elif target_assignment == "confusion_max":
                # Use class with the highest prediction probability (excluding true class)
                probs = np.array(candidate['prediction_probs'])
                probs[true_label] = 0  # Exclude true class
                target_label = np.argmax(probs)

            else:
                raise ValueError(f"Unknown target assignment: {self.config.target_assignment}")

            # Update candidate with assigned target label
            assigned_candidate = candidate.copy()
            assigned_candidate['target_label'] = int(target_label)
            assigned_candidates.append(assigned_candidate)



        self.logger.info(f"Assigned target labels nodes: {len(assigned_candidates)} ")

        return assigned_candidates



    
    def run_poison_selection_pipeline(self, data):
        """
        Run the complete poison node selection pipeline.
        Loads existing candidates and dimensions if available, generates missing ones.
        """
        self.logger.info("Starting poison node selection pipeline ...")

        # Load existing data if available
        assigned_candidates, sorted_dims, node_filepath, dims_filepath = self._load_candidates_and_dims()
        
        # Determine what needs to be generated
        need_candidates = assigned_candidates is None
        need_dims = sorted_dims is None
        need_surrogate_model = need_candidates or need_dims
        surrogate_model = None
        
        # Train surrogate model if needed for either candidates or dimensions
        if need_surrogate_model:
            surrogate_model = self._train_surrogate_model(data)


        # Generate feature importance if not loaded
        if need_dims:
            sorted_dims = self._calculate_feature_importance(data, surrogate_model)
            # Save dimensions immediately

            os.makedirs(os.path.dirname(dims_filepath), exist_ok=True)
            torch.save(sorted_dims, dims_filepath)
            self.logger.info(f"Saved feature dimensions to: {dims_filepath}")

        
        # Generate poison candidates if not loaded
        if need_candidates:
            self.logger.info("Selecting poison candidates...")
            candidates = self._select_poison_candidates(data)
            assigned_candidates = self._assign_target_labels(candidates, data.num_classes)
            
            # Save candidates immediately
            os.makedirs(os.path.dirname(node_filepath), exist_ok=True)
            save_data = {
                'dataset_name': self.config.dataset_name,
                'surrogate_model': self.config.surrogate_model,
                'target_assignment': self.config.target_assignment,
                'poison_num': self.config.poison_node_num,
                'uncertainty_method': self.config.uncertainty_method,
                'min_class_coverage': self.config.min_class_coverage,
                'num_candidates': len(assigned_candidates),
                'timestamp': datetime.now().isoformat(),
                'candidates': assigned_candidates
            }
            with open(node_filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(assigned_candidates)} poison candidates to: {node_filepath}")


        # Create poison mask
        poison_mask = torch.zeros(len(data.x), dtype=torch.bool)
        poison_indices = [c['node_idx'] for c in assigned_candidates]
        poison_mask[poison_indices] = True

        self.logger.info(f"Poison distribution: {dict(Counter(c['true_label'] for c in assigned_candidates))}")
        return assigned_candidates, poison_mask, sorted_dims
