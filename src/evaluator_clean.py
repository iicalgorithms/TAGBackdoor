"""Evaluation module for TAG backdoor attack experiments."""

import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
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


class CleanEvaluator:
    """Evaluator for backdoor attack effectiveness and model performance."""
    
    def __init__(self, config: ExperimentConfig):
        self.logger = logging.getLogger(__name__)
        self.device = config.device
        self.config = config
        self.num_runs = config.num_runs

    def _evaluate_accuracy(self, target_model, data):
        """
        Evaluate accuracy on clean data.
        """
        # Use test_mask to identify test nodes
        test_mask = data.test_mask

        self.logger.info(f"Evaluating clean data accuracy: {torch.sum(test_mask).item()}...")

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

        self.logger.info(f"Clean results: {results}")
        return results

    def _train_target_model(self, data, model_name, run_idx=0):
        """
        Train target model with optional run index for reproducibility.
        """
        data = data.to(self.device)

        # Set random seed for reproducibility across runs
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run_idx)

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

        train_mask = data.train_mask
        val_mask = data.val_mask
        
        # Early stopping parameters
        patience = self.config.patience
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.train_edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
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
                train_acc = (out[train_mask].argmax(dim=1) == data.y[train_mask]).float().sum() / train_mask.sum()
                
                print(f"Run {run_idx+1} - Clean data on target model - Epoch {epoch + 1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc.item():.4f}, "
                      f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
            
            # Early stopping check
            if epochs_without_improvement >= patience:
                print(f"Run {run_idx+1} - Early stopping at epoch {epoch + 1}. Best validation accuracy: {best_val_acc:.4f}")
                break
        
        # 加载最佳模型权重
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Run {run_idx+1} - Loaded best model weights with validation accuracy: {best_val_acc:.4f}")

        return model

    def _calculate_statistics(self, all_results: List[Dict]) -> Dict:
        """
        Calculate mean, std, and variance for multiple runs.
        """
        # Group results by model
        model_stats = {}
        
        # Get unique model names
        model_names = list(set([result['model_name'] for result in all_results]))
        
        for model_name in model_names:
            # Filter results for this model
            model_results = [result for result in all_results if result['model_name'] == model_name]
            
            # Extract overall accuracies
            overall_accs = [result['accuracy']['overall_accuracy'] for result in model_results]
            
            # Calculate statistics
            mean_acc = np.mean(overall_accs)
            std_acc = np.std(overall_accs, ddof=1) if len(overall_accs) > 1 else 0.0
            var_acc = np.var(overall_accs, ddof=1) if len(overall_accs) > 1 else 0.0
            
            # Calculate per-class statistics if available
            per_class_stats = {}
            if model_results and 'per_class_accuracy' in model_results[0]['accuracy']:
                # Get all class indices
                all_classes = set()
                for result in model_results:
                    all_classes.update(result['accuracy']['per_class_accuracy'].keys())
                
                for class_idx in all_classes:
                    class_accs = []
                    for result in model_results:
                        if class_idx in result['accuracy']['per_class_accuracy']:
                            class_accs.append(result['accuracy']['per_class_accuracy'][class_idx])
                    
                    if class_accs:
                        per_class_stats[class_idx] = {
                            'mean': np.mean(class_accs),
                            'std': np.std(class_accs, ddof=1) if len(class_accs) > 1 else 0.0,
                            'var': np.var(class_accs, ddof=1) if len(class_accs) > 1 else 0.0
                        }
            
            model_stats[model_name] = {
                'overall_accuracy': {
                    'mean': mean_acc,
                    'std': std_acc,
                    'var': var_acc,
                    'raw_values': overall_accs
                },
                'per_class_accuracy': per_class_stats,
                'num_runs': len(model_results)
            }
        
        return model_stats

    def run_evaluation_pipeline(self, data):
        """
        Run comprehensive evaluation with multiple runs.
        """
        self.logger.info(f"Running evaluation pipeline with {self.num_runs} runs...")

        all_results = []
        
        # Run multiple experiments
        for run_idx in range(self.num_runs):
            self.logger.info(f"Starting run {run_idx + 1}/{self.num_runs}")
            
            run_results = []
            for model_name in self.config.target_models:
                self.logger.info(f"Run {run_idx + 1} - Training model: {model_name}")

                # Train target model
                target_model = self._train_target_model(data, model_name, run_idx)
                target_model.eval()

                # Evaluate clean accuracy
                clean_accuracy_results = self._evaluate_accuracy(target_model, data)

                # Store results
                result = {
                    'run_idx': run_idx,
                    'model_name': model_name,
                    'accuracy': clean_accuracy_results
                }
                run_results.append(result)
                all_results.append(result)
                
                # Log individual run result
                self.logger.info(f"Run {run_idx + 1} - {model_name} accuracy: {clean_accuracy_results['overall_accuracy']:.4f}")
            
            print(f"\nCompleted run {run_idx + 1}/{self.num_runs}")
            print("-" * 50)

        # Calculate statistics across all runs
        statistics = self._calculate_statistics(all_results)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, stats in statistics.items():
            summary_data.append({
                'Model': model_name,
                'Mean_Accuracy': f"{stats['overall_accuracy']['mean']:.4f}",
                'Std_Accuracy': f"{stats['overall_accuracy']['std']:.4f}",
                'Var_Accuracy': f"{stats['overall_accuracy']['var']:.6f}",
                'Num_Runs': stats['num_runs']
            })
        
        # Display results
        print("\n" + "=" * 80)
        print(f"SUMMARY RESULTS ({self.num_runs} runs)")
        print("=" * 80)
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # # Display detailed statistics
        # print("\n" + "=" * 80)
        # print("DETAILED STATISTICS")
        # print("=" * 80)
        #
        # for model_name, stats in statistics.items():
        #     print(f"\n{model_name}:")
        #     print(f"  Overall Accuracy: {stats['overall_accuracy']['mean']:.4f} ± {stats['overall_accuracy']['std']:.4f}")
        #     print(f"  Variance: {stats['overall_accuracy']['var']:.6f}")
        #     print(f"  Raw values: {[f'{x:.4f}' for x in stats['overall_accuracy']['raw_values']]}")
        #
        #     if stats['per_class_accuracy']:
        #         print(f"  Per-class accuracy:")
        #         for class_idx, class_stats in stats['per_class_accuracy'].items():
        #             print(f"    Class {class_idx}: {class_stats['mean']:.4f} ± {class_stats['std']:.4f}")
        #
        # # Return comprehensive results
        return {
            'all_results': all_results,
            'statistics': statistics,
            'summary_df': df_summary
        }