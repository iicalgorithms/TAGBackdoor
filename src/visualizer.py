"""Visualization module for TAG backdoor attack experiment results."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from datetime import datetime
import logging

class ExperimentVisualizer:
    """Visualizer for backdoor attack experiment results."""
    
    def __init__(self, output_dir: str = './results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Figure settings
        self.figsize = (10, 6)
        self.dpi = 300
    
    def plot_single_experiment_results(self,
                                     experiment_results: Dict[str, Any],
                                     save_path: Optional[str] = None) -> str:
        """Plot results from a single experiment.
        
        Args:
            experiment_results: Single experiment results
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            exp_id = experiment_results.get('experiment_id', 'unknown')
            save_path = os.path.join(self.output_dir, f'single_experiment_{exp_id}.png')
        
        # Extract data
        eval_results = experiment_results['evaluation_results']
        models = list(eval_results.keys())
        
        clean_accs = [eval_results[model]['clean_accuracy']['overall_accuracy'] 
                     for model in models]
        attack_rates = [eval_results[model]['attack_success']['attack_success_rate'] 
                       for model in models]
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot clean accuracy
        bars1 = ax1.bar(models, clean_accs, alpha=0.7, color='skyblue')
        ax1.set_title('Clean Accuracy by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, clean_accs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Plot attack success rate
        bars2 = ax2.bar(models, attack_rates, alpha=0.7, color='salmon')
        ax2.set_title('Attack Success Rate by Model', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, attack_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Add experiment info
        exp_info = f"Dataset: {experiment_results.get('dataset_info', {}).get('name', 'Unknown')}\n"
        exp_info += f"Triggers: {experiment_results.get('poison_info', {}).get('num_dim', 'Unknown')}"
        fig.suptitle(f"Experiment Results\n{exp_info}", fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Single experiment plot saved: {save_path}")
        return save_path
    
    def plot_multiple_experiments_comparison(self,
                                           multiple_results: List[Dict[str, Any]],
                                           save_path: Optional[str] = None) -> str:
        """Plot comparison across multiple experiments.
        
        Args:
            multiple_results: List of experiment results
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f'multiple_experiments_{timestamp}.png')
        
        # Extract data for all models
        models = set()
        for result in multiple_results:
            models.update(result['evaluation_results'].keys())
        models = sorted(list(models))
        
        # Prepare data
        clean_acc_data = {model: [] for model in models}
        attack_rate_data = {model: [] for model in models}
        
        for result in multiple_results:
            eval_results = result['evaluation_results']
            for model in models:
                if model in eval_results:
                    clean_acc_data[model].append(
                        eval_results[model]['clean_accuracy']['overall_accuracy']
                    )
                    attack_rate_data[model].append(
                        eval_results[model]['attack_success']['attack_success_rate']
                    )
                else:
                    clean_acc_data[model].append(np.nan)
                    attack_rate_data[model].append(np.nan)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Box plot for clean accuracy
        clean_acc_df = pd.DataFrame(clean_acc_data)
        clean_acc_df.boxplot(ax=ax1)
        ax1.set_title('Clean Accuracy Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        
        # 2. Box plot for attack success rate
        attack_rate_df = pd.DataFrame(attack_rate_data)
        attack_rate_df.boxplot(ax=ax2)
        ax2.set_title('Attack Success Rate Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # 3. Mean comparison
        mean_clean_acc = [np.nanmean(clean_acc_data[model]) for model in models]
        mean_attack_rate = [np.nanmean(attack_rate_data[model]) for model in models]
        
        x_pos = np.arange(len(models))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, mean_clean_acc, width, 
                       label='Clean Accuracy', alpha=0.7, color='skyblue')
        bars2 = ax3.bar(x_pos + width/2, mean_attack_rate, width,
                       label='Attack Success Rate', alpha=0.7, color='salmon')
        
        ax3.set_title('Mean Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bar, val in zip(bars1, mean_clean_acc):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, val in zip(bars2, mean_attack_rate):
            if not np.isnan(val):
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Scatter plot: Clean Accuracy vs Attack Success Rate
        for i, model in enumerate(models):
            clean_vals = [x for x in clean_acc_data[model] if not np.isnan(x)]
            attack_vals = [x for x in attack_rate_data[model] if not np.isnan(x)]
            
            if clean_vals and attack_vals:
                ax4.scatter(clean_vals, attack_vals, label=model, alpha=0.7, s=50)
        
        ax4.set_title('Clean Accuracy vs Attack Success Rate', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Clean Accuracy', fontsize=12)
        ax4.set_ylabel('Attack Success Rate', fontsize=12)
        ax4.legend()
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle(f'Multiple Experiments Comparison ({len(multiple_results)} experiments)',
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Multiple experiments plot saved: {save_path}")
        return save_path
    
    def plot_parameter_sweep_results(self,
                                   sweep_results: List[Dict[str, Any]],
                                   sweep_config: Dict[str, Any],
                                   save_path: Optional[str] = None) -> str:
        """Plot parameter sweep results.
        
        Args:
            sweep_results: List of parameter sweep results
            sweep_config: Parameter sweep configuration
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f'parameter_sweep_{timestamp}.png')
        
        # Extract parameter names and values
        param_names = list(sweep_config.get('parameters', {}).keys())
        
        if not param_names:
            self.logger.warning("No parameters found in sweep config")
            return save_path
        
        # Prepare data for visualization
        sweep_data = []
        for result in sweep_results:
            if 'sweep_parameters' in result and 'evaluation_results' in result:
                row = result['sweep_parameters'].copy()
                
                # Add evaluation metrics for each model
                for model_name, eval_result in result['evaluation_results'].items():
                    row[f'{model_name}_clean_acc'] = eval_result['clean_accuracy']['overall_accuracy']
                    row[f'{model_name}_attack_rate'] = eval_result['attack_success']['attack_success_rate']
                
                sweep_data.append(row)
        
        if not sweep_data:
            self.logger.warning("No valid sweep data found")
            return save_path
        
        df = pd.DataFrame(sweep_data)
        
        # Determine number of subplots needed
        num_params = len(param_names)
        models = [col.replace('_clean_acc', '').replace('_attack_rate', '') 
                 for col in df.columns if '_clean_acc' in col]
        models = list(set(models))
        
        # Create figure with subplots
        fig_height = max(8, num_params * 3)
        fig, axes = plt.subplots(num_params, 2, figsize=(16, fig_height))
        
        if num_params == 1:
            axes = axes.reshape(1, -1)
        
        # Plot for each parameter
        for i, param_name in enumerate(param_names):
            param_values = df[param_name].unique()
            
            # Clean accuracy plot
            ax1 = axes[i, 0]
            for model in models:
                clean_acc_col = f'{model}_clean_acc'
                if clean_acc_col in df.columns:
                    mean_vals = []
                    std_vals = []
                    
                    for val in param_values:
                        subset = df[df[param_name] == val][clean_acc_col]
                        mean_vals.append(subset.mean())
                        std_vals.append(subset.std())
                    
                    ax1.errorbar(param_values, mean_vals, yerr=std_vals, 
                               label=model, marker='o', capsize=5)
            
            ax1.set_title(f'Clean Accuracy vs {param_name}', fontsize=12, fontweight='bold')
            ax1.set_xlabel(param_name, fontsize=10)
            ax1.set_ylabel('Clean Accuracy', fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Attack success rate plot
            ax2 = axes[i, 1]
            for model in models:
                attack_rate_col = f'{model}_attack_rate'
                if attack_rate_col in df.columns:
                    mean_vals = []
                    std_vals = []
                    
                    for val in param_values:
                        subset = df[df[param_name] == val][attack_rate_col]
                        mean_vals.append(subset.mean())
                        std_vals.append(subset.std())
                    
                    ax2.errorbar(param_values, mean_vals, yerr=std_vals,
                               label=model, marker='s', capsize=5)
            
            ax2.set_title(f'Attack Success Rate vs {param_name}', fontsize=12, fontweight='bold')
            ax2.set_xlabel(param_name, fontsize=10)
            ax2.set_ylabel('Attack Success Rate', fontsize=10)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
        
        # Add overall title
        fig.suptitle('Parameter Sweep Results', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.show()
        plt.close()
        
        self.logger.info(f"Parameter sweep plot saved: {save_path}")
        return save_path
    
    def plot_trigger_statistics(self,
                              trigger_stats: Dict[str, Any],
                              save_path: Optional[str] = None) -> str:
        """Plot trigger generation statistics.
        
        Args:
            trigger_stats: Trigger statistics dictionary
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f'trigger_stats_{timestamp}.png')
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Semantic similarity distribution
        if 'semantic_similarities' in trigger_stats:
            similarities = trigger_stats['semantic_similarities']
            ax1.hist(similarities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title('Semantic Similarity Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Semantic Similarity', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.axvline(np.mean(similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(similarities):.3f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Target class distribution
        if 'target_class_distribution' in trigger_stats:
            target_dist = trigger_stats['target_class_distribution']
            classes = list(target_dist.keys())
            counts = list(target_dist.values())
            
            bars = ax2.bar(classes, counts, alpha=0.7, color='lightcoral')
            ax2.set_title('Target Class Distribution', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Target Class', fontsize=10)
            ax2.set_ylabel('Number of Triggers', fontsize=10)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontsize=9)
        
        # 3. Replacement statistics
        if 'replacement_stats' in trigger_stats:
            repl_stats = trigger_stats['replacement_stats']
            
            # Number of replacements per trigger
            if 'replacements_per_trigger' in repl_stats:
                repl_counts = repl_stats['replacements_per_trigger']
                ax3.hist(repl_counts, bins=range(1, max(repl_counts)+2), 
                        alpha=0.7, color='lightgreen', edgecolor='black')
                ax3.set_title('Replacements per Trigger', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Number of Replacements', fontsize=10)
                ax3.set_ylabel('Frequency', fontsize=10)
                ax3.grid(True, alpha=0.3)
        
        # 4. Quality metrics
        if 'quality_metrics' in trigger_stats:
            quality = trigger_stats['quality_metrics']
            
            metrics = list(quality.keys())
            values = list(quality.values())
            
            bars = ax4.bar(metrics, values, alpha=0.7, color='gold')
            ax4.set_title('Trigger Quality Metrics', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Score', fontsize=10)
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add overall title
        fig.suptitle('Trigger Generation Statistics', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Trigger statistics plot saved: {save_path}")
        return save_path
    
    def create_experiment_report(self,
                               experiment_data: Dict[str, Any],
                               save_path: Optional[str] = None) -> str:
        """Create a comprehensive experiment report with multiple visualizations.
        
        Args:
            experiment_data: Experiment data (single, multiple, or sweep)
            save_path: Optional custom save path
            
        Returns:
            Path to saved report
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(self.output_dir, f'experiment_report_{timestamp}.html')
        
        # Generate plots
        plot_paths = []
        
        # Determine experiment type and generate appropriate plots
        if 'individual_results' in experiment_data:
            # Multiple experiments or parameter sweep
            results = experiment_data['individual_results']
            
            if 'sweep_config' in experiment_data:
                # Parameter sweep
                plot_path = self.plot_parameter_sweep_results(
                    results, experiment_data['sweep_config']
                )
                plot_paths.append(plot_path)
            else:
                # Multiple experiments
                plot_path = self.plot_multiple_experiments_comparison(results)
                plot_paths.append(plot_path)
        
        elif 'evaluation_results' in experiment_data:
            # Single experiment
            plot_path = self.plot_single_experiment_results(experiment_data)
            plot_paths.append(plot_path)
        
        # Generate HTML report
        html_content = self._generate_html_report(experiment_data, plot_paths)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Experiment report saved: {save_path}")
        return save_path
    
    def _generate_html_report(self,
                            experiment_data: Dict[str, Any],
                            plot_paths: List[str]) -> str:
        """Generate HTML report content.
        
        Args:
            experiment_data: Experiment data
            plot_paths: List of plot file paths
            
        Returns:
            HTML content string
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TAG Backdoor Attack Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; border-bottom: 2px solid #333; }
                h2 { color: #666; border-bottom: 1px solid #666; }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
                .plot { text-align: center; margin: 20px 0; }
                .plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
        """
        
        # Title
        html += "<h1>TAG Backdoor Attack Experiment Report</h1>"
        
        # Timestamp
        html += f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        
        # Experiment summary
        html += "<h2>Experiment Summary</h2>"
        html += "<div class='summary'>"
        
        if 'individual_results' in experiment_data:
            num_experiments = len(experiment_data['individual_results'])
            html += f"<p><strong>Number of Experiments:</strong> {num_experiments}</p>"
            
            if 'sweep_config' in experiment_data:
                html += "<p><strong>Experiment Type:</strong> Parameter Sweep</p>"
                sweep_params = experiment_data['sweep_config'].get('parameters', {})
                html += f"<p><strong>Sweep Parameters:</strong> {list(sweep_params.keys())}</p>"
            else:
                html += "<p><strong>Experiment Type:</strong> Multiple Experiments</p>"
        else:
            html += "<p><strong>Experiment Type:</strong> Single Experiment</p>"
            if 'experiment_id' in experiment_data:
                html += f"<p><strong>Experiment ID:</strong> {experiment_data['experiment_id']}</p>"
        
        # Dataset info
        if 'dataset_info' in experiment_data:
            dataset_info = experiment_data['dataset_info']
            html += f"<p><strong>Dataset:</strong> {dataset_info.get('name', 'Unknown')}</p>"
            html += f"<p><strong>Nodes:</strong> {dataset_info.get('num_nodes', 'Unknown')}</p>"
            html += f"<p><strong>Edges:</strong> {dataset_info.get('num_edges', 'Unknown')}</p>"
        
        html += "</div>"
        
        # Add plots
        html += "<h2>Visualizations</h2>"
        for i, plot_path in enumerate(plot_paths):
            # Convert absolute path to relative for HTML
            plot_filename = os.path.basename(plot_path)
            html += f"<div class='plot'>"
            html += f"<h3>Plot {i+1}</h3>"
            html += f"<img src='{plot_filename}' alt='Experiment Plot {i+1}'>"
            html += "</div>"
        
        # Results table (if available)
        if 'evaluation_results' in experiment_data:
            html += "<h2>Detailed Results</h2>"
            html += self._generate_results_table(experiment_data['evaluation_results'])
        elif 'individual_results' in experiment_data and experiment_data['individual_results']:
            html += "<h2>Aggregated Results</h2>"
            if 'aggregated_statistics' in experiment_data:
                html += self._generate_aggregated_table(experiment_data['aggregated_statistics'])
        
        html += "</body></html>"
        
        return html
    
    def _generate_results_table(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate HTML table for evaluation results."""
        html = "<table>"
        html += "<tr><th>Model</th><th>Clean Accuracy</th><th>Attack Success Rate</th></tr>"
        
        for model_name, results in evaluation_results.items():
            clean_acc = results['clean_accuracy']['overall_accuracy']
            attack_rate = results['attack_success']['attack_success_rate']
            
            html += f"<tr>"
            html += f"<td>{model_name}</td>"
            html += f"<td>{clean_acc:.4f}</td>"
            html += f"<td>{attack_rate:.4f}</td>"
            html += f"</tr>"
        
        html += "</table>"
        return html
    
    def _generate_aggregated_table(self, aggregated_stats: Dict[str, Any]) -> str:
        """Generate HTML table for aggregated statistics."""
        html = "<table>"
        html += "<tr><th>Model</th><th>Clean Acc (Mean±Std)</th><th>Attack Rate (Mean±Std)</th></tr>"
        
        model_stats = aggregated_stats.get('model_statistics', {})
        
        for model_name, stats in model_stats.items():
            clean_mean = stats['clean_accuracy']['mean']
            clean_std = stats['clean_accuracy']['std']
            attack_mean = stats['attack_success_rate']['mean']
            attack_std = stats['attack_success_rate']['std']
            
            html += f"<tr>"
            html += f"<td>{model_name}</td>"
            html += f"<td>{clean_mean:.4f}±{clean_std:.4f}</td>"
            html += f"<td>{attack_mean:.4f}±{attack_std:.4f}</td>"
            html += f"</tr>"
        
        html += "</table>"
        return html
    
    def load_and_visualize_results(self, results_path: str) -> str:
        """Load experiment results from file and create visualizations.
        
        Args:
            results_path: Path to results JSON file
            
        Returns:
            Path to generated report
        """
        with open(results_path, 'r') as f:
            experiment_data = json.load(f)
        
        return self.create_experiment_report(experiment_data)