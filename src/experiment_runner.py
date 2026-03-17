"""Experiment runner for TAG backdoor attack experiments."""

import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from itertools import product

import src
import src.trigger_generator
import src.trigger_generator3
from config import ExperimentConfig, ParameterSweepConfig
from src.data_processor import TAGDataProcessor
from src.poison_node_selector import PoisonNodeSelector
from src.trigger_generator2 import TextBackdoor
from src.evaluator_clean import CleanEvaluator


class ExperimentRunner:
    """Main experiment runner for TAG backdoor attacks."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        self.experiment_results = []
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = os.path.join('results', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'experiment_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger.info(f"Experiment logging: {log_file}")

    
    def run_single_experiment(self, experiment_id: Optional[str] = None) -> Dict[str, Any]:
        """Run a single backdoor attack experiment.
        
        Args:
            experiment_id: Optional experiment identifier
            
        Returns:
            Dictionary containing experiment results
        """
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting experiment: {experiment_id}")
        
        start_time = time.time()

        # Set random seeds
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)

        # 1. Load and process dataset
        data, text_autoencoder= TAGDataProcessor(self.config).run_process_dataset_pipeline()

        # # clean evaluation
        # clean_data_results = CleanEvaluator(self.config).run_evaluation_pipeline(data)

        # 2. Select poison nodes
        poison_candidates, poison_mask, sorted_dims = PoisonNodeSelector(self.config).run_poison_selection_pipeline(data)

        # 3. MLP trigger model with appending text
        evaluation_results = src.trigger_generator3.TextBackdoor(self.config, text_autoencoder).run_trigger_generation_pipeline(data, poison_candidates, poison_mask, sorted_dims)

        # # # 3. GNN trigger model (use text embedding for trigger generation)
        # evaluation_results = src.trigger_generator2.TextBackdoor(self.config, text_autoencoder).run_trigger_generation_pipeline(data, poison_candidates, poison_mask, sorted_dims)


        # Compile experiment results
        experiment_result = {
            'experiment_id': experiment_id,
            'config': self.config.__dict__,
            'dataset_info': {
                'name': self.config.dataset_name,
                'num_nodes': len(data.x),
                'num_edges': data.edge_index.shape[1],
                'num_classes': len(torch.unique(data.y)),
                'num_features': data.x.shape[1]
            },
            'evaluation_results': evaluation_results,
            'execution_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }

        # Save experiment results
        self.save_experiment_results(experiment_result)
        self.experiment_results.append(experiment_result)

        self.logger.info(f"Experiment {experiment_id}: {experiment_result['execution_time']:.2f} seconds")

        return experiment_result

    def run_multiple_experiments(self, num_experiments: int) -> List[Dict[str, Any]]:
        """Run multiple experiments with different random seeds.
        
        Args:
            num_experiments: Number of experiments to run
            
        Returns:
            List of experiment results
        """
        self.logger.info(f"Staring run {num_experiments} experiments")
        
        all_results = []
        original_seed = self.config.random_seed
        
        for i in range(num_experiments):
            # Update random seed for each experiment
            self.config.random_seed = original_seed + i*10
            
            experiment_id = f"Multi_exp_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            try:
                results = self.run_single_experiment(experiment_id)
                all_results.append(results)
                
                self.logger.info(f"Completed experiment {i+1}/{num_experiments}")
                
            except Exception as e:
                self.logger.error(f"Experiment {i+1} failed: {str(e)}")
                continue
        
        # Restore original seed
        self.config.random_seed = original_seed
        
        # Save aggregated results
        self.save_multiple_experiment_results(all_results)
        
        self.logger.info(f"Completed {len(all_results)}/{num_experiments} experiments")
        
        return all_results
    
    def run_parameter_sweep(self, sweep_config: ParameterSweepConfig):
        """Run parameter sweep experiments.
        
        Args:
            sweep_config: Parameter sweep configuration
            
        Returns:
            List of experiment results
        """
        # Generate parameter combinations
        param_names = list(sweep_config.parameters.keys())
        param_values = list(sweep_config.parameters.values())
        param_combinations = list(product(*param_values))
        
        self.logger.info(f"Starting parameter sweep generated {len(param_combinations)} parameter combinations")
        
        all_results = []
        original_config = self.config.__dict__.copy()
        
        for i, param_combo in enumerate(param_combinations):
            # Update config with current parameter combination
            for param_name, param_value in zip(param_names, param_combo):
                setattr(self.config, param_name, param_value)
            
            experiment_id = f"sweep_{i+1:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"Running sweep experiment {i+1}/{len(param_combinations)}")
            self.logger.info(f"Parameters: {dict(zip(param_names, param_combo))}")
            
            try:
                results = self.run_single_experiment(experiment_id)
                
                # Add parameter information to results
                results['sweep_parameters'] = dict(zip(param_names, param_combo))
                results['sweep_index'] = i
                
                all_results.append(results)
                
            except Exception as e:
                self.logger.error(f"Sweep experiment {i+1} failed: {str(e)}")
                continue
        
        # Restore original config
        for key, value in original_config.items():
            setattr(self.config, key, value)
        
        # # Save sweep results
        filepath=self.save_parameter_sweep_results(all_results, sweep_config)
        
        self.logger.info(f"Parameter sweep completed: {len(all_results)}/{len(param_combinations)} successful")
        
        return filepath, all_results
    
    def save_experiment_results(self, results: Dict[str, Any]) -> None:
        """Save single experiment results.
        
        Args:
            results: Experiment results dictionary
        """
        output_dir = os.path.join('results', 'exps_single')
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{results['experiment_id']}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str) # type: ignore
        
        self.logger.info(f"Experiment results saved: {filepath}")
    
    def save_multiple_experiment_results(self, all_results: List[Dict[str, Any]]) -> None:
        """Save multiple experiment results with aggregated statistics.
        
        Args:
            all_results: List of experiment results
        """
        output_dir = os.path.join('results', 'exps_multi')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"multi_exps_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Compute aggregated statistics
        aggregated_stats = self.compute_aggregated_statistics(all_results)
        
        save_data = {
            'num_experiments': len(all_results),
            'aggregated_statistics': aggregated_stats,
            'individual_results': all_results,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str) # type: ignore
        
        self.logger.info(f"Multiple experiment results saved: {filepath}")
    
    def save_parameter_sweep_results(self,
                                   all_results: List[Dict[str, Any]],
                                   sweep_config: ParameterSweepConfig):
        """Save parameter sweep results.
        
        Args:
            all_results: List of experiment results
            sweep_config: Parameter sweep configuration
        """
        output_dir = os.path.join('results', 'exp_sweeps')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"parameter_sweep_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Analyze parameter sweep results
        sweep_analysis = self.analyze_parameter_sweep(all_results, sweep_config)
        
        save_data = {
            'sweep_config': sweep_config.__dict__,
            'num_experiments': len(all_results),
            'sweep_analysis': sweep_analysis,
            'individual_results': all_results,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str) # type: ignore
        
        self.logger.info(f"Parameter sweep results saved: {filepath}")

        return filepath

    def compute_aggregated_statistics(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated statistics across multiple experiments.
        
        Args:
            all_results: List of experiment results
            
        Returns:
            Dictionary containing aggregated statistics
        """
        if not all_results:
            return {}
        
        # Extract metrics for each model
        model_stats = {}
        
        for model_name in self.config.target_models:
            clean_accs = []
            attack_rates = []
            
            for result in all_results:
                if model_name in result['evaluation_results']:
                    eval_result = result['evaluation_results'][model_name]
                    clean_accs.append(eval_result['clean_accuracy']['overall_accuracy'])
                    attack_rates.append(eval_result['attack_success']['attack_success_rate'])
            
            if clean_accs and attack_rates:
                model_stats[model_name] = {
                    'clean_accuracy': {
                        'mean': np.mean(clean_accs),
                        'std': np.std(clean_accs),
                        'min': np.min(clean_accs),
                        'max': np.max(clean_accs)
                    },
                    'attack_success_rate': {
                        'mean': np.mean(attack_rates),
                        'std': np.std(attack_rates),
                        'min': np.min(attack_rates),
                        'max': np.max(attack_rates)
                    },
                    'num_experiments': len(clean_accs)
                }
        
        return {
            'model_statistics': model_stats,
            'total_experiments': len(all_results)
        }
    
    def analyze_parameter_sweep(self,
                              all_results: List[Dict[str, Any]],
                              sweep_config: ParameterSweepConfig) -> Dict[str, Any]:
        """Analyze parameter sweep results.
        
        Args:
            all_results: List of experiment results
            sweep_config: Parameter sweep configuration
            
        Returns:
            Dictionary containing sweep analysis
        """
        if not all_results:
            return {}

        print(f"===all_results: {all_results}")
        
        # Find the best parameter combinations
        best_configs = {}
        
        for model_name in self.config.target_models:
            best_clean_acc = -1
            best_attack_rate = -1
            best_clean_config = None
            best_attack_config = None


            for result in all_results:
                if model_name in result['evaluation_results']:
                    eval_result = result['evaluation_results'][model_name]
                    clean_acc = eval_result['clean_accuracy']['overall_accuracy']
                    attack_rate = eval_result['attack_success']['attack_success_rate']
                    
                    if clean_acc > best_clean_acc:
                        best_clean_acc = clean_acc
                        best_clean_config = result['sweep_parameters']
                    
                    if attack_rate > best_attack_rate:
                        best_attack_rate = attack_rate
                        best_attack_config = result['sweep_parameters']
            
            best_configs[model_name] = {
                'best_clean_accuracy': {
                    'value': best_clean_acc,
                    'parameters': best_clean_config
                },
                'best_attack_success_rate': {
                    'value': best_attack_rate,
                    'parameters': best_attack_config
                }
            }
        
        return {
            'best_configurations': best_configs,
            'parameter_space': sweep_config.parameters,
            'total_combinations': len(all_results)
        }


