#!/usr/bin/env python3
"""Main entry point for TAG backdoor attack experiments."""

import os
import sys

# Add src to path first
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import json
import logging
import pandas as pd
import gc
import torch

from config import ExperimentConfig, ParameterSweepConfig
from src.experiment_runner import ExperimentRunner
from src.visualizer import ExperimentVisualizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


gc.collect()
torch.cuda.empty_cache()

def setup_logging(log_level: str = 'INFO') -> None:
    """Setup basic logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from JSON file.
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create config object
    config = ExperimentConfig()
    
    # Update config with loaded values
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.__dict__[key] = value  # Handle dynamic attributes
            print(f"Warning: Unknown config parameter '{key}' ignored")

    return config



def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                  python main.py single 
                  python main.py single --config my_config.json
                  python main.py multiple --num_experiments 5
                  python main.py sweep --sweep_path sweep_params.json

                  python main.py visualize --results_path results.json
                  python main.py save_config --output config_template.json
                """
    )
    
    # Global arguments
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--output_dir', type=str, default='./results')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single experiment
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('--experiment_id', type=str, help='Custom experiment identifier')
    
    # Multiple experiments
    multiple_parser = subparsers.add_parser('multiple', help='Run multiple experiments')
    multiple_parser.add_argument('--num_experiments', type=int, default=5, help='Number of experiments to run')
    
    # Parameter sweep
    sweep_parser = subparsers.add_parser('sweep', help='Run parameter sweep')
    sweep_parser.add_argument('--sweep_path', type=str, default='./configs/sweep_config_pr.json')


    # Visualization
    viz_parser = subparsers.add_parser('visual', help='Create visualizations')
    viz_parser.add_argument('--results_path', type=str, default='./results/exp_sweeps/parameter_sweep_20250703_104549.json')
    viz_parser.add_argument('--output_dir', type=str, default='./results/visualizations')
    
    # Save config template
    config_parser = subparsers.add_parser('save_config', help='Save configuration template')
    config_parser.add_argument('--output', type=str, default='./configs/learning_trigger.json')
    
    # Parse arguments
    args = parser.parse_args()
    setup_logging(args.log_level) # Setup logging

    if args.command == 'visual':
        visualizer = ExperimentVisualizer(args.output_dir)
        report_path = visualizer.load_and_visualize_results(args.results_path)
        print(f"Visualization report created: {report_path}")
        return


    if args.command == 'save_config':
        config = ExperimentConfig()
        config_dict = config.__dict__
        with open(args.output, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration template saved to: {args.output}")
        return

    # Load configuration
    if args.config:
        config = load_config_from_file(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = ExperimentConfig()
        print("Using default configuration")

    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Models: {config.shadow_model}")
    print(f"  Poison num: {config.poison_node_num}")
    print(f"  embedding_model_name: {config.embedding_model_name}")
    print(f"  text_poison_mode: {config.text_poison_mode}")
    print(f"  Device: {os.environ.get('CUDA_VISIBLE_DEVICES', 'CPU')}")

    # print(f"  config: {config}")
    
    # Execute commands
    # ===============================Single experiment============================
    if args.command == 'single':
        print("Starting single experiment...")

        runner = ExperimentRunner(config)
        results=runner.run_single_experiment(args.experiment_id)
        print(f"Execution time: {results['execution_time']:.2f} seconds")

        # Print summary
        print("\nResults Summary:")

        results_data = []
        for model_name, eval_result in results['evaluation_results'].items():
            clean_acc = eval_result['clean_accuracy']['overall_accuracy']
            attack_rate = eval_result['attack_success']['attack_success_rate']
            results_data.append({
                'Target Model': model_name,
                'Clean Acc (CA)': f"{clean_acc:.4f}",
                'Attack Success Rate (ASR)': f"{attack_rate:.4f}"
            })

        df = pd.DataFrame(results_data)
        print(df.to_string(index=True))


    # ===============================Multiple experiments============================
    elif args.command == 'multiple':
        print(f"Starting {args.num_experiments} experiments...")

        runner = ExperimentRunner(config)
        all_results = runner.run_multiple_experiments(args.num_experiments)

        print(f"Multiple experiments completed: {len(all_results)}/{args.num_experiments} successful")

        # Print aggregated summary
        if all_results:
            summary = runner.compute_aggregated_statistics(all_results)
            print("\nAggregated Results Summary:")

            results_data = []
            for model_name, stats in summary.get('model_statistics', {}).items():
                clean_mean = stats['clean_accuracy']['mean']
                clean_std = stats['clean_accuracy']['std']
                attack_mean = stats['attack_success_rate']['mean']
                attack_std = stats['attack_success_rate']['std']
                results_data.append({
                    'Target Model': model_name,
                    'Clean Acc (CA)': f"{clean_mean:.4f} ± {clean_std:.4f}",
                    'Attack Success Rate (ASR)': f"{attack_mean:.4f} ± {attack_std:.4f}"
                })

            df = pd.DataFrame(results_data)
            print(df.to_string(index=True))


    # ===============================Parameter sweep============================
    elif args.command == 'sweep':
        print("Starting parameter sweep...")

        # Load sweep configuration
        with open(args.sweep_path, 'r') as f:
            sweep_dict = json.load(f)

        sweep_config = ParameterSweepConfig()
        for key, value in sweep_dict.items():
            if hasattr(sweep_config, key):
                setattr(sweep_config, key, value)

        print(f"Loaded sweep configuration from: {args.sweep_path}")
        print(f"Loaded sweep configuration: {sweep_config}")

        runner = ExperimentRunner(config)
        filepath, all_results = runner.run_parameter_sweep(sweep_config)

        print(f"Parameter sweep completed: {len(all_results)} experiments")

        # Print best configurations
        if all_results:
            analysis = runner.analyze_parameter_sweep(all_results, sweep_config)
            print("\nBest Configurations:")

            for model_name, best_configs in analysis.get('best_configurations', {}).items():
                print(f"\t{model_name}:")

                best_clean = best_configs['best_clean_accuracy']
                print(f"\t\tBest Clean Acc: {best_clean['value']:.4f}")
                print(f"\t\tParameters: {best_clean['parameters']}")

                best_attack = best_configs['best_attack_success_rate']
                print(f"\t\tBest Attack Rate: {best_attack['value']:.4f}")
                print(f"\t\tParameters: {best_attack['parameters']}")


        visualizer = ExperimentVisualizer(args.output_dir)
        visualizer.load_and_visualize_results(filepath)

if __name__ == '__main__':
    main()