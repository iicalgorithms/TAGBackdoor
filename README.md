# TAGBD Code

This folder contains the code for the paper:

**Graph-Aware Text-Only Backdoor Poisoning for Text-Attributed Graphs**

Project repository:

[https://github.com/iicalgorithms/TAGBackdoor.git](https://github.com/iicalgorithms/TAGBackdoor.git)

## Overview

The code implements `TAGBD`, a text-only backdoor poisoning framework for text-attributed graphs. The main workflow includes:

- selecting vulnerable training nodes
- generating poisoned text with a graph-aware trigger generator
- training and evaluating backdoored target GNNs
- comparing against baseline attacks and defenses

## Directory Structure

- `main.py`: main entry point for running experiments
- `config.py`: default experiment and sweep configuration
- `configs/`: JSON configuration files
- `src/`: core implementation
- `src/baselines/`: baseline models, attacks, and defenses
- `dataset/`: local datasets and processed files
- `results/`: output directory created during runs
- `TAGLAS/`: bundled TAGLAS code used by the project

## Main Commands

Run a single experiment:

```bash
python main.py single
```

Run a single experiment with a config file:

```bash
python main.py --config ./configs/default_config.json single
```

Run multiple experiments:

```bash
python main.py multiple --num_experiments 5
```

Run a parameter sweep:

```bash
python main.py sweep --sweep_path ./configs/sweep_config_pr.json
```

Generate visualizations from saved results:

```bash
python main.py visual --results_path ./results/your_results.json
```

Save a configuration template:

```bash
python main.py save_config --output ./configs/my_config.json
```

## Configuration

The main settings are defined in `config.py`. Important fields include:

- `dataset_name`: dataset to run, such as `cora`, `pubmed`, or `arxiv`
- `text_poison_mode`: `overwriting` or `appending`
- `poison_node_num`: number of poisoned training nodes
- `embedding_model_name`: text embedding model
- `shadow_model`: shadow GNN used during attack generation
- `target_models`: target GNNs used for evaluation
- `defense_mode`: optional defense, such as `prune` or `reconstruct`

You can either edit `config.py` directly or pass a JSON file with `--config`.

## Notes

- `requirements.txt` is currently empty, so dependencies are not fully pinned in this folder.
- At minimum, the project expects a Python environment with PyTorch, pandas, and the graph/text libraries imported by `src/`.
- The bundled `dataset/` folder currently includes local data files for running experiments directly in this project workspace.

## Minimal Example

```bash
cd code
python main.py single
```

This will run one experiment with the default settings defined in `config.py`.
