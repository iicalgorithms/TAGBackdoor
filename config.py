"""Configuration file for TAG Backdoor Attack experiments."""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ExperimentConfig:
    """Configuration for backdoor attack experiments."""

    # Experiment configuration
    experiment_id: str = "none average label distribution"  # Unique identifier for the experiment
    num_runs: int = 1  # Number of independent runs
    device: str = "cuda" #  cuda
    random_seed: int = 42

    # Dataset configuration
    dataset_name: str = "cora"   # "cora", "wikics", "pubmed", "arxiv"  products
    text_poison_mode: str = "overwriting"  # appending, overwriting
    poison_node_num: int = 27 # 80, 240, 800, 2400. #cora 27, pubmed 197, Arxiv 1693
    defense_mode: str = None  # "prune", "reconstruct", None
    target_models: List[str] = field(default_factory=lambda: ["GNNGuard"])  # backdoor target models


    root: str = "./dataset"

    # Text embedding configuration
    embedding_model_name: str = "vec2text" # sonar, vec2text (gtr-t5), bow, tfidf, bottleneck-t5, mini
    # text_encoder: str = "sentence-transformers/all-mpnet-base-v2" #facebook/bart-base, google-t5/t5-small, sentence-transformers/gtr-t5-base

    # surrogate model configuration
    surrogate_model: str = "GCN"  # "GCN", "GraphSAGE", "GAT", "MLP"
    hidden_dim: int = 1024
    # num_layers: int = 2
    # dropout: float = 0.5

    train_ratio: float = 0.1 # Ratio of training nodes in the dataset，labeled
    val_ratio: float = 0.1
    test_ratio: float = 0.2 # default 0.2

    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 300
    patience: int = 50

    # Poison node selection and target assignment configurations
    poison_node_mode: str = "uncertainty"  # random, uncertainty




    uncertainty_method: str = "entropy"  # "entropy" or "variance"
    min_class_coverage: int = 1  # Minimum number of classes to cover (Whether to select poison nodes uniformly across classes)
    dims_importance_method: str = "statistics" # statistics, random_forest,        gradient-based, shapley
    target_assignment: str = "fixed"  # "fixed", "random", "confusion_max"
    fixed_target_class: Optional[int] = 2  #None

    # Trigger generation configuration
    feature_similarity_weight: float = 1 # poisoned text should be semantically similar to the original text.
    # text_reconstruction_weight: float = 1  # Weight for text reconstruction loss
    # neighbor_similarity_weight: float = 1  # Weight for neighbor similarity loss in GNNGuard

    max_seq_length_overwriting: int = 1024 # 生成poison text的长度, 根据上面判断，15，128
    max_seq_length_appending: int = 512 # 生成poison text的长度, 根据上面判断，15，128

    shadow_model: str="GCN"  # backdoor shadow model name

    # Trojan
    trojan_num_layers: int = 2
    trojan_dropout: float = 0.0  # GCN其他越大越好
    trojan_epochs: int = 150 # 150 Number of epochs for training the trojan model
    poison_dim_num: int = 84  # number of dimensions to poison
    outer_poison_num: int = 512 # number of outer poison nodes
    
    # OOD Detection configuration
    use_ood_detection: bool = True  # Enable OOD detection for adversarial training
    ood_lambda: float = 0.1  # Weight for OOD loss in discriminator training
    discriminator_lr: float = 0.0002  # Learning rate for OOD discriminator
    adversarial_lambda: float = 0.05  # Weight for adversarial loss in generator training
    ood_hidden_dim: int = 128  # Hidden dimension for OOD discriminator
    ood_num_layers: int = 3  # Number of layers in OOD discriminator
    ood_dropout: float = 0.3  # Dropout rate for OOD discriminator

    # Defence configuration




    prune_thr: float = 0.5  # Threshold for pruning edges in the graph
    rec_epochs: int = 50 # Number of epochs for training reconstruction model

    # Evaluation Target model configuration
    # target_models: List[str] = field(default_factory=lambda: ["GCN", "GraphSAGE", "GAT","GIN", "ChebNet","APPNPNet","GraphTransformer","RobustGCN", "GNNGuard"])  # backdoor target models
    # target_models: List[str] = field(default_factory=lambda: ["RobustGCN", "GNNGuard"])  # backdoor target models
    # target_models: List[str] = field(default_factory=lambda: ["GCN", "GraphSAGE", "RobustGCN","GNNGuard"])  # backdoor target models
    # target_models: List[str] = field(default_factory=lambda: ["GraphTransformer", "RobustGCN", "GNNGuard"])  # backdoor target models

    target_epochs: int= 100
    target_lr: float = 0.01
    target_weight_decay: float = 5e-4

    # Output configuration
    log_level: str = "INFO"
    output_dir: str = "./results"  # Directory to save results

    # Attack word selection configuration
    num_keywords_per_class: int = 500  # Number of keywords per class for word selection
    top_k_words: int = 100  # Number of important words to consider for replacement
    word_importance_class: str = "TF-IDF"  # TF-IDF, logistic-regression
    word_importance_text: str = "None"  # shap, lime, None
    shap_model: str = "distilbert/distilbert-base-uncased"  # "linear", "tree", "deep", "gradient"
    
    def __post_init__(self):
        """Post-initialization validation and setup. This method is automatically called whenever you create an instance of ExperimentConfig."""
        
        # Validate ratios
        if abs(self.train_ratio + self.val_ratio + self.test_ratio) > 1.0:
            raise ValueError("Train, validation, and test ratios must sum less than 1.0")
        
        # Validate poison ratio
        # if self.poison_ratio is not None and not 0 < self.poison_ratio < 1:
        #     raise ValueError("Poison ratio must be between 0 and 1")

        if self.target_assignment == "fixed" and self.fixed_target_class is None:
            raise ValueError("Fixed target class must be specified when target_assignment is 'fixed'")


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweep experiments."""
    
    # Parameters to sweep
    parameters: Dict[str, List[Any]] = field(default_factory=lambda: {
        # "poison_ratio": field(default_factory=lambda: [0.05, 0.1, 0.2, 0.5, 1]),
        # "poison_node_num": field(default_factory=lambda: [9, 19, 39, 98, 197, 394, 985]),
        # "poison_node_num": field(default_factory=lambda: [84, 169, 338, 846, 1693, 3386, 8467]),
        "max_seq_length_overwriting": field(default_factory=lambda: [128, 254, 512, 1024, 2048]),
        "max_seq_length_appending": field(default_factory=lambda: [16, 32, 64, 128, 254, 512]),
        # "importance_method": ["shap", "integrated_gradients", "lime"],
        # "target_assignment": ["fixed", "random", "confusion_max"],
        # "surrogate_models": field(default_factory=lambda: ["GCN", "GraphSAGE", "GAT", "MLP"]),
        # "datasets": field(default_factory=lambda: ["arxiv", "cora", "products", "pubmed", "wikics"]),
    })
    # poison_ratios: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])
    # surrogate_models: List[str] = field(default_factory=lambda: ["GCN", "GraphSAGE", "GAT", "MLP"])
    # datasets: List[str] = field(default_factory=lambda: ["arxiv","cora","products", "pubmed", "wikics"])
    # top_k_words_list: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Base configuration
    base_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Sweep configuration
    parallel_jobs: int = 4
    save_all_results: bool = True
