"""Base configuration classes for MINOTAUR."""
from dataclasses import dataclass, field
from typing import List, Optional, Union
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "/SAN/colcc/Hormad1/data/embeddings"))
    database: str = "lmdb"  # 'lmdb', 'disk', 'rocksdb'
    target: str = "Survival"
    bag_n: int = 3000  # Number of tiles per bag
    k: int = 0  # Additional parameter (used in some contexts)
    test_size: float = 0.2  # For train/val split
    stratify_random_state: int = 42  # Random seed for stratification
    k_fold: int = 8  # For cross-validation
    k_fold_random_state: int = 1  # Random seed for k-fold


@dataclass
class ConceptConfig:
    """Concept configuration."""
    cpt_ids: List[str] = field(default_factory=lambda: ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter'])
    n_concepts: int = 4
    concept_states: List[int] = field(default_factory=lambda: [4, 3, 10, 3])
    # concept_states: number of classes for each concept (0 = continuous, 1 = binary, >1 = categorical)
    cpt_cls_weights: Optional[List[float]] = None  # Class weights for concept classification
    no_concepts: bool = False  # If True, run without concepts


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    n_tasks: int = 1
    h_dim: int = 1024  # Hidden dimension
    emb_dim: int = 8  # Embedding dimension
    emb_activ: str = "LeakyReLU"  # Embedding activation
    pre: bool = True  # Pre-processing
    pre_bn_mlp: bool = True  # Pre-batch-norm MLP
    
    # Attention mechanism
    attn: bool = True  # Use attention
    n_attn_heads: int = 4
    attn_dim: int = 256
    attn_dropout: float = 0.3
    
    # Regularization
    dropout: Union[bool, float] = True  # Can be True/False or a float
    
    # Loss weights
    c_weight: float = 1.0  # Concept loss weight
    task_loss_weight: float = 1.0  # Task loss weight
    
    # Additional parameters
    shared_prob_gen: bool = False
    c2y_model: Optional[str] = None
    c2y_layers: Optional[List[int]] = None
    top_k_accuracy: Optional[int] = None


@dataclass
class TrainingConfig:
    """Training configuration."""
    optimizer: str = "adam"  # 'adam', 'sgd', etc.
    learning_rate: float = 0.001
    momentum: float = 0.9  # For SGD
    weight_decay: float = 0.00004
    
    max_epochs: int = 7
    check_val_every_n_epoch: int = 7
    
    batch_size: int = 64
    val_batch_size: int = 4
    num_workers: Optional[int] = None  # None = auto (cpu_count()/6)
    drop_last: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    save_path: Optional[str] = None
    log_path: Optional[str] = None
    name: str = "experiment"  # Experiment name
    categorical: bool = True  # Whether concepts are categorical


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    accelerator: str = "gpu"  # 'gpu', 'cpu'
    devices: Union[str, int] = "auto"  # Number of devices or 'auto'
    enable_progress_bar: bool = True
    cuda_launch_blocking: bool = False


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    concepts: ConceptConfig = field(default_factory=ConceptConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # Task type: 'binary', 'multiclass', 'continuous', or 'cox'
    task_type: str = "binary"
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    description: Optional[str] = None
    random_seed: int = 42
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set categorical flag based on concept_states
        if self.concepts.concept_states:
            # If any concept has > 1 states (or 0 for continuous), it's categorical
            self.output.categorical = any(s > 1 or s == 0 for s in self.concepts.concept_states)
        
        # Set log_path from save_path if not set
        if self.output.save_path and not self.output.log_path:
            self.output.log_path = str(Path(self.output.save_path) / "logs")
        
        # Set experiment name from output.name if not set
        if not self.experiment_name:
            self.experiment_name = self.output.name


