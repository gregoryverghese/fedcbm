"""Configuration loader for MINOTAUR."""
import yaml
import os
import re
from pathlib import Path
from typing import Union, Dict, Any, Optional
from .base_config import Config, DataConfig, ConceptConfig, ModelConfig, TrainingConfig, OutputConfig, HardwareConfig


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in configuration values.
    
    Supports ${VAR} syntax for environment variable expansion.
    
    Args:
        value: Configuration value (may be dict, list, or string)
        
    Returns:
        Value with environment variables expanded
    """
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    elif isinstance(value, str):
        # Expand ${VAR} syntax
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_env, value)
    else:
        return value


def load_config_from_yaml(config_path: Union[str, Path]) -> Config:
    """Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Expand environment variables
    config_dict = expand_env_vars(config_dict)
    
    return dict_to_config(config_dict)


def dict_to_config(config_dict: Dict[str, Any]) -> Config:
    """Convert a dictionary to a Config object.
    
    Args:
        config_dict: Dictionary with configuration values
        
    Returns:
        Config object
    """
    # Extract top-level task_type and experiment metadata if present
    task_type = config_dict.pop("task_type", "binary")
    experiment_name = config_dict.pop("experiment_name", None)
    description = config_dict.pop("description", None)
    random_seed = config_dict.pop("random_seed", 42)
    
    # Handle legacy "experiment" key
    if "experiment" in config_dict:
        exp_dict = config_dict.pop("experiment", {})
        experiment_name = exp_dict.get("name", experiment_name)
        description = exp_dict.get("description", description)
        random_seed = exp_dict.get("random_seed", random_seed)
    
    # Create sub-configs
    data_config = DataConfig(**config_dict.pop("data", {}))
    concepts_config = ConceptConfig(**config_dict.pop("concepts", {}))
    model_config = ModelConfig(**config_dict.pop("model", {}))
    
    # Extract training dict and remove hyperparameter_search (not part of TrainingConfig)
    training_dict = config_dict.pop("training", {})
    training_dict.pop("hyperparameter_search", None)  # Remove if present
    training_config = TrainingConfig(**training_dict)
    
    output_config = OutputConfig(**config_dict.pop("output", {}))
    hardware_config = HardwareConfig(**config_dict.pop("hardware", {}))
    
    # Create main config
    config = Config(
        data=data_config,
        concepts=concepts_config,
        model=model_config,
        training=training_config,
        output=output_config,
        hardware=hardware_config,
        task_type=task_type,
        experiment_name=experiment_name,
        description=description,
        random_seed=random_seed,
    )
    
    # Handle any remaining keys (legacy support)
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def create_config_from_args(args) -> Config:
    """Create a Config object from an Args object (backward compatibility).
    
    This function helps migrate from the old Args() classes to the new Config system.
    
    Args:
        args: An Args object (from arg_def.py or arg_def_cat.py)
        
    Returns:
        Config object
    """
    # Map old Args attributes to new Config structure
    data_config = DataConfig(
        db_path=getattr(args, 'db_path', None),
        database=getattr(args, 'database', 'lmdb'),
        target=getattr(args, 'target', 'Survival'),
        bag_n=getattr(args, 'bag_n', 3000),
        k=getattr(args, 'k', 0),
    )
    
    concepts_config = ConceptConfig(
        cpt_ids=getattr(args, 'cpt_ids', []),
        n_concepts=getattr(args, 'n_concepts', 4),
        concept_states=getattr(args, 'concept_states', []),
        cpt_cls_weights=getattr(args, 'cpt_cls_weights', None),
        no_concepts=getattr(args, 'no_concepts', False),
    )
    
    model_config = ModelConfig(
        n_tasks=getattr(args, 'n_tasks', 1),
        h_dim=getattr(args, 'h_dim', 1024),
        emb_dim=getattr(args, 'emb_dim', 8),
        emb_activ=getattr(args, 'emb_activ', 'LeakyReLU'),
        pre=getattr(args, 'pre', True),
        dropout=getattr(args, 'dropout', True),
        c_weight=getattr(args, 'c_weight', 1.0),
        attn=getattr(args, 'attn', True),
        n_attn_heads=getattr(args, 'n_attn_heads', 4),
        attn_dim=getattr(args, 'attn_dim', 256),
        attn_dropout=getattr(args, 'attn_dropout', 0.3),
    )
    
    training_config = TrainingConfig(
        batch_size=getattr(args, 'batch_size', 64),
    )
    
    output_config = OutputConfig(
        save_path=getattr(args, 'save_path', None),
        log_path=getattr(args, 'log_path', None),
        name=getattr(args, 'name', 'experiment'),
        categorical=getattr(args, 'categorical', True),
    )
    
    task_type = getattr(args, 'task_type', 'binary')
    
    return Config(
        data=data_config,
        concepts=concepts_config,
        model=model_config,
        training=training_config,
        output=output_config,
        task_type=task_type,
        experiment_name=getattr(args, 'name', None),
    )


def save_config_to_yaml(config: Config, output_path: Union[str, Path]) -> None:
    """Save a Config object to a YAML file.
    
    Args:
        config: Config object to save
        output_path: Path where to save the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        "task_type": config.task_type,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "random_seed": config.random_seed,
        "data": {
            "db_path": config.data.db_path,
            "database": config.data.database,
            "target": config.data.target,
            "bag_n": config.data.bag_n,
            "k": config.data.k,
            "test_size": config.data.test_size,
            "stratify_random_state": config.data.stratify_random_state,
            "k_fold": config.data.k_fold,
            "k_fold_random_state": config.data.k_fold_random_state,
        },
        "concepts": {
            "cpt_ids": config.concepts.cpt_ids,
            "n_concepts": config.concepts.n_concepts,
            "concept_states": config.concepts.concept_states,
            "cpt_cls_weights": config.concepts.cpt_cls_weights,
            "no_concepts": config.concepts.no_concepts,
        },
        "model": {
            "n_tasks": config.model.n_tasks,
            "h_dim": config.model.h_dim,
            "emb_dim": config.model.emb_dim,
            "emb_activ": config.model.emb_activ,
            "pre": config.model.pre,
            "pre_bn_mlp": config.model.pre_bn_mlp,
            "attn": config.model.attn,
            "n_attn_heads": config.model.n_attn_heads,
            "attn_dim": config.model.attn_dim,
            "attn_dropout": config.model.attn_dropout,
            "dropout": config.model.dropout,
            "c_weight": config.model.c_weight,
            "task_loss_weight": config.model.task_loss_weight,
        },
        "training": {
            "optimizer": config.training.optimizer,
            "learning_rate": config.training.learning_rate,
            "momentum": config.training.momentum,
            "weight_decay": config.training.weight_decay,
            "max_epochs": config.training.max_epochs,
            "check_val_every_n_epoch": config.training.check_val_every_n_epoch,
            "batch_size": config.training.batch_size,
            "val_batch_size": config.training.val_batch_size,
            "num_workers": config.training.num_workers,
            "drop_last": config.training.drop_last,
        },
        "output": {
            "save_path": config.output.save_path,
            "log_path": config.output.log_path,
            "name": config.output.name,
            "categorical": config.output.categorical,
        },
        "hardware": {
            "accelerator": config.hardware.accelerator,
            "devices": config.hardware.devices,
            "enable_progress_bar": config.hardware.enable_progress_bar,
            "cuda_launch_blocking": config.hardware.cuda_launch_blocking,
        },
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

