"""
Configuration utilities for loading model hyperparameters.
Supports both YAML config files and checkpoint hyperparameters.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Import AttentionHook for model creation
try:
    from minotaur.training import AttentionHook
except ImportError:
    # Fallback for backward compatibility
    try:
        from cem_mil.utilities import AttentionHook
    except ImportError:
        # Handle case where utilities module is not available
        AttentionHook = None


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with configuration parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load configuration from checkpoint hyperparameters.
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Dictionary with hyperparameters
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('hyper_parameters', {})


def get_model_kwargs_from_config(
    config_dict: Dict[str, Any],
    concept_ids: list,
    concept_states: list,
    checkpoint_hyperparams: Optional[Dict[str, Any]] = None,
    create_attention_hook: bool = True
) -> Dict[str, Any]:
    """
    Extract model kwargs from config dictionary.
    
    Args:
        config_dict: Configuration dictionary from YAML or Args class
        concept_ids: List of concept names
        concept_states: List of concept states
        checkpoint_hyperparams: Optional checkpoint hyperparameters to override config
        create_attention_hook: Whether to create attention hook if attention is enabled
        
    Returns:
        Dictionary with model initialization kwargs
    """
    # Use checkpoint hyperparameters if available, otherwise use config
    if checkpoint_hyperparams:
        hyper_params = checkpoint_hyperparams
    else:
        hyper_params = config_dict
    
    # Create attention hook if needed
    attention_hook = None
    if create_attention_hook and AttentionHook is not None:
        attn_enabled = hyper_params.get('attn', True) if 'attn' in hyper_params else True
        if attn_enabled:
            attention_hook = AttentionHook()
            print("Created attention hook for model")
    
    # Handle different config formats
    if 'model' in config_dict:
        # YAML config format
        model_config = config_dict['model']
        model_kwargs = {
            'n_concepts': len(concept_ids),
            'n_tasks': model_config.get('n_tasks', 1),
            'h_dim': model_config.get('h_dim', 1024),
            'emb_size': model_config.get('emb_dim', 8),
            'concept_states': concept_states,
            'embedding_activation': model_config.get('emb_activ', 'LeakyReLU'),
            'n_att_heads': model_config.get('n_attn_heads', 4),
            'attn_dim': model_config.get('attn_dim', 256),
            'attn_dropout': model_config.get('attn_dropout', 0.3),
            'dropout': model_config.get('dropout', 0.3),
            'pre_bn_mlp': model_config.get('pre_bn_mlp', True),
            'concept_loss_weight': model_config.get('concept_loss_weight', 1),
            'task_loss_weight': model_config.get('task_loss_weight', 1),
            'shared_prob_gen': model_config.get('shared_prob_gen', False),
            'c2y_model': model_config.get('c2y_model', None),
            'c2y_layers': model_config.get('c2y_layers', None),
            'attention_hook': attention_hook,
        }
        
        # Add training config if available
        if 'training' in config_dict:
            training_config = config_dict['training']
            model_kwargs.update({
                'optimizer': training_config.get('optimizer', 'adam'),
                'learning_rate': training_config.get('learning_rate', 0.001),
                'weight_decay': training_config.get('weight_decay', 4e-05),
            })
    else:
        # Args class format or direct hyperparameters
        model_kwargs = {
            'n_concepts': len(concept_ids),
            'n_tasks': hyper_params.get('n_tasks', 1),
            'h_dim': hyper_params.get('h_dim', 1024),
            'emb_size': hyper_params.get('emb_size', 8),
            'concept_states': concept_states,
            'embedding_activation': hyper_params.get('embedding_activation', 'LeakyReLU'),
            'n_att_heads': hyper_params.get('n_att_heads', 4),
            'attn_dim': hyper_params.get('attn_dim', 256),
            'attn_dropout': hyper_params.get('attn_dropout', 0.3),
            'dropout': hyper_params.get('dropout', 0.3),
            'pre_bn_mlp': hyper_params.get('pre_bn_mlp', True),
            'concept_loss_weight': hyper_params.get('concept_loss_weight', 1),
            'task_loss_weight': hyper_params.get('task_loss_weight', 1),
            'shared_prob_gen': hyper_params.get('shared_prob_gen', False),
            'c2y_model': hyper_params.get('c2y_model', None),
            'c2y_layers': hyper_params.get('c2y_layers', None),
            'optimizer': hyper_params.get('optimizer', 'adam'),
            'learning_rate': hyper_params.get('learning_rate', 0.001),
            'weight_decay': hyper_params.get('weight_decay', 4e-05),
            'attention_hook': attention_hook,
        }
    
    return model_kwargs


def load_model_config(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    concept_ids: list = None,
    concept_states: list = None
) -> Dict[str, Any]:
    """
    Load model configuration from either config file or checkpoint.
    
    Args:
        config_path: Optional path to YAML config file
        checkpoint_path: Optional path to model checkpoint
        concept_ids: List of concept names (required)
        concept_states: List of concept states (required)
        
    Returns:
        Dictionary with model initialization kwargs
    """
    if not concept_ids or not concept_states:
        raise ValueError("concept_ids and concept_states are required")
    
    config_dict = {}
    checkpoint_hyperparams = None
    
    # Load from config file if provided
    if config_path and Path(config_path).exists():
        print(f"Loading config from YAML file: {config_path}")
        config_dict = load_config_from_yaml(config_path)
    
    # Load from checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading hyperparameters from checkpoint: {checkpoint_path}")
        checkpoint_hyperparams = load_config_from_checkpoint(checkpoint_path)
    
    # If neither config nor checkpoint provided, use defaults
    if not config_dict and not checkpoint_hyperparams:
        print("No config file or checkpoint provided, using defaults")
        config_dict = {}
    
    # Extract model kwargs
    model_kwargs = get_model_kwargs_from_config(
        config_dict, concept_ids, concept_states, checkpoint_hyperparams
    )
    
    print(f"Model kwargs: {model_kwargs}")
    return model_kwargs


# Example usage and testing
if __name__ == "__main__":
    # Test with default parameters
    concept_ids = ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter']
    concept_states = [4, 3, 10, 3]
    
    # Test loading from checkpoint
    checkpoint_path = "/analysis/minotaur/results/uni/cat_model_exv28/1/7_10_1_7_3_True_8_10_0.3_attention_exp1_noweights_checkmodel.ckpt"
    
    try:
        model_kwargs = load_model_config(
            checkpoint_path=checkpoint_path,
            concept_ids=concept_ids,
            concept_states=concept_states
        )
        print("Successfully loaded model kwargs from checkpoint")
        print(f"Number of concepts: {model_kwargs['n_concepts']}")
        print(f"Concept states: {model_kwargs['concept_states']}")
        print(f"Embedding size: {model_kwargs['emb_size']}")
        print(f"Dropout: {model_kwargs['dropout']}")
    except Exception as e:
        print(f"Error loading from checkpoint: {e}")
        print("Using default configuration")
        
        model_kwargs = load_model_config(
            concept_ids=concept_ids,
            concept_states=concept_states
        )
        print("Successfully loaded default model kwargs")
