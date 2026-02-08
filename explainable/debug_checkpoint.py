#!/usr/bin/env python3
"""
Debug script to examine checkpoint structure and understand attention block names.
"""

import torch
import sys
import os

# Updated imports for new package structure
from minotaur.models import ConceptEmbeddingModel
from minotaur.training import AttentionHook, SaveAttentionCallback

# For backward compatibility with old Args classes
try:
    import cem_mil.arg_def_cat as arg_def_cat
    OLD_ARGS_AVAILABLE = True
except ImportError:
    OLD_ARGS_AVAILABLE = False

def examine_checkpoint(checkpoint_path):
    """Examine the checkpoint structure."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    state_dict = checkpoint['state_dict']
    print(f"State dict keys: {list(state_dict.keys())}")
    
    # Find attention-related keys
    attention_keys = [k for k in state_dict.keys() if 'attention' in k or 'atn' in k]
    print(f"\nAttention-related keys ({len(attention_keys)}):")
    for key in attention_keys:
        print(f"  {key}: {state_dict[key].shape}")
    
    return attention_keys

def create_model_and_compare(config, attention_keys):
    """Create a model and compare its state dict with the checkpoint."""
    print(f"\nCreating model with config...")
    
    # Create dummy data
    import pandas as pd
    dummy_data = pd.DataFrame({'ID': ['dummy'], 'Survival': [0]})
    for cpt in config.cpt_ids:
        dummy_data[cpt] = [0]

    if config.attn:
        attention_hook = AttentionHook()
        atten_callback = SaveAttentionCallback(attention_hook=attention_hook)
    else:
        None
    
    # Create model
    model = ConceptEmbeddingModel(
        n_concepts=config.n_concepts,
        n_tasks=config.n_tasks,
        h_dim=config.h_dim,
        emb_size=config.emb_dim,
        concept_states=config.concept_states,
        embedding_activation=config.emb_activ,
        shared_prob_gen=False,
        concept_loss_weight=config.c_weight,
        task_loss_weight=1,
        task_class_weights=None,
        concept_class_weights=config.cpt_cls_weights,
        n_att_heads=config.n_attn_heads,
        attn_dim=config.attn_dim,
        attn_dropout=config.attn_dropout,
        dropout=config.dropout if isinstance(config.dropout, float) else 0.4,
        pre_bn_mlp=config.pre,
        c2y_model=None,
        c2y_layers=None,
        optimizer='adam',
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        top_k_accuracy=None,
        attention_hook=attention_hook
    )
    
    print(f"Model created successfully")
    print(f"Model has attention_blocks: {hasattr(model, 'attention_blocks')}")
    if hasattr(model, 'attention_blocks'):
        print(f"Number of attention blocks: {len(model.attention_blocks)}")
        for i, block in enumerate(model.attention_blocks):
            print(f"  Block {i}: {type(block)}")
            print(f"    atn_1_linear: {hasattr(block, 'atn_1_linear')}")
            print(f"    atn_2_linear: {hasattr(block, 'atn_2_linear')}")
    
    # Get model state dict
    model_state_dict = model.state_dict()
    model_attention_keys = [k for k in model_state_dict.keys() if 'attention' in k or 'atn' in k]
    
    print(f"\nModel attention keys ({len(model_attention_keys)}):")
    for key in model_attention_keys:
        print(f"  {key}: {model_state_dict[key].shape}")
    
    # Compare
    missing_in_model = set(attention_keys) - set(model_attention_keys)
    missing_in_checkpoint = set(model_attention_keys) - set(attention_keys)
    
    print(f"\nMissing in model: {missing_in_model}")
    print(f"Missing in checkpoint: {missing_in_checkpoint}")
    
    return model

if __name__ == "__main__":
    checkpoint_path = "/analysis/minotaur/results/uni/cat_model_exv18/0/3_1_0_3_3000_True_8_1_0.3_attention_exp0_noweights_checkmodel.ckpt"
    
    # Examine checkpoint
    attention_keys = examine_checkpoint(checkpoint_path)
    
    # Create config
    config = arg_def_cat.Args()
    print(f"\nConfig: {config.__dict__}")
    
    # Create model and compare
    model = create_model_and_compare(config, attention_keys) 