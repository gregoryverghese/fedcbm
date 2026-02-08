"""PyTorch Lightning callbacks for MINOTAUR."""
import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Dict


class AttentionHook:
    """Hook for capturing attention scores during training."""
    
    def __init__(self) -> None:
        self.attention_scores = {'train': [], 'val': []}
        self.current_batch_scores = []
        self.phase = 'train'

    def hook_fn(self, module, input, output):
        """Hook function to capture attention outputs."""
        self.current_batch_scores.append(output.detach())

    def start_new_batch(self):
        """Reset for a new batch."""
        self.current_batch_scores = []
    
    def finalise_batch(self):
        """Finalize current batch and store scores."""
        self.attention_scores[self.phase].append(self.current_batch_scores)
        self.current_batch_scores = []


class SaveAttentionCallback(pl.Callback):
    """PyTorch Lightning callback to save attention scores."""
    
    def __init__(self, attention_hook: AttentionHook):
        """Initialize callback with attention hook.
        
        Args:
            attention_hook: AttentionHook instance to store scores
        """
        self.attention_hook = attention_hook

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Reset hook storage for new batch."""
        self.attention_hook.phase = 'train'

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Save attention scores for this batch."""
        self.attention_hook.finalise_batch()
        
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        """Reset hook storage for new batch."""
        self.attention_hook.phase = 'val'
        self.attention_hook.start_new_batch()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Save attention scores for this batch."""
        self.attention_hook.finalise_batch()


