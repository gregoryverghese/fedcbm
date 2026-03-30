"""Flower federated client wrapping CBM local training."""
import warnings
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple

import pytorch_lightning as pl
from flwr.client import NumPyClient

from fedcbm.models.cem_mil import ConceptEmbeddingModel
from fedcbm.data.loaders import get_data_loaders
from fedcbm.training.utils import get_concept_weights
from fedcbm.training.callbacks import AttentionHook

warnings.filterwarnings("ignore")


def _get_accelerator():
    """Return ('gpu', 1) if a CUDA GPU is visible to this process, else ('cpu', 'auto')."""
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", "auto"


def build_model(config) -> ConceptEmbeddingModel:
    """Instantiate a ConceptEmbeddingModel from config."""
    attention_hook = None
    if getattr(config.model, "attn", False):
        attention_hook = AttentionHook()

    model = ConceptEmbeddingModel(
        n_concepts=config.concepts.n_concepts,
        concept_states=config.concepts.concept_states,
        n_tasks=config.model.n_tasks,
        h_dim=config.model.h_dim,
        pre_bn_mlp=config.model.pre,
        emb_size=config.model.emb_dim,
        embedding_activation=config.model.emb_activ,
        concept_loss_weight=config.model.c_weight,
        learning_rate=config.training.learning_rate,
        dropout=config.model.dropout,
        optimizer=config.training.optimizer,
        attention_hook=attention_hook,
        no_concepts=config.concepts.no_concepts,
        task_type=config.task_type,
    )
    return model


def get_parameters(model: ConceptEmbeddingModel) -> List[np.ndarray]:
    """Extract model parameters as a list of numpy arrays."""
    return [v.cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model: ConceptEmbeddingModel, parameters: List[np.ndarray]):
    """Load parameters (numpy arrays) into a model's state dict."""
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


class FedCBMClient(NumPyClient):
    """Flower NumPyClient that trains a local CBM partition.

    Each client holds a fixed local dataset (one client's CSV partition).
    On each FL round:
      1. Server sends global parameters → set_parameters
      2. Client trains for `local_epochs` epochs → fit
      3. Client returns updated parameters + train metrics

    Local evaluation (evaluate) runs on a held-aside local validation split.
    GPU is used when available in the Ray worker process.
    """

    def __init__(
        self,
        client_id: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        config,
    ):
        self.client_id = client_id
        self.train_df = train_df
        self.val_df = val_df
        self.config = config
        self.accelerator, self.devices = _get_accelerator()
        self.model = build_model(config)

        # Pre-build data loaders once — reused across rounds
        # num_workers=0: Ray actor processes can't fork DataLoader workers
        self._train_loader, self._val_loader = get_data_loaders(
            config.data.db_path,
            train_data=train_df,
            val_data=val_df,
            config=config,
            num_workers=0,
        )
        self._n_train = len(self._train_loader.dataset)
        self._n_val = len(self._val_loader.dataset)

    def _make_trainer(self, max_epochs: int) -> pl.Trainer:
        return pl.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=max_epochs,
            check_val_every_n_epoch=max_epochs,
            enable_progress_bar=False,
            enable_checkpointing=False,
            logger=False,
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)

        local_epochs = config.get("local_epochs", self.config.training.max_epochs)
        fl_round = config.get("server_round", 0)

        trainer = self._make_trainer(max_epochs=local_epochs)
        trainer.fit(self.model, self._train_loader, self._val_loader)

        metrics = {"client_id": float(self.client_id), "round": float(fl_round)}
        return get_parameters(self.model), self._n_train, metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        set_parameters(self.model, parameters)

        trainer = self._make_trainer(max_epochs=1)
        results = trainer.validate(self.model, self._val_loader, verbose=False)
        metrics = results[0] if results else {}

        loss = metrics.get("val_loss", metrics.get("val_task_loss", 0.0))
        return float(loss), self._n_val, {k: float(v) for k, v in metrics.items()}
