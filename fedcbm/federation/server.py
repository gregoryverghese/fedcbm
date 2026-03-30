"""Flower server-side evaluation and strategy for federated CBM training."""
import warnings
import numpy as np
import pandas as pd
import torch
from typing import Callable, Dict, List, Optional, Tuple

from flwr.server.strategy import FedAvg
from flwr.common import (
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)

from fedcbm.federation.client import build_model, set_parameters
from fedcbm.training.trainer import predict
from fedcbm.training.metrics import compute_task_metrics_all, compute_concept_metrics
from fedcbm.data.loaders import get_data_loaders

warnings.filterwarnings("ignore")


def make_test_loader(test_df: pd.DataFrame, config):
    """Build a DataLoader for the global test set."""
    # num_workers=0: server evaluate_fn runs in main process; avoid fork issues
    _, test_loader = get_data_loaders(
        config.data.db_path,
        train_data=test_df,
        val_data=test_df,
        config=config,
        num_workers=0,
    )
    return test_loader


def get_evaluate_fn(test_df: pd.DataFrame, config) -> Callable:
    """Return a server-side evaluation function called after each FL round.

    Evaluates the global model on the held-out test set and reports:
      - loss  : 1 - c_index  (lower is better, as Flower expects)
      - metrics: c_index, mean_concept_f1
    """
    test_loader = make_test_loader(test_df, config)

    def evaluate_fn(
        server_round: int,
        parameters: NDArrays,
        eval_config: Dict[str, Scalar],
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        model = build_model(config)
        set_parameters(model, parameters)
        model.eval()

        results = predict(model, test_loader, config)
        c_probs, c_test, y_probs, y_test, c_embs, ctxts, caw, patient_ids = results

        task_metrics = compute_task_metrics_all(y_probs, y_test, config.task_type)
        c_index = task_metrics.get("c_index", 0.0) if task_metrics else 0.0

        mean_concept_f1 = 0.0
        if c_probs is not None and not config.concepts.no_concepts:
            concept_metrics = compute_concept_metrics(
                c_probs, c_test, config.concepts.concept_states, is_logits=True
            )
            f1s = [m["f1"] for m in concept_metrics.values() if "f1" in m]
            mean_concept_f1 = float(np.mean(f1s)) if f1s else 0.0

        loss = 1.0 - c_index
        metrics = {
            "c_index": float(c_index),
            "mean_concept_f1": float(mean_concept_f1),
            "server_round": float(server_round),
        }
        print(
            f"  [Server round {server_round:>2d}] "
            f"C-index: {c_index*100:.2f}%  "
            f"Concept F1: {mean_concept_f1*100:.2f}%"
        )
        return float(loss), metrics

    return evaluate_fn


def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate client metrics by sample-count weighted average."""
    totals: Dict[str, float] = {}
    total_n = sum(n for n, _ in metrics)
    if total_n == 0:
        return {}
    for n, m in metrics:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0.0) + float(v) * n
    return {k: v / total_n for k, v in totals.items()}


def build_strategy(
    initial_parameters: Parameters,
    evaluate_fn: Callable,
    n_clients: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> FedAvg:
    """Build a FedAvg strategy with server-side global evaluation each round."""
    return FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=n_clients,
        min_evaluate_clients=int(n_clients * fraction_evaluate),
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_parameters,
    )
