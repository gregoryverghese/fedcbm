"""Flower simulation runner for federated CBM experiments."""
import os
import warnings
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fedcbm.federation.client import FedCBMClient, build_model, get_parameters
from fedcbm.federation.server import build_strategy, get_evaluate_fn
from fedcbm.training.utils import get_concept_weights

warnings.filterwarnings("ignore")


@dataclass
class FedConfig:
    """Runtime federated experiment configuration."""
    n_rounds: int = 20
    n_clients: int = 6
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    local_epochs: int = 5
    client_num_cpus: int = 2
    client_num_gpus: float = 0.0


def load_client_partitions(partition_dir: str, n_clients: int) -> List[pd.DataFrame]:
    """Load pre-saved client partition CSVs from partition_dir."""
    dfs = []
    for i in range(n_clients):
        path = os.path.join(partition_dir, f"client_{i}.csv")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Client partition not found: {path}")
        dfs.append(pd.read_csv(path))
    return dfs


def run_federated_simulation(
    partition_dir: str,
    global_test_path: str,
    config,
    fed_config: FedConfig,
    results_dir: Optional[str] = None,
) -> None:
    """Run a federated CBM simulation using Flower.

    Args:
        partition_dir: Directory containing client_0.csv ... client_N.csv
        global_test_path: Path to global_test.csv
        config: MINOTAUR/fedcbm model+training config
        fed_config: Federated learning hyperparameters
        results_dir: Where to save per-round metrics (optional)
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    client_dfs = load_client_partitions(partition_dir, fed_config.n_clients)
    test_df = pd.read_csv(global_test_path)

    # ── Build a train/val split for each client (80/20 within each partition) ─
    from sklearn.model_selection import train_test_split
    client_splits = []
    for df in client_dfs:
        if len(df) < 4:
            # Too small to split — use all for both train and val
            client_splits.append((df, df))
            continue
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=config.random_seed,
            stratify=df["event"] if "event" in df.columns else None,
        )
        client_splits.append((
            train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
        ))

    # ── Set concept class weights from combined training data ─────────────────
    all_train = pd.concat([tr for tr, _ in client_splits], ignore_index=True)
    if not config.concepts.no_concepts:
        config.concepts.cpt_cls_weights = get_concept_weights(
            all_train, config.concepts.cpt_ids
        )

    # ── Initial global model parameters ──────────────────────────────────────
    init_model = build_model(config)
    init_params = ndarrays_to_parameters(get_parameters(init_model))
    del init_model

    # ── Server-side evaluate function (global test set) ───────────────────────
    evaluate_fn = get_evaluate_fn(test_df, config)

    # ── Build strategy ────────────────────────────────────────────────────────
    strategy = build_strategy(
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        n_clients=fed_config.n_clients,
        fraction_fit=fed_config.fraction_fit,
        fraction_evaluate=fed_config.fraction_evaluate,
    )

    # ── Client factory ────────────────────────────────────────────────────────
    def client_fn(context: Context) -> NumPyClient:
        node_id = int(context.node_config["partition-id"])
        train_df, val_df = client_splits[node_id]
        client_config = config  # shared config — no data inside
        return FedCBMClient(
            client_id=node_id,
            train_df=train_df,
            val_df=val_df,
            config=client_config,
        ).to_client()

    # ── Server factory ────────────────────────────────────────────────────────
    def server_fn(context: Context) -> ServerAppComponents:
        local_epochs = fed_config.local_epochs
        fit_config = {"local_epochs": local_epochs}
        # Patch strategy on_fit_config_fn to send local_epochs to clients
        nonlocal strategy
        strategy.on_fit_config_fn = lambda rnd: {**fit_config, "server_round": rnd}
        server_cfg = ServerConfig(num_rounds=fed_config.n_rounds)
        return ServerAppComponents(strategy=strategy, config=server_cfg)

    # ── Assemble apps and run ─────────────────────────────────────────────────
    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    backend_config = {
        "client_resources": {
            "num_cpus": fed_config.client_num_cpus,
            "num_gpus": fed_config.client_num_gpus,
        }
    }

    from flwr.simulation import run_simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=fed_config.n_clients,
        backend_config=backend_config,
    )
