#!/usr/bin/env python
"""Run a federated CBM experiment using Flower simulation.

Examples:
  # IID federation (20 rounds, 6 clients)
  python scripts/run_federated.py \
      --partition-dir data/partitions/iid \
      --test-path     data/partitions/global_test.csv \
      --config        configs/federated.yaml \
      --save-path     results/federated_iid \
      --n-rounds 20 --local-epochs 5 --experiment-name fed_iid

  # Non-IID federation
  python scripts/run_federated.py \
      --partition-dir data/partitions/noniid \
      --test-path     data/partitions/global_test.csv \
      --config        configs/federated.yaml \
      --save-path     results/federated_noniid \
      --n-rounds 20 --local-epochs 5 --experiment-name fed_noniid
"""
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

from fedcbm.config.loader import load_config_from_yaml
from fedcbm.federation.simulation import run_federated_simulation, FedConfig


def main():
    parser = argparse.ArgumentParser(description="Run federated CBM simulation")
    parser.add_argument("--partition-dir", required=True,
                        help="Directory of client_0.csv ... client_N.csv")
    parser.add_argument("--test-path", required=True,
                        help="Path to global test CSV")
    parser.add_argument("--config", default="configs/federated.yaml",
                        help="Path to YAML config")
    parser.add_argument("--save-path", default="results/federated",
                        help="Directory to save results")
    parser.add_argument("--experiment-name", default="federated",
                        help="Experiment label")
    # Federated hyperparameters (override YAML)
    parser.add_argument("--n-rounds", type=int, default=None,
                        help="Number of FL rounds (overrides config)")
    parser.add_argument("--n-clients", type=int, default=None,
                        help="Number of clients (overrides config, must match partition-dir)")
    parser.add_argument("--local-epochs", type=int, default=None,
                        help="Local training epochs per round (overrides config)")
    parser.add_argument("--client-gpus", type=float, default=None,
                        help="GPU fraction per simulated client")
    parser.add_argument("--client-cpus", type=int, default=None,
                        help="CPU cores per simulated client")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    config.output.save_path = args.save_path
    config.output.name = args.experiment_name
    os.makedirs(args.save_path, exist_ok=True)

    # Build federated config from YAML federated section + CLI overrides
    yaml_fed = getattr(config, "_raw_fed", {})  # fallback

    # Read federated section directly from YAML
    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)
    fed_yaml = raw.get("federated", {})

    fed_config = FedConfig(
        n_rounds=args.n_rounds or fed_yaml.get("n_rounds", 20),
        n_clients=args.n_clients or fed_yaml.get("n_clients", 6),
        fraction_fit=fed_yaml.get("fraction_fit", 1.0),
        fraction_evaluate=fed_yaml.get("fraction_evaluate", 1.0),
        local_epochs=args.local_epochs or fed_yaml.get("local_epochs", 5),
        client_num_cpus=args.client_cpus or fed_yaml.get("client_num_cpus", 2),
        client_num_gpus=args.client_gpus if args.client_gpus is not None else fed_yaml.get("client_num_gpus", 0.0),
    )

    print(f"\n{'='*60}")
    print(f"  Federated CBM experiment: {args.experiment_name}")
    print(f"  Partition: {args.partition_dir}")
    print(f"  Rounds: {fed_config.n_rounds}  Clients: {fed_config.n_clients}  Local epochs: {fed_config.local_epochs}")
    print(f"{'='*60}\n")

    run_federated_simulation(
        partition_dir=args.partition_dir,
        global_test_path=args.test_path,
        config=config,
        fed_config=fed_config,
        results_dir=args.save_path,
    )


if __name__ == "__main__":
    main()
