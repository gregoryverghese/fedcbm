#!/usr/bin/env python
"""Partition dataset for federated learning scenarios."""
import os
import sys

# Add fedcbm/data to path for standalone import (avoids minotaur/LMDB deps)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_data_dir = os.path.join(_project_root, "fedcbm", "data")
if _data_dir not in sys.path:
    sys.path.insert(0, _data_dir)

from partitioning import (
    partition_iid_from_csv,
    save_client_partitions,
    save_global_test,
)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Partition dataset for federated learning"
    )
    parser.add_argument(
        "-dp",
        "--data-path",
        required=True,
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save client partition CSVs",
    )
    parser.add_argument(
        "-n",
        "--n-clients",
        type=int,
        default=6,
        help="Number of clients (default: 6)",
    )
    parser.add_argument(
        "--scenario",
        choices=["iid"],
        default="iid",
        help="Partitioning scenario: iid (random/IID) or non_iid (future)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--prefix",
        default="client",
        help="Filename prefix for output files (default: client)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for global test set (0 = disabled, default: 0.2)",
    )
    parser.add_argument(
        "--stratify-col",
        default="event",
        help="Column for stratification when splitting (default: event for Cox)",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratification (use random split)",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.data_path):
        parser.error(f"Data file not found: {args.data_path}")

    stratify_col = None if args.no_stratify else args.stratify_col

    if args.scenario == "iid":
        print(f"Partitioning dataset (IID) into {args.n_clients} clients...")
        if args.test_size > 0:
            print(f"  Holding out {args.test_size:.0%} for global test set")
        global_test_df, client_dfs = partition_iid_from_csv(
            args.data_path,
            n_clients=args.n_clients,
            random_seed=args.seed,
            test_size=args.test_size,
            stratify_col=stratify_col,
        )
    else:
        parser.error(f"Scenario '{args.scenario}' not yet implemented")

    # Save global test set
    if global_test_df is not None and len(global_test_df) > 0:
        global_test_path = save_global_test(
            global_test_df,
            output_dir=args.output_dir,
        )
        print(f"\nGlobal test set: {os.path.basename(global_test_path)} ({len(global_test_df)} samples)")

    paths = save_client_partitions(
        client_dfs,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )

    print(f"\nClient partitions ({len(paths)} files) in {args.output_dir}:")
    for i, (path, df) in enumerate(zip(paths, client_dfs)):
        print(f"  {os.path.basename(path)}: {len(df)} samples")


if __name__ == "__main__":
    main()
