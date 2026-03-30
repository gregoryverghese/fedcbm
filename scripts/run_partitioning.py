#!/usr/bin/env python
"""Generate all data partitions for the federated CBM experiments.

Creates:
  <output_dir>/global_test.csv
  <output_dir>/iid/client_0.csv  ...  client_5.csv
  <output_dir>/noniid/client_0.csv  ...  client_5.csv
  <output_dir>/partition_summary.csv
"""
import argparse
import os
import sys
import pandas as pd

from fedcbm.data.partitioning import (
    partition_iid_from_csv,
    partition_noniid_from_csv,
    save_global_test,
    save_client_partitions,
    summarise_partitions,
)


def main():
    parser = argparse.ArgumentParser(description="Partition data for federated CBM experiments")
    parser.add_argument("-dp", "--data-path", required=True, help="Path to dataset CSV")
    parser.add_argument("-o", "--output-dir", default="data/partitions", help="Output directory")
    parser.add_argument("-n", "--n-clients", type=int, default=6, help="Number of clients")
    parser.add_argument("--test-size", type=float, default=0.2, help="Global test holdout fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--stratify-col", default="event", help="Column to stratify on for test split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── IID partition ──────────────────────────────────────────────────────────
    print(f"\n[1/2] Generating IID partition ({args.n_clients} clients)...")
    test_df, iid_dfs = partition_iid_from_csv(
        args.data_path,
        n_clients=args.n_clients,
        random_seed=args.seed,
        test_size=args.test_size,
        stratify_col=args.stratify_col,
    )

    test_path = save_global_test(test_df, args.output_dir)
    print(f"  Global test set: {len(test_df)} patients → {test_path}")

    iid_dir = os.path.join(args.output_dir, "iid")
    iid_paths = save_client_partitions(iid_dfs, iid_dir)
    print(f"  IID clients:")
    for i, (p, df) in enumerate(zip(iid_paths, iid_dfs)):
        er = df["event"].mean() if "event" in df.columns else float("nan")
        print(f"    client_{i}: {len(df)} patients  event_rate={er:.2f}")

    # ── Non-IID partition (same global test set) ───────────────────────────────
    print(f"\n[2/2] Generating non-IID partition (cancer-type based)...")
    # Re-run with same seed so test set is identical
    _, noniid_dfs = partition_noniid_from_csv(
        args.data_path,
        n_clients=args.n_clients,
        random_seed=args.seed,
        test_size=args.test_size,
        stratify_col=args.stratify_col,
    )

    noniid_dir = os.path.join(args.output_dir, "noniid")
    noniid_paths = save_client_partitions(noniid_dfs, noniid_dir)
    print(f"  Non-IID clients:")
    for i, (p, df) in enumerate(zip(noniid_paths, noniid_dfs)):
        er = df["event"].mean() if "event" in df.columns else float("nan")
        cancers = sorted(df["Cancer"].unique().tolist()) if "Cancer" in df.columns else []
        print(f"    client_{i}: {len(df)} patients  event_rate={er:.2f}  cancers={cancers}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n── Partition Summary ──────────────────────────────────────────────")
    print("\nIID:")
    iid_summary = summarise_partitions(iid_dfs, global_test_df=test_df)
    print(iid_summary.to_string(index=False))

    print("\nNon-IID:")
    noniid_summary = summarise_partitions(noniid_dfs, global_test_df=test_df)
    print(noniid_summary.to_string(index=False))

    # Save combined summary
    iid_summary["scheme"] = "iid"
    noniid_summary["scheme"] = "noniid"
    summary = pd.concat([iid_summary, noniid_summary], ignore_index=True)
    summary_path = os.path.join(args.output_dir, "partition_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
