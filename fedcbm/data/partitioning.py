"""Data partitioning for federated learning experiments.

Supports:
- IID: random stratified split into N clients
- Non-IID: cancer-type based assignment to 6 dummy centres (via TSS grouping)
- Global holdout test set creation
"""
import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split


# Cancer-code to centre mapping for 6-centre non-IID partition.
# Integer Cancer values (0-9) in the dataset.
# Design: 3 single-cancer centres + 3 multi-cancer centres:
#   Centre 0: Cancer 0 (n~1001) — large single-cancer hub
#   Centre 1: Cancer 1, 2 (n~798) — two medium cancers
#   Centre 2: Cancer 3, 4 (n~565) — rare + dominant pairing
#   Centre 3: Cancer 7   (n~456) — single medium cancer
#   Centre 4: Cancer 5, 6 (n~666) — two medium cancers
#   Centre 5: Cancer 8, 9 (n~311) — two smaller cancer types
DEFAULT_CANCER_TO_CENTRE = {
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 4,
    6: 4,
    7: 3,
    8: 5,
    9: 5,
}


def _split_off_test(df, test_size, stratify_col, random_seed):
    """Split a dataframe into (test, train) with optional stratification."""
    stratify = df[stratify_col] if (stratify_col and stratify_col in df.columns) else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=stratify, random_state=random_seed
    )
    return test_df.reset_index(drop=True), train_df.reset_index(drop=True)


def partition_iid_from_csv(
    data_path,
    n_clients=6,
    random_seed=42,
    test_size=0.2,
    stratify_col="event",
):
    """Partition CSV into IID client splits with optional global test holdout.

    Returns (global_test_df, client_dfs).  global_test_df is None if test_size==0.
    """
    df = pd.read_csv(data_path)
    global_test_df = None
    train_df = df

    if test_size > 0:
        global_test_df, train_df = _split_off_test(
            df, test_size=test_size, stratify_col=stratify_col, random_seed=random_seed
        )

    rng = np.random.default_rng(random_seed)
    idx = rng.permutation(len(train_df))
    splits = np.array_split(idx, n_clients)
    client_dfs = [train_df.iloc[split].reset_index(drop=True) for split in splits]

    return global_test_df, client_dfs


def partition_noniid_from_csv(
    data_path,
    n_clients=6,
    random_seed=42,
    test_size=0.2,
    stratify_col="event",
    cancer_col="Cancer",
    cancer_to_centre=None,
):
    """Partition CSV into non-IID client splits using cancer-type grouping.

    Patients are assigned to dummy centres based on their Cancer column value,
    which in TCGA maps 1-to-1 with TSS site codes.  The DEFAULT_CANCER_TO_CENTRE
    mapping creates heterogeneous centres: some with one cancer type, some mixed.

    Returns (global_test_df, client_dfs).
    """
    df = pd.read_csv(data_path)
    global_test_df = None
    train_df = df

    if test_size > 0:
        global_test_df, train_df = _split_off_test(
            df, test_size=test_size, stratify_col=stratify_col, random_seed=random_seed
        )

    mapping = cancer_to_centre if cancer_to_centre is not None else DEFAULT_CANCER_TO_CENTRE

    train_df = train_df.copy()
    train_df["_centre"] = train_df[cancer_col].map(mapping)

    unmapped = train_df["_centre"].isna()
    if unmapped.any():
        bad = train_df.loc[unmapped, cancer_col].unique().tolist()
        print(f"  WARNING: {unmapped.sum()} patients with unmapped cancer codes {bad} assigned to centre 0.")
        train_df["_centre"] = train_df["_centre"].fillna(0)

    train_df["_centre"] = train_df["_centre"].astype(int)

    client_dfs = []
    for cid in range(n_clients):
        cdf = train_df[train_df["_centre"] == cid].drop(columns=["_centre"]).reset_index(drop=True)
        client_dfs.append(cdf)

    return global_test_df, client_dfs


def save_global_test(global_test_df, output_dir, filename="global_test.csv"):
    """Save the global holdout test set to CSV. Returns path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    global_test_df.to_csv(path, index=False)
    return path


def save_client_partitions(client_dfs, output_dir, prefix="client"):
    """Save each client partition to a numbered CSV. Returns list of paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, df in enumerate(client_dfs):
        path = os.path.join(output_dir, f"{prefix}_{i}.csv")
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def summarise_partitions(client_dfs, global_test_df=None, cancer_col="Cancer", event_col="event"):
    """Return a DataFrame summarising each client partition (n, event rate, cancer counts)."""
    all_dfs = []
    if global_test_df is not None:
        all_dfs.append(("global_test", global_test_df))
    all_dfs += [(f"client_{i}", df) for i, df in enumerate(client_dfs)]

    all_cancers = sorted(set(c for _, df in all_dfs for c in df[cancer_col].unique()))
    rows = []
    for name, df in all_dfs:
        row = {"split": name, "n_patients": len(df)}
        if event_col in df.columns:
            row["event_rate"] = round(df[event_col].mean(), 3)
        for c in all_cancers:
            row[f"cancer_{c}"] = int((df[cancer_col] == c).sum())
        rows.append(row)
    return pd.DataFrame(rows)
