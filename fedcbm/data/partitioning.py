"""Federated data partitioning for fedCBM."""
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _stratified_split(
    df: pd.DataFrame,
    stratify_col: str,
    test_size: float,
    random_seed: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split without sklearn."""
    federated_indices = []
    test_indices = []

    rng = np.random.default_rng(random_seed)

    for _, group in df.groupby(stratify_col, group_keys=False):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n_test = (
            max(1, int(len(indices) * test_size))
            if len(indices) > 1
            else 0
        )
        if n_test > 0:
            test_indices.extend(indices[:n_test])
            federated_indices.extend(indices[n_test:])
        else:
            federated_indices.extend(indices)

    federated_pool = df.loc[sorted(federated_indices)].reset_index(drop=True)
    global_test = df.loc[sorted(test_indices)].reset_index(drop=True)

    return federated_pool, global_test


def split_global_test(
    df: pd.DataFrame,
    test_size: float = 0.3,
    stratify_col: Optional[str] = None,
    random_seed: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into federated pool and global test set.

    The global test set is held out for centralized evaluation and is not
    partitioned across clients. Clients will split their own data for train/val.

    Args:
        df: Full dataset DataFrame
        test_size: Fraction of data for global test (0.0 to 1.0). If 0, returns
            (df, empty DataFrame).
        stratify_col: Column name for stratification (e.g. 'event' for Cox).
            If None, uses random split. For Cox survival, use 'event' to preserve
            censored vs observed distribution.
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (federated_pool, global_test) DataFrames
    """
    if test_size <= 0:
        return df, pd.DataFrame()

    if test_size >= 1.0:
        return pd.DataFrame(), df

    use_stratify = (
        stratify_col is not None
        and stratify_col in df.columns
        and df[stratify_col].notna().all()
    )

    if use_stratify:
        federated_pool, global_test = _stratified_split(
            df,
            stratify_col=stratify_col,
            test_size=test_size,
            random_seed=random_seed,
        )
    else:
        shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        n_test = int(len(shuffled) * test_size)
        global_test = shuffled.iloc[:n_test].reset_index(drop=True)
        federated_pool = shuffled.iloc[n_test:].reset_index(drop=True)

    return federated_pool, global_test


def partition_iid(
    df: pd.DataFrame,
    n_clients: int = 6,
    random_seed: Optional[int] = 42,
) -> List[pd.DataFrame]:
    """
    Partition DataFrame into n_clients subsets with IID (random) sampling.

    Each client receives a random subset of the data, approximating the global
    distribution (IID assumption).

    Args:
        df: Full dataset DataFrame
        n_clients: Number of client partitions
        random_seed: Random seed for reproducibility. If None, shuffling is non-deterministic.

    Returns:
        List of n_clients DataFrames, one per client
    """
    n_samples = len(df)
    if n_samples < n_clients:
        raise ValueError(
            f"Dataset has {n_samples} samples but {n_clients} clients requested. "
            "Each client needs at least 1 sample."
        )

    # Shuffle
    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)

    # Split into nearly equal parts
    base_size = n_samples // n_clients
    remainder = n_samples % n_clients

    partitions: List[pd.DataFrame] = []
    start = 0
    for i in range(n_clients):
        # First 'remainder' clients get one extra sample
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        partitions.append(shuffled.iloc[start:end])
        start = end

    return partitions


def partition_iid_from_csv(
    csv_path: str,
    n_clients: int = 6,
    random_seed: Optional[int] = 42,
    test_size: float = 0.2,
    stratify_col: Optional[str] = "event",
) -> Tuple[Optional[pd.DataFrame], List[pd.DataFrame]]:
    """
    Load CSV, optionally hold out global test set, and partition remainder into
    n_clients IID subsets.

    Args:
        csv_path: Path to dataset CSV file
        n_clients: Number of client partitions
        random_seed: Random seed for reproducibility
        test_size: Fraction for global test set (0 = no holdout). Default 0.2.
        stratify_col: Column for stratification when splitting (e.g. 'event' for
            Cox). If None or column missing, uses random split. Default 'event'.

    Returns:
        Tuple of (global_test_df or None, list of client DataFrames).
        global_test_df is None if test_size is 0.
    """
    df = pd.read_csv(csv_path)
    federated_pool = df
    global_test = None

    if test_size > 0:
        federated_pool, global_test = split_global_test(
            df,
            test_size=test_size,
            stratify_col=stratify_col,
            random_seed=random_seed,
        )

    client_dfs = partition_iid(
        federated_pool,
        n_clients=n_clients,
        random_seed=random_seed,
    )

    return global_test, client_dfs


def save_global_test(
    global_test_df: pd.DataFrame,
    output_dir: str,
    filename: str = "global_test.csv",
) -> str:
    """
    Save global test set to CSV.

    Args:
        global_test_df: Global test DataFrame
        output_dir: Directory to save file
        filename: Output filename (default: global_test.csv)

    Returns:
        Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    global_test_df.to_csv(path, index=False)
    return path


def save_client_partitions(
    client_dfs: List[pd.DataFrame],
    output_dir: str,
    prefix: str = "client",
) -> List[str]:
    """
    Save each client's DataFrame to a separate CSV file.

    Args:
        client_dfs: List of DataFrames, one per client
        output_dir: Directory to save CSV files
        prefix: Filename prefix (files will be {prefix}_0.csv, {prefix}_1.csv, ...)

    Returns:
        List of paths to saved files
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for i, client_df in enumerate(client_dfs):
        path = os.path.join(output_dir, f"{prefix}_{i}.csv")
        client_df.to_csv(path, index=False)
        paths.append(path)
    return paths
