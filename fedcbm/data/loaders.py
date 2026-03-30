"""Data loader utilities for MINOTAUR."""
import os
import glob
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from .datasets import TileDataset, MultinominalSampler


def validate_batch_size(dataset_size: int, batch_size: int) -> int:
    """
    Validate that batch size won't cause issues with BatchNorm.
    
    Args:
        dataset_size: Size of the dataset
        batch_size: Desired batch size
        
    Returns:
        Validated batch size
        
    Raises:
        ValueError: If dataset is too small for BatchNorm
    """
    if dataset_size < 2:
        raise ValueError(
            f"Dataset size ({dataset_size}) must be at least 2 for BatchNorm to work"
        )
    elif dataset_size < batch_size:
        return 2  # Use minimum batch size of 2 for BatchNorm
    elif dataset_size % batch_size == 1:
        return batch_size - 1  # Avoid single-sample final batch
    return batch_size


def get_data_loaders(
    db_path: str,
    train_data,
    val_data,
    config,
    num_workers: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Set up data loaders for training and validation.
    
    Args:
        db_path: Path to database directory
        train_data: Training DataFrame
        val_data: Validation DataFrame
        config: Configuration object with required attributes:
            - db_path: Database path
            - target: Target column name
            - cpt_ids: List of concept IDs
            - bag_n: Number of tiles per bag
            - batch_size: Batch size for training
            - task_type: Task type (optional, defaults to 'binary')
        num_workers: Number of DataLoader workers (default: cpu_count()/6)
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if num_workers is None:
        num_workers = int(os.cpu_count() / 6)
    
    # Get database paths
    db_paths = glob.glob(os.path.join(db_path, 'features', '*'))
    
    # Filter to match training data IDs
    train_dbs = [
        p
        for p in db_paths
        for db in list(train_data['ID'])
        if db == os.path.basename(p)[:-4]
    ]
    
    train_weights = [1.0 for _ in range(len(train_dbs))]
    mn_sampler = MultinominalSampler(
        train_weights,
        len(train_dbs)
    )

    # Create datasets
    task_type = getattr(config, 'task_type', 'binary')
    train_dataset = TileDataset(
        train_data,
        config.data.db_path,
        config.data.target,
        config.concepts.cpt_ids,
        config.data.bag_n,
        task_type=task_type
    )
    val_dataset = TileDataset(
        val_data,
        config.data.db_path,
        config.data.target,
        config.concepts.cpt_ids,
        config.data.bag_n,
        task_type=task_type
    )

    # Validate batch sizes
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    validated_train_batch_size = validate_batch_size(train_size, config.training.batch_size)
    validated_val_batch_size = validate_batch_size(val_size, config.training.val_batch_size)

    print(f"Train size: {train_size}, Original batch: {config.training.batch_size}, "
          f"Validated batch: {validated_train_batch_size}")
    print(f"Val size: {val_size}, Original batch: 4, "
          f"Validated batch: {validated_val_batch_size}")

    # Create data loaders
    train_dloader = DataLoader(
        train_dataset,
        batch_size=validated_train_batch_size,
        sampler=mn_sampler,
        num_workers=num_workers,
        drop_last=True
    )

    val_dloader = DataLoader(
        val_dataset,
        batch_size=validated_val_batch_size,
        num_workers=num_workers,
        drop_last=False
    )

    return train_dloader, val_dloader


