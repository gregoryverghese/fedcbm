"""Training utilities for MINOTAUR."""
import os
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold
from typing import List, Optional, Dict, Any, Tuple, Sequence
from datetime import datetime
from scipy.special import softmax

# Note: AttentionHook is in callbacks.py, not imported here to avoid circular dependency


def get_concept_weights(cohort, concepts, categorical=False):
    """Calculate concept class weights for imbalanced datasets.
    
    Args:
        cohort: DataFrame with concept columns
        concepts: List of concept column names
        categorical: Whether concepts are categorical
        
    Returns:
        List of class weights for each concept
    """
    col_weights_dic = {}
    for column in concepts:
        value_counts = cohort[column].value_counts().sort_index()
        if categorical:
            total_counts = sum(value_counts)
            weights = {cls: total_counts / (len(value_counts) * v) for cls, v in value_counts.items()}
            col_weights_dic[column] = weights
        else:
            count_0 = value_counts.get(0, 0)
            count_1 = value_counts.get(1, 0)
            weight = count_0 / count_1 if count_1 > 0 else 1.0
            col_weights_dic[column] = {1: weight}
 
    cpt_class_weights = [v2 for k, v in col_weights_dic.items() for k2, v2 in v.items()]
    return cpt_class_weights


def get_cptprobs(batch_results, cat=False):
    """Extract concept probabilities from batch results.
    
    Args:
        batch_results: List of batch prediction results
        cat: Whether concepts are categorical
        
    Returns:
        numpy array of concept probabilities
    """
    c_probs = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)), axis=0
    )
    return c_probs


def get_yprobs(batch_results, task_type='binary'):
    """Extract y predictions from batch results.
    
    Args:
        batch_results: List of batch prediction results
        task_type: Type of task - 'binary', 'multiclass', 'continuous', or 'cox'
                   For 'cox', returns raw hazard predictions (no sigmoid)
                   For other types, applies sigmoid for probabilities
    
    Returns:
        numpy array of predictions
    """
    if task_type == 'cox':
        # For Cox regression, return raw hazard predictions without sigmoid
        y_probs = np.concatenate(
            list(map(lambda x: x[2].detach().cpu().numpy(), batch_results)),
            axis=0,
        )
        # Debug: Print prediction statistics
        print(f"\n[COX DEBUG] get_yprobs: Extracted hazard predictions")
        print(f"  - y_probs shape: {y_probs.shape}")
        print(f"  - y_probs stats: min={y_probs.min():.4f}, max={y_probs.max():.4f}, "
              f"mean={y_probs.mean():.4f}, std={y_probs.std():.4f}")
        print(f"  - First 10 predictions: {y_probs[:10].flatten().tolist()}\n")
    else:
        # For binary/multiclass, apply sigmoid to get probabilities
        y_probs = np.concatenate(
            list(map(lambda x: torch.nn.Sigmoid()(x[2]).detach().cpu().numpy(), batch_results)),
            axis=0,
        )
    return y_probs


def get_cembs(batch_results):
    """Extract concept embeddings from batch results.
    
    Args:
        batch_results: List of batch prediction results
        
    Returns:
        numpy array of concept embeddings
    """
    c_embs = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    return c_embs


def get_contexts(batch_results):
    """Extract contexts from batch results.
    
    Args:
        batch_results: List of batch prediction results
        
    Returns:
        numpy array of contexts
    """
    contexts = np.concatenate(
        list(map(lambda x: x[5].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    return contexts


def get_concept_attention_weights(batch_results):
    """Extract concept attention weights from batch results.
    
    Args:
        batch_results: List of tuples from model prediction
        
    Returns:
        numpy array of shape (n_samples, n_concepts) with attention weights
        Returns None if no_concepts mode or attention weights not available
    """
    attention_weights_list = []
    for x in batch_results:
        if len(x) > 6 and x[6] is not None:  # x[6] is concept_attention_weights
            attention_weights_list.append(x[6].detach().cpu().numpy())
        else:
            return None
    
    if not attention_weights_list:
        return None
        
    attention_weights = np.concatenate(attention_weights_list, axis=0)
    return attention_weights


def stratif_kfold(dataset, target, k, random_state=1):
    """Perform Stratified K-Folds cross-validation splitting of the dataset.
    
    This function splits the dataset into `k` stratified folds for cross-validation, 
    ensuring that each fold maintains the distribution of the target class.

    Args:
        dataset: The input dataset containing the features and target
        target: The name of the target column in the dataset
        k: The number of folds to split the dataset into
        random_state: Random seed for reproducibility of the splits

    Returns:
        DataFrame with an additional column, `test`, indicating the fold number
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    
    x, _ = list(zip(*enumerate(dataset['ID'])))
    y = dataset[target].values
    test_folds = np.zeros(len(y), dtype=int)
    for fold_idx, (train, test) in enumerate(skf.split(x, y)):
        for i in test:
            test_folds[i] = fold_idx

    dataset['test'] = test_folds
    return dataset


def prepare_kfold_splits(
    train_cohort: pd.DataFrame,
    target: str,
    k_fold: int,
    random_state: int,
    save_path: str
) -> pd.DataFrame:
    """
    Prepare and save k-fold cross-validation splits.
    
    Args:
        train_cohort: Full training dataset DataFrame
        target: Target column name for stratification
        k_fold: Number of folds
        random_state: Random seed for reproducibility
        save_path: Path to save the fold assignments CSV
        
    Returns:
        DataFrame with 'test' column indicating fold assignments
    """
    kfold_dataset = stratif_kfold(train_cohort.copy(), target, k_fold, random_state=random_state)
    
    # Save fold assignments
    os.makedirs(save_path, exist_ok=True)
    strat_csv_path = os.path.join(save_path, 'strat.csv')
    kfold_dataset.to_csv(strat_csv_path, index=False)
    
    # Print fold distribution
    fold_counts = kfold_dataset['test'].value_counts().sort_index()
    print(f"Fold distribution:")
    for fold_idx, count in fold_counts.items():
        print(f"  Fold {fold_idx}: {count} samples")
    
    return kfold_dataset


def get_fold_split(kfold_dataset: pd.DataFrame, fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get train and validation data for a specific fold.
    
    Args:
        kfold_dataset: Dataset with 'test' column indicating fold assignments
        fold_idx: Index of the fold to use as validation (0 to k-1)
        
    Returns:
        Tuple of (train_data, val_data) DataFrames
    """
    train_data = kfold_dataset[kfold_dataset['test'] != fold_idx].copy()
    val_data = kfold_dataset[kfold_dataset['test'] == fold_idx].copy()
    return train_data, val_data


def get_sweep_id() -> str:
    """Generate unique sweep ID from current date and time.
    
    Returns:
        String in format 'sweep_YYYY-MM-DD_HH-MM-SS' based on current datetime
    """
    return f"sweep_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def build_trial_name(config) -> str:
    """Build trial name from hyperparameters in decimal format.
    
    Args:
        config: Configuration object with training, model, and data settings
        
    Returns:
        String with hyperparameters in format 'lr=X_bs=Y_wd=Z_seed=N_n_bags=...'
    """
    lr = config.training.learning_rate
    bs = config.training.batch_size
    wd = config.training.weight_decay
    seed = config.random_seed
    n_bags = config.data.bag_n
    c_weight = config.model.c_weight
    emb_dim = config.model.emb_dim
    
    # Handle dropout: convert boolean to float if needed
    dropout = config.model.dropout
    if isinstance(dropout, bool):
        dropout = 0.3 if dropout else 0.0
    dropout = float(dropout)
    
    return f"lr={lr}_bs={bs}_wd={wd}_seed={seed}_n_bags={n_bags}_c_weight={c_weight}_emb_dim={emb_dim}_dropout={dropout}"


def collect_hyperparameters(config) -> Dict[str, Any]:
    """Collect hyperparameters for logging to TensorBoard.
    
    Args:
        config: Configuration object with training, model, and data settings
        
    Returns:
        Dictionary mapping hyperparameter names to their values
    """
    hparams = {
        'learning_rate': config.training.learning_rate,
        'batch_size': config.training.batch_size,
        'weight_decay': config.training.weight_decay,
        'random_seed': config.random_seed,
        'bag_n': config.data.bag_n,
        'c_weight': config.model.c_weight,
        'emb_dim': config.model.emb_dim,
        'h_dim': config.model.h_dim,
        'dropout': config.model.dropout if isinstance(config.model.dropout, (int, float)) else (0.3 if config.model.dropout else 0.0),
        'optimizer': config.training.optimizer,
        'max_epochs': config.training.max_epochs,
    }
    return hparams


def build_predictions_dataframe(
    patient_ids: Sequence[str],
    results: List,
    config: Any,
    run_id: Optional[Any] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
    c_embs: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build a DataFrame of test-set predictions (task + concepts).

    One row per test sample. Columns: patient_id, optional run_id and run_metadata,
    task ground truth/pred/probs, then per-concept ground truth/pred/probs,
    and optionally per-concept embedding dimensions (e.g. c_emb_<name>_0 .. c_emb_<name>_7).
    Caller is responsible for saving (e.g. df.to_csv()).

    Args:
        patient_ids: Patient/sample IDs in same order as test set (length N).
        results: List [c_probs, c_test, y_probs, y_test, ...] from predict().
        config: Config with task_type, concepts.cpt_ids, concepts.concept_states,
                concepts.no_concepts, model.emb_dim.
        run_id: Optional identifier for this run (e.g. repeat index, or (repeat, fold)).
        run_metadata: Optional dict of extra columns (e.g. fold, repeat, combination).
        c_embs: Optional concept embeddings array of shape (n, n_concepts, emb_dim).

    Returns:
        DataFrame with predictions.
    """
    c_probs, c_test, y_probs, y_test = results[0], results[1], results[2], results[3]
    task_type = getattr(config, "task_type", "binary")
    n = len(patient_ids)

    # Ensure arrays are numpy and 1d/2d as expected
    y_probs = np.asarray(y_probs)
    if y_probs.ndim == 1:
        y_probs = np.atleast_2d(y_probs).T
    c_test = np.asarray(c_test)

    data = {"patient_id": list(patient_ids)}

    if run_id is not None:
        data["run_id"] = [run_id] * n

    if run_metadata:
        for k, v in run_metadata.items():
            data[k] = [v] * n

    # Task columns
    if task_type == "cox":
        events, survtimes = y_test
        events = np.asarray(events).flatten()
        survtimes = np.asarray(survtimes).flatten()
        hazard = np.asarray(y_probs).flatten()
        data["event"] = events
        data["time"] = survtimes
        data["hazard"] = hazard
    elif task_type == "continuous":
        y_true = np.asarray(y_test).flatten()
        y_pred = y_probs.flatten()
        data["y_true"] = y_true
        data["y_pred"] = y_pred
    else:
        # Binary or multiclass
        y_true = np.asarray(y_test).flatten()
        if y_probs.shape[1] == 1:
            prob = y_probs[:, 0]
            data["y_true"] = y_true
            data["y_pred"] = (prob >= 0.5).astype(int)
            data["y_prob"] = prob
        else:
            probs = softmax(y_probs, axis=-1) if y_probs.shape[1] > 1 else y_probs
            data["y_true"] = y_true
            data["y_pred"] = np.argmax(probs, axis=1)
            for j in range(probs.shape[1]):
                data[f"y_prob_{j}"] = probs[:, j]

    # Concept columns (skip if no concepts)
    if not getattr(config.concepts, "no_concepts", False) and c_probs is not None:
        cpt_ids = getattr(config.concepts, "cpt_ids", [])
        concept_states = getattr(config.concepts, "concept_states", [])
        c_probs = np.asarray(c_probs)
        idx = 0
        for i, name in enumerate(cpt_ids):
            if i >= len(concept_states):
                break
            n_states = concept_states[i]
            c_true = c_test[:, i]

            if n_states == 0:
                # Continuous
                c_pred = c_probs[:, idx]
                data[f"c_true_{name}"] = c_true
                data[f"c_pred_{name}"] = c_pred
                idx += 1
            elif n_states == 1:
                # Binary: one prob column
                logit = c_probs[:, idx]
                prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500)))
                pred = (prob >= 0.5).astype(int)
                data[f"c_true_{name}"] = c_true
                data[f"c_pred_{name}"] = pred
                data[f"c_prob_{name}"] = prob
                idx += 1
            else:
                # Categorical: softmax and one column per class
                n_idx = idx + n_states
                slice_ = c_probs[:, idx:n_idx]
                probs = softmax(slice_, axis=-1)
                pred = np.argmax(probs, axis=1)
                data[f"c_true_{name}"] = c_true
                data[f"c_pred_{name}"] = pred
                for j in range(n_states):
                    data[f"c_prob_{name}_{j}"] = probs[:, j]
                idx = n_idx

    # Concept embedding columns (when c_embs provided and concepts enabled)
    if c_embs is not None and not getattr(config.concepts, "no_concepts", False):
        c_embs = np.asarray(c_embs)
        if c_embs.ndim == 3 and c_embs.shape[0] == n:
            n_concepts = c_embs.shape[1]
            emb_dim = getattr(config.model, "emb_dim", c_embs.shape[2])
            cpt_ids = getattr(config.concepts, "cpt_ids", [])
            for i in range(n_concepts):
                name = cpt_ids[i] if i < len(cpt_ids) else str(i)
                for j in range(emb_dim):
                    data[f"c_emb_{name}_{j}"] = c_embs[:, i, j]

    return pd.DataFrame(data)

