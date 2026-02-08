"""Evaluation metrics for MINOTAUR."""
import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# =============================================================================
# INPUT NORMALIZATION HELPERS
# =============================================================================

def _normalize_concept_predictions(c_pred, c_true, concept_states, is_logits=False):
    """Normalize concept predictions to standardized format.
    
    Args:
        c_pred: Concept predictions - can be logits (tensor) or probs (numpy array, concatenated)
        c_true: Ground truth concept values (tensor or numpy array)
        concept_states: List of concept states (0=continuous, 1=binary, >1=categorical)
        is_logits: If True, c_pred contains logits (apply sigmoid/softmax)
    
    Returns:
        List of dicts, one per concept: [{'pred': array, 'true': array, 'type': 'binary'/'categorical'/'continuous'}, ...]
    """
    # Convert to numpy
    if isinstance(c_pred, torch.Tensor):
        c_pred_np = c_pred.cpu().detach().numpy()
    else:
        c_pred_np = np.array(c_pred)
    
    if isinstance(c_true, torch.Tensor):
        c_true_np = c_true.cpu().detach().numpy()
    else:
        c_true_np = np.array(c_true)
    
    normalized = []
    idx = 0
    
    for i, n_states in enumerate(concept_states):
        if n_states == 0:
            # Continuous concept
            c_pred_val = c_pred_np[:, idx]
            c_true_val = c_true_np[:, i]
            normalized.append({
                'pred': c_pred_val,
                'true': c_true_val,
                'type': 'continuous'
            })
            idx += 1
        elif n_states == 1:
            # Binary concept
            if is_logits:
                c_logit = torch.tensor(c_pred_np[:, idx])
                c_prob = torch.nn.Sigmoid()(c_logit).cpu().numpy()
            else:
                # Assume probabilities already
                c_prob = c_pred_np[:, idx]
            c_true_val = c_true_np[:, i]
            normalized.append({
                'pred': c_prob,
                'true': c_true_val,
                'type': 'binary'
            })
            idx += 1
        else:
            # Categorical concept
            n_idx = idx + n_states
            if is_logits:
                c_logits = c_pred_np[:, idx:n_idx]
                c_probs = softmax(c_logits, axis=-1)
            else:
                # Assume probabilities already (or apply softmax if needed)
                c_probs = softmax(c_pred_np[:, idx:n_idx], axis=-1)
            c_true_val = c_true_np[:, i]
            normalized.append({
                'pred': c_probs,
                'true': c_true_val,
                'type': 'categorical',
                'n_classes': n_states
            })
            idx = n_idx
    
    return normalized


def _normalize_task_predictions(y_pred, y_true, task_type, is_logits=False):
    """Normalize task predictions to standardized format.
    
    Args:
        y_pred: Task predictions - can be logits (tensor) or probs/predictions (numpy)
        y_true: Ground truth task labels (tensor, numpy, or tuple for Cox)
        task_type: 'binary', 'multiclass', 'continuous', or 'cox'
        is_logits: If True, y_pred contains logits (apply sigmoid/softmax for classification)
    
    Returns:
        Tuple: (y_pred_norm, y_true_norm) as numpy arrays (or tuple for Cox)
    """
    # Convert to numpy
    if isinstance(y_pred, torch.Tensor):
        y_pred_np = y_pred.cpu().detach().numpy()
    else:
        y_pred_np = np.array(y_pred)
    
    if task_type == 'cox':
        # For Cox, y_true is tuple (events, survtimes) - return as-is
        # Flatten y_pred to 1D for concordance_index_censored
        y_pred_np = y_pred_np.flatten()
        return y_pred_np, y_true
    elif task_type == 'continuous':
        # For regression, predictions are already values
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().detach().numpy()
        else:
            y_true_np = np.array(y_true)
        return y_pred_np.flatten(), y_true_np.flatten()
    else:
        # Classification - apply sigmoid or softmax if logits
        if is_logits:
            if y_pred_np.ndim == 1 or (y_pred_np.ndim == 2 and y_pred_np.shape[1] == 1):
                # Binary
                y_logit = torch.tensor(y_pred_np)
                y_prob = torch.nn.Sigmoid()(y_logit).cpu().numpy()
            else:
                # Multiclass
                y_probs = softmax(y_pred_np, axis=-1)
                y_prob = y_probs
        else:
            y_prob = y_pred_np
        
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.cpu().detach().numpy()
        else:
            y_true_np = np.array(y_true)
        
        return y_prob, y_true_np


# =============================================================================
# CORE METRIC FUNCTIONS (Standardized Inputs/Outputs)
# =============================================================================

def _compute_concept_metrics_normalized(normalized_concepts):
    """Compute all concept metrics from normalized inputs.
    
    Args:
        normalized_concepts: List from _normalize_concept_predictions()
    
    Returns:
        Dict: {'concept_0': {'f1': X, 'accuracy': Y, 'roc_auc': Z}, 
               'concept_1': {'mse': X, 'mae': Y, 'r2': Z}, ...}
    """
    metrics = {}
    
    for i, concept_data in enumerate(normalized_concepts):
        c_pred = concept_data['pred']
        c_true = concept_data['true']
        c_type = concept_data['type']
        
        if c_type == 'continuous':
            metrics[f'concept_{i}'] = {
                'mse': mean_squared_error(c_true, c_pred),
                'mae': mean_absolute_error(c_true, c_pred),
                'r2': r2_score(c_true, c_pred),
            }
        elif c_type == 'binary':
            c_pred_binary = (c_pred >= 0.5).astype(int)
            metrics[f'concept_{i}'] = {
                'accuracy': accuracy_score(c_true, c_pred_binary),
                'f1': f1_score(c_true, c_pred_binary),
                'roc_auc': roc_auc_score(c_true, c_pred) if len(np.unique(c_true)) > 1 else accuracy_score(c_true, c_pred_binary),
            }
        else:  # categorical
            c_pred_classes = np.argmax(c_pred, axis=-1)
            metrics[f'concept_{i}'] = {
                'accuracy': accuracy_score(c_true, c_pred_classes),
                'f1': f1_score(c_true, c_pred_classes, average='macro'),
            }
    
    return metrics


def _compute_task_metrics_normalized(y_pred_norm, y_true_norm, task_type):
    """Compute all task metrics from normalized inputs.
    
    Args:
        y_pred_norm: Normalized predictions from _normalize_task_predictions()
        y_true_norm: Normalized true values from _normalize_task_predictions()
        task_type: 'binary', 'multiclass', 'continuous', or 'cox'
    
    Returns:
        Dict with all computed metrics
    """
    if task_type == 'cox':
        try:
            from sksurv.metrics import concordance_index_censored
            events, survtimes = y_true_norm
            
            # Debug: Print information before computing c-index
            # y_pred_norm here is the hazard / risk score output from the model.
            # CoxLoss treats larger values as higher risk (theta), and
            # sksurv.metrics.concordance_index_censored expects that larger
            # estimates correspond to higher risk as well. Therefore we should
            # pass y_pred_norm directly, without negating it.
            print(f"\n[COX DEBUG] _compute_task_metrics_normalized:")
            print(f"  - y_pred_norm shape: {y_pred_norm.shape}, dtype: {y_pred_norm.dtype}")
            print(f"  - y_pred_norm stats: min={y_pred_norm.min():.4f}, max={y_pred_norm.max():.4f}, "
                  f"mean={y_pred_norm.mean():.4f}, std={y_pred_norm.std():.4f}")
            print(f"  - events shape: {events.shape}, dtype: {events.dtype}")
            print(f"  - events stats: min={events.min()}, max={events.max()}, "
                  f"sum={events.sum()} (n_events), n_censored={len(events) - events.sum()}")
            print(f"  - survtimes shape: {survtimes.shape}, dtype: {survtimes.dtype}")
            print(f"  - survtimes stats: min={survtimes.min():.2f}, max={survtimes.max():.2f}, "
                  f"mean={survtimes.mean():.2f}, median={np.median(survtimes):.2f}")
            print(f"  - Risk scores (y_pred_norm) stats: min={y_pred_norm.min():.4f}, "
                  f"max={y_pred_norm.max():.4f}, mean={y_pred_norm.mean():.4f}")
            
            c_index, _, _, _, _ = concordance_index_censored(
                events.astype(bool), survtimes, y_pred_norm
            )
            print(f"  - Computed c-index: {c_index:.4f} ({c_index*100:.2f}%)\n")
            return {'c_index': c_index}
        except ImportError:
            return None
    elif task_type == 'continuous':
        return {
            'mse': mean_squared_error(y_true_norm, y_pred_norm),
            'mae': mean_absolute_error(y_true_norm, y_pred_norm),
            'r2': r2_score(y_true_norm, y_pred_norm),
        }
    else:
        # Binary/multiclass
        if y_pred_norm.ndim == 1 or (y_pred_norm.ndim == 2 and y_pred_norm.shape[1] == 1):
            y_pred_binary = (y_pred_norm >= 0.5).astype(int).flatten()
        else:
            y_pred_binary = np.argmax(y_pred_norm, axis=-1)
        
        result = {
            'accuracy': accuracy_score(y_true_norm, y_pred_binary),
            'f1': f1_score(y_true_norm, y_pred_binary, average='macro' if len(np.unique(y_true_norm)) > 2 else 'binary'),
        }
        
        # ROC-AUC for binary/multiclass
        if len(np.unique(y_true_norm)) > 1:
            if y_pred_norm.ndim == 1 or (y_pred_norm.ndim == 2 and y_pred_norm.shape[1] == 1):
                result['roc_auc'] = roc_auc_score(y_true_norm, y_pred_norm.flatten())
            else:
                try:
                    result['roc_auc'] = roc_auc_score(y_true_norm, y_pred_norm, multi_class='ovo')
                except:
                    result['roc_auc'] = 0.0
        else:
            result['roc_auc'] = result['accuracy']
        
        return result


# =============================================================================
# PUBLIC API FUNCTIONS (Handle Normalization Internally)
# =============================================================================

def compute_concept_metrics(c_pred, c_true, concept_states, is_logits=False):
    """Compute concept metrics for all concepts individually.
    
    Args:
        c_pred: Concept predictions (logits tensor or probs numpy array)
        c_true: Ground truth concept values
        concept_states: List of concept states
        is_logits: Whether c_pred contains logits (vs probabilities)
    
    Returns:
        Dict: {'concept_0': {'f1': X, 'accuracy': Y, 'roc_auc': Z}, 
               'concept_1': {'mse': X, 'mae': Y, 'r2': Z}, ...}
    """
    normalized = _normalize_concept_predictions(c_pred, c_true, concept_states, is_logits)
    return _compute_concept_metrics_normalized(normalized)


def compute_task_metrics_all(y_pred, y_true, task_type='binary', is_logits=False):
    """Compute all task metrics.
    
    Args:
        y_pred: Task predictions (logits tensor or probs numpy)
        y_true: Ground truth labels
        task_type: 'binary', 'multiclass', 'continuous', or 'cox'
        is_logits: Whether y_pred contains logits
    
    Returns:
        Dict with all computed metrics:
        - Binary/multiclass: {'accuracy': X, 'f1': Y, 'roc_auc': Z}
        - Continuous: {'mse': X, 'mae': Y, 'r2': Z}
        - Cox: {'c_index': X} or None if sksurv not available
    """
    y_pred_norm, y_true_norm = _normalize_task_predictions(y_pred, y_true, task_type, is_logits)
    return _compute_task_metrics_normalized(y_pred_norm, y_true_norm, task_type)


# =============================================================================
# TRAINING CONVENIENCE FUNCTIONS (Single Aggregated Metrics)
# =============================================================================

def compute_concept_metric_single(c_pred, c_true, concept_states):
    """For training: compute single aggregated concept metric.
    
    Args:
        c_pred: Concept logits (tensor)
        c_true: Ground truth concept values (tensor)
        concept_states: List of concept states
    
    Returns:
        Tuple: (aggregated_value, metric_type) where metric_type is 'f1' or 'mse'
    """
    metrics = compute_concept_metrics(c_pred, c_true, concept_states, is_logits=True)
    
    # Aggregate: average F1 for binary/categorical, average MSE for continuous
    has_continuous = any(s == 0 for s in concept_states)
    
    if has_continuous:
        mse_values = [m['mse'] for m in metrics.values() if 'mse' in m]
        return (np.mean(mse_values) if mse_values else 0.0, 'mse')
    else:
        f1_values = [m['f1'] for m in metrics.values() if 'f1' in m]
        return (np.mean(f1_values) if f1_values else 0.0, 'f1')


def compute_task_metric_single(y_pred, y_true, task_type='binary'):
    """For training: compute single task metric.
    
    Args:
        y_pred: Task logits (tensor)
        y_true: Ground truth labels (tensor or tuple for Cox)
        task_type: 'binary', 'multiclass', 'continuous', or 'cox'
    
    Returns:
        Single scalar metric value (F1, MSE, or 0.0 for Cox during batch)
    """
    if task_type == 'cox':
        # C-index requires all samples - return 0 during batch training
        return 0.0
    
    metrics = compute_task_metrics_all(y_pred, y_true, task_type, is_logits=True)
    
    if metrics is None:
        return 0.0
    
    if task_type == 'continuous':
        return metrics['mse']
    else:
        return metrics['f1']


# =============================================================================
# BACKWARD COMPATIBILITY WRAPPERS (Temporary - for explainable/ scripts)
# =============================================================================

def get_cat_concept_metrics(c_probs, c_test, concept_states):
    """Backward compatibility wrapper - returns lists per concept.
    
    Returns:
        Tuple: (c_accs, c_f1s, c_probs_lst)
    """
    metrics = compute_concept_metrics(c_probs, c_test, concept_states, is_logits=False)
    
    c_accs = []
    c_f1s = []
    c_probs_lst = []
    idx = 0
    
    for i, n in enumerate(concept_states):
        if n == 0:
            continue  # Skip continuous
        key = f'concept_{i}'
        c_accs.append(metrics[key]['accuracy'])
        c_f1s.append(metrics[key]['f1'])
        
        # Extract probabilities for this concept
        if n == 1:
            n_idx = idx + 1
        else:
            n_idx = idx + n
        c_probs_lst.append(softmax(c_probs[:, idx:n_idx], axis=-1))
        idx = n_idx
    
    return c_accs, c_f1s, c_probs_lst


def get_bin_concept_metrics(c_probs, c_test, concept_states):
    """Backward compatibility wrapper for binary concepts.
    
    Returns:
        Tuple: (c_accuracies, c_f1s, cpt_probs)
    """
    metrics = compute_concept_metrics(c_probs, c_test, concept_states, is_logits=False)
    
    c_accuracies = []
    c_f1s = []
    cpt_probs = []
    
    for i, n in enumerate(concept_states):
        if n == 0:
            continue
        key = f'concept_{i}'
        c_accuracies.append(metrics[key]['accuracy'])
        c_f1s.append(metrics[key]['f1'])
        
        # Get probabilities
        if isinstance(c_probs, torch.Tensor):
            cpt_prob = torch.nn.Sigmoid()(c_probs[:, i]).cpu().numpy()
        else:
            cpt_prob = c_probs[:, i]
        cpt_probs.append(cpt_prob)
    
    return c_accuracies, c_f1s, (cpt_probs[0] if cpt_probs else None)


def get_regression_concept_metrics(c_probs, c_test, concept_states, concept_idx):
    """Backward compatibility wrapper for continuous concepts.
    
    Returns:
        Tuple: (mse, mae, r2, c_pred)
    """
    metrics = compute_concept_metrics(c_probs, c_test, concept_states, is_logits=False)
    
    key = f'concept_{concept_idx}'
    concept_metrics = metrics.get(key, {})
    
    # Extract prediction
    idx = 0
    for i, n in enumerate(concept_states):
        if i == concept_idx:
            break
        idx += n if n > 0 else 1
    
    c_pred = c_probs[:, idx] if isinstance(c_probs, np.ndarray) else c_probs[:, idx].cpu().numpy()
    
    return (
        concept_metrics.get('mse', 0.0),
        concept_metrics.get('mae', 0.0),
        concept_metrics.get('r2', 0.0),
        c_pred
    )
