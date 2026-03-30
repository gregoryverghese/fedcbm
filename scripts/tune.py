#!/usr/bin/env python
"""Hyperparameter tuning script for MINOTAUR."""
import os
import sys
import warnings
import argparse
import pandas as pd
import numpy as np
import itertools
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

warnings.filterwarnings("ignore")

# Set environment variables if needed
#if not os.getenv("CUDA_LAUNCH_BLOCKING"):
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from fedcbm.config import load_config_from_yaml
from fedcbm.training import train
from fedcbm.training.metrics import compute_task_metrics_all, compute_concept_metrics
from fedcbm.training.utils import (
    get_sweep_id, 
    build_trial_name, 
    collect_hyperparameters,
    prepare_kfold_splits,
    get_fold_split
)


def extract_concept_primary_metrics(c_probs, c_test, config) -> Dict[str, Any]:
    """
    Extract primary concept metrics for tuning output.

    For each concept:
      - If continuous (state == 0): use MSE
      - If binary/categorical (state > 0): use F1

    Returns:
        Dict mapping column names to metric values, e.g.:
        {
            "concept_Stage_f1": 0.82,
            "concept_Age_mse": 0.15,
            ...
        }
    """
    concept_metrics = compute_concept_metrics(
        c_probs,
        c_test,
        config.concepts.concept_states,
        is_logits=False,
    )

    cols: Dict[str, Any] = {}
    concept_ids = config.concepts.cpt_ids

    for i, concept_id in enumerate(concept_ids):
        key = f"concept_{i}"
        metrics = concept_metrics.get(key, {})
        n_states = config.concepts.concept_states[i]

        # Make column name safe (no spaces)
        concept_name = str(concept_id).replace(" ", "_")

        if n_states == 0:
            # Continuous concept: store MSE
            cols[f"concept_{concept_name}_mse"] = metrics.get("mse", None)
        else:
            # Binary / categorical concept: store F1
            cols[f"concept_{concept_name}_f1"] = metrics.get("f1", None)

    return cols


def generate_hyperparameter_combinations(config, hyperparams):
    """Generate all combinations of hyperparameters from config."""
    
    # Extract search spaces
    n_bags = hyperparams.get('n_bags', [config.data.bag_n])
    batches = hyperparams.get('batches', [config.training.batch_size])
    cpt_weights = hyperparams.get('cpt_weights', [config.model.c_weight])
    emb_sizes = hyperparams.get('emb_sizes', [config.model.emb_dim])
    dropouts = hyperparams.get('dropouts', [config.model.dropout])
    attns = hyperparams.get('attns', [config.model.attn])
    
    # Generate all combinations
    combinations = list(itertools.product(
        n_bags, batches, cpt_weights, emb_sizes, dropouts, attns
    ))
    
    return combinations


def apply_hyperparameters(config, combination):
    """Apply a hyperparameter combination to config."""
    n_bags, batches, cpt_weights, emb_sizes, dropouts, attns = combination
    
    # Update config with hyperparameters
    config.data.bag_n = n_bags
    config.training.batch_size = batches
    config.model.c_weight = cpt_weights
    config.model.emb_dim = emb_sizes
    config.model.dropout = dropouts
    config.model.attn = attns
    
    return config


def create_config_for_run(
    config_path: str,
    combination: Tuple,
    args: argparse.Namespace,
    results_dir: str,
    combo_idx: int,
    fold_idx: Optional[int] = None,
    repeat: int = 1
) -> Tuple[Any, str, str]:
    """
    Create and configure a config object for a single training run.
    
    Args:
        config_path: Path to base config YAML
        combination: Hyperparameter combination tuple
        args: Command line arguments
        results_dir: Directory to save results
        combo_idx: Index of the combination
        fold_idx: Optional fold index (None if not using k-fold)
        repeat: Repeat number
        
    Returns:
        Tuple of (config, save_path, trial_name)
    """
    # Create a copy of config and apply hyperparameters
    config = load_config_from_yaml(config_path)
    config = apply_hyperparameters(config, combination)
    config.data.db_path = args.db_path
    # For tuning, use the common results_dir as the base save path so that
    # all TensorBoard logs live under:
    #   results_dir/sweep_YYYY-MM-DD/<trial_name>/
    config.output.save_path = results_dir
    if not config.output.log_path:
        config.output.log_path = os.path.join(results_dir, 'logs')
    config.task_type = args.task_type
    
    # Ensure the common results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate trial name (used as TensorBoard 'version')
    # Always include repeat number to avoid overwriting repeats
    trial_name = build_trial_name(config)
    if fold_idx is not None:
        # With k-fold: include both fold and repeat
        trial_name = f"{trial_name}_fold{fold_idx+1}_rep{repeat}"
    else:
        # Without k-fold: include repeat number
        trial_name = f"{trial_name}_rep{repeat}"
    
    # For backwards compatibility, we still return a save_path value, but this
    # now points to the common results_dir rather than a per-combination folder.
    save_path = results_dir
    return config, save_path, trial_name


def extract_metrics_from_results(results: List, config) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
    """
    Extract task and concept metrics from training results.
    
    Args:
        results: Results tuple from training
        config: Configuration object
        
    Returns:
        Tuple of (metric_name, metric_value, concept_cols_dict)
    """
    c_probs, c_test, y_probs, y_test, c_embs, ctxts, concept_attention_weights, patient_ids = results
    
    # Compute task metrics
    task_metrics = compute_task_metrics_all(y_probs, y_test, config.task_type)
    
    if task_metrics is None:
        task_metric = None
        metric_name = None
    else:
        # Extract primary metric based on task type
        if config.task_type == 'cox':
            metric_name = 'c_index'
            task_metric = task_metrics.get('c_index')
        elif config.task_type == 'continuous':
            metric_name = 'mse'
            task_metric = task_metrics.get('mse')
        else:
            # Binary/multiclass - use roc_auc as primary metric
            metric_name = 'roc_auc'
            task_metric = task_metrics.get('roc_auc')
    
    # Extract per-concept primary metrics
    concept_cols = extract_concept_primary_metrics(c_probs, c_test, config)
    
    return metric_name, task_metric, concept_cols


def run_single_experiment(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    concepts: List[str],
    db_path: str,
    config_path: str,
    combination: Tuple,
    args: argparse.Namespace,
    results_dir: str,
    sweep_dir: str,
    combo_idx: int,
    sweep_id: str,
    fold_idx: Optional[int] = None,
    repeat: int = 1,
    use_kfold: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run a single training experiment and return results.
    
    Args:
        train_data: Training data DataFrame (or full cohort if not using k-fold)
        val_data: Optional validation data DataFrame (None for standard mode)
        concepts: List of concept column names
        db_path: Path to embeddings database
        config_path: Path to base config YAML
        combination: Hyperparameter combination tuple
        args: Command line arguments
        results_dir: Directory to save results
        sweep_dir: Sweep directory (results_dir/sweep_id) for predictions and summary
        combo_idx: Index of the combination
        sweep_id: Sweep ID for TensorBoard
        fold_idx: Optional fold index
        repeat: Repeat number
        use_kfold: Whether using k-fold CV
        
    Returns:
        Dictionary with results or None if training failed
    """
    n_bags, batches, cpt_weights, emb_sizes, dropouts, attns = combination
    
    # Create config for this run
    config, save_path, trial_name = create_config_for_run(
        config_path, combination, args, results_dir, combo_idx, fold_idx, repeat
    )

    run_metadata = {
        "fold": (fold_idx + 1) if fold_idx is not None else None,
        "repeat": repeat + 1,  # 1-indexed for clarity
    }
    run_metadata["combination"] = combo_idx + 1
    run_id = (repeat + 1, fold_idx + 1) if fold_idx is not None else (repeat + 1,)
    
    try:
        # Run training
        if use_kfold and val_data is not None:
            # K-fold mode: use pre-split data
            results, trainer, predictions_df = train(
                train_cohort=None,  # Not needed when using pre-split data
                concepts=concepts,
                db_path=db_path,
                config=config,
                experiment_name=sweep_id,
                version=trial_name,
                train_data=train_data,
                val_data=val_data,
                run_id=run_id,
                run_metadata=run_metadata,
            )
        else:
            # Standard mode: train() will split internally
            results, trainer, predictions_df = train(
                train_cohort=train_data,
                concepts=concepts,
                db_path=db_path,
                config=config,
                experiment_name=sweep_id,
                version=trial_name,
                run_id=run_id,
                run_metadata=run_metadata,
            )
        
        # Log hyperparameters to TensorBoard
        hparams = collect_hyperparameters(config)
        trainer.logger.log_hyperparams(hparams)

        # Save final model checkpoint next to TensorBoard logs
        # trainer.logger.log_dir points to: {save_dir}/{sweep_id}/{trial_name}
        if getattr(trainer, "logger", None) is not None and getattr(trainer.logger, "log_dir", None):
            checkpoint_path = os.path.join(trainer.logger.log_dir, "model.ckpt")
        else:
            # Fallback: save under run-specific save_path
            checkpoint_path = os.path.join(save_path, "model.ckpt")
        trainer.save_checkpoint(checkpoint_path)
        print(f"  Model checkpoint saved to: {checkpoint_path}")
        
        # Extract metrics
        metric_name, task_metric, concept_cols = extract_metrics_from_results(results, config)
        
        # Build result dictionary (exclude predictions_df from metrics dict; collected separately)
        result = {
            'combination': combo_idx + 1,
            'fold': fold_idx + 1 if fold_idx is not None else None,
            'repeat': repeat,
            'predictions_df': predictions_df,
            'bag_n': n_bags,
            'batch_size': batches,
            'c_weight': cpt_weights,
            'emb_dim': emb_sizes,
            'dropout': dropouts,
            'attn': attns,
            'metric_name': metric_name,
            'metric_value': task_metric,
            'save_path': save_path,
            'checkpoint_path': checkpoint_path,
        }
        # Add concept-level metrics
        result.update(concept_cols)
        
        # Print result
        if task_metric is not None:
            print(f"  {metric_name}: {task_metric*100:.2f}%")
        else:
            print(f"  Metric calculation skipped")
        
        return result
        
    except Exception as e:
        fold_info = f", fold {fold_idx + 1}" if fold_idx is not None else ""
        print(f"ERROR: Training failed for combination {combo_idx + 1}{fold_info}, repeat {repeat}")
        print(f"  Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main hyperparameter tuning function."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for MINOTAUR')
    parser.add_argument('-dp', '--data_path', required=True, help='Path to dataset CSV')
    parser.add_argument('-db', '--db_path', required=True, help='Path to embeddings database')
    parser.add_argument('-sp', '--save_path', required=True, help='Path to save outputs')
    parser.add_argument('-cp', '--config_path', required=True, help='Path to YAML config file')
    parser.add_argument('--task-type', default='binary',
                       choices=['binary', 'multiclass', 'continuous', 'cox'],
                       help='Task type (default: binary)')

    args = parser.parse_args()

    # Load data
    train_cohort = pd.read_csv(args.data_path)

    # Load base configuration from YAML
    base_config = load_config_from_yaml(args.config_path)
    base_config.data.db_path = args.db_path
    base_config.output.save_path = args.save_path
    if not base_config.output.log_path:
        base_config.output.log_path = os.path.join(args.save_path, 'logs')
    base_config.task_type = args.task_type

    # Load raw YAML to get hyperparameter_search (not in Config dataclass)
    with open(args.config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    hyperparams = config_dict.get('training', {}).get('hyperparameter_search', {})
    
    # Check if hyperparameter search is configured
    if not hyperparams:
        print("ERROR: No hyperparameter_search section found in config.training")
        print("Please add a hyperparameter_search section to your YAML config file.")
        sys.exit(1)

    # Setup k-fold CV if needed
    k_fold = base_config.data.k_fold
    use_kfold = k_fold > 1
    kfold_dataset = None
    
    if use_kfold:
        print(f"\nPreparing {k_fold}-fold cross-validation splits...")
        kfold_dataset = prepare_kfold_splits(
            train_cohort=train_cohort,
            target=base_config.data.target,
            k_fold=k_fold,
            random_state=base_config.data.k_fold_random_state,
            save_path=args.save_path
        )
        print(f"Fold assignments saved to: {os.path.join(args.save_path, 'strat.csv')}")

    # Generate hyperparameter combinations
    concepts = base_config.concepts.cpt_ids
    combinations = generate_hyperparameter_combinations(base_config, hyperparams)
    n_repeats = hyperparams.get('n_repeats', 1)
    
    # Print setup information
    print(f"\nTotal hyperparameter combinations: {len(combinations)}")
    print(f"Repeats per combination: {n_repeats}")
    if use_kfold:
        print(f"K-fold CV: {k_fold} folds")
        print(f"Total training runs: {len(combinations) * k_fold * n_repeats}")
    else:
        print(f"K-fold CV: Disabled (k_fold={k_fold})")
        print(f"Total training runs: {len(combinations) * n_repeats}")
    
    # Setup output directory
    results_dir = os.path.join(args.save_path, 'tuning_results')
    os.makedirs(results_dir, exist_ok=True)
    sweep_id = get_sweep_id()
    sweep_dir = os.path.join(results_dir, sweep_id)
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"Sweep ID: {sweep_id}")
    
    # Run experiments
    all_results = []
    all_predictions_dfs = []

    for combo_idx, combination in enumerate(combinations):
        n_bags, batches, cpt_weights, emb_sizes, dropouts, attns = combination
        
        print(f"\n{'='*80}")
        print(f"Combination {combo_idx + 1}/{len(combinations)}")
        print(f"  bag_n: {n_bags}, batch_size: {batches}, c_weight: {cpt_weights}")
        print(f"  emb_dim: {emb_sizes}, dropout: {dropouts}, attn: {attns}")
        print(f"{'='*80}")
        
        if use_kfold:
            # K-fold CV mode: iterate over folds, then repeats
            for fold_idx in range(k_fold):
                print(f"\nFold {fold_idx + 1}/{k_fold}")
                
                # Get train/val split for this fold (same splits reused for all repeats)
                train_fold_data, val_fold_data = get_fold_split(kfold_dataset, fold_idx)
                
                for repeat in range(n_repeats):
                    print(f"  Repeat {repeat + 1}/{n_repeats}")
                    
                    result = run_single_experiment(
                        train_data=train_fold_data,
                        val_data=val_fold_data,
                        concepts=concepts,
                        db_path=args.db_path,
                        config_path=args.config_path,
                        combination=combination,
                        args=args,
                        results_dir=results_dir,
                        sweep_dir=sweep_dir,
                        combo_idx=combo_idx,
                        sweep_id=sweep_id,
                        fold_idx=fold_idx,
                        repeat=repeat,
                        use_kfold=True,
                    )

                    if result is not None:
                        pred_df = result.pop('predictions_df', None)
                        if pred_df is not None:
                            all_predictions_dfs.append(pred_df)
                        all_results.append(result)
        else:
            # Standard mode: single train/val split, iterate over repeats
            for repeat in range(n_repeats):
                print(f"\nRepeat {repeat + 1}/{n_repeats}")
                
                result = run_single_experiment(
                    train_data=train_cohort,  # Full cohort, train() will split internally
                    val_data=None,  # Not used in standard mode
                    concepts=concepts,
                    db_path=args.db_path,
                    config_path=args.config_path,
                    combination=combination,
                    args=args,
                    results_dir=results_dir,
                    sweep_dir=sweep_dir,
                    combo_idx=combo_idx,
                    sweep_id=sweep_id,
                    fold_idx=None,
                    repeat=repeat,
                    use_kfold=False,
                )

                if result is not None:
                    pred_df = result.pop('predictions_df', None)
                    if pred_df is not None:
                        all_predictions_dfs.append(pred_df)
                    all_results.append(result)

    # Save predictions CSV (all folds/repeats combined)
    if all_predictions_dfs:
        combined_predictions = pd.concat(all_predictions_dfs, ignore_index=True)
        predictions_path = os.path.join(sweep_dir, 'predictions.csv')
        combined_predictions.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

    # Save summary results
    results_df = pd.DataFrame(all_results)
    results_summary_path = os.path.join(sweep_dir, 'tuning_summary.csv')
    results_df.to_csv(results_summary_path, index=False)
    print(f"\n{'='*80}")
    print(f"Tuning complete! Results saved to: {results_summary_path}")
    
    # Print best combination
    if not results_df.empty and results_df['metric_value'].notna().any():
        best_idx = results_df['metric_value'].idxmax()
        best_result = results_df.loc[best_idx]
        print(f"\nBest combination (by {best_result['metric_name']}):")
        if use_kfold:
            print(f"  Combination {best_result['combination']}, "
                  f"Fold {best_result['fold']}, Repeat {best_result['repeat']}")
        else:
            print(f"  Combination {best_result['combination']}, "
                  f"Repeat {best_result['repeat']}")
        print(f"  bag_n: {best_result['bag_n']}, batch_size: {best_result['batch_size']}")
        print(f"  c_weight: {best_result['c_weight']}, emb_dim: {best_result['emb_dim']}")
        print(f"  dropout: {best_result['dropout']}, attn: {best_result['attn']}")
        print(f"  {best_result['metric_name']}: {best_result['metric_value']*100:.2f}%")
        print(f"  Save path: {best_result['save_path']}")
    
    return all_results


if __name__ == '__main__':
    main()

