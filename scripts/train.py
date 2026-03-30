#!/usr/bin/env python
"""Training script for MINOTAUR."""
import os
import sys
import warnings
import argparse
import pandas as pd

warnings.filterwarnings("ignore")

# Set environment variables if needed
#if not os.getenv("CUDA_LAUNCH_BLOCKING"):
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from fedcbm.config import load_config_from_yaml
from fedcbm.training import train
from fedcbm.training.metrics import compute_task_metrics_all
from fedcbm.training.utils import get_sweep_id, build_trial_name, collect_hyperparameters
from fedcbm.data import get_data_loaders


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MINOTAUR model')
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

    # Load configuration from YAML
    config = load_config_from_yaml(args.config_path)
    config.data.db_path = args.db_path
    config.output.save_path = args.save_path
    if not config.output.log_path:
        config.output.log_path = os.path.join(args.save_path, 'logs')
    config.task_type = args.task_type

    # Create output directory
    os.makedirs(config.output.log_path, exist_ok=True)

    # Get concepts
    concepts = config.concepts.cpt_ids

    # Generate experiment name and version for TensorBoard
    sweep_id = get_sweep_id()
    trial_name = build_trial_name(config)
    
    # Run training
    print(f"Starting training with {len(concepts)} concepts")
    print(f"Task type: {config.task_type}")
    print(f"Save path: {config.output.save_path}")
    print(f"Experiment: {sweep_id}, Trial: {trial_name}")

    results, trainer, predictions_df = train(train_cohort, concepts, args.db_path, config, experiment_name=sweep_id, version=trial_name)

    # Save predictions CSV once
    if predictions_df is not None and config.output.save_path:
        os.makedirs(config.output.save_path, exist_ok=True)
        predictions_path = os.path.join(config.output.save_path, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

    # Log hyperparameters to TensorBoard
    hparams = collect_hyperparameters(config)
    trainer.logger.log_hyperparams(hparams)

    # Save model checkpoint
    checkpoint_path = os.path.join(config.output.save_path, "model.ckpt")
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

    # Print metrics
    c_probs, c_test, y_probs, y_test, c_embs, ctxts, concept_attention_weights = results
    
    task_metrics = compute_task_metrics_all(y_probs, y_test, config.task_type)
    
    if task_metrics is None:
        print("WARNING: Could not compute task metrics (sksurv not available for Cox)")
    elif config.task_type == 'cox':
        print(f"Test C-index: {task_metrics['c_index']*100:.2f}%")
    else:
        print(f"Test task accuracy: {task_metrics['accuracy']*100:.2f}%")
        print(f"Test task F1: {task_metrics['f1']*100:.2f}%")
        print(f"Test task AUC-ROC: {task_metrics['roc_auc']*100:.2f}%")

    return results, trainer


if __name__ == '__main__':
    main()

