#!/usr/bin/env python
"""Train centralised CBM baseline on all training data, evaluate on global test set.

Usage:
  python scripts/run_centralised.py \
      --train-path data/partitions/iid/  \   # any directory; we concat all client_*.csv
      --test-path  data/partitions/global_test.csv \
      --config     configs/federated.yaml \
      --save-path  results/centralised
"""
import argparse
import os
import sys
import warnings
import glob
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fedcbm.config.loader import load_config_from_yaml
from fedcbm.training.trainer import train, predict, train_model
from fedcbm.training.metrics import compute_task_metrics_all, compute_concept_metrics
from fedcbm.training.utils import get_concept_weights
from fedcbm.data.loaders import get_data_loaders
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Centralised CBM baseline")
    parser.add_argument("--train-path", required=True,
                        help="Path to training CSV or directory of client_*.csv files to concatenate")
    parser.add_argument("--test-path", required=True,
                        help="Path to global test CSV")
    parser.add_argument("--config", default="configs/federated.yaml",
                        help="Path to YAML config")
    parser.add_argument("--save-path", default="results/centralised",
                        help="Directory to save results")
    parser.add_argument("--experiment-name", default="centralised",
                        help="Experiment label for logging")
    args = parser.parse_args()

    config = load_config_from_yaml(args.config)
    config.output.save_path = args.save_path
    config.output.name = args.experiment_name
    os.makedirs(args.save_path, exist_ok=True)

    # ── Load training data ─────────────────────────────────────────────────────
    if os.path.isdir(args.train_path):
        csv_files = sorted(glob.glob(os.path.join(args.train_path, "client_*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No client_*.csv found in {args.train_path}")
        train_cohort = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f"Loaded {len(train_cohort)} training patients from {len(csv_files)} files")
    else:
        train_cohort = pd.read_csv(args.train_path)
        print(f"Loaded {len(train_cohort)} training patients from {args.train_path}")

    test_df = pd.read_csv(args.test_path)
    print(f"Loaded {len(test_df)} test patients from {args.test_path}")

    # ── Internal train/val split ───────────────────────────────────────────────
    stratify = train_cohort["event"] if "event" in train_cohort.columns else None
    train_df, val_df = train_test_split(
        train_cohort, test_size=config.data.test_size,
        stratify=stratify, random_state=config.data.stratify_random_state,
    )

    # ── Concept weights ────────────────────────────────────────────────────────
    if not config.concepts.no_concepts:
        config.concepts.cpt_cls_weights = get_concept_weights(train_df, config.concepts.cpt_ids)

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\nTraining centralised model: {len(train_df)} train / {len(val_df)} val")
    concepts = config.concepts.cpt_ids

    results, trainer, predictions_df = train(
        train_cohort=train_cohort,
        concepts=concepts,
        db_path=config.data.db_path,
        config=config,
        experiment_name=args.experiment_name,
        train_data=train_df.reset_index(drop=True),
        val_data=val_df.reset_index(drop=True),
    )

    # ── Evaluate on global test set ────────────────────────────────────────────
    print(f"\n── Global Test Set Evaluation ─────────────────────────────────────")
    model = trainer.model  # trained Lightning module

    test_loader_cfg = config  # reuse config for db_path etc.
    # Build test loader — use test_df for both train+val slots, only val_loader used
    _, test_loader = get_data_loaders(
        config.data.db_path,
        train_data=test_df,
        val_data=test_df,
        config=config,
    )

    test_results = predict(model, test_loader, config)
    c_probs, c_test, y_probs, y_test, c_embs, ctxts, caw, patient_ids = test_results

    task_metrics = compute_task_metrics_all(y_probs, y_test, config.task_type)
    print(f"  C-index: {task_metrics['c_index']*100:.2f}%")

    if c_probs is not None and not config.concepts.no_concepts:
        concept_metrics = compute_concept_metrics(
            c_probs, c_test, config.concepts.concept_states, is_logits=True
        )
        f1s = [m["f1"] for m in concept_metrics.values() if "f1" in m]
        print(f"  Mean concept F1: {np.mean(f1s)*100:.2f}%")
        for i, (k, m) in enumerate(concept_metrics.items()):
            if "f1" in m:
                print(f"    {config.concepts.cpt_ids[i]}: F1={m['f1']*100:.2f}%")

    # ── Save outputs ───────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.save_path, "test_metrics.csv")
    pd.DataFrame([{
        "experiment": args.experiment_name,
        "c_index": task_metrics["c_index"],
        "n_test": len(test_df),
    }]).to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")

    ckpt_path = os.path.join(args.save_path, "model.ckpt")
    trainer.save_checkpoint(ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
