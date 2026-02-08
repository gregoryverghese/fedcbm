#!/usr/bin/env python3
"""
Batch intervention analysis script.

This script runs concept interventions across multiple models and test folds,
aggregating results into a comprehensive summary.

Usage:
    python batch_intervention_analysis.py \
        --models_base_dir /path/to/models \
        --test_data_path /path/to/full_test_data.csv \
        --output_base_dir /path/to/output \
        --concept_ids "Stage Age Cancer RNA_Bio_ter" \
        --concept_states "4 3 10 3" \
        --db_path /path/to/lmdb \
        --config_path /path/to/config.yaml
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import subprocess
import json
from datetime import datetime

# Updated imports - use proper package structure
from intervention.run_intervention_pipeline import InterventionPipeline


def find_model_files(models_dir: str) -> List[str]:
    """Find all .ckpt model files in the directory."""
    pattern = os.path.join(models_dir, "**", "*.ckpt")
    model_files = glob.glob(pattern, recursive=True)
    return sorted(model_files)


def extract_experiment_info(model_path: str, target_lambda: str) -> Dict[str, str]:
    """Extract repeat, fold, and lambda information from model path."""
    path_parts = Path(model_path).parts
    
    # Extract repeat from folder name (e.g., "0" from path/to/0/model.ckpt)
    repeat = None
    for part in path_parts:
        if part.isdigit() and len(part) == 1:  # Single digit folder
            repeat = part
            break
    
    # Extract info from filename
    filename = Path(model_path).stem
    # Expected format: fold_lambda_0_4_3_True_8_0.1_0.3_attention_exp0_noweights_checkmodel
    parts = filename.split('_')
    
    # Fold is always the first element (index 0)
    fold = parts[0] if len(parts) > 0 else 'unknown'
    
    # Lambda is always the second element (index 1)
    model_lambda = parts[1] if len(parts) > 1 else 'unknown'
    
    # Repeat is always the third element (index 2)
    filename_repeat = parts[2] if len(parts) > 2 else 'unknown'
    
    return {
        'repeat': filename_repeat or 'unknown',
        'fold': fold or 'unknown',
        'lambda': model_lambda or 'unknown',
        'model_path': model_path,
        'filename': filename
    }


def subset_test_data(test_data_path: str, fold: int) -> pd.DataFrame:
    """Subset test data by fold and return DataFrame directly."""
    df = pd.read_csv(test_data_path)
    
    # Filter by fold (assuming there's a 'fold' or 'test' column)
    if 'fold' in df.columns:
        subset_df = df[df['fold'] == fold]
    elif 'test' in df.columns:
        subset_df = df[df['test'] == fold]
    else:
        # If no fold column, assume all data is for this fold
        subset_df = df
    
    return subset_df


def create_shared_dataloader(
    test_data_subset: pd.DataFrame,
    db_path: str,
    concept_ids: List[str],
    bag_num: int = 3000,
    batch_size: int = 1,
    use_whole_slide: bool = False
) -> DataLoader:
    """Create a shared dataloader for a fold that can be reused across repeats."""
    
    if use_whole_slide:
        # Import the WholeSlideDataset from concept_intervention
        from concept_intervention import WholeSlideDataset
        
        # Create dataset with all tiles per slide (only for slides with available databases)
        dataset = WholeSlideDataset(
            dataset_df=test_data_subset,
            db_path=db_path,
            target='Survival',
            cpt_ids=concept_ids
        )
    else:
        # Use traditional random sampling approach
        from minotaur.data import TileDataset
        dataset = TileDataset(
            dataset=test_data_subset,
            db_path=db_path,
            target='Survival',
            cpt_ids=concept_ids,
            bag_num=bag_num
        )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    return dataloader


def run_intervention_analysis_with_shared_dataloader(
    model_path: str,
    test_data_df: pd.DataFrame,
    output_dir: str,
    concept_ids: List[str],
    concept_states: List[int],
    db_path: str,
    config_path: str = None,
    bag_num: int = 3000,
    batch_size: int = 1,
    threshold: float = 0.5,
    seed: int = 42,
    shared_dataloader: DataLoader = None,
    use_whole_slide: bool = False
) -> Dict[str, Any]:
    """Run intervention analysis for a single model/test fold combination with shared dataloader."""
    
    print(f"\n{'='*60}")
    print(f"Running intervention analysis with shared dataloader:")
    print(f"  Model: {model_path}")
    print(f"  Test data: {len(test_data_df)} patients")
    print(f"  Output: {output_dir}")
    print(f"  Using shared dataloader: {shared_dataloader is not None}")
    print(f"  Using whole slide mode: {use_whole_slide}")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the provided DataFrame directly
    test_dataset = test_data_df
    
    # We don't need to create the pipeline since we're handling everything manually
    
    # If we have a shared dataloader, we need to use it for both baseline and interventions
    if shared_dataloader is not None:
        print("Using shared dataloader for both baseline and interventions...")
        
        # We need to manually run baseline and interventions to use the shared dataloader
        # First, create the baseline inference instance
        from baseline_inference import BaselineInference
        baseline = BaselineInference(
            model_path=model_path,
            test_dataset=test_dataset,
            db_path=db_path,
            concept_ids=concept_ids,
            concept_states=concept_states,
            bag_num=bag_num,
            batch_size=batch_size,
            threshold=threshold,
            device='cuda',
            config_path=config_path,
            seed=seed,
            use_whole_slide=use_whole_slide
        )
        
        # Override the dataloader with our shared one
        baseline.test_loader = shared_dataloader
        
        # Run baseline inference with shared dataloader
        baseline_results = baseline.run_baseline_inference()
        
        # Save baseline results
        baseline_path = os.path.join(output_dir, 'baseline_results.csv')
        baseline.save_results(baseline_results, baseline_path)
        
        # Now run interventions with the same shared dataloader
        from concept_intervention import ConceptIntervention
        intervention = ConceptIntervention(
            model_path=model_path,
            test_dataset=test_dataset,
            db_path=db_path,
            concept_ids=concept_ids,
            concept_states=concept_states,
            bag_num=bag_num,
            batch_size=batch_size,
            threshold=threshold,
            device='cuda',
            seed=seed,
            config_path=config_path,
            model=baseline.get_model(),  # Use the same model
            dataloader=shared_dataloader,  # Use the same shared dataloader
            use_whole_slide=use_whole_slide
        )
        
        # Run interventions
        intervention_results = intervention.run_interventions()
        
        # Save intervention results
        intervention_path = os.path.join(output_dir, 'intervention_results.csv')
        intervention.save_results(intervention_results, intervention_path)
        
        # Run evaluation
        from evaluation_utils import InterventionEvaluator
        evaluator = InterventionEvaluator(intervention_results, baseline_results)
        evaluator.print_summary()
        
        # Generate plots
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        evaluator.plot_metrics_evolution(os.path.join(plots_dir, 'metrics_evolution.png'))
        evaluator.plot_error_flips(os.path.join(plots_dir, 'error_flips.png'))
        evaluator.plot_concept_importance(os.path.join(plots_dir, 'concept_importance.png'))
        evaluator.generate_confusion_matrices(os.path.join(plots_dir, 'confusion_matrices.png'))
        
        # Save detailed report
        report_path = os.path.join(output_dir, 'detailed_report.xlsx')
        evaluator.save_detailed_report(report_path)
        
    # Note: We always use the shared dataloader approach now
    
    return intervention_results




def aggregate_results(results_dir: str) -> pd.DataFrame:
    """Aggregate results from all experiments into a summary DataFrame."""
    
    all_results = []
    
    # Find all experiment directories
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        
        # Extract repeat and fold from directory name
        parts = exp_dir.split('_')
        repeat = parts[0] if len(parts) > 0 else 'unknown'
        fold = parts[1] if len(parts) > 1 else 'unknown'
        
        print(f"\nProcessing experiment: {exp_dir} (repeat={repeat}, fold={fold})")
        
        # Load baseline results
        baseline_path = os.path.join(exp_path, 'baseline_results.csv')
        intervention_path = os.path.join(exp_path, 'intervention_results.csv')
        
        if os.path.exists(baseline_path) and os.path.exists(intervention_path):
            # Load baseline metrics
            baseline_df = pd.read_csv(baseline_path)
            
            # Load intervention results
            intervention_df = pd.read_csv(intervention_path)
            
            # Calculate summary metrics
            baseline_accuracy = (baseline_df['survival_prediction'] == baseline_df['survival_truth']).mean()
            baseline_f1 = calculate_f1_score(baseline_df['survival_truth'], baseline_df['survival_prediction'])
            
            # Ensure patient order matches between baseline and intervention
            # Sort both DataFrames by patient_id to ensure consistent ordering
            baseline_df_sorted = baseline_df.sort_values('patient_id').reset_index(drop=True)
            intervention_df_sorted = intervention_df.sort_values('patient_id').reset_index(drop=True)
            
            # Calculate intervention metrics for each time step
            for t in sorted(intervention_df_sorted['time_step'].unique()):
                
                t_data = intervention_df_sorted[intervention_df_sorted['time_step'] == t]
                t_accuracy = (t_data['survival_prediction'] == t_data['survival_truth']).mean()
                t_f1 = calculate_f1_score(t_data['survival_truth'], t_data['survival_prediction'])
                
                # Calculate error flips vs baseline with proper patient ordering
                baseline_preds = baseline_df_sorted['survival_prediction'].values
                t_preds = t_data['survival_prediction'].values
                ground_truth = t_data['survival_truth'].values
                
                # Verify patient IDs match
                baseline_patient_ids = baseline_df_sorted['patient_id'].values
                t_patient_ids = t_data['patient_id'].values
                
                if not np.array_equal(baseline_patient_ids, t_patient_ids):
                    print(f"    Warning: Patient ID mismatch at time step {t}")
                    print(f"      Baseline patients: {baseline_patient_ids[:5].tolist()}")
                    print(f"      Intervention patients: {t_patient_ids[:5].tolist()}")
                else:
                    print(f"    Patient IDs match at time step {t}")
                
                # For time step 0, error flips should be 0 since it's baseline
                if t == 0:
                    # Compare baseline vs time step 0 data to verify they're identical
                    baseline_preds_check = baseline_df_sorted['survival_prediction'].values
                    baseline_truth_check = baseline_df_sorted['survival_truth'].values
                    t_preds_check = t_data['survival_prediction'].values
                    t_truth_check = t_data['survival_truth'].values
                    
                    preds_match = np.array_equal(baseline_preds_check, t_preds_check)
                    truth_match = np.array_equal(baseline_truth_check, t_truth_check)
                    
                    print(f"    Time step 0 comparison (repeat={repeat}, fold={fold}):")
                    print(f"      Predictions match: {preds_match}")
                    print(f"      Truth match: {truth_match}")
                    print(f"      Baseline F1: {baseline_f1:.6f}")
                    print(f"      Time step 0 F1: {t_f1:.6f}")
                    print(f"      F1 difference: {abs(baseline_f1 - t_f1):.6f}")
                    
                    if not preds_match or not truth_match:
                        print(f"      WARNING: Time step 0 data doesn't match baseline!")
                        print(f"      Baseline preds shape: {baseline_preds_check.shape}")
                        print(f"      Time step 0 preds shape: {t_preds_check.shape}")
                        print(f"      First 5 baseline preds: {baseline_preds_check[:5].tolist()}")
                        print(f"      First 5 time step 0 preds: {t_preds_check[:5].tolist()}")
                        
                        # Find exactly where the differences are
                        diff_indices = np.where(baseline_preds_check != t_preds_check)[0]
                        print(f"      Number of differences: {len(diff_indices)}")
                        if len(diff_indices) > 0:
                            print(f"      First 10 difference indices: {diff_indices[:10].tolist()}")
                            print(f"      Baseline values at differences: {baseline_preds_check[diff_indices[:10]].tolist()}")
                            print(f"      Time step 0 values at differences: {t_preds_check[diff_indices[:10]].tolist()}")
                    
                    fn_to_tp = fp_to_tn = tp_to_fn = tn_to_fp = 0
                    print(f"    Time step 0: Error flips = 0 (baseline)")
                else:
                    # Calculate error flips for intervention steps
                    fn_to_tp = np.sum((baseline_preds == 0) & (t_preds == 1) & (ground_truth == 1))
                    fp_to_tn = np.sum((baseline_preds == 1) & (t_preds == 0) & (ground_truth == 0))
                    tp_to_fn = np.sum((baseline_preds == 1) & (t_preds == 0) & (ground_truth == 1))
                    tn_to_fp = np.sum((baseline_preds == 0) & (t_preds == 1) & (ground_truth == 0))
                
                net_improvement = fn_to_tp + fp_to_tn - tp_to_fn - tn_to_fp
                
                all_results.append({
                    'repeat': repeat,
                    'fold': fold,
                    'time_step': t,
                    'baseline_accuracy': baseline_accuracy,
                    'baseline_f1': baseline_f1,
                    'intervention_accuracy': t_accuracy,
                    'intervention_f1': t_f1,
                    'fn_to_tp': fn_to_tp,
                    'fp_to_tn': fp_to_tn,
                    'tp_to_fn': tp_to_fn,
                    'tn_to_fp': tn_to_fp,
                    'net_improvement': net_improvement,
                    'experiment_dir': exp_dir
                })
    
    return pd.DataFrame(all_results)


def calculate_f1_score(y_true, y_pred):
    """Calculate F1 score."""
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description='Batch intervention analysis')
    
    # Required arguments
    parser.add_argument('--models_base_dir', type=str, required=True,
                       help='Base directory containing model folders (0-5)')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to full test data CSV')
    parser.add_argument('--output_base_dir', type=str, required=True,
                       help='Base directory for output results')
    parser.add_argument('--concept_ids', type=str, required=True,
                       help='Space-separated concept IDs')
    parser.add_argument('--concept_states', type=str, required=True,
                       help='Space-separated concept states')
    parser.add_argument('--db_path', type=str, required=True,
                       help='Path to LMDB database')
    
    # Optional arguments
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--bag_num', type=int, default=3000,
                       help='Number of tiles per bag')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Decision threshold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_whole_slide', action='store_true',
                       help='Use all tiles from each WSI instead of random sampling')
    parser.add_argument('--repeats', type=str, default='0,1,2,3,4',
                       help='Comma-separated list of repeat numbers to process')
    parser.add_argument('--lambda', type=str, required=True,
                       help='Lambda value to filter models (e.g., "0.1")')
    
    args = parser.parse_args()
    
    # Parse arguments
    concept_ids = args.concept_ids.split()
    concept_states = [int(x) for x in args.concept_states.split()]
    repeats = [x.strip() for x in args.repeats.split(',')]
    target_lambda = getattr(args, 'lambda')  # Handle the --lambda argument
    
    print(f"Starting batch intervention analysis...")
    print(f"Models base dir: {args.models_base_dir}")
    print(f"Test data: {args.test_data_path}")
    print(f"Output base dir: {args.output_base_dir}")
    print(f"Repeats to process: {repeats}")
    print(f"Target lambda: {target_lambda}")
    print(f"Concept IDs: {concept_ids}")
    print(f"Concept states: {concept_states}")
    
    # Create output base directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Process each fold first, then repeats within each fold
    # This allows us to use the same dataloader across repeats for the same fold
    folds = list(range(8))  # Assuming 8 folds (0-7)
    
    for fold in folds:
        print(f"\n{'='*80}")
        print(f"Processing fold {fold}")
        print(f"{'='*80}")
        
        # Subset test data for this fold (same across all repeats)
        test_data_subset_df = subset_test_data(args.test_data_path, fold)
        print(f"Created test data subset for fold {fold} with {len(test_data_subset_df)} patients")
        
        # Create shared dataloader for this fold (to be reused across repeats)
        print(f"Creating shared dataloader for fold {fold}...")
        shared_dataloader = create_shared_dataloader(
            test_data_subset=test_data_subset_df,
            db_path=args.db_path,
            concept_ids=concept_ids,
            bag_num=args.bag_num,
            batch_size=args.batch_size,
            use_whole_slide=args.use_whole_slide
        )
        
        # Find all models for this fold across all repeats
        fold_models = []
        for repeat in repeats:
            repeat_dir = os.path.join(args.models_base_dir, repeat)
            if not os.path.exists(repeat_dir):
                print(f"Warning: Repeat directory {repeat_dir} not found, skipping...")
                continue
            
            # Find all model files in this repeat
            all_model_files = find_model_files(repeat_dir)
            
            print('target_lambda,fold',target_lambda,fold)
            # Filter by lambda and fold
            for model_path in all_model_files:
                exp_info = extract_experiment_info(model_path, target_lambda)
                print(f"Exp info: {exp_info}")
                if exp_info['lambda'] == target_lambda and int(exp_info['fold']) == fold:
                    fold_models.append((repeat, model_path, exp_info))
        
        print(f"Found {len(fold_models)} models for fold {fold} with lambda={target_lambda}")
        
        # Process each model for this fold
        for repeat, model_path, exp_info in fold_models:
            print(f"\nProcessing model: {model_path}")
            print(f"Extracted info: repeat={repeat}, fold={fold}, lambda={exp_info['lambda']}")
            
            # Create output directory for this experiment
            exp_name = f"{repeat}_{fold}"
            output_dir = os.path.join(args.output_base_dir, exp_name)
            
            try:
                # Run intervention analysis with shared dataloader
                results = run_intervention_analysis_with_shared_dataloader(
                    model_path=model_path,
                    test_data_df=test_data_subset_df,  # Same test data DataFrame for all repeats of this fold
                    output_dir=output_dir,
                    concept_ids=concept_ids,
                    concept_states=concept_states,
                    db_path=args.db_path,
                    config_path=args.config_path,
                    bag_num=args.bag_num,
                    batch_size=args.batch_size,
                    threshold=args.threshold,
                    seed=args.seed,
                    shared_dataloader=shared_dataloader,  # Pass the shared dataloader
                    use_whole_slide=args.use_whole_slide
                )
                
                print(f"✓ Completed experiment {exp_name}")
                
            except Exception as e:
                print(f"✗ Error in experiment {exp_name}: {e}")
                continue
            
    
    # Aggregate all results
    print(f"\n{'='*80}")
    print("Aggregating results...")
    print(f"{'='*80}")
    
    summary_df = aggregate_results(args.output_base_dir)
    
    # Save summary
    summary_path = os.path.join(args.output_base_dir, 'intervention_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary saved to: {summary_path}")
    print(f"Total experiments: {len(summary_df)}")
    print(f"Repeats: {summary_df['repeat'].unique()}")
    print(f"Folds: {summary_df['fold'].unique()}")
    print(f"Time steps: {summary_df['time_step'].unique()}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Average baseline accuracy: {summary_df['baseline_accuracy'].mean():.4f}")
    print(f"Average baseline F1: {summary_df['baseline_f1'].mean():.4f}")
    print(f"Average final intervention accuracy: {summary_df[summary_df['time_step'] == summary_df['time_step'].max()]['intervention_accuracy'].mean():.4f}")
    print(f"Average final intervention F1: {summary_df[summary_df['time_step'] == summary_df['time_step'].max()]['intervention_f1'].mean():.4f}")
    
    print(f"\nBatch intervention analysis completed!")


if __name__ == "__main__":
    main()
