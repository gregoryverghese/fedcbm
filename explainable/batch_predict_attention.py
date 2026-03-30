#!/usr/bin/env python3
"""
Batch attention prediction script for multiple models.

This script runs attention prediction across multiple models (8 folds x 5 repeats = 40 models),
processing each model with its corresponding test fold data and saving results with fold/repeat information.

Usage:
    python batch_predict_attention.py \
        --models_base_dir /path/to/models \
        --test_data_path /path/to/strat.csv \
        --output_base_dir /path/to/output \
        --db_path /path/to/lmdb \
        --config_path /path/to/config.yaml \
        --target_lambda 1
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add parent directory to path
# Updated imports for new package structure
from fedcbm.config import load_config_from_yaml, create_config_from_args
from fedcbm.models import ConceptEmbeddingModel
from fedcbm.training.utils import get_cptprobs, get_yprobs, get_cembs, get_contexts
from fedcbm.training.metrics import get_cat_concept_metrics
from fedcbm.data.io.lmdb import LMDBRead

# Fix for pickle module import issue (for compatibility with pickled LMDB files)
import fedcbm.data.io.lmdb as lmdb_io
import sys
sys.modules['lmdb_io'] = lmdb_io

# For backward compatibility with old Args classes
try:
    import cem_mil.arg_def as arg_def
    import cem_mil.arg_def_cat as arg_def_cat
    OLD_ARGS_AVAILABLE = True
except ImportError:
    OLD_ARGS_AVAILABLE = False

# Import functions from the original predict_attention_batch.py
from predict_attention_batch import (
    load_config_from_yaml,
    load_slide_feature_vec,
    SlideDataset,
    load_trained_model,
    get_model_predictions,
    extract_attention_scores,
    get_true_values,
    get_survival_prediction_status,
    get_concept_prediction_status,
    create_hierarchical_save_path,
    convert_concept_probs_to_array
)


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
        print(f"Warning: No fold column found in {test_data_path}, using all data")
        subset_df = df
    
    return subset_df


def calculate_f1_score(tp, fp, fn):
    """
    Calculate F1 score from true positives, false positives, and false negatives.
    
    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives
        
    Returns:
        float: F1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def process_single_model_attention(
    model_path: str,
    test_data_df: pd.DataFrame,
    db_path: str,
    config_path: str,
    output_dir: str,
    fold: int,
    repeat: str
) -> Dict[str, Any]:
    """
    Process attention prediction for a single model.
    
    Args:
        model_path: Path to the model checkpoint
        test_data_df: DataFrame containing test data for this fold
        db_path: Path to the LMDB database
        config_path: Path to YAML configuration file
        output_dir: Directory to save results
        fold: Fold number
        repeat: Repeat number
        
    Returns:
        Dictionary with processing results
    """
    print(f"Processing model: {model_path}")
    print(f"Fold: {fold}, Repeat: {repeat}")
    
    # Load config
    config = load_config_from_yaml(config_path)
    
    # Load model
    model = load_trained_model(model_path, config)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all slide IDs from the test data
    slide_ids = test_data_df['ID'].tolist()
    
    # Append "-01Z" to slide IDs if not already present
    slide_ids = [slide_id if slide_id.endswith('-01Z') else f"{slide_id}-01Z" for slide_id in slide_ids]
    
    # Process each slide
    results = {
        'total_slides': len(slide_ids),
        'processed_slides': [],
        'failed_slides': [],
        'survival_stats': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
        'concept_stats': {},
        'predictions_data': [],
        'all_concept_probs': [],
        'all_concept_trues': []
    }
    
    # Initialize concept stats
    for concept_name in config.cpt_ids:
        results['concept_stats'][concept_name] = {'True': 0, 'False': 0, 'TP': 0, 'FP': 0, 'FN': 0}
    
    for idx, slide_id in enumerate(tqdm(slide_ids, desc=f"Processing slides (Fold {fold}, Repeat {repeat})")):
        try:
            # Get cancer type for this slide from the dataset
            original_slide_id = test_data_df['ID'].iloc[idx]  # Get original ID without -01Z
            cancer_type = test_data_df['project_id_x'].iloc[idx] if 'project_id_x' in test_data_df.columns else 'unknown'
            
            # Create slide dataset
            slide_dataset = SlideDataset(slide_id, db_path, config.target, config.cpt_ids, test_data_df)
            
            # Get model predictions
            survival_pred, concept_preds, concept_logits, survival_probs, concept_probs = get_model_predictions(model, slide_dataset, config)
            
            # Get true values
            true_survival, true_concepts = get_true_values(slide_dataset)
            
            # Store concept probabilities and true values for metrics calculation
            results['all_concept_probs'].append(concept_probs)
            results['all_concept_trues'].append(true_concepts)
            
            # Extract attention scores
            attention_scores = extract_attention_scores(model, slide_dataset, config)
            
            # Determine survival prediction status
            survival_status = get_survival_prediction_status(true_survival, survival_pred)
            survival_folder = f"{true_survival}_{survival_status}"
            
            # Update survival statistics
            results['survival_stats'][survival_status] += 1
            
            # Save attention scores for each concept
            for i, concept_name in enumerate(config.cpt_ids):
                # Determine if this concept is continuous or categorical
                concept_type = 'continuous' if config.concept_states[i] == 0 else 'categorical'
                
                # Get concept prediction status
                concept_status = get_concept_prediction_status(true_concepts[i], concept_preds[i], concept_type)
                
                # Update concept statistics
                if concept_type == 'categorical':
                    # For categorical concepts, track True/False
                    if concept_status not in results['concept_stats'][concept_name]:
                        results['concept_stats'][concept_name][concept_status] = 0
                    results['concept_stats'][concept_name][concept_status] += 1
                    
                    # For F1 calculation, we need to track TP, FP, FN for each class
                    if concept_status == 'True':
                        results['concept_stats'][concept_name]['TP'] += 1
                    else:
                        results['concept_stats'][concept_name]['FP'] += 1
                        results['concept_stats'][concept_name]['FN'] += 1
                else:
                    # For continuous concepts, track MSE
                    if 'mse_values' not in results['concept_stats'][concept_name]:
                        results['concept_stats'][concept_name]['mse_values'] = []
                    mse_value = float(concept_status.split('_')[1])  # Extract MSE value from status
                    results['concept_stats'][concept_name]['mse_values'].append(mse_value)
                
                # Create hierarchical save path
                concept_save_path = create_hierarchical_save_path(output_dir, cancer_type, survival_folder, concept_name)
                
                # Save attention scores for this concept with fold/repeat in filename
                attention_file = os.path.join(concept_save_path, f'{slide_id}_{concept_status}_attention_fold{fold}_repeat{repeat}.npy')
                np.save(attention_file, attention_scores[:, i])
                
                # Save tile keys for reference
                keys_file = os.path.join(concept_save_path, f'{slide_id}_{concept_status}_tile_keys_fold{fold}_repeat{repeat}.txt')
                with open(keys_file, 'w') as f:
                    for key in slide_dataset.keys:
                        f.write(f'{key}\n')
            
            # Store predictions data for CSV output
            prediction_row = {
                'slide_id': slide_id,
                'cancer_type': cancer_type,
                'fold': fold,
                'repeat': repeat,
                'survival_pred': survival_pred,
                'survival_prob': survival_probs,
                'survival_true': true_survival,
                'survival_status': survival_status
            }
            
            # Add concept predictions, probabilities, and true values
            for i, concept_name in enumerate(config.cpt_ids):
                prediction_row[f'{concept_name}_pred'] = concept_preds[i]
                prediction_row[f'{concept_name}_true'] = true_concepts[i]
                concept_type = 'continuous' if config.concept_states[i] == 0 else 'categorical'
                prediction_row[f'{concept_name}_status'] = get_concept_prediction_status(true_concepts[i], concept_preds[i], concept_type)
                
                # Add concept probabilities
                if concept_type == 'continuous':
                    # For continuous concepts, store the prediction as probability
                    prediction_row[f'{concept_name}_prob'] = concept_probs[concept_name]
                else:
                    # For categorical concepts, store individual class probabilities
                    for prob_key, prob_value in concept_probs[concept_name].items():
                        prediction_row[prob_key] = prob_value
            
            results['predictions_data'].append(prediction_row)
            results['processed_slides'].append(slide_id)
            
            print(f"✓ {slide_id}: Processed - Survival: {survival_folder}")
            
        except Exception as e:
            import traceback
            print(f"Error processing slide {slide_id}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            results['failed_slides'].append(slide_id)
            continue
    
    # Save summary for this model
    summary_path = os.path.join(output_dir, f'processing_summary_fold{fold}_repeat{repeat}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Processing Summary (Fold {fold}, Repeat {repeat})\n")
        f.write(f"============================================\n")
        f.write(f"Total slides to process: {len(slide_ids)}\n")
        f.write(f"Successfully processed: {len(results['processed_slides'])}\n")
        f.write(f"Failed to process: {len(results['failed_slides'])}\n\n")
        
        # Survival statistics
        f.write(f"Survival Prediction Statistics:\n")
        f.write(f"  True Positives (TP): {results['survival_stats']['TP']}\n")
        f.write(f"  True Negatives (TN): {results['survival_stats']['TN']}\n")
        f.write(f"  False Positives (FP): {results['survival_stats']['FP']}\n")
        f.write(f"  False Negatives (FN): {results['survival_stats']['FN']}\n")
        
        total_survival = sum(results['survival_stats'].values())
        if total_survival > 0:
            survival_accuracy = (results['survival_stats']['TP'] + results['survival_stats']['TN']) / total_survival * 100
            survival_f1 = calculate_f1_score(results['survival_stats']['TP'], results['survival_stats']['FP'], results['survival_stats']['FN'])
            f.write(f"  Survival Accuracy: {survival_accuracy:.2f}%\n")
            f.write(f"  Survival F1 Score: {survival_f1:.4f}\n")
        
        # Calculate concept metrics using get_cat_concept_metrics
        if results['all_concept_probs'] and results['all_concept_trues']:
            # Convert dictionary-based concept probabilities to numpy array format
            all_concept_probs_array = convert_concept_probs_to_array(results['all_concept_probs'], config)
            all_concept_trues = np.array(results['all_concept_trues'])
            
            # Calculate metrics using the evaluate function
            concept_accuracies, concept_f1s, _ = get_cat_concept_metrics(
                all_concept_probs_array, 
                all_concept_trues, 
                config.concept_states
            )
            
            # Update concept statistics with the calculated metrics
            for i, concept_name in enumerate(config.cpt_ids):
                if i < len(concept_accuracies):
                    results['concept_stats'][concept_name]['accuracy'] = concept_accuracies[i]
                    results['concept_stats'][concept_name]['f1_score'] = concept_f1s[i]
            
            # Write concept metrics to summary
            f.write(f"\nConcept Metrics (using get_cat_concept_metrics):\n")
            for i, concept_name in enumerate(config.cpt_ids):
                if i < len(concept_accuracies):
                    f.write(f"  {concept_name}:\n")
                    f.write(f"    - Accuracy: {concept_accuracies[i]:.4f}\n")
                    f.write(f"    - F1 Score: {concept_f1s[i]:.4f}\n")
        
        if results['failed_slides']:
            f.write(f"\nFailed slides:\n")
            for slide_id in results['failed_slides']:
                f.write(f"  - {slide_id}\n")
    
    # Save predictions CSV for this model
    if results['predictions_data']:
        predictions_df = pd.DataFrame(results['predictions_data'])
        csv_path = os.path.join(output_dir, f'predictions_summary_fold{fold}_repeat{repeat}.csv')
        predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions CSV saved to: {csv_path}")
    
    return results


def run_batch_attention_prediction(args):
    """
    Run attention prediction for all models across folds and repeats.
    
    Args:
        args: Command line arguments
    """
    print(f"Starting batch attention prediction...")
    print(f"Models base directory: {args.models_base_dir}")
    print(f"Test data path: {args.test_data_path}")
    print(f"Output base directory: {args.output_base_dir}")
    print(f"Target lambda: {args.target_lambda}")
    
    # Define folds and repeats
    folds = list(range(8))  # 8 folds
    repeats = ['0', '1', '2', '3', '4']  # 5 repeats
    
    # Create output base directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Process each fold
    for fold in folds:
        print(f"\n{'='*80}")
        print(f"Processing fold {fold}")
        print(f"{'='*80}")
        
        # Subset test data for this fold (same across all repeats)
        test_data_subset_df = subset_test_data(args.test_data_path, fold)
        print(f"Created test data subset for fold {fold} with {len(test_data_subset_df)} patients")
        
        # Find all models for this fold across all repeats
        fold_models = []
        for repeat in repeats:
            repeat_dir = os.path.join(args.models_base_dir, repeat)
            if not os.path.exists(repeat_dir):
                print(f"Warning: Repeat directory {repeat_dir} not found, skipping...")
                continue
            
            # Find all model files in this repeat
            all_model_files = find_model_files(repeat_dir)
            
            print(f'Looking for models with lambda={args.target_lambda}, fold={fold}')
            # Filter by lambda and fold
            for model_path in all_model_files:
                exp_info = extract_experiment_info(model_path, args.target_lambda)
                print(f"Exp info: {exp_info}")
                if exp_info['lambda'] == args.target_lambda and int(exp_info['fold']) == fold:
                    fold_models.append((repeat, model_path, exp_info))
        
        print(f"Found {len(fold_models)} models for fold {fold} with lambda={args.target_lambda}")
        
        # Process each model for this fold
        for repeat, model_path, exp_info in fold_models:
            print(f"\nProcessing model: {model_path}")
            print(f"Extracted info: repeat={repeat}, fold={fold}, lambda={exp_info['lambda']}")
            
            # Create output directory for this experiment
            exp_name = f"fold{fold}_repeat{repeat}"
            output_dir = os.path.join(args.output_base_dir, exp_name)
            
            try:
                # Run attention prediction for this model
                results = process_single_model_attention(
                    model_path=model_path,
                    test_data_df=test_data_subset_df,
                    db_path=args.db_path,
                    config_path=args.config_path,
                    output_dir=output_dir,
                    fold=fold,
                    repeat=repeat
                )
                
                print(f"✓ Completed fold {fold}, repeat {repeat}")
                print(f"  Processed: {len(results['processed_slides'])} slides")
                print(f"  Failed: {len(results['failed_slides'])} slides")
                
                # Print survival statistics
                total_survival = sum(results['survival_stats'].values())
                if total_survival > 0:
                    survival_accuracy = (results['survival_stats']['TP'] + results['survival_stats']['TN']) / total_survival * 100
                    survival_f1 = calculate_f1_score(results['survival_stats']['TP'], results['survival_stats']['FP'], results['survival_stats']['FN'])
                    print(f"  Survival Accuracy: {survival_accuracy:.2f}%")
                    print(f"  Survival F1 Score: {survival_f1:.4f}")
                
                # Print concept metrics if available
                if results['all_concept_probs'] and results['all_concept_trues']:
                    print(f"  Concept Metrics:")
                    for concept_name, stats in results['concept_stats'].items():
                        if 'accuracy' in stats and 'f1_score' in stats:
                            print(f"    {concept_name}: Accuracy={stats['accuracy']:.4f}, F1={stats['f1_score']:.4f}")
                
            except Exception as e:
                import traceback
                print(f"Error processing fold {fold}, repeat {repeat}: {e}")
                print("Full traceback:")
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print(f"Batch attention prediction complete!")
    print(f"Results saved to: {args.output_base_dir}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Batch attention prediction across multiple models and folds')
    parser.add_argument('--models_base_dir', required=True, help='Base directory containing model folders (0, 1, 2, 3, 4)')
    parser.add_argument('--test_data_path', required=True, help='Path to stratified test data CSV')
    parser.add_argument('--output_base_dir', required=True, help='Base directory to save results')
    parser.add_argument('--db_path', required=True, help='Path to LMDB database')
    parser.add_argument('--config_path', required=True, help='Path to YAML configuration file')
    parser.add_argument('--target_lambda', type=str, default='1', help='Target lambda value to filter models')
    
    args = parser.parse_args()
    
    # Run batch attention prediction
    run_batch_attention_prediction(args)


if __name__ == '__main__':
    main()
