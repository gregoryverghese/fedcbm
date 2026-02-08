#!/usr/bin/env python3
"""
Wrapper script to run attention analysis over the entire file structure.

This script loops over the attention directory structure:
attentions/
├── Cancer_type/
│   ├── predict_result/
│   │   └── concept/
│   │       ├── patient1_attention.npy
│   │       ├── patient1_tile_keys.txt
│   │       └── ...
│   └── ...
└── ...

And calls attention_analysis.py for each combination, then combines results.
"""

import os
import glob
import subprocess
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Tuple


def get_cancer_types(attentions_dir: str) -> List[str]:
    """Get list of cancer type directories."""
    cancer_dirs = []
    for item in os.listdir(attentions_dir):
        item_path = os.path.join(attentions_dir, item)
        if os.path.isdir(item_path):
            cancer_dirs.append(item)
    return cancer_dirs


def get_predict_results(cancer_dir: str) -> List[str]:
    """Get list of predict result directories (0_FP, 0_TN, 1_FN, 1_TP)."""
    predict_dirs = []
    for item in os.listdir(cancer_dir):
        item_path = os.path.join(cancer_dir, item)
        if os.path.isdir(item_path):
            predict_dirs.append(item)
    return predict_dirs


def get_concepts(predict_dir: str) -> List[str]:
    """Get list of concept directories."""
    concept_dirs = []
    for item in os.listdir(predict_dir):
        item_path = os.path.join(predict_dir, item)
        if os.path.isdir(item_path):
            concept_dirs.append(item)
    return concept_dirs


def run_attention_analysis(
    attention_dir: str,
    segmentation_dir: str,
    output_file: str,
    wsi_dir: str
) -> pd.DataFrame:
    """
    Run the attention analysis script and return the results DataFrame.
    
    Args:
        attention_dir: Directory containing attention scores and tile IDs
        segmentation_dir: Directory containing segmentation CSV files
        output_file: Temporary output file for this run
        
    Returns:
        DataFrame with results, or None if failed
    """
    try:
        # Run the attention analysis script
        cmd = [
            "python", "/projects/MINOTAUR/analysis/attention_analysis.py",
            attention_dir,
            segmentation_dir,
            wsi_dir,
            "--output", output_file
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        
        # Read the results
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            # Clean up temporary file
            os.remove(output_file)
            return df
        else:
            print(f"Warning: Output file {output_file} was not created")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running attention analysis: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def process_attention_structure(
    attentions_dir: str,
    segmentation_dir: str,
    wsi_dir: str,
    output_file: str
) -> pd.DataFrame:
    """
    Process the entire attention directory structure.
    
    Args:
        attentions_dir: Root directory containing the attention structure
        segmentation_dir: Directory containing segmentation CSV files
        output_file: Final output file path
        
    Returns:
        Combined DataFrame with all results
    """
    all_results = []
    
    # Get cancer types
    cancer_types = get_cancer_types(attentions_dir)
    print(f"Found cancer types: {cancer_types}")
    
    for cancer_type in cancer_types:
        cancer_path = os.path.join(attentions_dir, cancer_type)
        
        # Get predict results
        predict_results = get_predict_results(cancer_path)
        print(f"  Cancer {cancer_type}: Found predict results {predict_results}")
        
        for predict_result in predict_results:
            predict_path = os.path.join(cancer_path, predict_result)
            
            # Get concepts
            print('pattthhhhh',predict_path)
            concepts = get_concepts(predict_path)
            print(f"    Predict {predict_result}: Found concepts {concepts}")
            
            for concept in concepts:
                concept_path = os.path.join(predict_path, concept)
                print(f"      Processing concept: {concept}")
                
                # Check if this directory contains attention files
                attention_files = glob.glob(os.path.join(concept_path, "*_attention_*.npy"))
                if not attention_files:
                    print(f"        No attention files found in {concept_path}")
                    continue
                
                print(f"        Found {len(attention_files)} attention files")
                
                # Extract prediction correctness from filename
                # Example: TCGA-A1-A0SK-01Z_True_attention.npy -> True
                prediction_correctness = None
                for attn_file in attention_files:
                    filename = os.path.basename(attn_file)
                    # Look for _True_ or _False_ in the filename
                    if "_True_" in filename:
                        prediction_correctness = "True"
                        break
                    elif "_False_" in filename:
                        prediction_correctness = "False"
                        break
                
                if prediction_correctness is None:
                    print(f"        Warning: Could not determine prediction correctness from filenames in {concept_path}")
                    prediction_correctness = "Unknown"
                
                # Run attention analysis for this combination
                temp_output = f"temp_{cancer_type}_{predict_result}_{concept}.csv"
                result_df = run_attention_analysis(
                    concept_path, 
                    segmentation_dir, 
                    temp_output,
                    wsi_dir
                )
                
                if result_df is not None and not result_df.empty:
                    # Add metadata columns
                    result_df['cancer_type'] = cancer_type
                    result_df['surv_result'] = predict_result
                    result_df['concept'] = concept
                    result_df['ccpt_predict_correct'] = prediction_correctness
                    
                    all_results.append(result_df)
                    print(f"        Successfully processed {len(result_df)} tiles with prediction correctness: {prediction_correctness}")
                else:
                    print(f"        No results for {concept}")
    
    if not all_results:
        print("No results were generated from any combination")
        return pd.DataFrame()
    
    # Combine all results
    print(f"Combining {len(all_results)} result sets...")
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns to put metadata first
    column_order = ['cancer_type', 'surv_result', 'concept', 'ccpt_predict_correct', 'patient_id', 
                   'id', 'attention', 'attention_group', 'x', 'y', 'tissue_class']
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[existing_columns]
    
    return combined_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run attention analysis over the entire file structure"
    )
    parser.add_argument(
        "attentions_dir",
        help="Root directory containing the attention structure (e.g., /path/to/attentions)"
    )
    parser.add_argument(
        "segmentation_dir",
        help="Directory containing segmentation CSV files"
    )
    parser.add_argument(
        "wsi_dir",
        help="Directory containing WSI files"
    )
    parser.add_argument(
        "--output",
        default="combined_attention_results.csv",
        help="Output CSV file path (default: combined_attention_results.csv)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.attentions_dir):
        print(f"Error: Attentions directory {args.attentions_dir} does not exist")
        return
        
    if not os.path.exists(args.segmentation_dir):
        print(f"Error: Segmentation directory {args.segmentation_dir} does not exist")
        return
        
    if not os.path.exists(args.wsi_dir):
        print(f"Error: WSI directory {args.wsi_dir} does not exist")
        return
    
    print(f"Processing attention structure in: {args.attentions_dir}")
    print(f"Using segmentation data from: {args.segmentation_dir}")
    print(f"Using WSI files from: {args.wsi_dir}")
    print(f"Output will be saved to: {args.output}")
    print("-" * 80)
    
    # Process the entire structure
    results_df = process_attention_structure(
        args.attentions_dir,
        args.segmentation_dir,
        args.wsi_dir,
        args.output
    )
    
    if not results_df.empty:
        # Save combined results
        results_df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")
        print(f"Total tiles processed: {len(results_df)}")
        print(f"Cancer types: {results_df['cancer_type'].nunique()}")
        print(f"Survival results: {results_df['surv_result'].nunique()}")
        print(f"Concepts: {results_df['concept'].nunique()}")
        print(f"Concept prediction correctness: {results_df['ccpt_predict_correct'].value_counts().to_dict()}")
        print(f"Patients: {results_df['patient_id'].nunique()}")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
