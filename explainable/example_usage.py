#!/usr/bin/env python3
"""
Example usage of the attention prediction scripts.

This script demonstrates how to use the prediction scripts to extract attention scores
from trained models and shows the expected output format.
"""
import sys
# Updated: No need for sys.path.append with proper package structure

import os
import numpy as np
import pandas as pd
#from predict_attention import predict_slide_attention
from predict_attention_batch import predict_batch_attention_with_validation, calculate_f1_score


def example_single_slide():
    """
    Example of processing a single slide.
    """
    print("=== Single Slide Example ===")
    
    # Example parameters (replace with your actual paths)
    checkpoint_path = "/analysis/minotaur/results/uni/cat_model_exv18/0/3_1_0_3_3000_True_8_1_0.3_attention_exp0_noweights_checkmodel.ckpt"
    slide_id = "TCGA-E2-A2P5-01Z"
    db_path = "/data/embeddings/uni/224-20x-0.2"
    data_path = "/data/datasets/cat_stcanage_60surv_train_cohort.csv"
    save_path = "/analysis/minotaur/attentions"
    
    try:
        # Predict attention scores for a single slide
        attention_scores, tile_keys = predict_slide_attention(
            checkpoint_path=checkpoint_path,
            slide_id=slide_id,
            db_path=db_path,
            data_path=data_path,
            concept_type='categorical'  # or specify concept type if needed
        )
        
        print(f"Attention scores shape: {attention_scores.shape}")
        print(f"Number of tiles: {len(tile_keys)}")
        print(f"Number of concepts: {attention_scores.shape[1]}")
        
        # attention_scores has shape (n_tiles, n_concepts)
        # Each row represents a tile, each column represents a concept
        print(f"\nFirst 5 tiles, first 3 concepts:")
        print(attention_scores[:5, :3])
        
        # Save results
        print(f"Saving results to: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, f'{slide_id}_attention_scores.npy'), attention_scores)
        
        # Save tile keys for reference
        with open(os.path.join(save_path, f'{slide_id}_tile_keys.txt'), 'w') as f:
            for key in tile_keys:
                f.write(f'{key}\n')
                
        print(f"\nResults saved to: {save_path}")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("Please check your file paths and model configuration.")


def example_batch_processing():
    """
    Example of processing multiple slides in batch with validation.
    """
    print("\n=== Batch Processing Example ===")
    
    
    checkpoint_path = "/analysis/minotaur/results/uni/mm_model_exv14/0/3_10_0_3_3000_True_8_10_0.3_attention_exp0_noweights_checkmodel.ckpt"
    db_path = "/data/embeddings/uni/224-20x-0.2"
    data_path =  "/data/datasets/cat_stcanagerna_60surv_train_cohort.csv"
    slide_ids_csv = "/data/datasets/r0_f3_test_BLCH.csv" 
    save_path = "/analysis/minotaur/attentions2"
    
    try:
        # Process multiple slides with validation
        results = predict_batch_attention_with_validation(
            checkpoint_path=checkpoint_path,
            db_path=db_path,
            data_path=data_path,
            slide_ids_csv=slide_ids_csv,
            save_path=save_path,
            concept_type=None  # or specify concept type if needed
        )
        
        print(f"Successfully processed {len(results['processed_slides'])} slides")
        print(f"Total slides to process: {results['total_slides']}")
        
        # Show survival statistics
        survival_stats = results['survival_stats']
        print(f"\nSurvival Prediction Statistics:")
        print(f"  - True Positives (TP): {survival_stats['TP']}")
        print(f"  - True Negatives (TN): {survival_stats['TN']}")
        print(f"  - False Positives (FP): {survival_stats['FP']}")
        print(f"  - False Negatives (FN): {survival_stats['FN']}")
        
        # Show concept statistics
        print(f"\nConcept Prediction Statistics:")
        for concept_name, stats in results['concept_stats'].items():
            print(f"  {concept_name}:")
            print(f"    - Correct: {stats['True']}, Incorrect: {stats['False']}")
            total_concept = stats['True'] + stats['False']
            if total_concept > 0:
                concept_accuracy = stats['True'] / total_concept * 100
                concept_f1 = calculate_f1_score(stats['TP'], stats['FP'], stats['FN'])
                print(f"    - Accuracy: {concept_accuracy:.2f}%")
                print(f"    - F1 Score: {concept_f1:.4f}")
        
        print(f"\nResults saved to: {save_path}")
        print(f"Predictions CSV saved to: {save_path}/attentions/predictions_summary.csv")
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("Full traceback:")
        traceback.print_exc()
        print("Please check your file paths and model configuration.")


def analyze_attention_scores(attention_scores, concept_names):
    """
    Analyze attention scores to understand model behavior.
    
    Args:
        attention_scores (np.ndarray): Shape (n_tiles, n_concepts)
        concept_names (list): List of concept names
    """
    print("\n=== Attention Score Analysis ===")
    
    n_tiles, n_concepts = attention_scores.shape
    
    print(f"Total tiles: {n_tiles}")
    print(f"Total concepts: {n_concepts}")
    
    # Analyze each concept
    for i, concept_name in enumerate(concept_names):
        concept_scores = attention_scores[:, i]
        
        print(f"\nConcept: {concept_name}")
        print(f"  - Mean attention: {concept_scores.mean():.4f}")
        print(f"  - Std attention: {concept_scores.std():.4f}")
        print(f"  - Min attention: {concept_scores.min():.4f}")
        print(f"  - Max attention: {concept_scores.max():.4f}")
        
        # Find tiles with highest attention for this concept
        top_indices = np.argsort(concept_scores)[-5:][::-1]
        print(f"  - Top 5 tile indices: {top_indices}")
        print(f"  - Top 5 attention scores: {concept_scores[top_indices]}")
    
    # Analyze attention distribution
    print(f"\nOverall attention statistics:")
    print(f"  - Global mean: {attention_scores.mean():.4f}")
    print(f"  - Global std: {attention_scores.std():.4f}")
    print(f"  - Sparsity (zeros): {(attention_scores == 0).sum() / attention_scores.size:.2%}")


def create_heatmap_data(attention_scores, tile_keys, concept_names):
    """
    Prepare data for creating attention heatmaps.
    
    Args:
        attention_scores (np.ndarray): Shape (n_tiles, n_concepts)
        tile_keys (list): List of tile keys/identifiers
        concept_names (list): List of concept names
        
    Returns:
        dict: Dictionary with data ready for heatmap visualization
    """
    print("\n=== Heatmap Data Preparation ===")
    
    # Create a DataFrame for easy manipulation
    heatmap_df = pd.DataFrame(attention_scores, columns=concept_names)
    heatmap_df['tile_key'] = tile_keys
    
    print(f"Heatmap DataFrame shape: {heatmap_df.shape}")
    print(f"Columns: {list(heatmap_df.columns)}")
    
    # Example: Get top tiles for each concept
    for concept in concept_names:
        top_tiles = heatmap_df.nlargest(10, concept)[['tile_key', concept]]
        print(f"\nTop 10 tiles for {concept}:")
        print(top_tiles.head())
    
    return heatmap_df


def create_slide_ids_csv():
    """
    Example of creating a CSV file with slide IDs to process.
    """
    print("\n=== Creating Slide IDs CSV Example ===")
    
    # Example slide IDs and cancer types (replace with your actual data)
    slide_ids = [
        "TCGA-E2-A2P5",
        "TCGA-E2-A2P6", 
        "TCGA-E2-A2P7",
        "TCGA-E2-A2P8",
        "TCGA-E2-A2P9"
    ]
    
    cancer_types = [
        "BRCA",
        "LUAD",
        "BRCA",
        "LUAD",
        "BRCA"
    ]
    
    # Create DataFrame
    slide_ids_df = pd.DataFrame({
        'ID': slide_ids,
        'cancer': cancer_types
    })
    
    # Save to CSV
    csv_path = "slide_ids_to_process.csv"
    slide_ids_df.to_csv(csv_path, index=False)
    
    print(f"Created slide IDs CSV: {csv_path}")
    print(f"Contains {len(slide_ids)} slide IDs")
    print(f"CSV format:")
    print(slide_ids_df.head())
    print(f"\nNote: The script will automatically append '-01Z' to slide IDs if needed.")
    
    return csv_path


if __name__ == "__main__":
    print("Attention Score Prediction Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    #example_single_slide()
    #create_slide_ids_csv()  # Uncomment to create example slide IDs CSV
    example_batch_processing()
    
    # Example of analyzing saved results
    # attention_scores = np.load("/path/to/slide_001_attention_scores.npy")
    # concept_names = ["Stage", "Age", "Cancer_Type"]  # Replace with your concept names
    # tile_keys = ["tile_001", "tile_002", ...]  # Load from saved file
    
    # analyze_attention_scores(attention_scores, concept_names)
    # heatmap_data = create_heatmap_data(attention_scores, tile_keys, concept_names)
    
    print("\nTo use these scripts:")
    print("1. Update the file paths in the examples")
    print("2. Create a CSV file with slide IDs (see create_slide_ids_csv() function)")
    print("3. Uncomment the example functions you want to run")
    print("4. Run: python example_usage.py")
    print("\nExpected folder structure:")
    print("attentions/")
    print("├── predictions_summary.csv")
    print("├── BRCA/")
    print("│   ├── 1_TP/ (True Positives - correctly predicted dead)")
    print("│   ├── 1_FN/ (False Negatives - incorrectly predicted alive when dead)")
    print("│   ├── 0_TN/ (True Negatives - correctly predicted alive)")
    print("│   └── 0_FP/ (False Positives - incorrectly predicted dead when alive)")
    print("│       └── concept_name/")
    print("│           └── slide_id_STATUS_attention.npy")
    print("└── LUAD/")
    print("    └── ... (same structure)") 