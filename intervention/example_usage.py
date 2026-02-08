"""
Example usage script for concept intervention pipeline.
Demonstrates how to use the intervention scripts with sample data.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# No need for sys.path.append with proper package structure

from run_intervention_pipeline import InterventionPipeline


def create_sample_data():
    """Create sample test data for demonstration."""
    # Create sample patient data
    np.random.seed(42)
    n_patients = 100
    
    # Sample patient IDs
    patient_ids = [f"patient_{i:03d}" for i in range(n_patients)]
    
    # Sample concept labels
    # Stage: 0-3 (4 classes)
    stage_labels = np.random.randint(0, 4, n_patients)
    
    # Age: 0-2 (3 classes)
    age_labels = np.random.randint(0, 3, n_patients)
    
    # Cancer: 0-9 (10 classes)
    cancer_labels = np.random.randint(0, 10, n_patients)
    
    # RNA_Bio_ter: 0-2 (3 classes)
    rna_labels = np.random.randint(0, 3, n_patients)
    
    # Survival: 0-1 (binary)
    survival_labels = np.random.randint(0, 2, n_patients)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'ID': patient_ids,
        'Stage': stage_labels,
        'Age': age_labels,
        'Cancer': cancer_labels,
        'RNA_Bio_ter': rna_labels,
        'Survival': survival_labels
    })
    
    return sample_data


def main():
    """Example usage of the intervention pipeline."""
    print("Concept Intervention Pipeline - Example Usage")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample test data...")
    sample_data = create_sample_data()
    
    # Save sample data
    sample_data_path = "sample_test_data.csv"
    sample_data.to_csv(sample_data_path, index=False)
    print(f"Sample data saved to {sample_data_path}")
    
    # Example configuration
    config = {
        'model_path': '/path/to/your/model.ckpt',  # Update this path
        'test_data_path': sample_data_path,
        'db_path': '/path/to/your/database',  # Update this path
        'output_dir': './intervention_results',
        'concept_ids': ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter'],
        'concept_states': [4, 3, 10, 3],
        'bag_num': 1000,  # Smaller for demo
        'batch_size': 1,
        'threshold': 0.5,
        'seed': 42,
        'device': 'cpu'  # Use CPU for demo
    }
    
    print("\nExample configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nTo run the pipeline with your actual data:")
    print("1. Update the model_path to point to your trained model checkpoint")
    print("2. Update the db_path to point to your database")
    print("3. Replace sample_test_data.csv with your actual test data")
    print("4. Adjust concept_ids and concept_states to match your model")
    print("5. Run the pipeline:")
    print()
    print("python run_intervention_pipeline.py \\")
    print("    --model_path /path/to/your/model.ckpt \\")
    print("    --test_data /path/to/your/test_data.csv \\")
    print("    --db_path /path/to/your/database \\")
    print("    --output_dir ./intervention_results \\")
    print("    --concept_ids Stage Age Cancer RNA_Bio_ter \\")
    print("    --concept_states 4 3 10 3 \\")
    print("    --bag_num 3000 \\")
    print("    --batch_size 1 \\")
    print("    --threshold 0.5 \\")
    print("    --seed 42")
    print()
    print("Or run individual components:")
    print()
    print("# Baseline inference only:")
    print("python baseline_inference.py \\")
    print("    --model_path /path/to/your/model.ckpt \\")
    print("    --test_data /path/to/your/test_data.csv \\")
    print("    --db_path /path/to/your/database \\")
    print("    --output_path ./baseline_results.pkl")
    print()
    print("# Concept interventions only:")
    print("python concept_intervention.py \\")
    print("    --model_path /path/to/your/model.ckpt \\")
    print("    --test_data /path/to/your/test_data.csv \\")
    print("    --db_path /path/to/your/database \\")
    print("    --output_path ./intervention_results.pkl")
    print()
    print("# Evaluation only:")
    print("python evaluation_utils.py \\")
    print("    --intervention_results ./intervention_results.pkl \\")
    print("    --baseline_results ./baseline_results.pkl \\")
    print("    --output_dir ./analysis \\")
    print("    --save_plots")


if __name__ == "__main__":
    main()
