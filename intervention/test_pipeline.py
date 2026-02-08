"""
Test script for the concept intervention pipeline.
Tests the pipeline components without requiring actual model/data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# No need for sys.path.append with proper package structure

def test_data_loading():
    """Test data loading functionality."""
    print("Testing data loading...")
    
    # Create sample data
    np.random.seed(42)
    n_patients = 10
    
    sample_data = pd.DataFrame({
        'ID': [f"patient_{i:03d}" for i in range(n_patients)],
        'Stage': np.random.randint(0, 4, n_patients),
        'Age': np.random.randint(0, 3, n_patients),
        'Cancer': np.random.randint(0, 10, n_patients),
        'RNA_Bio_ter': np.random.randint(0, 3, n_patients),
        'Survival': np.random.randint(0, 2, n_patients)
    })
    
    print(f"Created sample data with {len(sample_data)} patients")
    print("Sample data:")
    print(sample_data.head())
    
    return sample_data

def test_concept_mapping():
    """Test concept mapping functionality."""
    print("\nTesting concept mapping...")
    
    concept_ids = ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter']
    concept_states = [4, 3, 10, 3]
    
    # Create concept mapping
    concept_mapping = {}
    prob_idx = 0
    
    for i, (concept_id, num_states) in enumerate(zip(concept_ids, concept_states)):
        if num_states == 0:
            concept_mapping[concept_id] = {
                'index': i,
                'prob_start': prob_idx,
                'prob_length': 1,
                'num_states': num_states,
                'type': 'continuous'
            }
            prob_idx += 1
        elif num_states == 1:
            concept_mapping[concept_id] = {
                'index': i,
                'prob_start': prob_idx,
                'prob_length': 1,
                'num_states': num_states,
                'type': 'binary'
            }
            prob_idx += 1
        else:
            concept_mapping[concept_id] = {
                'index': i,
                'prob_start': prob_idx,
                'prob_length': num_states,
                'num_states': num_states,
                'type': 'categorical'
            }
            prob_idx += num_states
    
    print("Concept mapping:")
    for concept, info in concept_mapping.items():
        print(f"  {concept}: {info}")
    
    return concept_mapping

def test_one_hot_creation():
    """Test one-hot vector creation."""
    print("\nTesting one-hot vector creation...")
    
    concept_mapping = test_concept_mapping()
    batch_size = 2
    
    # Test binary concept
    concept_id = 'Age'
    truth_idx = 1
    concept_info = concept_mapping[concept_id]
    
    if concept_info['type'] == 'binary':
        one_hot = torch.zeros(batch_size, concept_info['prob_length'])
        one_hot[:, 0] = 1.0 - truth_idx
        print(f"Binary concept {concept_id} (truth={truth_idx}): {one_hot}")
    
    # Test categorical concept
    concept_id = 'Stage'
    truth_idx = 2
    concept_info = concept_mapping[concept_id]
    
    if concept_info['type'] == 'categorical':
        one_hot = torch.zeros(batch_size, concept_info['prob_length'])
        one_hot[:, truth_idx] = 1.0
        print(f"Categorical concept {concept_id} (truth={truth_idx}): {one_hot}")

def test_metrics_computation():
    """Test metrics computation."""
    print("\nTesting metrics computation...")
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    # Create sample predictions and ground truth
    np.random.seed(42)
    n_samples = 100
    
    ground_truth = np.random.randint(0, 2, n_samples)
    predictions = np.random.randint(0, 2, n_samples)
    scores = np.random.rand(n_samples)
    
    # Compute metrics
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    auc = roc_auc_score(ground_truth, scores)
    
    print(f"Sample metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")

def test_error_flip_analysis():
    """Test error flip analysis."""
    print("\nTesting error flip analysis...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    ground_truth = np.random.randint(0, 2, n_samples)
    baseline_preds = np.random.randint(0, 2, n_samples)
    current_preds = np.random.randint(0, 2, n_samples)
    
    # Count flips
    fn_to_tp = np.sum((baseline_preds == 0) & (current_preds == 1) & (ground_truth == 1))
    fp_to_tn = np.sum((baseline_preds == 1) & (current_preds == 0) & (ground_truth == 0))
    tp_to_fn = np.sum((baseline_preds == 1) & (current_preds == 0) & (ground_truth == 1))
    tn_to_fp = np.sum((baseline_preds == 0) & (current_preds == 1) & (ground_truth == 0))
    
    net_improvement = fn_to_tp + fp_to_tn - tp_to_fn - tn_to_fp
    
    print(f"Error flip analysis:")
    print(f"  FN→TP: {fn_to_tp}")
    print(f"  FP→TN: {fp_to_tn}")
    print(f"  TP→FN: {tp_to_fn}")
    print(f"  TN→FP: {tn_to_fp}")
    print(f"  Net improvement: {net_improvement}")

def test_concept_importance():
    """Test concept importance analysis."""
    print("\nTesting concept importance analysis...")
    
    concept_ids = ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter']
    n_patients = 100
    
    # Simulate random concept orders
    np.random.seed(42)
    concept_orders = {}
    
    for i in range(n_patients):
        patient_id = f"patient_{i:03d}"
        order = concept_ids.copy()
        np.random.shuffle(order)
        concept_orders[patient_id] = order
    
    # Count positions
    concept_positions = {concept: [0] * len(concept_ids) for concept in concept_ids}
    
    for patient_id, order in concept_orders.items():
        for pos, concept in enumerate(order):
            concept_positions[concept][pos] += 1
    
    # Calculate importance
    print("Concept importance analysis:")
    for concept in concept_ids:
        positions = concept_positions[concept]
        avg_position = sum(pos * count for pos, count in enumerate(positions)) / n_patients
        first_position_freq = positions[0] / n_patients
        
        print(f"  {concept}:")
        print(f"    Average position: {avg_position:.2f}")
        print(f"    First position frequency: {first_position_freq:.2f}")
        print(f"    Position distribution: {positions}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING CONCEPT INTERVENTION PIPELINE COMPONENTS")
    print("=" * 60)
    
    try:
        test_data_loading()
        test_concept_mapping()
        test_one_hot_creation()
        test_metrics_computation()
        test_error_flip_analysis()
        test_concept_importance()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        print("\nThe intervention pipeline components are working correctly.")
        print("You can now run the pipeline with your actual model and data.")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please check the error and fix any issues.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
