"""
Evaluation utilities for concept intervention analysis.
Provides functions for analyzing intervention results and generating reports.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
# import pickle  # No longer needed - using CSV format
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')


class InterventionEvaluator:
    """
    Evaluation class for analyzing concept intervention results.
    """
    
    def __init__(self, intervention_results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator with intervention results.
        
        Args:
            intervention_results: Results from concept intervention
            baseline_results: Optional baseline results for comparison
        """
        self.intervention_results = intervention_results
        self.baseline_results = baseline_results
        
        # Extract key data
        self.predictions_by_t = intervention_results['predictions_by_t']
        self.scores_by_t = intervention_results['scores_by_t']
        self.ground_truth = intervention_results['ground_truth']
        self.metrics_by_t = intervention_results['metrics_by_t']
        self.error_flips_by_t = intervention_results['error_flips_by_t']
        self.concept_orders = intervention_results['concept_orders']
        self.concept_ids = intervention_results['concept_ids']
        self.concept_states = intervention_results['concept_states']
        self.threshold = intervention_results['threshold']
        
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate summary report of intervention results.
        
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for t in sorted(self.metrics_by_t.keys()):
            metrics = self.metrics_by_t[t]
            flips = self.error_flips_by_t.get(t, {})
            
            summary_data.append({
                'Step': t,
                'Concepts_Fixed': t,
                'Accuracy': metrics['accuracy'],
                'F1_Score': metrics['f1'],
                'AUC': metrics['auc'],
                'FN_to_TP': flips.get('fn_to_tp', 0),
                'FP_to_TN': flips.get('fp_to_tn', 0),
                'TP_to_FN': flips.get('tp_to_fn', 0),
                'TN_to_FP': flips.get('tn_to_fp', 0),
                'Net_Improvement': flips.get('net_improvement', 0)
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_concept_importance(self) -> pd.DataFrame:
        """
        Analyze concept importance based on intervention order frequency.
        
        Returns:
            DataFrame with concept importance analysis
        """
        # Count how often each concept appears in each position
        concept_positions = {concept: [0] * len(self.concept_ids) for concept in self.concept_ids}
        
        for patient_id, order in self.concept_orders.items():
            for pos, concept in enumerate(order):
                concept_positions[concept][pos] += 1
        
        # Calculate importance metrics
        importance_data = []
        for concept in self.concept_ids:
            positions = concept_positions[concept]
            total_patients = len(self.concept_orders)
            
            # Calculate average position (lower is more important)
            avg_position = sum(pos * count for pos, count in enumerate(positions)) / total_patients
            
            # Calculate how often concept is in first position
            first_position_freq = positions[0] / total_patients
            
            importance_data.append({
                'Concept': concept,
                'Avg_Position': avg_position,
                'First_Position_Freq': first_position_freq,
                'Total_Patients': total_patients,
                'Position_Distribution': positions
            })
        
        return pd.DataFrame(importance_data)
    
    def analyze_error_correction(self) -> Dict[str, Any]:
        """
        Analyze how many errors are corrected at each step.
        
        Returns:
            Dictionary with error correction analysis
        """
        baseline_preds = self.predictions_by_t[0]
        baseline_errors = baseline_preds != self.ground_truth
        
        error_correction = {}
        
        for t in sorted(self.metrics_by_t.keys()):
            if t == 0:
                continue
                
            current_preds = self.predictions_by_t[t]
            current_errors = current_preds != self.ground_truth
            
            # Count errors corrected
            errors_corrected = np.sum(baseline_errors & ~current_errors)
            errors_introduced = np.sum(~baseline_errors & current_errors)
            net_correction = errors_corrected - errors_introduced
            
            error_correction[t] = {
                'errors_corrected': int(errors_corrected),
                'errors_introduced': int(errors_introduced),
                'net_correction': int(net_correction),
                'baseline_errors': int(np.sum(baseline_errors)),
                'current_errors': int(np.sum(current_errors))
            }
        
        return error_correction
    
    def plot_metrics_evolution(self, save_path: Optional[str] = None):
        """
        Plot evolution of metrics across intervention steps.
        
        Args:
            save_path: Optional path to save the plot
        """
        steps = sorted(self.metrics_by_t.keys())
        metrics = ['accuracy', 'f1', 'auc']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [self.metrics_by_t[t][metric] for t in steps]
            axes[i].plot(steps, values, marker='o', linewidth=2, markersize=8)
            axes[i].set_title(f'{metric.upper()} Evolution')
            axes[i].set_xlabel('Intervention Step')
            axes[i].set_ylabel(metric.upper())
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics evolution plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_flips(self, save_path: Optional[str] = None):
        """
        Plot error flips across intervention steps.
        
        Args:
            save_path: Optional path to save the plot
        """
        steps = [t for t in sorted(self.error_flips_by_t.keys()) if t > 0]
        
        if not steps:
            print("No error flip data available")
            return
        
        fn_to_tp = [self.error_flips_by_t[t]['fn_to_tp'] for t in steps]
        fp_to_tn = [self.error_flips_by_t[t]['fp_to_tn'] for t in steps]
        tp_to_fn = [self.error_flips_by_t[t]['tp_to_fn'] for t in steps]
        tn_to_fp = [self.error_flips_by_t[t]['tn_to_fp'] for t in steps]
        net_improvement = [self.error_flips_by_t[t]['net_improvement'] for t in steps]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot individual flips
        width = 0.2
        x = np.arange(len(steps))
        
        ax1.bar(x - 1.5*width, fn_to_tp, width, label='FN→TP', color='green', alpha=0.7)
        ax1.bar(x - 0.5*width, fp_to_tn, width, label='FP→TN', color='blue', alpha=0.7)
        ax1.bar(x + 0.5*width, tp_to_fn, width, label='TP→FN', color='red', alpha=0.7)
        ax1.bar(x + 1.5*width, tn_to_fp, width, label='TN→FP', color='orange', alpha=0.7)
        
        ax1.set_xlabel('Intervention Step')
        ax1.set_ylabel('Number of Flips')
        ax1.set_title('Error Flips by Type')
        ax1.set_xticks(x)
        ax1.set_xticklabels(steps)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot net improvement
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in net_improvement]
        ax2.bar(steps, net_improvement, color=colors, alpha=0.7)
        ax2.set_xlabel('Intervention Step')
        ax2.set_ylabel('Net Improvement')
        ax2.set_title('Net Error Correction')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error flips plot saved to {save_path}")
        
        plt.show()
    
    def plot_concept_importance(self, save_path: Optional[str] = None):
        """
        Plot concept importance analysis.
        
        Args:
            save_path: Optional path to save the plot
        """
        importance_df = self.analyze_concept_importance()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot average position (lower is more important)
        concepts = importance_df['Concept']
        avg_positions = importance_df['Avg_Position']
        
        bars1 = ax1.bar(concepts, avg_positions, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Concept')
        ax1.set_ylabel('Average Position')
        ax1.set_title('Concept Importance (Lower Position = More Important)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pos in zip(bars1, avg_positions):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{pos:.2f}', ha='center', va='bottom')
        
        # Plot first position frequency
        first_pos_freq = importance_df['First_Position_Freq']
        bars2 = ax2.bar(concepts, first_pos_freq, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Concept')
        ax2.set_ylabel('First Position Frequency')
        ax2.set_title('How Often Each Concept is Fixed First')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, freq in zip(bars2, first_pos_freq):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{freq:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Concept importance plot saved to {save_path}")
        
        plt.show()
    
    def generate_confusion_matrices(self, save_path: Optional[str] = None):
        """
        Generate confusion matrices for each intervention step.
        
        Args:
            save_path: Optional path to save the plot
        """
        n_steps = len(self.metrics_by_t)
        fig, axes = plt.subplots(1, n_steps, figsize=(5*n_steps, 4))
        
        if n_steps == 1:
            axes = [axes]
        
        for i, t in enumerate(sorted(self.metrics_by_t.keys())):
            predictions = self.predictions_by_t[t]
            cm = confusion_matrix(self.ground_truth, predictions)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Step {t} (Concepts Fixed: {t})')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def save_detailed_report(self, output_path: str):
        """
        Save detailed report to file.
        
        Args:
            output_path: Path to save the report
        """
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate reports
        summary_df = self.generate_summary_report()
        importance_df = self.analyze_concept_importance()
        error_correction = self.analyze_error_correction()
        
        # Save to CSV files (replace .xlsx with .csv in path)
        base_path = str(output_path).replace('.xlsx', '')
        
        # Save summary report
        summary_path = f"{base_path}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        # Save concept importance analysis
        importance_path = f"{base_path}_concept_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        
        # Save error correction analysis
        error_df = pd.DataFrame([
            {**{'step': t}, **data} 
            for t, data in error_correction.items()
        ])
        error_path = f"{base_path}_error_correction.csv"
        error_df.to_csv(error_path, index=False)
        
        print(f"Detailed reports saved to:")
        print(f"  - Summary: {summary_path}")
        print(f"  - Concept Importance: {importance_path}")
        print(f"  - Error Correction: {error_path}")
    
    def print_summary(self):
        """Print summary of intervention results."""
        print("=" * 60)
        print("CONCEPT INTERVENTION ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Basic info
        print(f"Number of patients: {len(self.ground_truth)}")
        print(f"Number of concepts: {len(self.concept_ids)}")
        print(f"Concepts: {', '.join(self.concept_ids)}")
        print(f"Decision threshold: {self.threshold}")
        print()
        
        # Metrics evolution
        print("METRICS EVOLUTION:")
        print("-" * 40)
        summary_df = self.generate_summary_report()
        print(summary_df.to_string(index=False, float_format='%.4f'))
        print()
        
        # Concept importance
        print("CONCEPT IMPORTANCE:")
        print("-" * 40)
        importance_df = self.analyze_concept_importance()
        for _, row in importance_df.iterrows():
            print(f"{row['Concept']}: Avg Position = {row['Avg_Position']:.2f}, "
                  f"First Position Freq = {row['First_Position_Freq']:.2f}")
        print()
        
        # Error correction
        print("ERROR CORRECTION:")
        print("-" * 40)
        error_correction = self.analyze_error_correction()
        for t, data in error_correction.items():
            print(f"Step {t}: {data['errors_corrected']} errors corrected, "
                  f"{data['errors_introduced']} errors introduced, "
                  f"net improvement = {data['net_correction']}")


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load intervention results from CSV file.
    
    Args:
        results_path: Path to CSV results file
        
    Returns:
        Loaded results dictionary
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    # Determine if this is baseline or intervention results based on filename
    if 'baseline' in results_path:
        return _load_baseline_results(results_path)
    elif 'intervention' in results_path:
        return _load_intervention_results(results_path)
    else:
        raise ValueError(f"Cannot determine result type from path: {results_path}")


def _load_baseline_results(csv_path: str) -> Dict[str, Any]:
    """Load baseline results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Extract basic columns
    patient_ids = df['patient_id'].tolist()
    predictions = df['survival_prediction'].values
    scores = df['survival_score'].values
    ground_truth = df['survival_truth'].values
    
    # Extract concept predictions and truths
    concept_predictions = []
    concept_truths = []
    
    # Get concept columns
    concept_pred_cols = [col for col in df.columns if col.endswith('_prediction')]
    concept_truth_cols = [col for col in df.columns if col.endswith('_truth')]
    
    for pred_col, truth_col in zip(concept_pred_cols, concept_truth_cols):
        concept_predictions.append(df[pred_col].values)
        concept_truths.append(df[truth_col].values)
    
    concept_truths = np.column_stack(concept_truths) if concept_truths else np.array([])
    
    return {
        'patient_ids': patient_ids,
        'predictions': predictions,
        'scores': scores,
        'ground_truth': ground_truth,
        'concept_predictions': concept_predictions,
        'concept_truths': concept_truths,
        'metrics': {}  # Will be computed when needed
    }


def _load_intervention_results(csv_path: str) -> Dict[str, Any]:
    """Load intervention results from CSV."""
    df = pd.read_csv(csv_path)
    
    # Get unique patients and time steps
    patient_ids = df['patient_id'].unique().tolist()
    time_steps = sorted(df['time_step'].unique())
    
    # Organize data by time step
    predictions = {}
    scores = {}
    ground_truth = df['survival_truth'].values[:len(patient_ids)]  # Same for all time steps
    
    for t in time_steps:
        t_data = df[df['time_step'] == t]
        predictions[t] = t_data['survival_prediction'].values
        scores[t] = t_data['survival_score'].values
    
    # Extract concept orders
    concept_orders = {}
    order_cols = [col for col in df.columns if 'concept_' in col and 'order' in col]
    
    for _, row in df[df['time_step'] == 0].iterrows():  # Orders are same for all time steps
        patient_id = row['patient_id']
        order = [row[col] for col in order_cols if pd.notna(row[col])]
        concept_orders[patient_id] = order
    
    return {
        'patient_ids': patient_ids,
        'predictions': predictions,
        'scores': scores,
        'ground_truth': ground_truth,
        'concept_orders': concept_orders,
        'time_steps': time_steps,
        'metrics': {}  # Will be computed when needed
    }


def compare_baseline_vs_intervention(
    baseline_path: str, 
    intervention_path: str
) -> pd.DataFrame:
    """
    Compare baseline vs intervention results.
    
    Args:
        baseline_path: Path to baseline results
        intervention_path: Path to intervention results
        
    Returns:
        Comparison DataFrame
    """
    # Load results
    baseline_results = load_results(baseline_path)
    intervention_results = load_results(intervention_path)
    
    # Extract baseline metrics
    baseline_metrics = baseline_results['metrics']
    
    # Extract intervention metrics
    intervention_metrics = intervention_results['metrics_by_t']
    
    # Create comparison
    comparison_data = []
    
    # Add baseline
    comparison_data.append({
        'Step': 'Baseline',
        'Concepts_Fixed': 0,
        'Accuracy': baseline_metrics['survival_accuracy'],
        'F1_Score': baseline_metrics['survival_f1'],
        'AUC': baseline_metrics['survival_auc']
    })
    
    # Add intervention steps
    for t in sorted(intervention_metrics.keys()):
        metrics = intervention_metrics[t]
        comparison_data.append({
            'Step': f'Intervention_{t}',
            'Concepts_Fixed': t,
            'Accuracy': metrics['accuracy'],
            'F1_Score': metrics['f1'],
            'AUC': metrics['auc']
        })
    
    return pd.DataFrame(comparison_data)


def main():
    """Example usage of evaluation utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate concept intervention results')
    parser.add_argument('--intervention_results', type=str, required=True, 
                       help='Path to intervention results file')
    parser.add_argument('--baseline_results', type=str, 
                       help='Path to baseline results file (optional)')
    parser.add_argument('--output_dir', type=str, default='./intervention_analysis',
                       help='Output directory for analysis')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to files')
    
    args = parser.parse_args()
    
    # Load results
    intervention_results = load_results(args.intervention_results)
    baseline_results = None
    if args.baseline_results:
        baseline_results = load_results(args.baseline_results)
    
    # Create evaluator
    evaluator = InterventionEvaluator(intervention_results, baseline_results)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate analysis
    evaluator.print_summary()
    
    if args.save_plots:
        evaluator.plot_metrics_evolution(f"{args.output_dir}/metrics_evolution.png")
        evaluator.plot_error_flips(f"{args.output_dir}/error_flips.png")
        evaluator.plot_concept_importance(f"{args.output_dir}/concept_importance.png")
        evaluator.generate_confusion_matrices(f"{args.output_dir}/confusion_matrices.png")
    
    # Save detailed report
    evaluator.save_detailed_report(f"{args.output_dir}/detailed_report.xlsx")


if __name__ == "__main__":
    main()
