"""
Main script to run the complete concept intervention pipeline.
Orchestrates baseline inference, concept interventions, and evaluation.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import time

# Updated imports - no need for sys.path.append with proper package structure
# Fix for pickle module import issue (for compatibility with pickled LMDB files)
import minotaur.data.io.lmdb as lmdb_io
import sys
sys.modules['lmdb_io'] = lmdb_io

# Use relative imports since these files are in the same directory
from baseline_inference import BaselineInference
from concept_intervention import ConceptIntervention
from evaluation_utils import InterventionEvaluator, load_results


class InterventionPipeline:
    """
    Complete pipeline for concept intervention analysis.
    """
    
    def __init__(
        self,
        model_path: str,
        test_data_path: str,
        db_path: str,
        output_dir: str,
        concept_ids: list = ['Stage', 'Age', 'Cancer', 'RNA_Bio_ter'],
        concept_states: list = [4, 3, 10, 3],
        bag_num: int = 3000,
        batch_size: int = 1,
        threshold: float = 0.5,
        seed: int = 42,
        device: str = 'cuda',
        config_path: str = None
    ):
        """
        Initialize intervention pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            test_data_path: Path to test dataset CSV
            db_path: Path to database containing tile embeddings
            output_dir: Output directory for results
            concept_ids: List of concept names
            concept_states: List of number of states for each concept
            bag_num: Number of tiles per bag
            batch_size: Batch size for inference
            threshold: Decision threshold for binary classification
            seed: Random seed for reproducibility
            device: Device to run inference on
            config_path: Optional path to YAML config file
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.db_path = db_path
        self.output_dir = output_dir
        self.concept_ids = concept_ids
        self.concept_states = concept_states
        self.bag_num = bag_num
        self.batch_size = batch_size
        self.threshold = threshold
        self.seed = seed
        self.device = device
        self.config_path = config_path
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load test dataset
        self.test_dataset = pd.read_csv(self.test_data_path)
    
    def run_baseline_inference(self, force_rerun: bool = False) -> Dict[str, Any]:
        """
        Run baseline inference on test set.
        
        Args:
            force_rerun: Force rerun even if results exist
            
        Returns:
            Baseline results
        """
        baseline_path = os.path.join(self.output_dir, 'baseline_results.csv')
        
        if os.path.exists(baseline_path) and not force_rerun:
            from evaluation_utils import load_results
            return load_results(baseline_path)
        
        start_time = time.time()
        
        # Create baseline inference instance
        baseline = BaselineInference(
            model_path=self.model_path,
            test_dataset=self.test_dataset,
            db_path=self.db_path,
            concept_ids=self.concept_ids,
            concept_states=self.concept_states,
            bag_num=self.bag_num,
            batch_size=self.batch_size,
            threshold=self.threshold,
            device=self.device,
            config_path=self.config_path
        )
        
        # Run baseline inference
        results = baseline.run_baseline_inference()
        
        # Save results
        baseline.save_results(results, baseline_path)
        
        elapsed_time = time.time() - start_time
        
        # Store the baseline model and dataloader for reuse in interventions
        self.baseline_model = baseline.get_model()
        self.baseline_dataloader = baseline.get_dataloader()
        
        return results
    
    def run_concept_interventions(self, force_rerun: bool = False) -> Dict[str, Any]:
        """
        Run concept interventions on test set.
        
        Args:
            force_rerun: Force rerun even if results exist
            
        Returns:
            Intervention results
        """
        intervention_path = os.path.join(self.output_dir, 'intervention_results.csv')
        
        if os.path.exists(intervention_path) and not force_rerun:
            from evaluation_utils import load_results
            return load_results(intervention_path)
        
        start_time = time.time()
        
        # Create intervention instance with baseline model and dataloader
        baseline_model = getattr(self, 'baseline_model', None)
        baseline_dataloader = getattr(self, 'baseline_dataloader', None)
        
        intervention = ConceptIntervention(
            model_path=self.model_path,
            test_dataset=self.test_dataset,
            db_path=self.db_path,
            concept_ids=self.concept_ids,
            concept_states=self.concept_states,
            bag_num=self.bag_num,
            batch_size=self.batch_size,
            threshold=self.threshold,
            device=self.device,
            seed=self.seed,
            config_path=self.config_path,
            model=baseline_model,  # Use baseline model if available
            dataloader=baseline_dataloader  # Use baseline dataloader if available
        )
        
        # Run interventions
        results = intervention.run_interventions()
        
        # Save results
        intervention.save_results(results, intervention_path)
        
        elapsed_time = time.time() - start_time
        
        return results
    
    def run_evaluation(self, baseline_results: Dict[str, Any], intervention_results: Dict[str, Any]) -> None:
        """
        Run evaluation and generate reports.
        
        Args:
            baseline_results: Baseline inference results
            intervention_results: Concept intervention results
        """
        
        # Create evaluator
        evaluator = InterventionEvaluator(intervention_results, baseline_results)
        
        # Print summary
        evaluator.print_summary()
        
        # Generate plots
        plots_dir = os.path.join(self.output_dir, 'plots')
        Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        evaluator.plot_metrics_evolution(os.path.join(plots_dir, 'metrics_evolution.png'))
        evaluator.plot_error_flips(os.path.join(plots_dir, 'error_flips.png'))
        evaluator.plot_concept_importance(os.path.join(plots_dir, 'concept_importance.png'))
        evaluator.generate_confusion_matrices(os.path.join(plots_dir, 'confusion_matrices.png'))
        
        # Save detailed report
        report_path = os.path.join(self.output_dir, 'detailed_report.xlsx')
        evaluator.save_detailed_report(report_path)
        
    
    def run_complete_pipeline(self, force_rerun: bool = False) -> None:
        """
        Run the complete intervention pipeline.
        
        Args:
            force_rerun: Force rerun all steps even if results exist
        """
        print("=" * 60)
        print("STARTING CONCEPT INTERVENTION PIPELINE")
        print("=" * 60)
        
        total_start_time = time.time()
        
        # Step 1: Baseline inference
        print("\n" + "=" * 40)
        print("STEP 1: BASELINE INFERENCE")
        print("=" * 40)
        baseline_results = self.run_baseline_inference(force_rerun)
        
        # Step 2: Concept interventions
        print("\n" + "=" * 40)
        print("STEP 2: CONCEPT INTERVENTIONS")
        print("=" * 40)
        intervention_results = self.run_concept_interventions(force_rerun)
        
        # Step 3: Evaluation
        print("\n" + "=" * 40)
        print("STEP 3: EVALUATION AND REPORTING")
        print("=" * 40)
        self.run_evaluation(baseline_results, intervention_results)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"\n" + "=" * 60)
        print(f"PIPELINE COMPLETED IN {total_elapsed_time:.2f} SECONDS")
        print("=" * 60)
        
        # Print final summary
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Main function for running the intervention pipeline."""
    parser = argparse.ArgumentParser(description='Run complete concept intervention pipeline')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, 
                       help='Path to test dataset CSV')
    parser.add_argument('--db_path', type=str, required=True, 
                       help='Path to database containing tile embeddings')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--concept_ids', nargs='+', 
                       default=['Stage', 'Age', 'Cancer', 'RNA_Bio_ter'],
                       help='List of concept IDs')
    parser.add_argument('--concept_states', nargs='+', type=int, 
                       default=[4, 3, 10, 3],
                       help='List of concept states')
    parser.add_argument('--bag_num', type=int, default=3000, 
                       help='Number of tiles per bag')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Decision threshold for binary classification')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to run inference on')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to YAML config file (optional)')
    parser.add_argument('--force_rerun', action='store_true', 
                       help='Force rerun all steps even if results exist')
    
    # Pipeline control
    parser.add_argument('--skip_baseline', action='store_true', 
                       help='Skip baseline inference step')
    parser.add_argument('--skip_intervention', action='store_true', 
                       help='Skip concept intervention step')
    parser.add_argument('--skip_evaluation', action='store_true', 
                       help='Skip evaluation step')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = InterventionPipeline(
        model_path=args.model_path,
        test_data_path=args.test_data,
        db_path=args.db_path,
        output_dir=args.output_dir,
        concept_ids=args.concept_ids,
        concept_states=args.concept_states,
        bag_num=args.bag_num,
        batch_size=args.batch_size,
        threshold=args.threshold,
        seed=args.seed,
        device=args.device,
        config_path=args.config_path
    )
    
    # Run pipeline steps
    if not args.skip_baseline and not args.skip_intervention and not args.skip_evaluation:
        # Run complete pipeline
        pipeline.run_complete_pipeline(args.force_rerun)
    else:
        # Run individual steps
        baseline_results = None
        intervention_results = None
        
        if not args.skip_baseline:
            baseline_results = pipeline.run_baseline_inference(args.force_rerun)
        
        if not args.skip_intervention:
            intervention_results = pipeline.run_concept_interventions(args.force_rerun)
        
        if not args.skip_evaluation and baseline_results and intervention_results:
            pipeline.run_evaluation(baseline_results, intervention_results)


if __name__ == "__main__":
    main()
