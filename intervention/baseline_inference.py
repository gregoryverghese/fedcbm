"""
Baseline inference script for CEM-MIL model.
Runs inference on test set to establish baseline performance.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from tqdm import tqdm

# Updated imports for new package structure
from fedcbm.models import ConceptEmbeddingModel
from fedcbm.data import TileDataset
from fedcbm.training import AttentionHook
from fedcbm.data.io.lmdb import LMDBRead

# Fix for pickle module import issue (for compatibility with pickled LMDB files)
import fedcbm.data.io.lmdb as lmdb_io
import sys
sys.modules['lmdb_io'] = lmdb_io
from torch.utils.data import DataLoader


def load_slide_feature_vec(db_path):
    """Load all feature vectors from a slide's database and convert to tensors."""
    import lmdb
    import pickle
    import torch
    import numpy as np
    
    features = []
    keys = []
    
    env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            key_str = key.decode('utf-8')
            ndarray = pickle.loads(value)
            
            # Convert NpyObject to tensor (same as TileDataset._buildbag)
            if hasattr(ndarray, 'get_ndarray'):
                feature_vec = ndarray.get_ndarray()
            else:
                feature_vec = ndarray
            
            feature_vec = torch.Tensor(np.array(feature_vec))
            feature_vec = torch.squeeze(feature_vec, 0)
            
            features.append(feature_vec)
            keys.append(key_str)
    env.close()
    
    # Stack all features into a single tensor (same as TileDataset._buildbag)
    if features:
        features = torch.stack(features, 0)
        features = torch.squeeze(features)
    
    return features, keys


class WholeSlideDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading all tiles from multiple slides (only those with available databases).
    Uses the same logic as TileDataset to find matching databases.
    """
    def __init__(self, dataset_df, db_path, target, cpt_ids):
        self.dataset_df = dataset_df
        self.target = target
        self.cpt_ids = cpt_ids
        
        # Find all available database paths using the same logic as TileDataset
        import glob
        db_paths = glob.glob(os.path.join(db_path, 'features', '*'))
        
        # Find matching databases for slides in our dataset (same logic as TileDataset)
        self.ws_dbs = [
            p 
            for p in db_paths 
            for db in list(dataset_df['ID']) 
            if db == os.path.basename(p)[:-4]
        ]
        
        print(f"Found {len(self.ws_dbs)} slides with available databases out of {len(dataset_df)} total slides")
        
        # Create mapping from database path to slide data
        self.slide_data = {}
        for db_path in self.ws_dbs:
            slide_id = os.path.basename(db_path)[:-4]  # Remove -01Z
            slide_row = dataset_df[dataset_df['ID'] == slide_id].iloc[0]
            self.slide_data[db_path] = {
                'slide_id': slide_id,
                'target': slide_row[target],
                'concepts': [slide_row[c] for c in cpt_ids]
            }
        
        # Load all features upfront
        self.all_features = {}
        self.all_keys = {}
        for db_path in self.ws_dbs:
            slide_id = os.path.basename(db_path)[:-4]
            features, keys = load_slide_feature_vec(db_path)
            self.all_features[db_path] = features
            self.all_keys[db_path] = keys
            print(f"Loaded {len(features)} tiles from slide {slide_id}")
        
    def __len__(self):
        return len(self.ws_dbs)  # Number of slides with available databases
    
    def __getitem__(self, idx):
        # Get the database path for this index
        db_path = self.ws_dbs[idx]
        
        # Get slide data
        slide_info = self.slide_data[db_path]
        
        # Get features for this slide
        features = self.all_features[db_path]
        
        # Convert to tensors (same format as TileDataset)
        target = torch.tensor(slide_info['target'], dtype=torch.float32)
        concepts = torch.tensor(slide_info['concepts'], dtype=torch.float32)
        
        # Return same format as TileDataset: bag, target, cpts, wsi_id
        return features, target, concepts, slide_info['slide_id']


class BaselineInference:
    """
    Baseline inference class for running CEM-MIL model on test set.
    """
    
    def __init__(
        self,
        model_path: str,
        test_dataset: pd.DataFrame,
        db_path: str,
        concept_ids: List[str],
        concept_states: List[int],
        bag_num: int = 3000,
        batch_size: int = 1,
        threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config_path: str = None,
        seed: int = 42,
        use_whole_slide: bool = False
    ):
        """
        Initialize baseline inference.
        
        Args:
            model_path: Path to trained model checkpoint
            test_dataset: Test dataset DataFrame with patient IDs and labels
            db_path: Path to database containing tile embeddings
            concept_ids: List of concept names
            concept_states: List of number of states for each concept
            bag_num: Number of tiles per bag
            batch_size: Batch size for inference
            threshold: Decision threshold for binary classification
            device: Device to run inference on
            config_path: Optional path to YAML config file
        """
        self.model_path = model_path
        self.test_dataset = test_dataset
        self.db_path = db_path
        self.concept_ids = concept_ids
        self.concept_states = concept_states
        self.bag_num = bag_num
        self.batch_size = batch_size
        self.threshold = threshold
        self.device = device
        self.config_path = config_path
        self.use_whole_slide = use_whole_slide
        
        # Set comprehensive random seeds for full reproducibility
        self._set_reproducibility(seed)
        
        # Load model
        self.model = self._load_model()
        
        # Create test dataloader
        self.test_loader = self._create_dataloader()
    
    def _set_reproducibility(self, seed: int):
        """Set comprehensive random seeds for full reproducibility."""
        import random
        import numpy as np
        import torch
        import os
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Set deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variables for reproducibility
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Enable deterministic algorithms (PyTorch 1.7+)
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        
    def _load_model(self) -> ConceptEmbeddingModel:
        """Load the trained model from checkpoint."""
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model hyperparameters from checkpoint with proper defaults from config template
        hyper_params = checkpoint.get('hyper_parameters', {})
        
        # Create attention hook if attention is enabled
        attention_hook = None
        if hyper_params.get('attn', True):  # Default to True if not specified
            attention_hook = AttentionHook()
        
        model_kwargs = {
            'n_concepts': len(self.concept_ids),
            'n_tasks': hyper_params.get('n_tasks', 1),  # Binary survival prediction
            'h_dim': hyper_params.get('h_dim', 1024),
            'emb_size': hyper_params.get('emb_size', 8),  # From config template
            'concept_states': self.concept_states,
            'embedding_activation': hyper_params.get('embedding_activation', 'LeakyReLU'),
            'n_att_heads': hyper_params.get('n_att_heads', 4),
            'attn_dim': hyper_params.get('attn_dim', 256),
            'attn_dropout': hyper_params.get('attn_dropout', 0.3),  # From config template
            'dropout': hyper_params.get('dropout', 0.3),  # From config template
            'pre_bn_mlp': hyper_params.get('pre_bn_mlp', True),
            'concept_loss_weight': hyper_params.get('concept_loss_weight', 1),
            'task_loss_weight': hyper_params.get('task_loss_weight', 1),
            'shared_prob_gen': hyper_params.get('shared_prob_gen', False),
            'c2y_model': hyper_params.get('c2y_model', None),
            'c2y_layers': hyper_params.get('c2y_layers', None),
            'optimizer': hyper_params.get('optimizer', 'adam'),
            'learning_rate': hyper_params.get('learning_rate', 0.001),
            'weight_decay': hyper_params.get('weight_decay', 4e-05),
            'attention_hook': attention_hook,
        }

        
        # Create model instance
        model = ConceptEmbeddingModel(**model_kwargs)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def get_model(self) -> ConceptEmbeddingModel:
        """Get the loaded model instance."""
        return self.model
    
    def get_dataloader(self) -> DataLoader:
        """Get the test dataloader instance."""
        return self.test_loader
    
    def _create_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        
        if self.use_whole_slide:
            # Create dataset with all tiles per slide (only for slides with available databases)
            test_dataset = WholeSlideDataset(
                dataset_df=self.test_dataset,
                db_path=self.db_path,
                target='Survival',
                cpt_ids=self.concept_ids
            )
        else:
            # Use traditional random sampling approach
            test_dataset = TileDataset(
                dataset=self.test_dataset,
                db_path=self.db_path,
                target='Survival',
                cpt_ids=self.concept_ids,
                bag_num=self.bag_num
            )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Set to 0 to avoid multiprocessing issues with whole slide mode
        )
        
        return test_loader
    
    def run_baseline_inference(self) -> Dict[str, Any]:
        """
        Run baseline inference on test set.
        
        Returns:
            Dictionary containing baseline results
        """
        
        all_predictions = []
        all_scores = []
        all_ground_truth = []
        all_concept_truths = []
        all_concept_predictions = []
        all_patient_ids = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Baseline inference")):
                x, y, c, patient_ids = batch
                x = x.to(self.device)
                y = y.to(self.device)
                c = c.to(self.device)
                
                # Forward pass
                outputs = self.model.forward(x, c=c, y=y, train=False)
                state_logits, cpt_embeds, y_logits, concept_states, attn_info, contexts = outputs
                
                # Get concept probabilities
                state_probs = []
                prob_idx = 0
                for i, num_states in enumerate(self.concept_states):
                    if num_states == 0:
                        # Continuous concept
                        state_probs.append(state_logits[:, prob_idx])
                        prob_idx += 1
                    elif num_states == 1:
                        # Binary concept
                        state_probs.append(torch.sigmoid(state_logits[:, prob_idx]))
                        prob_idx += 1
                    else:
                        # Categorical concept
                        start_idx = prob_idx
                        end_idx = prob_idx + num_states
                        state_probs.append(torch.softmax(state_logits[:, start_idx:end_idx], dim=-1))
                        prob_idx = end_idx
                
                # Get survival prediction
                survival_score = torch.sigmoid(y_logits).cpu().numpy()
                survival_pred = (survival_score >= self.threshold).astype(int)
                
                # Store results
                all_predictions.extend(survival_pred.flatten())
                all_scores.extend(survival_score.flatten())
                all_ground_truth.extend(y.cpu().numpy().flatten())
                all_concept_truths.extend(c.cpu().numpy())
                
                # Store concept predictions properly - each concept gets its own list
                if not all_concept_predictions:
                    # Initialize lists for each concept
                    all_concept_predictions = [[] for _ in range(len(self.concept_states))]
                
                for i, (prob, num_states) in enumerate(zip(state_probs, self.concept_states)):
                    if num_states == 0:
                        # Continuous concept - store as is
                        all_concept_predictions[i].append(prob.cpu().numpy().flatten())
                    elif num_states == 1:
                        # Binary concept - store probability
                        all_concept_predictions[i].append(prob.cpu().numpy().flatten())
                    else:
                        # Categorical concept - store the full probability distribution
                        all_concept_predictions[i].append(prob.cpu().numpy())
                
                # Get patient ID from the batch
                for i in range(len(patient_ids)):
                    all_patient_ids.append(patient_ids[i])
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_scores = np.array(all_scores)
        all_ground_truth = np.array(all_ground_truth)
        all_concept_truths = np.array(all_concept_truths)
        
        # Compute baseline metrics
        baseline_metrics = self._compute_metrics(
            all_predictions, all_scores, all_ground_truth, 
            all_concept_predictions, all_concept_truths
        )
        
        # Store results
        baseline_results = {
            'predictions': all_predictions,
            'scores': all_scores,
            'ground_truth': all_ground_truth,
            'concept_truths': all_concept_truths,
            'concept_predictions': all_concept_predictions,
            'patient_ids': all_patient_ids,
            'metrics': baseline_metrics,
            'threshold': self.threshold
        }
        
        print("Baseline inference completed!")
        
        return baseline_results
    
    def _compute_metrics(
        self, 
        predictions: np.ndarray, 
        scores: np.ndarray, 
        ground_truth: np.ndarray,
        concept_predictions: List[np.ndarray],
        concept_truths: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        # Survival metrics
        survival_accuracy = accuracy_score(ground_truth, predictions)
        survival_f1 = f1_score(ground_truth, predictions)
        survival_auc = roc_auc_score(ground_truth, scores)
        
        # Concept metrics
        concept_accuracies = []
        concept_f1s = []
        
        for i, concept_pred_list in enumerate(concept_predictions):
            concept_truth = concept_truths[:, i]  # Get truth for this concept
            
            if self.concept_states[i] == 0:
                # Continuous concept - skip for now
                continue
            elif self.concept_states[i] == 1:
                # Binary concept - stack all predictions
                concept_pred = np.concatenate(concept_pred_list)
                concept_pred_binary = (concept_pred >= 0.5).astype(int)
                concept_accuracies.append(accuracy_score(concept_truth, concept_pred_binary))
                concept_f1s.append(f1_score(concept_truth, concept_pred_binary))
            else:
                # Categorical concept - stack probability distributions and get argmax
                concept_pred_dist = np.vstack(concept_pred_list)  # Stack along batch dimension
                concept_pred_cat = np.argmax(concept_pred_dist, axis=1)
           
                concept_accuracies.append(accuracy_score(concept_truth, concept_pred_cat))
                concept_f1s.append(f1_score(concept_truth, concept_pred_cat, average='macro'))
        
        return {
            'survival_accuracy': survival_accuracy,
            'survival_f1': survival_f1,
            'survival_auc': survival_auc,
            'concept_accuracies': concept_accuracies,
            'concept_f1s': concept_f1s,
            'avg_concept_accuracy': np.mean(concept_accuracies) if concept_accuracies else 0.0,
            'avg_concept_f1': np.mean(concept_f1s) if concept_f1s else 0.0
        }
    
    
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save baseline results to file."""
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results as CSV
        results_df = pd.DataFrame({
            'patient_id': results['patient_ids'],
            'survival_prediction': results['predictions'],
            'survival_score': results['scores'],
            'survival_truth': results['ground_truth']
        })
        
        # Add concept predictions as columns
        for i, concept_id in enumerate(self.concept_ids):
            if i < len(results['concept_predictions']):
                concept_preds = results['concept_predictions'][i]
                if len(concept_preds) > 0:
                    if concept_preds[0].ndim > 1:
                        # Categorical concept - take argmax
                        concept_preds_cat = [np.argmax(pred) for pred in concept_preds]
                    else:
                        # Binary/continuous concept
                        concept_preds_cat = concept_preds
                    results_df[f'{concept_id}_prediction'] = concept_preds_cat
        
        # Add concept truths as columns
        for i, concept_id in enumerate(self.concept_ids):
            if i < results['concept_truths'].shape[1]:
                results_df[f'{concept_id}_truth'] = results['concept_truths'][:, i]
        
        results_df.to_csv(output_path, index=False)
        


def main():
    """Main function for running baseline inference."""
    parser = argparse.ArgumentParser(description='Run baseline inference on CEM-MIL model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test dataset CSV')
    parser.add_argument('--db_path', type=str, required=True, help='Path to database')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results')
    parser.add_argument('--concept_ids', nargs='+', default=['Stage', 'Age', 'Cancer', 'RNA_Bio_ter'], 
                       help='List of concept IDs')
    parser.add_argument('--concept_states', nargs='+', type=int, default=[4, 3, 10, 3],
                       help='List of concept states')
    parser.add_argument('--bag_num', type=int, default=3000, help='Number of tiles per bag')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    
    args = parser.parse_args()
    
    # Load test dataset
    test_dataset = pd.read_csv(args.test_data)
    
    # Create baseline inference instance
    baseline = BaselineInference(
        model_path=args.model_path,
        test_dataset=test_dataset,
        db_path=args.db_path,
        concept_ids=args.concept_ids,
        concept_states=args.concept_states,
        bag_num=args.bag_num,
        batch_size=args.batch_size,
        threshold=args.threshold
    )
    
    # Run baseline inference
    results = baseline.run_baseline_inference()
    
    # Save results
    baseline.save_results(results, args.output_path)


if __name__ == "__main__":
    main()
