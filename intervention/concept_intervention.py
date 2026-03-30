"""
Concept intervention script for CEM-MIL model.
Implements random concept ordering intervention to analyze concept importance.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
import random

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


class ConceptIntervention:
    """
    Concept intervention class for analyzing concept importance through random ordering.
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
        seed: int = 42,
        config_path: str = None,
        model: Optional[ConceptEmbeddingModel] = None,
        dataloader: Optional[DataLoader] = None,
        use_whole_slide: bool = False
    ):
        """
        Initialize concept intervention.
        
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
            seed: Random seed for reproducibility
            config_path: Optional path to YAML config file
            model: Optional pre-loaded model instance
            dataloader: Optional pre-created dataloader
            use_whole_slide: If True, use all tiles from each WSI; if False, use random sampling
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
        self.seed = seed
        self.config_path = config_path
        self.use_whole_slide = use_whole_slide
        
        # Set comprehensive random seeds for full reproducibility
        self._set_reproducibility(seed)
        
        # Load model or use provided model
        if model is not None:
            self.model = model
        else:
            self.model = self._load_model()
        
        # Create test dataloader or use provided one
        if dataloader is not None:
            self.test_loader = dataloader
        else:
            self.test_loader = self._create_dataloader()
        
        # Concept mapping for intervention
        self.concept_mapping = self._create_concept_mapping()
    
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
    
    def _create_concept_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Create mapping for concept intervention."""
        concept_mapping = {}
        prob_idx = 0
        
        for i, (concept_id, num_states) in enumerate(zip(self.concept_ids, self.concept_states)):
            if num_states == 0:
                # Continuous concept
                concept_mapping[concept_id] = {
                    'index': i,
                    'prob_start': prob_idx,
                    'prob_length': 1,
                    'num_states': num_states,
                    'type': 'continuous'
                }
                prob_idx += 1
            elif num_states == 1:
                # Binary concept
                concept_mapping[concept_id] = {
                    'index': i,
                    'prob_start': prob_idx,
                    'prob_length': 1,
                    'num_states': num_states,
                    'type': 'binary'
                }
                prob_idx += 1
            else:
                # Categorical concept
                concept_mapping[concept_id] = {
                    'index': i,
                    'prob_start': prob_idx,
                    'prob_length': num_states,
                    'num_states': num_states,
                    'type': 'categorical'
                }
                prob_idx += num_states
        
        return concept_mapping
    
    def _baseline_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass to get baseline predictions and contexts.
        
        Args:
            x: Input tensor (batch_size, n_tiles, embedding_dim)
            
        Returns:
            Tuple of (state_probs, contexts, y_logits, state_logits)
        """
        with torch.no_grad():
            outputs = self.model.forward(x, train=False)
            state_logits, cpt_embeds, y_logits, concept_states, attn_info, contexts = outputs
            
            # Convert logits to probabilities
            state_probs = []
            prob_idx = 0
            
            for i, num_states in enumerate(self.concept_states):
                if num_states == 0:
                    # Continuous concept - keep raw logits
                    state_probs.append(state_logits[:, prob_idx])
                    prob_idx += 1
                elif num_states == 1:
                    # Binary concept - apply sigmoid
                    state_probs.append(torch.sigmoid(state_logits[:, prob_idx]))
                    prob_idx += 1
                else:
                    # Categorical concept - apply softmax
                    start_idx = prob_idx
                    end_idx = prob_idx + num_states
                    state_probs.append(torch.softmax(state_logits[:, start_idx:end_idx], dim=-1))
                    prob_idx = end_idx
            
            # Concatenate all probabilities
            state_probs = torch.cat(state_probs, dim=1)
            
        return state_probs, contexts, y_logits, state_logits
    
    def _recompute_survival(self, state_probs: torch.Tensor, contexts: torch.Tensor) -> torch.Tensor:
        """
        Recompute survival prediction with modified concept probabilities.
        
        Args:
            state_probs: Modified concept state probabilities
            contexts: Original concept contexts
            
        Returns:
            Survival logits
        """
        with torch.no_grad():
            # Rebuild concept embeddings using modified probabilities
            cpt_embeds = self.model._get_concept_embeddings(state_probs, contexts)
            
            # Pass through survival head
            y_logits = self.model.c2y_model(cpt_embeds)
            
        return y_logits
    
    def _create_one_hot(self, concept_id: str, truth_idx: int, batch_size: int) -> torch.Tensor:
        """
        Create one-hot vector for concept intervention.
        
        Args:
            concept_id: Concept identifier
            truth_idx: Ground truth class index
            batch_size: Batch size
            
        Returns:
            One-hot tensor for the concept
        """
        concept_info = self.concept_mapping[concept_id]
        num_states = concept_info['num_states']
        prob_start = concept_info['prob_start']
        prob_length = concept_info['prob_length']
        
        if concept_info['type'] == 'continuous':
            # For continuous concepts, we don't create one-hot
            # This should not be called for continuous concepts
            raise ValueError(f"Cannot create one-hot for continuous concept {concept_id}")
        elif concept_info['type'] == 'binary':
            # Binary concept: create [0, 1] or [1, 0]
            one_hot = torch.zeros(batch_size, prob_length)
            one_hot[:, 0] = 1.0 - truth_idx  # 0 -> [1, 0], 1 -> [0, 1]
            return one_hot
        else:
            # Categorical concept: create one-hot vector
            one_hot = torch.zeros(batch_size, prob_length)
            one_hot[:, truth_idx] = 1.0
            return one_hot
    
    def _intervene_random_order(
        self, 
        x: torch.Tensor, 
        concept_truths: torch.Tensor, 
        patient_id: str
    ) -> Tuple[Dict[int, Tuple[int, float]], List[str]]:
        """
        Perform intervention with random concept ordering.
        
        Args:
            x: Input tensor
            concept_truths: Ground truth concept labels
            patient_id: Patient identifier
            
        Returns:
            Tuple of (predictions_by_t, concept_order, concept_probs_by_t)
        """
        # Get baseline prediction
        state_probs0, contexts, y0, state_logits0 = self._baseline_forward(x)
        
        # Record baseline prediction (t=0)
        preds_by_t = {}
        concept_probs_by_t = {}
        survival_score0 = torch.sigmoid(y0).cpu().numpy()
        survival_pred0 = (survival_score0 >= self.threshold).astype(int)
        preds_by_t[0] = (survival_pred0[0], survival_score0[0])
        concept_probs_by_t[0] = state_probs0[0].cpu().numpy()  # Store baseline concept probabilities
        
        
        # Create deterministic random order for this patient
        patient_rng = random.Random(self.seed + hash(patient_id))
        concept_order = self.concept_ids.copy()
        patient_rng.shuffle(concept_order)
        
        # Work on mutable copy of baseline probabilities
        state_probs_curr = state_probs0.clone()
        
        # Intervene on concepts one by one
        for t in range(1, len(self.concept_ids) + 1):
            concept = concept_order[t-1]
            concept_info = self.concept_mapping[concept]
            concept_idx = concept_info['index']
            # Skip continuous concepts for intervention
            if concept_info['type'] == 'continuous':
                continue
            
            # Get ground truth for this concept
            truth_idx = int(concept_truths[concept_idx].item())
            # Create one-hot vector for this concept
            one_hot = self._create_one_hot(concept, truth_idx, x.size(0))
            one_hot = one_hot.to(self.device)
            
            # Overwrite concept probabilities with one-hot
            prob_start = concept_info['prob_start']
            prob_length = concept_info['prob_length']
            state_probs_curr[:, prob_start:prob_start + prob_length] = one_hot
            # Recompute survival with this intervention
            y_t = self._recompute_survival(state_probs_curr, contexts)
            survival_score_t = torch.sigmoid(y_t).cpu().numpy()
            survival_pred_t = (survival_score_t >= self.threshold).astype(int)
            preds_by_t[t] = (survival_pred_t[0], survival_score_t[0])
            concept_probs_by_t[t] = state_probs_curr[0].cpu().numpy()  # Store intervention concept probabilities
        
        return preds_by_t, concept_order, concept_probs_by_t
    
    def run_interventions(self) -> Dict[str, Any]:
        """
        Run concept interventions on all test patients.
        
        Returns:
            Dictionary containing intervention results
        """
        
        # Initialize for all possible time steps (0 to num_concepts)
        num_time_steps = len(self.concept_ids) + 1
        all_preds = {t: [] for t in range(num_time_steps)}  # per-t predicted classes
        all_scores = {t: [] for t in range(num_time_steps)}  # per-t logits/probabilities
        all_truths = []  # survival ground-truth per patient
        all_patient_ids = []  # patient IDs
        orders_log = {}  # store the random order per patient
        all_concept_probs = {t: [] for t in range(num_time_steps)}  # per-t concept probabilities
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Running interventions")):
                x, y, c, patient_ids = batch
                x = x.to(self.device)
                y = y.to(self.device)
                c = c.to(self.device)
                
                
                # Process each sample in the batch
                batch_size = x.size(0)
                for sample_idx in range(batch_size):
                    # Get single sample
                    x_sample = x[sample_idx:sample_idx+1]  # Keep batch dimension
                    y_sample = y[sample_idx]
                    c_sample = c[sample_idx]
                    patient_id = patient_ids[sample_idx]
                    
                    # Run intervention for this patient
                    preds_by_t, order, concept_probs_by_t = self._intervene_random_order(x_sample, c_sample, patient_id)
                    orders_log[patient_id] = order
                    
                    # Store results for each time step
                    for t in range(num_time_steps):
                        if t in preds_by_t:
                            cls_t, score_t = preds_by_t[t]
                            all_preds[t].append(cls_t)
                            all_scores[t].append(score_t)
                            all_concept_probs[t].append(concept_probs_by_t[t])
                        else:
                            # If intervention was skipped (e.g., continuous concepts), use previous prediction
                            if t > 0:
                                all_preds[t].append(all_preds[t-1][-1])
                                all_scores[t].append(all_scores[t-1][-1])
                                all_concept_probs[t].append(all_concept_probs[t-1][-1])
                    
                    # Store ground truth and patient ID
                    all_truths.append(y_sample.cpu().numpy().item())
                    all_patient_ids.append(patient_id)
        
        # Convert to numpy arrays and squeeze extra dimensions
        for t in all_preds:
            all_preds[t] = np.array(all_preds[t]).squeeze()
            all_scores[t] = np.array(all_scores[t]).squeeze()
        all_truths = np.array(all_truths)
        
        
        # Compute metrics at each step
        metrics_by_t = self._compute_metrics_by_t(all_preds, all_scores, all_truths)
        
        # Compute error flips vs baseline
        flips_by_t = self._compute_error_flips(all_preds, all_truths)
        
        # Store results
        intervention_results = {
            'predictions_by_t': all_preds,
            'scores_by_t': all_scores,
            'ground_truth': all_truths,
            'patient_ids': all_patient_ids,
            'concept_orders': orders_log,
            'metrics_by_t': metrics_by_t,
            'error_flips_by_t': flips_by_t,
            'threshold': self.threshold,
            'concept_ids': self.concept_ids,
            'concept_states': self.concept_states
        }
        
        print("Concept interventions completed!")
        
        return intervention_results
    
    def _compute_metrics_by_t(
        self, 
        predictions_by_t: Dict[int, np.ndarray], 
        scores_by_t: Dict[int, np.ndarray], 
        ground_truth: np.ndarray
    ) -> Dict[int, Dict[str, float]]:
        """Compute metrics at each intervention step."""
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        metrics_by_t = {}
        
        for t in predictions_by_t:
            predictions = predictions_by_t[t]
            scores = scores_by_t[t]
            
            accuracy = accuracy_score(ground_truth, predictions)
            f1 = f1_score(ground_truth, predictions)
            auc = roc_auc_score(ground_truth, scores)
            
            metrics_by_t[t] = {
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc
            }
        
        return metrics_by_t
    
    def _compute_error_flips(
        self, 
        predictions_by_t: Dict[int, np.ndarray], 
        ground_truth: np.ndarray
    ) -> Dict[int, Dict[str, int]]:
        """Compute error flips vs baseline at each step."""
        baseline_preds = predictions_by_t[0]
        
        flips_by_t = {}
        
        for t in range(1, len(self.concept_ids) + 1):
            if t not in predictions_by_t:
                continue
                
            current_preds = predictions_by_t[t]
            
            # Count different types of flips
            fn_to_tp = np.sum((baseline_preds == 0) & (current_preds == 1) & (ground_truth == 1))
            fp_to_tn = np.sum((baseline_preds == 1) & (current_preds == 0) & (ground_truth == 0))
            tp_to_fn = np.sum((baseline_preds == 1) & (current_preds == 0) & (ground_truth == 1))
            tn_to_fp = np.sum((baseline_preds == 0) & (current_preds == 1) & (ground_truth == 0))
            
            flips_by_t[t] = {
                'fn_to_tp': int(fn_to_tp),
                'fp_to_tn': int(fp_to_tn),
                'tp_to_fn': int(tp_to_fn),
                'tn_to_fp': int(tn_to_fp),
                'net_improvement': int(fn_to_tp + fp_to_tn - tp_to_fn - tn_to_fp)
            }
        
        return flips_by_t
    
    
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save intervention results to file."""
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save results as CSV
        # Create a comprehensive DataFrame with all time steps
        all_data = []
        
        # Get the time steps from the results
        time_steps = sorted(results['predictions_by_t'].keys())
        
        for t in time_steps:
            for i, patient_id in enumerate(results['patient_ids']):
                row = {
                    'patient_id': patient_id,
                    'time_step': t,
                    'survival_prediction': results['predictions_by_t'][t][i],
                    'survival_score': results['scores_by_t'][t][i],
                    'survival_truth': results['ground_truth'][i]
                }
                
                # Add concept order information
                if patient_id in results['concept_orders']:
                    order = results['concept_orders'][patient_id]
                    for j, concept in enumerate(order):
                        row[f'concept_{j+1}_order'] = concept
                
                all_data.append(row)
        
        results_df = pd.DataFrame(all_data)
        results_df.to_csv(output_path, index=False)
        


def main():
    """Main function for running concept interventions."""
    parser = argparse.ArgumentParser(description='Run concept interventions on CEM-MIL model')
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
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load test dataset
    test_dataset = pd.read_csv(args.test_data)
    
    # Create intervention instance
    intervention = ConceptIntervention(
        model_path=args.model_path,
        test_dataset=test_dataset,
        db_path=args.db_path,
        concept_ids=args.concept_ids,
        concept_states=args.concept_states,
        bag_num=args.bag_num,
        batch_size=args.batch_size,
        threshold=args.threshold,
        seed=args.seed
    )
    
    # Run interventions
    results = intervention.run_interventions()
    
    # Save results
    intervention.save_results(results, args.output_path)


if __name__ == "__main__":
    main()
