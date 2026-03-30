import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import yaml

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


def convert_concept_probs_to_array(concept_probs_list, config):
    """
    Convert list of dictionary-based concept probabilities to numpy array format
    expected by get_cat_concept_metrics.
    
    Args:
        concept_probs_list: List of dictionaries, each containing concept probabilities
        config: Model configuration containing concept_states and cpt_ids
        
    Returns:
        numpy.ndarray: Array with shape (n_samples, n_total_concept_classes)
    """
    n_samples = len(concept_probs_list)
    n_total_classes = sum(config.concept_states)
    
    # Initialize array
    concept_probs_array = np.zeros((n_samples, n_total_classes))
    
    for sample_idx, concept_probs_dict in enumerate(concept_probs_list):
        class_idx = 0
        
        for concept_idx, (concept_name, n_states) in enumerate(zip(config.cpt_ids, config.concept_states)):
            if n_states == 0:
                # Skip continuous concepts for categorical metrics
                continue
                
            # Extract probabilities for this concept
            for state_idx in range(n_states):
                prob_key = f'{concept_name}_{state_idx}_prob'
                if prob_key in concept_probs_dict[concept_name]:
                    concept_probs_array[sample_idx, class_idx] = concept_probs_dict[concept_name][prob_key]
                class_idx += 1
    
    return concept_probs_array


def load_config_from_yaml(config_path):
    """
    Load configuration from YAML file and convert to config object.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        object: Configuration object with all parameters
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create a simple config object
    class Config:
        def __init__(self, config_dict):
            # Data configuration
            self.db_path = config_dict['data']['db_path']
            self.bag_n = config_dict['data']['bag_n']
            self.database = config_dict['data']['database']
            self.target = config_dict['data']['target']
            self.k = config_dict['data']['k']
            
            # Concept configuration
            self.cpt_ids = config_dict['concepts']['cpt_ids']
            self.n_concepts = config_dict['concepts']['n_concepts']
            self.concept_states = config_dict['concepts']['concept_states']
            self.cpt_cls_weights = config_dict['concepts']['cpt_cls_weights']
            
            # Model architecture
            self.n_tasks = config_dict['model']['n_tasks']
            self.h_dim = config_dict['model']['h_dim']
            self.emb_dim = config_dict['model']['emb_dim']
            self.emb_activ = config_dict['model']['emb_activ']
            self.pre = config_dict['model']['pre']
            self.pre_bn_mlp = config_dict['model']['pre_bn_mlp']
            self.attn = config_dict['model']['attn']
            self.n_attn_heads = config_dict['model']['n_attn_heads']
            self.attn_dim = config_dict['model']['attn_dim']
            self.attn_dropout = config_dict['model']['attn_dropout']
            self.dropout = config_dict['model']['dropout']
            self.concept_loss_weight = config_dict['model']['concept_loss_weight']
            self.task_loss_weight = config_dict['model']['task_loss_weight']
            self.shared_prob_gen = config_dict['model']['shared_prob_gen']
            self.c2y_model = config_dict['model']['c2y_model']
            self.c2y_layers = config_dict['model']['c2y_layers']
            self.top_k_accuracy = config_dict['model']['top_k_accuracy']
            
            # Training configuration
            self.optimizer = config_dict['training']['optimizer']
            self.learning_rate = config_dict['training']['learning_rate']
            self.momentum = config_dict['training']['momentum']
            self.weight_decay = config_dict['training']['weight_decay']
            self.batch_size = config_dict['training']['batch_size']
            
            # Output configuration
            self.categorical = config_dict['output']['categorical']
            self.name = config_dict['experiment']['name']
    
    return Config(config_dict)


def load_slide_feature_vec(db_path):
    """
    Load all tiles from a slide into a single feature vector.
    
    Args:
        db_path (str): Path to the LMDB database for the slide
        
    Returns:
        tuple: (feature_tensor, keys) where feature_tensor has shape (n_tiles, feature_dim)
    """
    db = LMDBRead(db_path)
    keys = db.get_keys()
    x_wsi = []
    for i, k in enumerate(keys):
        ndarray_obj = db.read_ndarray(k)
        feature_vec = torch.Tensor(ndarray_obj.get_ndarray())
        x_wsi.append(feature_vec)
    return torch.stack(x_wsi, 0), keys


class SlideDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading all tiles from a single slide.
    """
    def __init__(self, slide_id, db_path, target, cpt_ids, dataset_df):
        self.slide_id = slide_id
        self.db_path = os.path.join(db_path, 'features', f'{slide_id}')
        self.target = target
        self.cpt_ids = cpt_ids
        self.dataset_df = dataset_df
        
        # Load all tiles from the slide
        self.features, self.keys = load_slide_feature_vec(self.db_path)
        print(f"Loaded {len(self.features)} tiles from slide {slide_id}")
        
    def _get_concepts(self):
        """Get concept values for this slide"""
        # Try to find the slide with the current ID
        slide_data = self.dataset_df[self.dataset_df['ID'] == self.slide_id]
        
        # If not found, try without the -01Z suffix
        if len(slide_data) == 0 and self.slide_id.endswith('-01Z'):
            original_id = self.slide_id[:-4]  # Remove -01Z
            slide_data = self.dataset_df[self.dataset_df['ID'] == original_id]
        
        # If still not found, try adding -01Z to the ID
        if len(slide_data) == 0 and not self.slide_id.endswith('-01Z'):
            slide_with_suffix = f"{self.slide_id}-01Z"
            slide_data = self.dataset_df[self.dataset_df['ID'] == slide_with_suffix]
        
        if len(slide_data) == 0:
            raise ValueError(f"Slide {self.slide_id} not found in dataset")
        
        cpts = []
        for c in self.cpt_ids:
            cpts.append(slide_data[c].iloc[0])
        return cpts
    
    def _get_target(self):
        """Get target value for this slide"""
        # Try to find the slide with the current ID
        slide_data = self.dataset_df[self.dataset_df['ID'] == self.slide_id]
        
        # If not found, try without the -01Z suffix
        if len(slide_data) == 0 and self.slide_id.endswith('-01Z'):
            original_id = self.slide_id[:-4]  # Remove -01Z
            slide_data = self.dataset_df[self.dataset_df['ID'] == original_id]
        
        # If still not found, try adding -01Z to the ID
        if len(slide_data) == 0 and not self.slide_id.endswith('-01Z'):
            slide_with_suffix = f"{self.slide_id}-01Z"
            slide_data = self.dataset_df[self.dataset_df['ID'] == slide_with_suffix]
        
        if len(slide_data) == 0:
            raise ValueError(f"Slide {self.slide_id} not found in dataset")
        
        return slide_data[self.target].iloc[0]
    
    def __len__(self):
        return 1  # Only one sample (the entire slide)
    
    def __getitem__(self, idx):
        # Return all tiles as a single bag
        concepts = self._get_concepts()
        target = self._get_target()
        
        return self.features, target, concepts


def load_trained_model(checkpoint_path, config):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        config: Model configuration
        
    Returns:
        ConceptEmbeddingModel: Loaded model
    """
    # Load checkpoint first to get the model state
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create a dummy dataset for model initialization
    dummy_data = pd.DataFrame({'ID': ['dummy'], 'Survival': [0]})
    for cpt in config.cpt_ids:
        dummy_data[cpt] = [0]

    from fedcbm.training import AttentionHook

    if config.attn:
        attention_hook = AttentionHook()
        atten_callback = utils.SaveAttentionCallback(attention_hook=attention_hook)
    else:
        attention_hook = None
    
    # Initialize model with same config as training (updated to match main.py)
    model = ConceptEmbeddingModel(
        n_concepts=config.n_concepts,
        concept_states=config.concept_states,
        n_tasks=config.n_tasks, 
        h_dim=config.h_dim,
        pre_bn_mlp=config.pre,
        emb_size=config.emb_dim,
        embedding_activation=config.emb_activ,
        concept_loss_weight=config.concept_loss_weight,
        concept_class_weights=torch.tensor(config.cpt_cls_weights),
        learning_rate=config.learning_rate,  
        dropout=config.dropout,
        optimizer=config.optimizer,  
        attention_hook=attention_hook
    )
    
    # Load checkpoint state dict
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    
    return model


def get_model_predictions(model, slide_dataset, config):
    """
    Get model predictions for survival and concepts from a slide.
    
    Args:
        model: Trained ConceptEmbeddingModel
        slide_dataset: SlideDataset containing all tiles
        config: Model configuration
        
    Returns:
        tuple: (survival_pred, concept_preds, concept_logits, survival_probs, concept_probs) where:
               survival_pred is int, concept_preds is list of predictions (int/float),
               concept_logits is numpy array, survival_probs is float, concept_probs is dict
    """
    # Create dataloader with single batch containing all tiles
    dataloader = DataLoader(slide_dataset, batch_size=1, shuffle=False)
    
    # Get the single batch
    batch = next(iter(dataloader))
    x, y, c = batch
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    
    # Run inference
    with torch.no_grad():
        # Forward pass
        results = model._forward(x, c, y, train=False)
        
        # Extract predictions based on CEM model return format:
        # [state_logits, cpt_embedds, y, self.concept_states, a_w_prob_lst, contexts]
        concept_logits = results[0]  # Concept prediction logits (index 0)
        y_logits = results[2]  # Survival prediction logits (index 2)
        
        # Apply sigmoid for binary survival prediction
        survival_probs_tensor = torch.sigmoid(y_logits)
        survival_pred = (survival_probs_tensor > 0.5).int().item()
        survival_probs = survival_probs_tensor.item()
        
        concept_preds, concept_probs, idx = [], {}, 0
        for i, n in enumerate(config.concept_states):
            concept_name = config.cpt_ids[i]
            if n == 0:
                # Continuous concept - use direct regression output
                c_pred = concept_logits[:, idx].item()
                concept_preds.append(c_pred)
                concept_probs[concept_name] = c_pred  # For continuous, probability = prediction
                idx += 1
            else:
                # Categorical concept - use softmax and argmax
                n_idx = idx + n
                c_probs_states = concept_logits[:, idx:n_idx]
                c_probs_states = torch.softmax(c_probs_states, axis=-1)
                c_pred = torch.argmax(c_probs_states, axis=-1).item()
                concept_preds.append(c_pred)
                
                # Store individual class probabilities
                concept_probs[concept_name] = {}
                for class_idx in range(n):
                    concept_probs[concept_name][f'{concept_name}_{class_idx}_prob'] = c_probs_states[0, class_idx].item()
                
                idx = n_idx

        return survival_pred, concept_preds, concept_logits.cpu().numpy(), survival_probs, concept_probs


def extract_attention_scores(model, slide_dataset, config):
    """
    Extract attention scores for all concepts from a slide.
    
    Args:
        model: Trained ConceptEmbeddingModel
        slide_dataset: SlideDataset containing all tiles
        config: Model configuration
        
    Returns:
        numpy.ndarray: Attention scores with shape (n_tiles, n_concepts)
    """
    # Create dataloader with single batch containing all tiles
    dataloader = DataLoader(slide_dataset, batch_size=1, shuffle=False)
    
    # Get the single batch
    batch = next(iter(dataloader))
    x, y, c = batch
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    x = x.to(device)
    
    # Run inference
    with torch.no_grad():
        # Forward pass - this will return attention scores in the last element
        results = model._forward(x, c, y, train=False)
        
        # Extract attention scores from results
        # The model returns: (state_logits, cpt_embedds, y, concept_states, a_w_prob_lst, contexts)
        a_w_prob_lst = results[4]  # List of attention weights for each concept
        
        # Convert attention scores to numpy array
        n_tiles = x.shape[1]  # Number of tiles
        n_concepts = len(config.cpt_ids)
        
        attention_scores = np.zeros((n_tiles, n_concepts))
        
        for i, a_w_prob in enumerate(a_w_prob_lst):
            if a_w_prob is not None:
                # a_w_prob has shape (batch_size, n_att_heads, n_tiles)
                # Take mean across attention heads and apply softmax
                concept_scores = a_w_prob.mean(dim=1).squeeze(0)  # Shape: (n_tiles,)
                # Apply softmax to normalize to probabilities
                #softmax = torch.nn.Softmax(dim=0)
                #concept_scores = softmax(concept_scores).cpu().numpy()  # Shape: (n_tiles,)
                attention_scores[:, i] = concept_scores.cpu().numpy()
            else:
                # No attention for this concept (using mean pooling)
                attention_scores[:, i] = 1.0 / n_tiles  # Equal attention
        
        return attention_scores


def get_true_values(slide_dataset):
    """
    Get true values for survival and concepts for a slide.
    
    Args:
        slide_dataset: SlideDataset containing slide data
        
    Returns:
        tuple: (true_survival, true_concepts) where true_survival is int and true_concepts is list of ints
    """
    # Get true survival value
    true_survival = slide_dataset._get_target()
    
    # Get true concept values
    true_concepts = slide_dataset._get_concepts()
    
    return true_survival, true_concepts


def check_model_accuracy(survival_pred, concept_preds, true_survival, true_concepts):
    """
    Check if model predictions are correct for survival and all concepts.
    
    Args:
        survival_pred (int): Model's survival prediction
        concept_preds (list): Model's concept predictions
        true_survival (int): True survival value
        true_concepts (list): True concept values
        
    Returns:
        bool: True if all predictions are correct, False otherwise
    """
    # Check survival prediction
    survival_correct = (survival_pred == true_survival)
    
    # Check concept predictions
    concept_correct = all(pred == true for pred, true in zip(concept_preds, true_concepts))
    
    return survival_correct and concept_correct


def get_survival_prediction_status(true_survival, pred_survival):
    """
    Determine the prediction status for survival (TP, TN, FP, FN).
    
    Args:
        true_survival (int): True survival value (0=alive, 1=dead)
        pred_survival (int): Predicted survival value (0=alive, 1=dead)
        
    Returns:
        str: Prediction status ('TP', 'TN', 'FP', 'FN')
    """
    if true_survival == 1 and pred_survival == 1:
        return 'TP'  # True Positive (correctly predicted dead)
    elif true_survival == 0 and pred_survival == 0:
        return 'TN'  # True Negative (correctly predicted alive)
    elif true_survival == 0 and pred_survival == 1:
        return 'FP'  # False Positive (incorrectly predicted dead)
    elif true_survival == 1 and pred_survival == 0:
        return 'FN'  # False Negative (incorrectly predicted alive)
    else:
        raise ValueError(f"Invalid survival values: true={true_survival}, pred={pred_survival}")


def get_concept_prediction_status(true_concept, pred_concept, concept_type='categorical'):
    """
    Determine the prediction status for a concept.
    
    Args:
        true_concept: True concept value
        pred_concept: Predicted concept value
        concept_type (str): 'categorical' or 'continuous'
        
    Returns:
        str: Prediction status ('True'/'False' for categorical, 'MSE' for continuous)
    """
    if concept_type == 'continuous':
        # For continuous concepts, calculate MSE
        mse = (true_concept - pred_concept) ** 2
        return f'MSE_{mse:.4f}'
    else:
        # For categorical concepts, use binary true/false
        if true_concept == pred_concept:
            return 'True'  # Correct prediction
        else:
            return 'False'  # Incorrect prediction


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


def create_hierarchical_save_path(base_path, cancer_type, survival_status, concept_name):
    """
    Create hierarchical save path for attention scores.
    
    Args:
        base_path (str): Base path for attentions folder
        cancer_type (str): Type of cancer
        survival_status (str): '1_TP', '1_FN', '0_TN', '0_FP'
        concept_name (str): Name of the concept
        
    Returns:
        str: Full path for saving attention scores
    """
    print(cancer_type, survival_status, concept_name)
    # Create path: attentions/cancer_type/survival_status/concept_name/
    save_path = os.path.join(base_path, 'attentions', cancer_type, survival_status, concept_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


def predict_batch_attention_with_validation(checkpoint_path, db_path, data_path, save_path, config_path):
    """
    Predict attention scores for all slides in dataset CSV, validate predictions, and save results.
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        db_path (str): Path to embeddings database
        data_path (str): Path to dataset CSV containing all slides to process
        save_path (str): Path to save attention scores
        config_path (str): Path to YAML configuration file
        
    Returns:
        dict: Dictionary with processing results
    """
    # Load dataset
    dataset_df = pd.read_csv(data_path)
    
    # Get all slide IDs from the dataset
    slide_ids = dataset_df['ID'].tolist()
    
    # Append "-01Z" to slide IDs if not already present
    slide_ids = [slide_id if slide_id.endswith('-01Z') else f"{slide_id}-01Z" for slide_id in slide_ids]
    
    # Load config from YAML file
    print(f"Loading configuration from: {config_path}")
    config = load_config_from_yaml(config_path)
    print(f"Loaded config: {config.name}")
    print(f"Concept states: {config.concept_states}")
    print(f"Number of concepts: {config.n_concepts}")
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(checkpoint_path, config)
    
    # Create base save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Process each slide
    results = {
        'total_slides': len(slide_ids),
        'processed_slides': [],
        'failed_slides': [],
        'survival_stats': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0},
        'concept_stats': {},
        'predictions_data': [],  # Store predictions for CSV output
        'all_concept_probs': [],  # Store all concept probabilities
        'all_concept_trues': []   # Store all true concept values
    }
    
    # Initialize concept stats
    for concept_name in config.cpt_ids:
        results['concept_stats'][concept_name] = {'True': 0, 'False': 0, 'TP': 0, 'FP': 0, 'FN': 0}
    
    for idx, slide_id in enumerate(tqdm(slide_ids, desc="Processing slides")):
        try:
            # Get cancer type for this slide from the dataset
            original_slide_id = dataset_df['ID'].iloc[idx]  # Get original ID without -01Z
            cancer_type = dataset_df['project_id_x'].iloc[idx] if 'project_id_x' in dataset_df.columns else 'unknown'
            
            # Create slide dataset
            slide_dataset = SlideDataset(slide_id, db_path, config.target, config.cpt_ids, dataset_df)
            
            # Get model predictions
            survival_pred, concept_preds, concept_logits, survival_probs, concept_probs = get_model_predictions(model, slide_dataset, config)
            print('concept_probs', concept_probs)
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
                concept_save_path = create_hierarchical_save_path(save_path, cancer_type, survival_folder, concept_name)
                
                # Save attention scores for this concept with status in filename
                attention_file = os.path.join(concept_save_path, f'{slide_id}_{concept_status}_attention.npy')
                np.save(attention_file, attention_scores[:, i])
                
                # Save tile keys for reference
                keys_file = os.path.join(concept_save_path, f'{slide_id}_{concept_status}_tile_keys.txt')
                with open(keys_file, 'w') as f:
                    for key in slide_dataset.keys:
                        f.write(f'{key}\n')
            
            # Store predictions data for CSV output
            prediction_row = {
                'slide_id': slide_id,
                'cancer_type': cancer_type,
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
            print(f"  Survival: Predicted={survival_pred} (prob={survival_probs:.4f}), True={true_survival} ({survival_status})")
            print(f"  Concepts: Predicted={concept_preds}, True={true_concepts}")
            for i, concept_name in enumerate(config.cpt_ids):
                concept_type = 'continuous' if config.concept_states[i] == 0 else 'categorical'
                concept_status = get_concept_prediction_status(true_concepts[i], concept_preds[i], concept_type)
                if concept_type == 'continuous':
                    print(f"    {concept_name}: {concept_status} (prob={concept_probs[concept_name]:.4f})")
                else:
                    # Show the predicted class probability
                    pred_class = concept_preds[i]
                    pred_prob = concept_probs[concept_name][f'{concept_name}_{pred_class}_prob']
                    print(f"    {concept_name}: {concept_status} (pred_class={pred_class}, prob={pred_prob:.4f})")
            
        except Exception as e:
            import traceback
            print(f"Error processing slide {slide_id}: {e}")
            print("Full traceback:")
            traceback.print_exc()
            results['failed_slides'].append(slide_id)
            continue
    
    # Save summary
    summary_path = os.path.join(save_path, 'processing_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Processing Summary\n")
        f.write(f"================\n")
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
        
        
        if results['failed_slides']:
            f.write(f"\nFailed slides:\n")
            for slide_id in results['failed_slides']:
                f.write(f"  - {slide_id}\n")
        
        # Calculate concept metrics using get_cat_concept_metrics
        print('results GREG', results['all_concept_probs'])
        
        if results['all_concept_probs'] and results['all_concept_trues']:
            # Convert dictionary-based concept probabilities to numpy array format
            all_concept_probs_array = convert_concept_probs_to_array(results['all_concept_probs'], config)
            all_concept_trues = np.array(results['all_concept_trues'])
            
            # Debug: Print array shapes and sample values
            print(f"DEBUG: all_concept_probs_array shape: {all_concept_probs_array.shape}")
            print(f"DEBUG: all_concept_trues shape: {all_concept_trues.shape}")
            print(f"DEBUG: concept_states: {config.concept_states}")
            print(f"DEBUG: Sample concept probs (first 3 samples, first 10 classes): {all_concept_probs_array[:3, :10]}")
            print(f"DEBUG: Sample true values (first 3 samples): {all_concept_trues[:3]}")
            
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
    
    # Save predictions CSV
    if results['predictions_data']:
        predictions_df = pd.DataFrame(results['predictions_data'])
        csv_path = os.path.join(save_path, 'attentions', 'predictions_summary.csv')
        predictions_df.to_csv(csv_path, index=False)
        print(f"Predictions CSV saved to: {csv_path}")
    
    print(f"\nProcessing complete!")
    print(f"Total slides: {len(slide_ids)}")
    print(f"Successfully processed: {len(results['processed_slides'])}")
    
    # Print survival statistics
    total_survival = sum(results['survival_stats'].values())
    if total_survival > 0:
        survival_accuracy = (results['survival_stats']['TP'] + results['survival_stats']['TN']) / total_survival * 100
        survival_f1 = calculate_f1_score(results['survival_stats']['TP'], results['survival_stats']['FP'], results['survival_stats']['FN'])
        print(f"Survival Accuracy: {survival_accuracy:.2f}%")
        print(f"Survival F1 Score: {survival_f1:.4f}")
        print(f"Survival Stats - TP: {results['survival_stats']['TP']}, TN: {results['survival_stats']['TN']}, FP: {results['survival_stats']['FP']}, FN: {results['survival_stats']['FN']}")
    
    # Print concept metrics if available
    if results['all_concept_probs'] and results['all_concept_trues']:
        print(f"\nConcept Metrics (using get_cat_concept_metrics):")
        for concept_name, stats in results['concept_stats'].items():
            if 'accuracy' in stats and 'f1_score' in stats:
                print(f"  {concept_name}:")
                print(f"    - Accuracy: {stats['accuracy']:.4f}")
                print(f"    - F1 Score: {stats['f1_score']:.4f}")
    
    # Debug: Print survival statistics
    print(f"\nDEBUG: Survival Statistics:")
    print(f"  TP: {results['survival_stats']['TP']}")
    print(f"  TN: {results['survival_stats']['TN']}")
    print(f"  FP: {results['survival_stats']['FP']}")
    print(f"  FN: {results['survival_stats']['FN']}")
    print(f"  Total: {sum(results['survival_stats'].values())}")
    
    print(f"Results saved to: {save_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Extract attention scores from trained model for all slides in dataset with validation')
    parser.add_argument('-cp', '--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('-db', '--db_path', required=True, help='Path to embeddings database')
    parser.add_argument('-dp', '--data_path', required=True, help='Path to dataset CSV containing all slides to process')
    parser.add_argument('-sp', '--save_path', required=True, help='Path to save attention scores')
    parser.add_argument('-cfg', '--config_path', required=True, help='Path to YAML configuration file')
    
    args = parser.parse_args()
    
    # Predict attention scores for batch with validation
    results = predict_batch_attention_with_validation(
        args.checkpoint_path,
        args.db_path,
        args.data_path,
        args.save_path,
        args.config_path
    )
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {len(results['processed_slides'])} slides")
    print(f"Results saved to: {args.save_path}")


if __name__ == '__main__':
    main() 
