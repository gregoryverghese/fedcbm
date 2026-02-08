import sys
import os
import argparse
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import glob

# Updated imports for new package structure
from minotaur.models import ConceptEmbeddingModel
from minotaur.training.utils import get_cptprobs, get_yprobs, get_cembs, get_contexts
from minotaur.data.io.lmdb import LMDBRead

# Fix for pickle module import issue (for compatibility with pickled LMDB files)
import minotaur.data.io.lmdb as lmdb_io
import sys
sys.modules['lmdb_io'] = lmdb_io


def load_slide_feature_vec(db_path):
    """
    Load all tiles from a slide into a single feature vector.
    
    Args:
        db_path (str): Path to the LMDB database for the slide
        
    Returns:
        tuple: (feature_tensor, keys) where feature_tensor has shape (n_tiles, feature_dim)
    """
    print(f"Loading slide feature vectors from: {db_path}")
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
        slide_data = self.dataset_df[self.dataset_df['ID'] == self.slide_id[:-4]]
        if len(slide_data) == 0:
            raise ValueError(f"Slide {self.slide_id} not found in dataset")
        
        cpts = []
        for c in self.cpt_ids:
            cpts.append(slide_data[c].iloc[0])
        return cpts
    
    def _get_target(self):
        """Get target value for this slide"""
        slide_data = self.dataset_df[self.dataset_df['ID'] == self.slide_id[:-4]]
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
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Checkpoint exists: {os.path.exists(checkpoint_path)}")
    
    # Create a dummy dataset for model initialization
    dummy_data = pd.DataFrame({'ID': ['dummy'], 'Survival': [0]})
    for cpt in config.cpt_ids:
        dummy_data[cpt] = [0]
    
    print(f"Model config - n_concepts: {config.n_concepts}, n_tasks: {config.n_tasks}, h_dim: {config.h_dim}")
    print(f"Model config - emb_size: {config.emb_dim}, concept_states: {config.concept_states}")
    print(f"Model config - n_attn_heads: {config.n_attn_heads}, attn_dim: {config.attn_dim}, attn_dropout: {config.attn_dropout}")
    print(f"Model config - attn: {config.attn}, pre: {config.pre}, dropout: {config.dropout}")
    print(f"Model config - emb_activ: {config.emb_activ}, c_weight: {config.c_weight}")
    print(f"Model config - attention_hook: {config.attn}")
    print(f"Model config - h_dim: {config.h_dim}")
    print(f"Expected ctxt_in_dim: {config.n_attn_heads * config.h_dim}")

    import cem_mil.train.utils as utils

    if config.attn:
        attention_hook = utils.AttentionHook()
        atten_callback = utils.SaveAttentionCallback(attention_hook=attention_hook)
    else:
        None
    

    print(f"Config - concept_loss_weight: {config.concept_loss_weight}")
    # Initialize model with same config as training
    model = ConceptEmbeddingModel(
        n_concepts=config.n_concepts,
        n_tasks=config.n_tasks,
        h_dim=1024, 
        emb_size=config.emb_dim,
        concept_states=config.concept_states,
        embedding_activation=config.emb_activ,
        shared_prob_gen=False,
        concept_loss_weight=config.concept_loss_weight,
        task_loss_weight=1,
        task_class_weights=None,
        concept_class_weights=config.cpt_cls_weights,
        n_att_heads=config.n_attn_heads,
        attn_dim=config.attn_dim,
        attn_dropout=config.attn_dropout,
        dropout=config.dropout if isinstance(config.dropout, float) else 0.4,
        pre_bn_mlp=True,
        c2y_model=None,
        c2y_layers=None,
        optimizer='adam',
        momentum=0.9,
        learning_rate=0.01,
        weight_decay=4e-05,
        top_k_accuracy=None,
        attention_hook=attention_hook
    )
    
    print("Model initialized successfully")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys: {checkpoint.keys()}")
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded and set to eval mode")
    
    return model


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
    print("Creating dataloader...")
    # Create dataloader with single batch containing all tiles
    dataloader = DataLoader(slide_dataset, batch_size=1, shuffle=False)
    
    # Get the single batch
    print("Getting batch...")
    batch = next(iter(dataloader))
    x, y, c = batch
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"c: {c}")

    print(f"Input shapes - x: {x.shape}, y: {y.shape}, c: {c}")
    print(f"Input types - x: {x.dtype}, y: {y.dtype}, c: {c}")
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    x = x.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Forward pass - this will return attention scores in the last element
        results = model._forward(x, c, y, train=False)
        
        print(f"Forward pass completed. Results length: {len(results)}")
        print(f"Results types: {[type(r) for r in results]}")
        
        # Extract attention scores from results
        # The model returns: (state_logits, cpt_embedds, y, concept_states, a_w_prob_lst, contexts)
        a_w_prob_lst = results[4]  # List of attention weights for each concept
        
        print(f"Attention weights list length: {len(a_w_prob_lst)}")
        
        # Convert attention scores to numpy array
        n_tiles = x.shape[1]  # Number of tiles
        n_concepts = len(config.cpt_ids)
        
        print(f"Converting to numpy array - n_tiles: {n_tiles}, n_concepts: {n_concepts}")
        
        attention_scores = np.zeros((n_tiles, n_concepts))
        
        for i, a_w_prob in enumerate(a_w_prob_lst):
            print(f"Processing concept {i}, attention weights shape: {a_w_prob.shape if a_w_prob is not None else 'None'}")
            if a_w_prob is not None:
                # a_w_prob has shape (batch_size, n_att_heads, n_tiles)
                # Take mean across attention heads and apply softmax
                concept_scores = a_w_prob.mean(dim=1).squeeze(0)  # Shape: (n_tiles,)
                # Apply softmax to normalize to probabilities
                softmax = torch.nn.Softmax(dim=0)
                concept_scores = softmax(concept_scores).cpu().numpy()  # Shape: (n_tiles,)
                attention_scores[:, i] = concept_scores
                print(f"Concept {i} scores - min: {concept_scores.min():.4f}, max: {concept_scores.max():.4f}, mean: {concept_scores.mean():.4f}")
            else:
                # No attention for this concept (using mean pooling)
                attention_scores[:, i] = 1.0 / n_tiles  # Equal attention
                print(f"Concept {i} using equal attention (1/{n_tiles})")
        
        print(f"Final attention scores shape: {attention_scores.shape}")
        return attention_scores


def predict_slide_attention(checkpoint_path, slide_id, db_path, data_path, config_path=None, concept_type='categorical'):
    """
    Predict attention scores for all tiles in a slide.
    
    Args:
        checkpoint_path (str): Path to trained model checkpoint
        slide_id (str): ID of the slide to analyze
        db_path (str): Path to embeddings database
        data_path (str): Path to dataset CSV
        config_path (str): Path to config file (optional)
        concept_type (str): Concept type for config selection
        
    Returns:
        tuple: (attention_scores, tile_keys) where attention_scores has shape (n_tiles, n_concepts)
    """
    # Load dataset
    dataset_df = pd.read_csv(data_path)
    
    # Load config
    if concept_type:
        config = arg_def_cat.Args()
    else:
        config = arg_def.Args()
    
    print(f"Config loaded: {config.__dict__}")
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(checkpoint_path, config)
    print("Model loaded successfully")
    
    # Create slide dataset
    print(f"Creating dataset for slide {slide_id}...")
    slide_dataset = SlideDataset(slide_id, db_path, config.target, config.cpt_ids, dataset_df)
    print("Dataset created successfully")
    
    # Extract attention scores
    print("Extracting attention scores...")
    attention_scores = extract_attention_scores(model, slide_dataset, config)
    print("Attention scores extracted successfully")
    
    return attention_scores, slide_dataset.keys


def main():
    parser = argparse.ArgumentParser(description='Extract attention scores from trained model')
    parser.add_argument('-cp', '--checkpoint_path', required=True, help='Path to model checkpoint')
    parser.add_argument('-sid', '--slide_id', required=True, help='Slide ID to analyze')
    parser.add_argument('-db', '--db_path', required=True, help='Path to embeddings database')
    parser.add_argument('-dp', '--data_path', required=True, help='Path to dataset CSV')
    parser.add_argument('-sp', '--save_path', required=True, help='Path to save attention scores')
    parser.add_argument('-ct', '--concept_type', default=None, help='Concept type for config')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_path, exist_ok=True)
    
    try:
        # Predict attention scores
        attention_scores, tile_keys = predict_slide_attention(
            args.checkpoint_path,
            args.slide_id,
            args.db_path,
            args.data_path,
            concept_type=args.concept_type
        )
        
        # Save results
        np.save(os.path.join(args.save_path, f'{args.slide_id}_attention_scores.npy'), attention_scores)
        
        # Save tile keys for reference
        with open(os.path.join(args.save_path, f'{args.slide_id}_tile_keys.txt'), 'w') as f:
            for key in tile_keys:
                f.write(f'{key}\n')
        
        print(f"Attention scores shape: {attention_scores.shape}")
        print(f"Number of tiles: {len(tile_keys)}")
        print(f"Results saved to: {args.save_path}")
        
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main() 