#!/usr/bin/env python3
"""
Generate heatmap overlays for attention scores.

This script loops over all attention files in the attentions directory structure
and generates heatmap overlays using the Overlay class from wsi_heatmaps.py.
"""

import os
import sys
import argparse
import numpy as np
import glob
from tqdm import tqdm

# Add paths for imports
# Updated: No need for sys.path.append with proper package structure

# Import the Overlay class from wsi_heatmaps.py
from wsi_heatmaps import Overlay
import openslide
import cv2
import matplotlib.pyplot as plt
import traceback


def find_attention_files(attentions_dir):
    """
    Find all attention files in the attentions directory structure.
    
    Args:
        attentions_dir (str): Path to the attentions directory
        
    Returns:
        list: List of tuples (file_path, cancer_type, survival_status, concept_name, patient_id, prediction_status)
    """
    attention_files = []
    
    # Walk through the directory structure: attentions/cancer/survival_status/concept/
    for cancer_dir in glob.glob(os.path.join(attentions_dir, "*")):
        if not os.path.isdir(cancer_dir):
            continue
            
        cancer_type = os.path.basename(cancer_dir)
        
        for survival_dir in glob.glob(os.path.join(cancer_dir, "*")):
            if not os.path.isdir(survival_dir):
                continue
                
            survival_status = os.path.basename(survival_dir)
            
            for concept_dir in glob.glob(os.path.join(survival_dir, "*")):
                if not os.path.isdir(concept_dir):
                    continue
                    
                concept_name = os.path.basename(concept_dir)
                
                # Find attention files in this concept directory
                for attention_file in glob.glob(os.path.join(concept_dir, "*_attention.npy")):
                    # Extract patient ID and prediction status from filename
                    # Format: patient_id_STATUS_attention.npy
                    filename = os.path.basename(attention_file)
                    parts = filename.split('_')
                    
                    if len(parts) >= 3:
                        patient_id = parts[0]
                        prediction_status = parts[1]  # True or False
                        
                        attention_files.append({
                            'file_path': attention_file,
                            'cancer_type': cancer_type,
                            'survival_status': survival_status,
                            'concept_name': concept_name,
                            'patient_id': patient_id,
                            'prediction_status': prediction_status
                        })
    
    return attention_files


def get_wsi_path(patient_id, wsi_dir):
    """
    Find the WSI file for a given patient ID.
    
    Args:
        patient_id (str): Patient ID
        wsi_dir (str): Directory containing WSI files
        
    Returns:
        str: Path to WSI file or None if not found
    """
    # Look for WSI file with patient ID in the name
    wsi_files = glob.glob(os.path.join(wsi_dir, f"*{patient_id}*"))
    if wsi_files:
        return wsi_files[0]
    return None


def get_tile_keys_path(patient_id, keys_dir):
    """
    Find the tile keys file for a given patient ID.
    
    Args:
        patient_id (str): Patient ID
        keys_dir (str): Directory containing tile keys files
        
    Returns:
        str: Path to tile keys file or None if not found
    """
    # Look for tile keys file with patient ID in the name
    keys_files = glob.glob(os.path.join(keys_dir, f"*{patient_id}*_tile_keys.txt"))
    if keys_files:
        return keys_files[0]
    return None


def generate_heatmap_overlay(attention_file_info, wsi_dir, keys_dir, overlays_dir, args):
    """
    Generate heatmap overlay for a single attention file.
    
    Args:
        attention_file_info (dict): Information about the attention file
        wsi_dir (str): Directory containing WSI files
        keys_dir (str): Directory containing tile keys files
        overlays_dir (str): Directory to save overlays
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract information
        attention_file = attention_file_info['file_path']
        cancer_type = attention_file_info['cancer_type']
        survival_status = attention_file_info['survival_status']
        concept_name = attention_file_info['concept_name']
        patient_id = attention_file_info['patient_id']
        prediction_status = attention_file_info['prediction_status']
        
        # Create output directory structure
        output_dir = os.path.join(
            overlays_dir, 
            cancer_type, 
            survival_status, 
            concept_name,
            f"{patient_id}_{prediction_status}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Find WSI file
        wsi_path = get_wsi_path(patient_id, wsi_dir)
        if not wsi_path:
            print(f"WSI file not found for patient {patient_id}")
            return False
        
        # Find tile keys file
        keys_path = get_tile_keys_path(patient_id, keys_dir)
        if not keys_path:
            print(f"Tile keys file not found for patient {patient_id}")
            return False
        
        # Load attention scores
        attentions = np.load(attention_file, allow_pickle=True)
        
        # Load WSI
        wsi = openslide.OpenSlide(wsi_path)
        
        # Load tile keys
        with open(keys_path, 'r') as f:
            keys = [line.strip() for line in f.readlines()]
        
        # Convert keys to coordinates (assuming format like "x_y")
        keys = [(int(c.split('_')[0][2:]), int(c.split('_')[1][:-1])) for c in keys]
        
        # Get thumbnail
        thumb = wsi.get_thumbnail(wsi.level_dimensions[args.downsampling_level])
        thumb = np.array(thumb.convert('RGB'))
        
        # Create overlay
        overlay = Overlay(thumb, attentions, keys)
        
        # Generate score distribution plot
        scores_dist = overlay.plot_dist()
        plt.savefig(os.path.join(output_dir, f'{patient_id}_scoresdist.png'))
        plt.close()
        
        # Generate heatmap
        downsample = int(wsi.level_downsamples[args.downsampling_level])
        heatmap = overlay.get_overlay(args.tile_dims, downsample)
        overlay_heatmap = overlay.get_gaussian_heatmap(args.sigma, args.alpha)
        
        # Save heatmaps
        cv2.imwrite(os.path.join(output_dir, f'{patient_id}_heatmap.png'), overlay.heatmap)
        cv2.imwrite(os.path.join(output_dir, f'{patient_id}_gaussian.png'), overlay_heatmap)
        
        return True
        
    except Exception as e:
        print(f"Error generating heatmap for {patient_id}: {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate heatmap overlays for attention scores')
    parser.add_argument('--attentions_dir', required=True, help='Path to attentions directory')
    parser.add_argument('--wsi_dir', required=True, help='Path to WSI files directory')
    parser.add_argument('--keys_dir', required=True, help='Path to tile keys files directory')
    parser.add_argument('--overlays_dir', required=True, help='Path to save overlays')
    parser.add_argument('--downsampling_level', type=int, default=2, help='Downsampling level for WSI')
    parser.add_argument('--tile_dims', type=int, nargs=2, default=[224, 224], help='Tile dimensions')
    parser.add_argument('--sigma', type=float, default=10.0, help='Gaussian sigma for heatmap')
    parser.add_argument('--alpha', type=float, default=0.7, help='Alpha for heatmap overlay')
    
    args = parser.parse_args()
    
    print("Finding attention files...")
    attention_files = find_attention_files(args.attentions_dir)
    print(f"Found {len(attention_files)} attention files")
    
    # Create overlays directory
    os.makedirs(args.overlays_dir, exist_ok=True)
    
    # Process each attention file
    successful = 0
    failed = 0
    
    for file_info in tqdm(attention_files, desc="Generating heatmaps"):
        if generate_heatmap_overlay(file_info, args.wsi_dir, args.keys_dir, args.overlays_dir, args):
            successful += 1
        else:
            failed += 1
    
    print(f"\nHeatmap generation complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {args.overlays_dir}")


if __name__ == '__main__':
    main() 