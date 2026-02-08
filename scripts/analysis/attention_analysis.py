#!/usr/bin/env python3
"""
Posthoc analysis script for attention scores and tissue segmentation masks.

This script processes attention scores and tissue classifier segmentation data
to create a comprehensive dataframe for analysis.
"""

import argparse
import os
import glob
from pathlib import Path
from typing import Tuple, List, Dict, Any
import re

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import openslide


def df_to_tiled_segmentation(
    df: pd.DataFrame,
    x_col: str = "dim1",
    y_col: str = "dim2",
    label_col: str = "preds",
    tile_size: int = 1024,
    anchor: str = "topleft",
    overlap: str = "last",
    canvas: Any = None,
    origin: tuple[int, int] = (0, 0),
    size: tuple | None = None,
    dtype: np.dtype | None = None,
) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Build a segmentation label mask from tile coordinates and class labels.

    Args:
        df: DataFrame with coordinates and labels
        x_col: Column name for x coordinates
        y_col: Column name for y coordinates
        label_col: Column name for class labels
        tile_size: Size of tiles
        anchor: "topleft" or "center"
        overlap: "last" or "max_index"
        canvas: Canvas to paint on
        origin: Origin offset
        size: Output size if canvas not provided
        dtype: Output dtype

    Returns:
        label_mask: Segmentation mask
        idx_to_class: List mapping indices to class names
        class_to_idx: Dict mapping class names to indices
    """
    # ---- resolve canvas size ----
    def _canvas_hw(canvas_obj, size_obj):
        if canvas_obj is not None:
            if isinstance(canvas_obj, tuple) and len(canvas_obj) == 2:
                H, W = int(canvas_obj[0]), int(canvas_obj[1])
                return H, W
            if isinstance(canvas_obj, np.ndarray):
                if canvas_obj.ndim == 2:
                    H, W = canvas_obj.shape
                elif canvas_obj.ndim == 3:
                    H, W = canvas_obj.shape[:2]
                else:
                    raise ValueError("canvas ndarray must be 2D or 3D")
                return int(H), int(W)
            if isinstance(canvas_obj, Image.Image):
                W, H = canvas_obj.size
                return int(H), int(W)
            raise ValueError("canvas must be (H,W), numpy array, or PIL.Image")
        if size_obj is not None:
            H, W = int(size_obj[0]), int(size_obj[1])
            return H, W
        return None

    # ---- extract ----
    xs = df[x_col].astype(np.int64).to_numpy()
    ys = df[y_col].astype(np.int64).to_numpy()
    labels_raw = df[label_col].to_numpy()

    # coords offset if they're relative to a crop
    if origin != (0, 0):
        xs -= int(origin[0])
        ys -= int(origin[1])

    # class mapping: 0 reserved for background
    uniq = pd.unique(labels_raw)
    class_order = sorted(uniq, key=lambda z: str(z))
    class_to_idx = {c: i + 1 for i, c in enumerate(class_order)}
    idx_to_class = ['background'] + list(class_order)
    label_idx = np.vectorize(class_to_idx.get)(labels_raw).astype(np.int64)

    # anchor → top-left tile coords
    if anchor.lower() in ("topleft", "top-left", "top_left"):
        x0, y0 = xs, ys
    elif anchor.lower() == "center":
        half = tile_size // 2
        x0, y0 = xs - half, ys - half
    else:
        raise ValueError("anchor must be 'topleft' or 'center'")

    # decide output geometry
    dims = _canvas_hw(canvas, size)
    if dims is not None:
        height, width = dims
        ox, oy = 0, 0
    else:
        min_x, min_y = int(x0.min()), int(y0.min())
        max_x, max_y = int((x0 + tile_size).max()), int((y0 + tile_size).max())
        width, height = max_x - min_x, max_y - min_y
        ox, oy = min_x, min_y

    # shift to canvas coords
    x0c = (x0 - ox).astype(np.int64)
    y0c = (y0 - oy).astype(np.int64)

    # choose compact dtype automatically if not provided
    n_indices = len(idx_to_class)  # includes background
    if dtype is None:
        dtype = np.uint8 if n_indices <= 255 else np.uint16

    # allocate label mask
    label_mask = np.zeros((height, width), dtype=dtype)

    # paint tiles
    for xi, yi, li in zip(x0c, y0c, label_idx):
        x1, y1 = max(0, xi), max(0, yi)
        x2, y2 = min(width, xi + tile_size), min(height, yi + tile_size)
        if x1 >= x2 or y1 >= y2:
            continue
        if overlap == "last":
            label_mask[y1:y2, x1:x2] = li
        elif overlap == "max_index":
            block = label_mask[y1:y2, x1:x2]
            np.maximum(block, li, out=block)
        else:
            raise ValueError("overlap must be 'last' or 'max_index'")

    return label_mask, idx_to_class, class_to_idx


def percentile_clip(x: np.ndarray, lower: float = 1, upper: float = 99, eps: float = 1e-8) -> np.ndarray:
    """
    Clips and normalizes values in x to the [0,1] range based on the given percentile bounds.

    Args:
        x: Input array of attention scores
        lower: Lower percentile for clipping
        upper: Upper percentile for clipping
        eps: Small value to avoid division by zero

    Returns:
        Normalized attention scores in [0,1] after clipping
    """
    # Compute percentile bounds
    low_val = np.percentile(x, lower)
    high_val = np.percentile(x, upper)

    # Clip values outside the percentile range
    x_clipped = np.clip(x, low_val, high_val)

    # Normalize to [0, 1]
    return (x_clipped - low_val) / (high_val - low_val + eps)


def get_tile_class(coords: List[str], mask: np.ndarray, t_dim: int, ds: float = 16.002, idx_to_class: List[str] = None) -> List[str]:
    """
    Get tissue class for each tile based on majority vote in the segmentation mask.

    Args:
        coords: List of coordinate strings in format "x_y"
        mask: Resized segmentation mask
        t_dim: Tile dimension
        ds: Downsampling factor
        idx_to_class: List mapping mask indices to class names

    Returns:
        List of tissue class names for each tile
    """
    tissue_classes = []
    for coord in coords:
        c = coord.split('_')
        x, y = int(c[0]), int(c[1])
        x = int(x / ds)
        y = int(y / ds)
        
        # Extract tile region from mask
        test = mask[y:y + t_dim, x:x + t_dim]
        vals, counts = np.unique(test, return_counts=True)
        idx = counts.argmax()
        cls_idx = vals[idx]
        
        # Map back to class name if idx_to_class is provided
        if idx_to_class is not None and cls_idx < len(idx_to_class):
            tissue_class = idx_to_class[cls_idx]
        else:
            tissue_class = str(cls_idx)
        
        tissue_classes.append(tissue_class)
    
    return tissue_classes


def load_attention_data(attention_file: str, ids_file: str) -> pd.DataFrame:
    """
    Load attention scores and corresponding tile IDs.

    Args:
        attention_file: Path to .npy file with attention scores
        ids_file: Path to .txt file with tile IDs

    Returns:
        DataFrame with attention scores and tile IDs
    """
    # Load attention scores
    attention_scores = np.load(attention_file)
    
    # Load tile IDs
    with open(ids_file, "rb") as file:
        ids = [line.decode("utf-8").strip() for line in file]
    
    # Clean IDs (remove quotes and brackets)
    ids = [x[2:-1] for x in ids]
    
    # Normalize attention scores
    attention_norm = percentile_clip(attention_scores, lower=1, upper=99, eps=1e-8)
    
    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'attention': attention_norm
    })
    
    # Add attention groups (tertiles)
    df["attention_group"] = pd.qcut(df["attention"], q=3, labels=["Low", "Medium", "High"])
    
    # Extract x, y coordinates from tile IDs
    df['x'] = df['id'].apply(lambda x: int(x.split('_')[0]))
    df['y'] = df['id'].apply(lambda x: int(x.split('_')[1]))
    
    return df


def find_wsi_file(patient_id: str, wsi_dir: str) -> str:
    """
    Find the corresponding WSI file for a given patient ID.
    
    Args:
        patient_id: Patient identifier (e.g., TCGA-A1-A0SK)
        wsi_dir: Directory containing WSI files
        
    Returns:
        Path to WSI file if found, None otherwise
    """
    if not os.path.exists(wsi_dir):
        raise ValueError(f"WSI directory does not exist: {wsi_dir}")
    
    # Common WSI file extensions
    wsi_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', '.scn']
    
    # Look for files containing the patient ID in the specified directory
    for ext in wsi_extensions:
        pattern = f"*{patient_id}*{ext}"
        matches = glob.glob(os.path.join(wsi_dir, pattern))
        if matches:
            return matches[0]  # Return first match
    
    return None


def process_patient_data(
    attention_file: str,
    ids_file: str,
    segmentation_csv: str,
    patient_id: str,
    wsi_dir: str = None
) -> pd.DataFrame:
    """
    Process data for a single patient.

    Args:
        attention_file: Path to attention scores .npy file
        ids_file: Path to tile IDs .txt file
        segmentation_csv: Path to segmentation CSV file
        patient_id: Patient identifier

    Returns:
        DataFrame with patient_id, tile_id, attention, attention_group, x, y, tissue_class
    """
    print(f"Processing patient: {patient_id}")
    
    # Load attention data
    attention_df = load_attention_data(attention_file, ids_file)
    
    # Load segmentation data
    seg_df = pd.read_csv(segmentation_csv)
    print('classes: ', seg_df['preds'].value_counts())
    
    # Find and load the corresponding WSI file to get exact dimensions
    wsi_file = find_wsi_file(patient_id, wsi_dir)
    if wsi_file is None:
        raise FileNotFoundError(f"WSI file not found for {patient_id} in directory: {wsi_dir}")
    
    print(f"Found WSI file: {wsi_file}")
    wsi = openslide.OpenSlide(wsi_file)
    canvas_width, canvas_height = wsi.dimensions
    level2_dimensions = wsi.level_dimensions[2]
    print(f"WSI dimensions: {canvas_width} x {canvas_height}")
    print(f"Level 2 dimensions: {level2_dimensions}")
    wsi.close()
    
    # Create segmentation mask using exact WSI dimensions
    label_mask, idx_to_class, class_to_idx = df_to_tiled_segmentation(
        seg_df, canvas=(canvas_height, canvas_width)
    )
    print(f"Original mask shape: {label_mask.shape}")
    
    # Debug: Print class mapping information
    print(f"Class mapping: {idx_to_class}")
    print(f"Class to index: {class_to_idx}")
    
    # Resize mask to level 2 of the WSI for proper tile class extraction
    mask_resized = cv2.resize(label_mask, level2_dimensions)
    print(f"Resized mask shape: {mask_resized.shape}")
    
    # Get tissue classes for each tile using the resized mask
    tile_dim = int(224 / 4)  # Adjust based on your tile size
    tissue_classes = get_tile_class(
        list(attention_df['id']), 
        mask_resized,  # Use resized mask to level 2
        tile_dim, 
        ds=16.001,
        idx_to_class=idx_to_class
    )
    print('here')
    # Add tissue class to attention DataFrame
    attention_df['tissue_class'] = tissue_classes
    attention_df['patient_id'] = patient_id
    
    # Debug: Print tissue class distribution
    print(f"Tissue class distribution: {pd.Series(tissue_classes).value_counts().to_dict()}")
    
    # Reorder columns to match desired output
    result_df = attention_df[['patient_id', 'id', 'attention', 'attention_group', 'x', 'y', 'tissue_class']]
    
    return result_df


def find_matching_files(attention_dir: str, segmentation_dir: str) -> List[Tuple[str, str, str, str]]:
    """
    Find matching attention, IDs, and segmentation files for each patient.

    Args:
        attention_dir: Directory containing attention scores and tile IDs
        segmentation_dir: Directory containing segmentation CSV files

    Returns:
        List of tuples (patient_id, attention_file, ids_file, segmentation_csv)
    """
    matches = []
    
    # Find all attention files
    attention_files = glob.glob(os.path.join(attention_dir, "**/*_attention_*.npy"), recursive=True)
    
    for attention_file in attention_files:
        # Extract patient ID from filename
        filename = os.path.basename(attention_file)
        # Assuming format: TCGA-XXXX-XXXX_True_attention.npy
        patient_match = re.search(r'(TCGA-[A-Z0-9-]+)', filename)
        if not patient_match:
            continue
            
        patient_id = patient_match.group(1)
        
        # Find corresponding IDs file
        ids_file = attention_file.replace('_attention_', '_tile_keys_').replace('.npy', '.txt')
        if not os.path.exists(ids_file):
            print(f"Warning: IDs file not found for {patient_id}: {ids_file}")
            continue
        
        # Find corresponding segmentation CSV in the separate segmentation directory
        seg_pattern = f"{patient_id}*.csv"
        seg_files = glob.glob(os.path.join(segmentation_dir, "**", seg_pattern), recursive=True)
        
        if not seg_files:
            print(f"Warning: Segmentation CSV not found for {patient_id} in {segmentation_dir}")
            continue
        
        # Use the first matching segmentation file
        segmentation_csv = seg_files[0]
        
        matches.append((patient_id, attention_file, ids_file, segmentation_csv))
    
    return matches


def main():
    """Main function to process all patient data."""
    parser = argparse.ArgumentParser(
        description="Posthoc analysis of attention scores and tissue segmentation"
    )
    parser.add_argument(
        "attention_dir",
        help="Directory containing attention scores and tile IDs"
    )
    parser.add_argument(
        "segmentation_dir", 
        help="Directory containing segmentation CSV files"
    )
    parser.add_argument(
        "wsi_dir",
        help="Directory containing WSI files"
    )
    parser.add_argument(
        "--output",
        default="attention_analysis_results.csv",
        help="Output CSV file path (default: attention_analysis_results.csv)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.attention_dir):
        print(f"Error: Attention directory {args.attention_dir} does not exist")
        return
        
    if not os.path.exists(args.segmentation_dir):
        print(f"Error: Segmentation directory {args.segmentation_dir} does not exist")
        return
    
    # Find matching files for each patient
    print("Finding matching files...")
    matches = find_matching_files(args.attention_dir, args.segmentation_dir)
    
    if not matches:
        print("No matching files found. Please check your data directory structure.")
        return
    
    print(f"Found {len(matches)} patient datasets")
    
    # Process each patient
    all_results = []
    for patient_id, attention_file, ids_file, segmentation_csv in matches:
        try:
            result_df = process_patient_data(
                attention_file, ids_file, segmentation_csv, patient_id, args.wsi_dir
            )
            all_results.append(result_df)
            print(f"Successfully processed {patient_id}")
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue
    
    if not all_results:
        print("No data was successfully processed")
        return
    
    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    final_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(f"Total tiles processed: {len(final_df)}")
    print(f"Patients processed: {final_df['patient_id'].nunique()}")


if __name__ == "__main__":
    main()
