#!/usr/bin/env python3
"""
Simple script to extract tiles using WSIParser with coordinate-based methods.

This script demonstrates how to use the enhanced WSIParser class to extract tiles
at specific coordinates without requiring the full tiling process.
"""

import sys
import os
import argparse
import numpy as np
import cv2
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from minotaur.wsi.parser import WSIParser
from minotaur.wsi.utilities import TissueDetect


def parse_coordinates(coord_string):
    """
    Parse coordinates from string format.
    
    Args:
        coord_string: String containing coordinates in format "[(x1,y1),(x2,y2),...]"
        
    Returns:
        List of (x, y) coordinate tuples
    """
    try:
        # Remove brackets and split by ),(
        coord_string = coord_string.strip('[]')
        coords = []
        
        if coord_string:
            # Split by ),( pattern
            coord_pairs = coord_string.split('),(')
            for pair in coord_pairs:
                # Remove any remaining parentheses
                pair = pair.strip('()')
                x, y = pair.split(',')
                coords.append((int(x.strip()), int(y.strip())))
        
        return coords
    except Exception as e:
        raise ValueError(f"Could not parse coordinates: {coord_string}. Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract tiles using WSIParser at specific coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract single tile
    python extract_tiles_with_wsi_parser.py --wsi_path slide.svs --coordinates "[(10000,15000)]"
    
    # Extract multiple tiles
    python extract_tiles_with_wsi_parser.py --wsi_path slide.svs --coordinates "[(10000,15000),(20000,15000)]" --output_dir ./tiles
        """
    )
    
    parser.add_argument(
        '--wsi_path',
        required=True,
        help='Path to the whole slide image file'
    )
    
    parser.add_argument(
        '--coordinates',
        required=True,
        help='Coordinates in format "[(x1,y1),(x2,y2),...]" at level 0 (highest resolution)'
    )
    
    parser.add_argument(
        '--output_dir',
        default='./extracted_tiles',
        help='Directory to save extracted tiles (default: ./extracted_tiles)'
    )
    
    parser.add_argument(
        '--tile_size',
        type=int,
        default=512,
        help='Size of tiles to extract (default: 512)'
    )
    
    parser.add_argument(
        '--mag_level',
        type=int,
        default=0,
        help='Magnification level for extraction (default: 0)'
    )
    
    parser.add_argument(
        '--prefix',
        default='tile',
        help='Prefix for tile filenames (default: tile)'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display extracted tiles (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    # Validate WSI path
    if not os.path.exists(args.wsi_path):
        print(f"Error: WSI file not found: {args.wsi_path}")
        return 1
    
    # Parse coordinates
    try:
        coordinates = parse_coordinates(args.coordinates)
        print(f"Parsed {len(coordinates)} coordinate pairs")
    except ValueError as e:
        print(f"Error parsing coordinates: {e}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Open slide
        print(f"Loading WSI: {args.wsi_path}")
        from openslide import OpenSlide
        slide = OpenSlide(args.wsi_path)
        
        # Create tissue detector for border (minimal setup)
        detector = TissueDetect(slide)
        border = detector.border()
        
        # Initialize WSIParser
        wsi_parser = WSIParser(
            slide=slide,
            tile_dim=args.tile_size,
            border=border,
            mag_level=args.mag_level
        )
        
        print(f"WSIParser initialized with tile size: {args.tile_size}")
        print(f"Magnification level: {args.mag_level}")
        print(f"Downsample factor: {wsi_parser._downsample}")
        
        # Extract tiles using the new coordinate-based methods
        print("Extracting tiles...")
        tiles = wsi_parser.extract_multiple_tiles_at_coordinates(coordinates)
        
        print(f"Extracted {len(tiles)} tiles")
        
        # Save tiles
        saved_paths = []
        for (x, y), tile in tiles:
            filename = f"{args.prefix}_{x}_{y}.png"
            filepath = os.path.join(args.output_dir, filename)
            
            # Convert RGB to BGR for OpenCV
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, tile_bgr)
            saved_paths.append(filepath)
            print(f"Saved: {filename}")
        
        # Display tiles if requested
        if args.display:
            try:
                import matplotlib.pyplot as plt
                
                n_tiles = len(tiles)
                cols = min(4, n_tiles)
                rows = (n_tiles + cols - 1) // cols
                
                fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
                if rows == 1:
                    axes = [axes] if cols == 1 else axes
                else:
                    axes = axes.flatten()
                
                for i, ((x, y), tile) in enumerate(tiles):
                    if i < len(axes):
                        axes[i].imshow(tile)
                        axes[i].set_title(f'Tile at ({x}, {y})')
                        axes[i].axis('off')
                
                # Hide unused subplots
                for i in range(len(tiles), len(axes)):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("Matplotlib not available for display")
        
        print(f"All tiles saved to: {args.output_dir}")
        
        # Close slide
        slide.close()
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

