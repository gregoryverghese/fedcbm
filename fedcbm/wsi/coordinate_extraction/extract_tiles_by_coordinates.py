#!/usr/bin/env python3
"""
Simple script to extract tiles from WSI at specific coordinates.

Usage:
    python extract_tiles_by_coordinates.py --wsi_path /path/to/slide.svs --coordinates "[(10000,15000),(20000,15000)]" --output_dir ./tiles

This script provides a command-line interface for extracting tiles at specific coordinates
from whole slide images using the MINOTAUR tiling system.
"""

import sys
import os
import argparse
import json
from pathlib import Path

# Add the current directory to the path to import from tiler module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from minotaur.wsi.coordinate_extraction.coordinate_tile_extractor import CoordinateTileExtractor


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
        description="Extract tiles from WSI at specific coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract single tile
    python extract_tiles_by_coordinates.py --wsi_path slide.svs --coordinates "[(10000,15000)]"
    
    # Extract multiple tiles
    python extract_tiles_by_coordinates.py --wsi_path slide.svs --coordinates "[(10000,15000),(20000,15000)]" --output_dir ./my_tiles
    
    # Extract with custom tile size
    python extract_tiles_by_coordinates.py --wsi_path slide.svs --coordinates "[(10000,15000)]" --tile_size 1024
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
        '--check_borders',
        action='store_true',
        help='Check if tiles fit within slide boundaries'
    )
    
    parser.add_argument(
        '--save_metadata',
        action='store_true',
        help='Save metadata about extracted tiles to JSON file'
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
        # Initialize extractor
        print(f"Loading WSI: {args.wsi_path}")
        extractor = CoordinateTileExtractor(
            wsi_path=args.wsi_path,
            tile_size=args.tile_size,
            mag_level=args.mag_level
        )
        
        # Print slide information
        slide_info = extractor.get_slide_info()
        print(f"Slide dimensions: {slide_info['dimensions']}")
        print(f"Downsample factor: {slide_info['downsample_factor']}")
        
        # Extract tiles
        if args.check_borders:
            print("Extracting tiles with border checking...")
            tiles_with_check = extractor.extract_tiles_with_border_check(coordinates)
            
            # Filter valid tiles
            valid_tiles = [(coords, tile) for coords, tile, is_valid in tiles_with_check if is_valid]
            invalid_count = len(tiles_with_check) - len(valid_tiles)
            
            print(f"Valid tiles: {len(valid_tiles)}")
            print(f"Invalid tiles (outside boundaries): {invalid_count}")
            
            # Print invalid coordinates
            for (x, y), tile, is_valid in tiles_with_check:
                if not is_valid:
                    print(f"  Invalid: ({x}, {y})")
        else:
            print("Extracting tiles...")
            valid_tiles = extractor.extract_multiple_tiles_at_coordinates(coordinates)
        
        if not valid_tiles:
            print("No valid tiles extracted!")
            return 1
        
        # Save tiles
        print(f"Saving {len(valid_tiles)} tiles to {args.output_dir}")
        saved_paths = extractor.save_tiles(valid_tiles, args.output_dir, prefix=args.prefix)
        
        # Save metadata if requested
        if args.save_metadata:
            metadata = {
                'wsi_path': args.wsi_path,
                'slide_info': slide_info,
                'extraction_params': {
                    'tile_size': args.tile_size,
                    'mag_level': args.mag_level,
                    'check_borders': args.check_borders
                },
                'coordinates': coordinates,
                'extracted_tiles': [
                    {
                        'coordinates': coords,
                        'filename': os.path.basename(path)
                    }
                    for (coords, tile), path in zip(valid_tiles, saved_paths)
                ]
            }
            
            metadata_path = os.path.join(args.output_dir, 'extraction_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
        
        print("Extraction completed successfully!")
        
        # Close extractor
        extractor.close()
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
