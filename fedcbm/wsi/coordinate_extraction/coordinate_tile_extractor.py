"""
Coordinate-based tile extractor for whole slide images.

This module provides functionality to extract tiles from WSI at specific coordinates
without requiring the full tiling process. It's designed to work with the existing
WSIParser infrastructure while providing a simpler interface for coordinate-based extraction.
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from openslide import OpenSlide

# Add the parent directory to the path to import from tiler module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedcbm.wsi.parser import WSIParser
from fedcbm.wsi.utilities import TissueDetect, StainNormalizer


class CoordinateTileExtractor:
    """
    A class for extracting tiles from WSI at specific coordinates.
    
    This class provides a simplified interface for extracting tiles at specific
    coordinates without requiring the full tiling and tissue detection process.
    """
    
    def __init__(
        self,
        wsi_path: str,
        tile_size: int = 512,
        mag_level: int = 0,
        stain_normalizer: Optional[StainNormalizer] = None
    ):
        """
        Initialize the coordinate tile extractor.
        
        Args:
            wsi_path: Path to the whole slide image file
            tile_size: Size of tiles to extract (default: 512)
            mag_level: Magnification level for extraction (default: 0)
            stain_normalizer: Optional stain normalizer for preprocessing
        """
        self.wsi_path = wsi_path
        self.tile_size = tile_size
        self.mag_level = mag_level
        self.stain_normalizer = stain_normalizer
        
        # Open the slide
        try:
            self.slide = OpenSlide(wsi_path)
        except Exception as e:
            raise ValueError(f"Could not open WSI at {wsi_path}: {e}")
        
        # Get slide properties
        self.base_mag = self._get_property('openslide.objective-power', 20)
        self.mpp = self._get_property('openslide.mpp-x', 0.25)
        
        # Calculate downsample factor for the specified magnification level
        self.downsample = int(self.slide.level_downsamples[mag_level])
        
        print(f"Slide loaded: {os.path.basename(wsi_path)}")
        print(f"Dimensions: {self.slide.dimensions}")
        print(f"Levels: {len(self.slide.level_dimensions)}")
        print(f"Downsample factor for level {mag_level}: {self.downsample}")
    
    def _get_property(self, key: str, default_value):
        """Get a property from the slide with a default value."""
        try:
            return self.slide.properties[key]
        except KeyError:
            return default_value
    
    def extract_tile_at_coordinates(
        self,
        x: int,
        y: int,
        tile_size: Optional[int] = None,
        normalize: bool = False
    ) -> np.ndarray:
        """
        Extract a tile at specific coordinates from the WSI.
        
        Args:
            x: X coordinate at level 0 (highest resolution)
            y: Y coordinate at level 0 (highest resolution)
            tile_size: Size of the tile to extract (defaults to self.tile_size)
            normalize: Whether to apply stain normalization
            
        Returns:
            numpy array containing the extracted tile
        """
        if tile_size is None:
            tile_size = self.tile_size
        
        # Convert coordinates to the appropriate level
        x_at_level = int(x / self.downsample)
        y_at_level = int(y / self.downsample)
        
        # Extract tile at the specified coordinates
        tile = self.slide.read_region(
            (y_at_level, x_at_level),
            self.mag_level,
            (tile_size, tile_size)
        )
        tile = np.array(tile.convert('RGB'))
        
        # Apply normalization if requested
        if normalize and self.stain_normalizer is not None:
            tile = self.stain_normalizer.normalize(tile)
            
        return tile
    
    def extract_multiple_tiles_at_coordinates(
        self,
        coordinates: List[Tuple[int, int]],
        tile_size: Optional[int] = None,
        normalize: bool = False
    ) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """
        Extract multiple tiles at specific coordinates from the WSI.
        
        Args:
            coordinates: List of (x, y) coordinate tuples at level 0
            tile_size: Size of the tile to extract (defaults to self.tile_size)
            normalize: Whether to apply stain normalization
            
        Returns:
            List of (coordinates, tile) tuples
        """
        tiles = []
        for x, y in coordinates:
            tile = self.extract_tile_at_coordinates(x, y, tile_size, normalize)
            tiles.append(((x, y), tile))
        return tiles
    
    def extract_tiles_with_border_check(
        self,
        coordinates: List[Tuple[int, int]],
        tile_size: Optional[int] = None,
        normalize: bool = False
    ) -> List[Tuple[Tuple[int, int], np.ndarray, bool]]:
        """
        Extract multiple tiles with border checking to ensure they fit within the slide.
        
        Args:
            coordinates: List of (x, y) coordinate tuples at level 0
            tile_size: Size of the tile to extract (defaults to self.tile_size)
            normalize: Whether to apply stain normalization
            
        Returns:
            List of (coordinates, tile, is_valid) tuples where is_valid indicates
            if the tile fits within the slide boundaries
        """
        if tile_size is None:
            tile_size = self.tile_size
        
        tiles = []
        for x, y in coordinates:
            # Check if tile fits within slide boundaries
            x_at_level = int(x / self.downsample)
            y_at_level = int(y / self.downsample)
            
            slide_width, slide_height = self.slide.level_dimensions[self.mag_level]
            is_valid = (x_at_level + tile_size <= slide_width and 
                       y_at_level + tile_size <= slide_height and
                       x_at_level >= 0 and y_at_level >= 0)
            
            if is_valid:
                tile = self.extract_tile_at_coordinates(x, y, tile_size, normalize)
            else:
                # Create a black tile if coordinates are invalid
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            
            tiles.append(((x, y), tile, is_valid))
        return tiles
    
    def save_tiles(
        self,
        tiles: List[Tuple[Tuple[int, int], np.ndarray]],
        output_dir: str,
        prefix: str = "tile"
    ) -> List[str]:
        """
        Save extracted tiles to disk.
        
        Args:
            tiles: List of (coordinates, tile) tuples
            output_dir: Directory to save tiles
            prefix: Prefix for tile filenames
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for (x, y), tile in tiles:
            filename = f"{prefix}_{x}_{y}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Convert RGB to BGR for OpenCV
            tile_bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, tile_bgr)
            saved_paths.append(filepath)
        
        return saved_paths
    
    def get_slide_info(self) -> dict:
        """
        Get information about the loaded slide.
        
        Returns:
            Dictionary containing slide information
        """
        return {
            'path': self.wsi_path,
            'name': os.path.basename(self.wsi_path),
            'dimensions': self.slide.dimensions,
            'level_dimensions': self.slide.level_dimensions,
            'level_downsamples': self.slide.level_downsamples,
            'base_magnification': self.base_mag,
            'mpp': self.mpp,
            'current_level': self.mag_level,
            'downsample_factor': self.downsample,
            'tile_size': self.tile_size
        }
    
    def close(self):
        """Close the slide object."""
        if hasattr(self, 'slide'):
            self.slide.close()


def create_stain_normalizer(
    target_image_path: str,
    method: str = 'macenko'
) -> StainNormalizer:
    """
    Create a stain normalizer for preprocessing tiles.
    
    Args:
        target_image_path: Path to target image for normalization
        method: Normalization method ('macenko', 'vahadane', 'reinhard')
        
    Returns:
        StainNormalizer object
    """
    return StainNormalizer(target_image_path, method)


# Example usage function for Jupyter notebooks
def extract_tiles_example(
    wsi_path: str,
    coordinates: List[Tuple[int, int]],
    output_dir: str = "./extracted_tiles",
    tile_size: int = 512,
    mag_level: int = 0,
    normalize: bool = False,
    target_image_path: Optional[str] = None
):
    """
    Example function showing how to use the CoordinateTileExtractor.
    
    Args:
        wsi_path: Path to the WSI file
        coordinates: List of (x, y) coordinate tuples
        output_dir: Directory to save extracted tiles
        tile_size: Size of tiles to extract
        mag_level: Magnification level for extraction
        normalize: Whether to apply stain normalization
        target_image_path: Path to target image for stain normalization
        
    Returns:
        CoordinateTileExtractor object and list of saved file paths
    """
    # Create stain normalizer if requested
    stain_normalizer = None
    if normalize and target_image_path:
        stain_normalizer = create_stain_normalizer(target_image_path)
    
    # Initialize extractor
    extractor = CoordinateTileExtractor(
        wsi_path=wsi_path,
        tile_size=tile_size,
        mag_level=mag_level,
        stain_normalizer=stain_normalizer
    )
    
    # Extract tiles
    tiles = extractor.extract_multiple_tiles_at_coordinates(
        coordinates=coordinates,
        tile_size=tile_size,
        normalize=normalize
    )
    
    # Save tiles
    saved_paths = extractor.save_tiles(tiles, output_dir)
    
    print(f"Extracted {len(tiles)} tiles")
    print(f"Saved to: {output_dir}")
    
    return extractor, saved_paths
