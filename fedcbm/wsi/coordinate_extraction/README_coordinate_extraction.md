# Coordinate-based Tile Extraction

This module provides functionality to extract tiles from whole slide images (WSI) at specific coordinates without requiring the full tiling and tissue detection process.

## Features

- Extract tiles at specific coordinates
- Support for multiple coordinates in a single operation
- Optional stain normalization
- Border checking to ensure tiles fit within slide boundaries
- Easy integration with existing MINOTAUR infrastructure
- Command-line interface for batch processing
- Jupyter notebook support for interactive use

## Files

- `coordinate_tile_extractor.py`: Main module with `CoordinateTileExtractor` class
- `extract_tiles_by_coordinates.py`: Command-line script for batch processing
- `coordinate_extraction_notebook.ipynb`: Jupyter notebook with examples
- `README_coordinate_extraction.md`: This documentation file

## Quick Start

### 1. Using the Python API

```python
from coordinate_tile_extractor import CoordinateTileExtractor

# Initialize extractor
extractor = CoordinateTileExtractor(
    wsi_path="/path/to/your/slide.svs",
    tile_size=512,
    mag_level=0
)

# Extract a single tile
tile = extractor.extract_tile_at_coordinates(x=10000, y=15000)

# Extract multiple tiles
coordinates = [(10000, 15000), (20000, 15000), (10000, 25000)]
tiles = extractor.extract_multiple_tiles_at_coordinates(coordinates)

# Save tiles
saved_paths = extractor.save_tiles(tiles, "./output_dir")

# Close extractor
extractor.close()
```

### 2. Using the Command Line

```bash
# Extract single tile
python extract_tiles_by_coordinates.py \
    --wsi_path /path/to/slide.svs \
    --coordinates "[(10000,15000)]" \
    --output_dir ./tiles

# Extract multiple tiles with border checking
python extract_tiles_by_coordinates.py \
    --wsi_path /path/to/slide.svs \
    --coordinates "[(10000,15000),(20000,15000),(10000,25000)]" \
    --output_dir ./tiles \
    --check_borders \
    --save_metadata
```

### 3. Using Jupyter Notebook

Open `coordinate_extraction_notebook.ipynb` and follow the examples provided.

## API Reference

### CoordinateTileExtractor Class

#### Constructor
```python
CoordinateTileExtractor(
    wsi_path: str,
    tile_size: int = 512,
    mag_level: int = 0,
    stain_normalizer: Optional[StainNormalizer] = None
)
```

#### Methods

##### `extract_tile_at_coordinates(x, y, tile_size=None, normalize=False)`
Extract a single tile at specific coordinates.

**Parameters:**
- `x`: X coordinate at level 0 (highest resolution)
- `y`: Y coordinate at level 0 (highest resolution)
- `tile_size`: Size of the tile to extract (defaults to constructor value)
- `normalize`: Whether to apply stain normalization

**Returns:** numpy array containing the extracted tile

##### `extract_multiple_tiles_at_coordinates(coordinates, tile_size=None, normalize=False)`
Extract multiple tiles at specific coordinates.

**Parameters:**
- `coordinates`: List of (x, y) coordinate tuples at level 0
- `tile_size`: Size of the tile to extract (defaults to constructor value)
- `normalize`: Whether to apply stain normalization

**Returns:** List of (coordinates, tile) tuples

##### `extract_tiles_with_border_check(coordinates, tile_size=None, normalize=False)`
Extract multiple tiles with border checking to ensure they fit within the slide.

**Parameters:**
- `coordinates`: List of (x, y) coordinate tuples at level 0
- `tile_size`: Size of the tile to extract (defaults to constructor value)
- `normalize`: Whether to apply stain normalization

**Returns:** List of (coordinates, tile, is_valid) tuples

##### `save_tiles(tiles, output_dir, prefix="tile")`
Save extracted tiles to disk.

**Parameters:**
- `tiles`: List of (coordinates, tile) tuples
- `output_dir`: Directory to save tiles
- `prefix`: Prefix for tile filenames

**Returns:** List of saved file paths

##### `get_slide_info()`
Get information about the loaded slide.

**Returns:** Dictionary containing slide information

##### `close()`
Close the slide object and free up resources.

## Coordinate System

- Coordinates are specified at **level 0** (highest resolution)
- The system automatically converts coordinates to the appropriate magnification level
- X and Y coordinates follow the standard image coordinate system (X=width, Y=height)

## Stain Normalization

To use stain normalization, create a `StainNormalizer` object:

```python
from coordinate_tile_extractor import create_stain_normalizer

# Create normalizer
stain_normalizer = create_stain_normalizer(
    target_image_path="/path/to/target/image.jpg",
    method="macenko"  # or "vahadane", "reinhard"
)

# Use with extractor
extractor = CoordinateTileExtractor(
    wsi_path="/path/to/slide.svs",
    stain_normalizer=stain_normalizer
)

# Extract with normalization
tile = extractor.extract_tile_at_coordinates(x, y, normalize=True)
```

## Integration with Existing MINOTAUR Code

The coordinate extraction functionality integrates seamlessly with the existing MINOTAUR tiling system:

1. **WSIParser Integration**: New methods added to `WSIParser` class:
   - `extract_tile_at_coordinates()`
   - `extract_multiple_tiles_at_coordinates()`

2. **Compatible with existing infrastructure**:
   - Uses the same tile extraction logic
   - Supports the same stain normalization
   - Compatible with existing feature extraction pipelines

3. **Minimal changes to existing code**: The new functionality is additive and doesn't modify existing behavior.

## Examples

### Example 1: Basic Tile Extraction

```python
from coordinate_tile_extractor import CoordinateTileExtractor

# Initialize
extractor = CoordinateTileExtractor("/path/to/slide.svs")

# Extract single tile
tile = extractor.extract_tile_at_coordinates(10000, 15000)

# Display
import matplotlib.pyplot as plt
plt.imshow(tile)
plt.show()

extractor.close()
```

### Example 2: Batch Processing

```python
# Define coordinates
coordinates = [
    (10000, 15000),
    (20000, 15000),
    (10000, 25000),
    (20000, 25000)
]

# Extract with border checking
tiles_with_check = extractor.extract_tiles_with_border_check(coordinates)

# Filter valid tiles
valid_tiles = [(coords, tile) for coords, tile, is_valid in tiles_with_check if is_valid]

# Save to disk
saved_paths = extractor.save_tiles(valid_tiles, "./output")
```

### Example 3: Command Line Usage

```bash
# Extract tiles with metadata
python extract_tiles_by_coordinates.py \
    --wsi_path /data/slides/slide1.svs \
    --coordinates "[(10000,15000),(20000,15000)]" \
    --output_dir ./extracted_tiles \
    --tile_size 512 \
    --check_borders \
    --save_metadata
```

## Troubleshooting

### Common Issues

1. **Coordinates outside slide boundaries**: Use `--check_borders` flag or `extract_tiles_with_border_check()` method
2. **Memory issues with large tiles**: Reduce `tile_size` or use higher `mag_level`
3. **Import errors**: Ensure the tiler directory is in your Python path

### Performance Tips

1. Use appropriate magnification levels for your use case
2. Batch coordinate extraction when possible
3. Close extractor objects when done to free memory
4. Use border checking to avoid processing invalid coordinates

## Dependencies

- openslide-python
- numpy
- opencv-python
- matplotlib (for visualization)
- pathlib
- json (for metadata)

## License

This module is part of the MINOTAUR project and follows the same licensing terms.
