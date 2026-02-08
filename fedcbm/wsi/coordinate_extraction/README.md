# Coordinate-based Tile Extraction

This folder contains standalone tools for extracting tiles from whole slide images at specific coordinates.

## Files in this folder:

- **`coordinate_tile_extractor.py`** - Main module with `CoordinateTileExtractor` class
- **`extract_tiles_by_coordinates.py`** - Command-line script for batch processing
- **`coordinate_extraction_notebook.ipynb`** - Jupyter notebook with examples
- **`README_coordinate_extraction.md`** - Comprehensive documentation

## Usage:

This is a standalone coordinate extraction system that can be used independently of the main MINOTAUR tiling pipeline. It provides:

- Extract tiles at specific coordinates
- Support for multiple coordinates
- Optional stain normalization
- Border checking
- Command-line interface
- Jupyter notebook support

## Integration:

These tools are designed to be portable and can be used in other projects. They provide a clean interface for coordinate-based tile extraction without requiring the full MINOTAUR infrastructure.

For more details, see `README_coordinate_extraction.md`.

