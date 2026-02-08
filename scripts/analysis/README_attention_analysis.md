# Attention Analysis Script

This script performs posthoc analysis on attention scores and tissue segmentation masks from a tissue classifier.

## Overview

The script processes three types of files for each patient:
1. **Attention scores** (`.npy` files) - Raw attention scores from the model
2. **Tile IDs** (`.txt` files) - Corresponding tile coordinates in format "x_y"
3. **Segmentation CSV** (`.csv` files) - Tissue classifier predictions with coordinates

## Output

### Individual Analysis Output
The `attention_analysis.py` script generates a CSV file with the following columns:
- `patient_id`: Patient identifier
- `id`: Tile ID (original format)
- `attention`: Normalized attention score (0-1)
- `attention_group`: Categorical grouping (Low/Medium/High based on tertiles)
- `x`: X coordinate extracted from tile ID
- `y`: Y coordinate extracted from tile ID
- `tissue_class`: Tissue class based on majority vote in segmentation mask

### Batch Processing Output
The `run_attention_analysis.py` wrapper script generates a combined CSV with additional metadata columns:
- `cancer_type`: Cancer type (e.g., TCGA_BRCA, TCGA_PAAD)
- `surv_result`: Survival prediction result (e.g., 0_FP, 0_TN, 1_FN, 1_TP)
- `concept`: Concept name (e.g., Age, Gender)
- `ccpt_predict_correct`: Whether the concept was correctly predicted (True/False/Unknown)
- `patient_id`: Patient identifier
- `id`: Tile ID (original format)
- `attention`: Normalized attention score (0-1)
- `attention_group`: Categorical grouping (Low/Medium/High based on tertiles)
- `x`: X coordinate extracted from tile ID
- `y`: Y coordinate extracted from tile ID
- `tissue_class`: Tissue class based on majority vote in segmentation mask

## Usage

### Individual Patient Analysis
```bash
python attention_analysis.py /path/to/attention/directory /path/to/segmentation/directory /path/to/wsi/directory
```

### Specify Output File
```bash
python attention_analysis.py /path/to/attention/directory /path/to/segmentation/directory /path/to/wsi/directory --output results.csv
```

### Batch Processing (Recommended)
Use the wrapper script to process the entire file structure:

```bash
python run_attention_analysis.py /path/to/attentions /path/to/segmentation/directory
```

### Help
```bash
python attention_analysis.py --help
python run_attention_analysis.py --help
```

## Data Directory Structure

### Individual Analysis
The `attention_analysis.py` script expects separate directories:

```
attention_directory/
├── patient1_attention.npy
├── patient1_tile_keys.txt
└── ...

segmentation_directory/
├── patient1_segmentation.csv
├── patient2_segmentation.csv
└── ...
```

### Batch Processing
The `run_attention_analysis.py` wrapper script processes the full structure:

```
attentions/
├── TCGA_BRCA/
│   ├── 0_FP/
│   │   ├── Age/
│   │   │   ├── TCGA-A1-A0SK_True_attention.npy
│   │   │   ├── TCGA-A1-A0SK_True_tile_keys.txt
│   │   │   └── ...
│   │   └── Gender/
│   │       └── ...
│   ├── 0_TN/
│   ├── 1_FN/
│   └── 1_TP/
├── TCGA_PAAD/
│   └── ...
└── ...
```

### File Naming Conventions

- **Attention files**: `*_attention.npy`
- **Tile ID files**: `*_tile_keys.txt` (same prefix as attention file)
- **Segmentation files**: `*_segmentation.csv` (should contain patient ID in filename)

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

## Key Functions

- `df_to_tiled_segmentation()`: Converts CSV coordinates to segmentation mask
- `percentile_clip()`: Normalizes attention scores using percentile clipping
- `get_tile_class()`: Determines tissue class for each tile using majority voting
- `load_attention_data()`: Loads and processes attention scores and tile IDs
- `process_patient_data()`: Main processing function for each patient
- `find_matching_files()`: Automatically finds matching files for each patient

## Configuration

You may need to adjust these parameters in the script based on your specific data:

- **Canvas size**: Default is `(7859, 4692)` - adjust based on your WSI dimensions
- **Tile dimension**: Default is `int(224/4)` - adjust based on your tile size
- **Downsampling factor**: Default is `16.001` - adjust based on your coordinate system

## Example Output

```csv
patient_id,tile_id,attention,attention_group,x,y,tissue_class
TCGA-A1-A0SK,1234_5678,0.85,High,1234,5678,2
TCGA-A1-A0SK,1235_5679,0.45,Medium,1235,5679,1
...
```

## Error Handling

The script includes comprehensive error handling:
- Continues processing if individual patients fail
- Reports warnings for missing files
- Provides detailed error messages for debugging

## Notes

- The script automatically handles file matching based on patient IDs
- Attention scores are normalized using 1st-99th percentile clipping
- Tissue classes are determined by majority voting within each tile region
- Background class is assigned index 0 in the segmentation mask
