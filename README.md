# MINOTAUR

**Multimodal learnIng cliNical cOncepts Transparent pAn-cancer sURvival**

MINOTAUR extends the Concept Embedding Model (CEM) to the multiple instance learning (MIL) setting and categorical concepts for pan-cancer survival prediction. The framework learns interpretable clinical concepts from histopathology images and uses them to predict patient survival outcomes.

## Installation

### Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `opencv-python>=4.5.0`
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `Pillow>=8.3.0`
- `openslide-python>=1.2.0`

### Setup

Install the package in development mode:

```bash
pip install -e .
```

## Codebase Structure

```
MINOTAUR/
├── minotaur/              # Core package
│   ├── config/            # Configuration management
│   ├── data/              # Data loaders and datasets
│   ├── models/            # Model architectures (CEM-MIL, attention)
│   ├── training/          # Training utilities, metrics, callbacks
│   ├── utils/             # Utility functions
│   └── wsi/               # Whole slide image processing (tile extraction)
├── scripts/               # Main entry point scripts
│   ├── train.py          # Training script
│   ├── tune.py           # Hyperparameter tuning
│   └── analysis/         # Attention analysis tools
├── intervention/          # Concept intervention analysis
├── explainable/          # Explainability tools (attention, heatmaps)
└── configs/              # YAML configuration files
```

## Quick Start

### Training a Model

Train a MINOTAUR model using a configuration file:

```bash
python scripts/train.py \
    -dp /path/to/train_data.csv \
    -db /path/to/embeddings/database \
    -sp /path/to/save/results \
    -cp configs/default.yaml \
    --task-type binary
```

**Required arguments:**
- `-dp, --data_path`: Path to dataset CSV file
- `-db, --db_path`: Path to embeddings database (LMDB, RocksDB, or disk)
- `-sp, --save_path`: Path to save model outputs and checkpoints
- `-cp, --config_path`: Path to YAML configuration file
- `--task-type`: Task type - `binary`, `multiclass`, `continuous`, or `cox`

**Example:**
```bash
python scripts/train.py \
    -dp data/train_cohort.csv \
    -db data/embeddings \
    -sp results/experiment_1 \
    -cp configs/categorical_concepts.yaml \
    --task-type cox
```

## Main Scripts

### 1. Training (`scripts/train.py`)

Train a MINOTAUR model with specified configuration:

```bash
python scripts/train.py -dp <data_path> -db <db_path> -sp <save_path> -cp <config_path> --task-type <type>
```

The script will:
- Load data and configuration
- Train the model
- Save checkpoint to `save_path/model.ckpt`
- Compute and display task metrics

### 2. Hyperparameter Tuning (`scripts/tune.py`)

Perform hyperparameter search with cross-validation:

```bash
python scripts/tune.py \
    -dp /path/to/train_data.csv \
    -db /path/to/embeddings/database \
    -cp configs/default.yaml \
    -sp /path/to/save/results \
    --task-type binary
```

See the script for additional arguments to customize the search space.

### 3. Concept Intervention (`intervention/run_intervention_pipeline.py`)

Run concept intervention analysis to study concept importance:

```bash
python intervention/run_intervention_pipeline.py \
    --model_path /path/to/model.ckpt \
    --test_data /path/to/test_data.csv \
    --db_path /path/to/embeddings/database \
    --output_dir ./intervention_results \
    --concept_ids Stage Age Cancer RNA_Bio_ter \
    --concept_states 4 3 10 3
```

See [`intervention/README.md`](intervention/README.md) for detailed documentation.

### 4. Attention Analysis (`scripts/analysis/run_attention_analysis.py`)

Analyze attention scores and tissue segmentation:

```bash
python scripts/analysis/run_attention_analysis.py \
    /path/to/attentions \
    /path/to/segmentation/directory
```

See [`scripts/analysis/README_attention_analysis.md`](scripts/analysis/README_attention_analysis.md) for details.

## Configuration

MINOTAUR uses YAML configuration files to specify model architecture, training parameters, and data settings.

### Configuration Files

Pre-configured examples are available in `configs/`:
- `default.yaml` - Default configuration with sensible defaults
- `categorical_concepts.yaml` - Configuration for categorical concepts
- `binary_concepts.yaml` - Configuration for binary concepts
- `cox_regression.yaml` - Configuration for Cox regression/survival analysis

### Configuration Structure

Key sections:
- **`data`**: Database paths, bag size, train/test split
- **`concepts`**: Concept IDs, number of concepts, concept states
- **`model`**: Architecture parameters (hidden dim, embedding dim, attention)
- **`training`**: Optimizer, learning rate, batch size, epochs
- **`output`**: Save paths and logging

See [`configs/README.md`](configs/README.md) for detailed configuration documentation and [`config_template.yaml`](config_template.yaml) for a comprehensive template.

### Using Configuration Files

```python
from minotaur.config import load_config_from_yaml

# Load configuration
config = load_config_from_yaml("configs/default.yaml")

# Override paths at runtime
config.data.db_path = "/path/to/database"
config.output.save_path = "/path/to/results"
```

Configuration files support environment variable expansion using `${VAR}` syntax.

## Data Requirements

### Input Data Format

1. **CSV file**: Contains patient metadata and labels
   - Must include patient IDs matching database keys
   - Must include target variable (e.g., `Survival`)
   - Must include concept columns (e.g., `Stage`, `Age`, `Cancer`)

2. **Embeddings database**: Contains pre-extracted tile embeddings
   - Supported formats: LMDB, RocksDB, or disk-based storage
   - Keys should match patient IDs from CSV
   - See `minotaur/data/io/` for database implementations

### Example Data Structure

```csv
patient_id,Survival,Stage,Age,Cancer,RNA_Bio_ter
TCGA-A1-A0SK,1,2,1,BRCA,2
TCGA-A2-A0T1,0,3,2,LUAD,1
...
```

## Package Modules

- **`minotaur.config`**: Configuration loading and management
- **`minotaur.data`**: Data loaders for MIL datasets
- **`minotaur.models`**: Model architectures (CEM-MIL, attention mechanisms)
- **`minotaur.training`**: Training loop, metrics, and callbacks
- **`minotaur.wsi`**: Whole slide image processing and tile extraction
- **`minotaur.utils`**: Utility functions for concepts and paths

## Additional Resources

- **Configuration Guide**: [`configs/README.md`](configs/README.md)
- **Intervention Pipeline**: [`intervention/README.md`](intervention/README.md)
- **Attention Analysis**: [`scripts/analysis/README_attention_analysis.md`](scripts/analysis/README_attention_analysis.md)
- **WSI Processing**: [`minotaur/wsi/coordinate_extraction/README.md`](minotaur/wsi/coordinate_extraction/README.md)

## Contact

- **Email**: [gregory.e.verghese@kcl.ac.uk](mailto:gregory.e.verghese@kcl.ac.uk)
