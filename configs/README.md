# MINOTAUR Configuration Files

This directory contains YAML configuration files for MINOTAUR experiments.

## Available Configurations

- **default.yaml** - Default configuration with sensible defaults
- **categorical_concepts.yaml** - Configuration for categorical concepts (based on arg_def_cat.py)
- **binary_concepts.yaml** - Configuration for binary concepts (based on arg_def.py)
- **cox_regression.yaml** - Configuration for Cox regression/survival analysis

## Usage

### Loading a Configuration

```python
from minotaur.config import load_config_from_yaml

# Load configuration from YAML file
config = load_config_from_yaml("configs/categorical_concepts.yaml")

# Access configuration values
print(config.data.db_path)
print(config.model.h_dim)
print(config.concepts.n_concepts)
```

### Environment Variables

Configuration files support environment variable expansion using `${VAR}` syntax:

```yaml
data:
  db_path: "${DB_PATH}"  # Will be replaced with value from DB_PATH env var
```

Set environment variables before loading:
```bash
export DB_PATH=/path/to/your/database
python your_script.py
```

Or use a `.env` file (see `.env.example` in project root).

### Backward Compatibility

To migrate from old `Args()` classes:

```python
from cem_mil.arg_def_cat import Args
from minotaur.config import create_config_from_args

# Old way
old_args = Args()

# Convert to new config
config = create_config_from_args(old_args)
```

## Configuration Structure

### Data Configuration (`data`)
- `db_path`: Path to database/embeddings
- `database`: Database type ('lmdb', 'disk', 'rocksdb')
- `target`: Target variable name (e.g., 'Survival')
- `bag_n`: Number of tiles per bag
- `test_size`: Fraction for train/val split
- `k_fold`: Number of folds for cross-validation

### Concept Configuration (`concepts`)
- `cpt_ids`: List of concept names
- `n_concepts`: Number of concepts
- `concept_states`: List of number of states per concept (0=continuous, 1=binary, >1=categorical)
- `cpt_cls_weights`: Optional class weights for concept classification
- `no_concepts`: If True, run without concepts

### Model Configuration (`model`)
- `n_tasks`: Number of tasks
- `h_dim`: Hidden dimension
- `emb_dim`: Embedding dimension
- `attn`: Use attention mechanism
- `n_attn_heads`: Number of attention heads
- `dropout`: Dropout rate (bool or float)
- `c_weight`: Concept loss weight

### Training Configuration (`training`)
- `optimizer`: Optimizer name ('adam', 'sgd')
- `learning_rate`: Learning rate
- `batch_size`: Training batch size
- `max_epochs`: Maximum number of epochs
- `num_workers`: DataLoader workers (null = auto)

### Output Configuration (`output`)
- `save_path`: Path to save results
- `log_path`: Path for logs (auto-set from save_path if null)
- `name`: Experiment name

### Hardware Configuration (`hardware`)
- `accelerator`: 'gpu' or 'cpu'
- `devices`: Number of devices or 'auto'
- `enable_progress_bar`: Show progress bar

## Creating Custom Configurations

1. Copy an existing config file
2. Modify values as needed
3. Use environment variables for paths
4. Load with `load_config_from_yaml()`

## Example

```python
from minotaur.config import load_config_from_yaml

# Load config
config = load_config_from_yaml("configs/categorical_concepts.yaml")

# Override save_path at runtime
config.output.save_path = "/path/to/results"

# Use in your scripts
from minotaur.models import ConceptEmbeddingModel

model = ConceptEmbeddingModel(
    n_concepts=config.concepts.n_concepts,
    concept_states=config.concepts.concept_states,
    n_tasks=config.model.n_tasks,
    h_dim=config.model.h_dim,
    # ... etc
)
```


