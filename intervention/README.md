# Concept Intervention Pipeline

This directory contains scripts for running concept interventions on the CEM-MIL model to analyze concept importance and their impact on survival predictions.

## Overview

The intervention pipeline implements a random concept ordering approach where:
1. For each patient, a random order of concepts is generated
2. Concepts are sequentially "fixed" to their ground truth values
3. Survival predictions are recomputed after each intervention
4. Performance metrics are tracked across all intervention steps

## Files

- `baseline_inference.py` - Runs baseline inference on test set
- `concept_intervention.py` - Implements concept interventions with random ordering
- `evaluation_utils.py` - Evaluation utilities and visualization tools
- `run_intervention_pipeline.py` - Main orchestration script
- `example_usage.py` - Example usage and configuration
- `README.md` - This documentation

## Quick Start

### 1. Run Complete Pipeline

```bash
python run_intervention_pipeline.py \
    --model_path /path/to/your/model.ckpt \
    --test_data /path/to/your/test_data.csv \
    --db_path /path/to/your/database \
    --output_dir ./intervention_results \
    --concept_ids Stage Age Cancer RNA_Bio_ter \
    --concept_states 4 3 10 3 \
    --bag_num 3000 \
    --batch_size 1 \
    --threshold 0.5 \
    --seed 42
```

### 2. Run Individual Components

#### Baseline Inference Only
```bash
python baseline_inference.py \
    --model_path /path/to/your/model.ckpt \
    --test_data /path/to/your/test_data.csv \
    --db_path /path/to/your/database \
    --output_path ./baseline_results.pkl \
    --concept_ids Stage Age Cancer RNA_Bio_ter \
    --concept_states 4 3 10 3
```

#### Concept Interventions Only
```bash
python concept_intervention.py \
    --model_path /path/to/your/model.ckpt \
    --test_data /path/to/your/test_data.csv \
    --db_path /path/to/your/database \
    --output_path ./intervention_results.pkl \
    --concept_ids Stage Age Cancer RNA_Bio_ter \
    --concept_states 4 3 10 3 \
    --seed 42
```

#### Evaluation Only
```bash
python evaluation_utils.py \
    --intervention_results ./intervention_results.pkl \
    --baseline_results ./baseline_results.pkl \
    --output_dir ./analysis \
    --save_plots
```

## Data Format

### Test Dataset CSV
The test dataset should be a CSV file with the following columns:
- `ID`: Patient identifier
- `Stage`: Stage concept (0-3 for 4 classes)
- `Age`: Age concept (0-2 for 3 classes)  
- `Cancer`: Cancer concept (0-9 for 10 classes)
- `RNA_Bio_ter`: RNA Bio concept (0-2 for 3 classes)
- `Survival`: Survival target (0-1 for binary)

### Model Checkpoint
The model checkpoint should be a PyTorch Lightning checkpoint file (`.ckpt`) containing:
- `state_dict`: Model parameters
- `hyper_parameters`: Model configuration

## Concept Configuration

### Concept IDs and States
- `concept_ids`: List of concept names (e.g., `['Stage', 'Age', 'Cancer', 'RNA_Bio_ter']`)
- `concept_states`: List of number of states for each concept (e.g., `[4, 3, 10, 3]`)

### Concept Types
- **Categorical**: Multiple states (e.g., Stage with 4 classes)
- **Binary**: 2 states (e.g., Age with 3 classes, but treated as binary)
- **Continuous**: Single continuous value (not supported in current intervention)

## Output Files

The pipeline generates the following output files:

### Results Files
- `baseline_results.pkl` - Baseline inference results
- `intervention_results.pkl` - Concept intervention results
- `detailed_report.xlsx` - Comprehensive analysis report

### Visualization Files
- `plots/metrics_evolution.png` - Metrics evolution across steps
- `plots/error_flips.png` - Error correction analysis
- `plots/concept_importance.png` - Concept importance analysis
- `plots/confusion_matrices.png` - Confusion matrices for each step

## Analysis Results

### Metrics Tracked
- **Accuracy**: Classification accuracy at each step
- **F1 Score**: F1 score for survival prediction
- **AUC**: Area under ROC curve
- **Error Flips**: FN→TP, FP→TN, TP→FN, TN→FP transitions

### Concept Importance
- **Average Position**: How early each concept is typically fixed
- **First Position Frequency**: How often each concept is fixed first
- **Position Distribution**: Distribution of concept positions across patients

### Error Correction
- **Errors Corrected**: Number of baseline errors fixed at each step
- **Errors Introduced**: Number of new errors introduced at each step
- **Net Improvement**: Net change in error count

## Example Analysis

```python
from evaluation_utils import load_results, InterventionEvaluator

# Load results
intervention_results = load_results('./intervention_results.pkl')
baseline_results = load_results('./baseline_results.pkl')

# Create evaluator
evaluator = InterventionEvaluator(intervention_results, baseline_results)

# Print summary
evaluator.print_summary()

# Generate plots
evaluator.plot_metrics_evolution()
evaluator.plot_error_flips()
evaluator.plot_concept_importance()

# Save detailed report
evaluator.save_detailed_report('./detailed_analysis.xlsx')
```

## Parameters

### Model Parameters
- `model_path`: Path to trained model checkpoint
- `test_data`: Path to test dataset CSV
- `db_path`: Path to database containing tile embeddings
- `output_dir`: Output directory for results

### Concept Parameters
- `concept_ids`: List of concept names
- `concept_states`: List of concept state counts
- `bag_num`: Number of tiles per bag (default: 3000)
- `batch_size`: Batch size for inference (default: 1)
- `threshold`: Decision threshold for binary classification (default: 0.5)

### Control Parameters
- `seed`: Random seed for reproducibility (default: 42)
- `device`: Device to run inference on (default: 'cuda')
- `force_rerun`: Force rerun even if results exist

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size` or `bag_num`
2. **Model Loading Error**: Check model checkpoint path and format
3. **Data Loading Error**: Verify test dataset format and database path
4. **Concept Mismatch**: Ensure `concept_ids` and `concept_states` match your model

### Performance Tips

1. Use GPU if available (`device='cuda'`)
2. Adjust `bag_num` based on available memory
3. Use `batch_size=1` for large bags
4. Set `num_workers=0` in DataLoader to avoid multiprocessing issues

## Dependencies

- PyTorch
- PyTorch Lightning
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- OpenPyXL (for Excel output)
- tqdm (for progress bars)

## Citation

If you use this intervention pipeline in your research, please cite the original CEM-MIL paper and mention the intervention analysis methodology.
