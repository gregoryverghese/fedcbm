#!/bin/bash

# Batch intervention analysis script
# This script runs concept interventions across multiple models and test folds

# Set default values
MODELS_BASE_DIR="/analysis/minotaur/results/uni/cat_model_exv28"
TEST_DATA_PATH="/data/predict-sets/repeat1_fold7_attentions_V3.csv"
OUTPUT_BASE_DIR="/analysis/minotaur/intervention_results"
CONCEPT_IDS="Stage Age Cancer RNA_Bio_ter"
CONCEPT_STATES="4 3 10 3"
DB_PATH="/data/features"
CONFIG_PATH="/analysis/minotaur/config.yaml"
BAG_NUM=3000
BATCH_SIZE=1
THRESHOLD=0.5
SEED=42
REPEATS="0,1,2,3,4"
LAMBDA="0.1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_base_dir)
            MODELS_BASE_DIR="$2"
            shift 2
            ;;
        --test_data_path)
            TEST_DATA_PATH="$2"
            shift 2
            ;;
        --output_base_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --concept_ids)
            CONCEPT_IDS="$2"
            shift 2
            ;;
        --concept_states)
            CONCEPT_STATES="$2"
            shift 2
            ;;
        --db_path)
            DB_PATH="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --bag_num)
            BAG_NUM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --repeats)
            REPEATS="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models_base_dir DIR    Base directory containing model folders (default: $MODELS_BASE_DIR)"
            echo "  --test_data_path PATH    Path to full test data CSV (default: $TEST_DATA_PATH)"
            echo "  --output_base_dir DIR    Base directory for output results (default: $OUTPUT_BASE_DIR)"
            echo "  --concept_ids STRING     Space-separated concept IDs (default: $CONCEPT_IDS)"
            echo "  --concept_states STRING  Space-separated concept states (default: $CONCEPT_STATES)"
            echo "  --db_path PATH          Path to LMDB database (default: $DB_PATH)"
            echo "  --config_path PATH      Path to config YAML file (default: $CONFIG_PATH)"
            echo "  --bag_num INT           Number of tiles per bag (default: $BAG_NUM)"
            echo "  --batch_size INT        Batch size (default: $BATCH_SIZE)"
            echo "  --threshold FLOAT       Decision threshold (default: $THRESHOLD)"
            echo "  --seed INT              Random seed (default: $SEED)"
            echo "  --repeats STRING        Comma-separated repeat numbers (default: $REPEATS)"
            echo "  --lambda STRING         Lambda value to filter models (default: $LAMBDA)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting batch intervention analysis..."
echo "Models base dir: $MODELS_BASE_DIR"
echo "Test data: $TEST_DATA_PATH"
echo "Output base dir: $OUTPUT_BASE_DIR"
echo "Repeats: $REPEATS"
echo "Lambda: $LAMBDA"
echo "Concept IDs: $CONCEPT_IDS"
echo "Concept states: $CONCEPT_STATES"
echo ""

# Run the batch intervention analysis
python batch_intervention_analysis.py \
    --models_base_dir "$MODELS_BASE_DIR" \
    --test_data_path "$TEST_DATA_PATH" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --concept_ids "$CONCEPT_IDS" \
    --concept_states "$CONCEPT_STATES" \
    --db_path "$DB_PATH" \
    --config_path "$CONFIG_PATH" \
    --bag_num "$BAG_NUM" \
    --batch_size "$BATCH_SIZE" \
    --threshold "$THRESHOLD" \
    --seed "$SEED" \
    --repeats "$REPEATS" \
    --lambda "$LAMBDA"

echo ""
echo "Batch intervention analysis completed!"
echo "Results saved to: $OUTPUT_BASE_DIR"
echo "Summary CSV: $OUTPUT_BASE_DIR/intervention_summary.csv"
