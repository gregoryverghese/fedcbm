#!/bin/bash

# Concept Intervention Pipeline - Singularity Execution Script
# This script runs the concept intervention pipeline inside a Singularity container

# Set up environment variables (adjust these paths as needed)
export WORKDIR="/SAN/colcc/Hormad1/working"
export PROJECTSDIR="/SAN/colcc/Hormad1/projects"
export ANALYSISDIR="/SAN/colcc/Hormad1/analysis"
export FOUNDATIONMODELSDIR="/SAN/colcc/Hormad1/foundation_models"
export CONTAINERSDIR="/SAN/colcc/Hormad1/containers"

# Model and data paths
MODEL_PATH="/analysis/minotaur/results/uni/cat_model_exv28/1/7_10_1_7_3_True_8_10_0.3_attention_exp1_noweights_checkmodel.ckpt"
TEST_DATA="/data/predict-sets/repeat1_fold7_attentions_V3.csv"
DB_PATH="/data/embeddings"
OUTPUT_DIR="/analysis/minotaur/intervention_results"

# Optional config file (if available)
CONFIG_PATH="/analysis/minotaur/results/uni/cat_model_exv28/1/config.yaml"

# Concept configuration (adjust based on your model)
CONCEPT_IDS="Stage Age Cancer RNA_Bio_ter"
CONCEPT_STATES="4 3 10 3"

# Pipeline parameters
BAG_NUM=3000
BATCH_SIZE=1
THRESHOLD=0.5
SEED=42

echo "Starting Concept Intervention Pipeline..."
echo "Model: $MODEL_PATH"
echo "Test Data: $TEST_DATA"
echo "Output: $OUTPUT_DIR"
echo "Config: $CONFIG_PATH"
echo "Concepts: $CONCEPT_IDS"
echo "Concept States: $CONCEPT_STATES"
echo ""

# Run the intervention pipeline in Singularity container
singularity exec --nv \
    --bind $WORKDIR:/scripts \
    --bind $PROJECTSDIR:/projects \
    --bind $ANALYSISDIR:/analysis \
    --bind $FOUNDATIONMODELSDIR:/fmodels \
    --bind /SAN/colcc/Hormad1/data:/data \
    $CONTAINERSDIR/cbm-pytorch2.5-cuda12.4-cudnn8-runtime-from-scratch-24102024.sif \
    python /scripts/MINOTAUR/intervention/run_intervention_pipeline.py \
        --model_path "$MODEL_PATH" \
        --test_data "$TEST_DATA" \
        --db_path "$DB_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --concept_ids $CONCEPT_IDS \
        --concept_states $CONCEPT_STATES \
        --bag_num $BAG_NUM \
        --batch_size $BATCH_SIZE \
        --threshold $THRESHOLD \
        --seed $SEED \
        --device cuda \
        --config_path "$CONFIG_PATH"

echo ""
echo "Intervention pipeline completed!"
echo "Results saved to: $OUTPUT_DIR"
