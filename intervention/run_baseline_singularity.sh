#!/bin/bash

# Baseline Inference - Singularity Execution Script

# Set up environment variables
export WORKDIR="/SAN/colcc/Hormad1/working"
export PROJECTSDIR="/SAN/colcc/Hormad1/projects"
export ANALYSISDIR="/SAN/colcc/Hormad1/analysis"
export FOUNDATIONMODELSDIR="/SAN/colcc/Hormad1/foundation_models"
export CONTAINERSDIR="/SAN/colcc/Hormad1/containers"

# Model and data paths
MODEL_PATH="/analysis/minotaur/results/uni/cat_model_exv28/1/7_10_1_7_3_True_8_10_0.3_attention_exp1_noweights_checkmodel.ckpt"
TEST_DATA="/data/predict-sets/repeat1_fold7_attentions_V3.csv"
DB_PATH="/data/embeddings"
OUTPUT_PATH="/analysis/minotaur/baseline_results.pkl"

# Concept configuration
CONCEPT_IDS="Stage Age Cancer RNA_Bio_ter"
CONCEPT_STATES="4 3 10 3"

# Pipeline parameters
BAG_NUM=3000
BATCH_SIZE=1
THRESHOLD=0.5

echo "Running Baseline Inference..."
echo "Model: $MODEL_PATH"
echo "Test Data: $TEST_DATA"
echo "Output: $OUTPUT_PATH"
echo ""

# Run baseline inference in Singularity container
singularity exec --nv \
    --bind $WORKDIR:/scripts \
    --bind $PROJECTSDIR:/projects \
    --bind $ANALYSISDIR:/analysis \
    --bind $FOUNDATIONMODELSDIR:/fmodels \
    --bind /SAN/colcc/Hormad1/data:/data \
    $CONTAINERSDIR/cbm-pytorch2.5-cuda12.4-cudnn8-runtime-from-scratch-24102024.sif \
    python /scripts/MINOTAUR/intervention/baseline_inference.py \
        --model_path "$MODEL_PATH" \
        --test_data "$TEST_DATA" \
        --db_path "$DB_PATH" \
        --output_path "$OUTPUT_PATH" \
        --concept_ids $CONCEPT_IDS \
        --concept_states $CONCEPT_STATES \
        --bag_num $BAG_NUM \
        --batch_size $BATCH_SIZE \
        --threshold $THRESHOLD

echo "Baseline inference completed!"
