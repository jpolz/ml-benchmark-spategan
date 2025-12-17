#!/bin/bash

# Local integration test script (non-SLURM version)
# This runs both training and comparison in a single script for testing purposes

# Print job information
echo "======================================================================"
echo "INTEGRATION TEST - Training + Comparison Pipeline (Local)"
echo "======================================================================"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Configuration
CONFIG_FILE="config_integration_test.yml"
DOMAIN="NZ"
VAR_TARGET="pr"
EXPERIMENT="ESD_pseudo_reality"
DATA_PATH="/bg/fast/aihydromet/cordexbench/"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "======================================================================"
echo "STEP 1: Training with $CONFIG_FILE"
echo "======================================================================"
echo ""

# Run training and capture both stdout and stderr
TRAIN_LOG="logs/integration_test_training_$(date +%Y%m%d_%H%M%S).log"
.venv/bin/python training/training.py --config $CONFIG_FILE 2>&1 | tee $TRAIN_LOG

# Check if training succeeded
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo ""
    echo "ERROR: Training failed!"
    exit 1
fi

# Extract the run ID from the training log (check for both formats)
RUN_ID=$(grep -E "Run ID:|run_id:" $TRAIN_LOG | head -1 | sed 's/.*Run ID: //' | sed 's/.*run_id: //' | tr -d ' ')
RUN_DIR=$(grep -E "Run directory:|run_dir:" $TRAIN_LOG | head -1 | sed 's/.*Run directory: //' | sed 's/.*run_dir: //' | tr -d ' ')

if [ -z "$RUN_ID" ]; then
    echo ""
    echo "ERROR: Could not extract run ID from training log!"
    exit 1
fi

echo ""
echo "======================================================================"
echo "Training completed successfully!"
echo "Run ID: $RUN_ID"
echo "Run directory: $RUN_DIR"
echo "======================================================================"
echo ""

# Wait a moment for files to sync
sleep 2

echo "======================================================================"
echo "STEP 2: Running comparison (without DeepESD)"
echo "======================================================================"
echo ""

# Set up comparison output directory
OUTPUT_DIR="./analysis/results/integration_test_$(date +%Y%m%d_%H%M)"

# Build the comparison command without DeepESD
CMD=".venv/bin/python analysis/compare_models.py \
    --domain $DOMAIN \
    --var-target $VAR_TARGET \
    --experiment $EXPERIMENT \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR \
    --gan-runs $RUN_DIR"

# Print and execute command
echo "Running comparison with the following settings:"
echo "  Domain: $DOMAIN"
echo "  Variable: $VAR_TARGET"
echo "  Experiment: $EXPERIMENT"
echo "  Output directory: $OUTPUT_DIR"
echo "  GAN run: $RUN_DIR"
echo "  DeepESD: Not included (integration test)"
echo ""
echo "Command: $CMD"
echo ""

COMPARISON_LOG="logs/integration_test_comparison_$(date +%Y%m%d_%H%M%S).log"
eval $CMD 2>&1 | tee $COMPARISON_LOG

# Check if comparison command succeeded
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Comparison command failed!"
    exit 1
fi

# Check if GAN was successfully loaded and evaluated
if grep -q "Error loading GAN" $COMPARISON_LOG; then
    echo ""
    echo "====================================================================="
    echo "ERROR: Integration test FAILED!"
    echo "====================================================================="
    echo "GAN model could not be loaded from: $RUN_DIR"
    echo "Check the comparison log for details: $COMPARISON_LOG"
    echo "====================================================================="
    exit 1
fi

# Check if GAN appears in the summary table
if ! grep -q "$RUN_ID" $COMPARISON_LOG; then
    echo ""
    echo "====================================================================="
    echo "WARNING: GAN model may not have been evaluated properly"
    echo "====================================================================="
    echo "Run ID $RUN_ID not found in comparison summary"
    echo "====================================================================="
fi

echo ""
echo "======================================================================"
echo "Integration test completed successfully!"
echo "======================================================================"
echo "Run ID: $RUN_ID"
echo "Results directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "======================================================================"
