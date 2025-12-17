#!/bin/bash
#SBATCH --job-name=spategan_integration_test
#SBATCH --partition=sockdolager
#SBATCH --time=4:00:00
#SBATCH --qos=sdlgpu
#SBATCH --output=logs/slurm_integration_test_%j.out
#SBATCH --error=logs/slurm_integration_test_%j.err

# Integration test submission script
# This runs both training and comparison in a single job for testing purposes

# Print job information
echo "======================================================================"
echo "INTEGRATION TEST - Training + Comparison Pipeline"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Change to project root directory (where this script is located)
cd $SLURM_SUBMIT_DIR

# Configuration
CONFIG_FILE="config_integration_test.yml"
DOMAIN="NZ"
VAR_TARGET="pr"

# Optional: Use specific checkpoint epoch for comparison (leave empty for final model)
CHECKPOINT_EPOCH=""

# Build command
CMD=".venv/bin/python integration_test.py --config $CONFIG_FILE --domain $DOMAIN --var-target $VAR_TARGET"

# Add checkpoint epoch if specified
if [ -n "$CHECKPOINT_EPOCH" ]; then
    CMD="$CMD --checkpoint-epoch $CHECKPOINT_EPOCH"
fi

# Print configuration
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Domain: $DOMAIN"
echo "  Variable: $VAR_TARGET"
if [ -n "$CHECKPOINT_EPOCH" ]; then
    echo "  Checkpoint epoch: $CHECKPOINT_EPOCH"
else
    echo "  Checkpoint epoch: final model"
fi
echo ""
echo "Command: $CMD"
echo ""
echo "======================================================================"
echo "Starting integration test..."
echo "======================================================================"
echo ""

# Run integration test
eval $CMD

# Check exit status
EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Integration test PASSED"
else
    echo "Integration test FAILED with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "======================================================================"

exit $EXIT_CODE
