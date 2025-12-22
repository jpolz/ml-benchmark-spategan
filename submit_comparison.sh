#!/bin/bash
#SBATCH --job-name=spategan_comp
##SBATCH --partition=ccgp
##SBATCH --partition=grace
#SBATCH --partition=sockdolager
#SBATCH --time=24:00:00
#SBATCH --exclusive
##SBATCH --qos=nvgpu
#SBATCH --qos=sdlgpu
#SBATCH --output=logs/slurm_compare_%j.out
#SBATCH --error=logs/slurm_compare_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Change to project root directory (where this script is located)
cd $SLURM_SUBMIT_DIR

# Configuration
DOMAIN="NZ"
# VAR_TARGET="tasmax" # tasmax or pr
VAR_TARGET="pr" # tasmax or pr
EXPERIMENT="ESD_pseudo_reality"
DATA_PATH="/bg/fast/aihydromet/cordexbench/"
# DEEPESD_MODEL="./training/models/model.pt"
DEEPESD_MODEL="./training/models/DeepESD_pr_NZ.pt"
OUTPUT_DIR="./analysis/results/comparison_$(date +%Y%m%d_%H%M)"

# GAN runs to compare (modify this list as needed)
GAN_RUNS=(
    # "./runs/20251220_1833_w99vqb1e" # Unet SA tasmax
    "./runs/20251219_2155_6key13zd" # Unet NZ pr
)

# Optional: Checkpoint epochs to load (one per run, or leave empty for final models)
# If specified, must have same length as GAN_RUNS
CHECKPOINT_EPOCHS=(
    # 200 # "./runs/20251220_1833_w99vqb1e"
    200
)

# Build the command
CMD=".venv/bin/python -m ml_benchmark_spategan.analysis.compare_models \
    --domain $DOMAIN \
    --var-target $VAR_TARGET \
    --experiment $EXPERIMENT \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR"

# Add DeepESD model if it exists
if [ -f "$DEEPESD_MODEL" ]; then
    CMD="$CMD --deepesd-model $DEEPESD_MODEL"
fi

# Add GAN runs
for run in "${GAN_RUNS[@]}"; do
    if [ -d "$run" ]; then
        CMD="$CMD --gan-runs $run"
    fi
done

# Add checkpoint epochs if specified
if [ ${#CHECKPOINT_EPOCHS[@]} -gt 0 ]; then
    for epoch in "${CHECKPOINT_EPOCHS[@]}"; do
        CMD="$CMD --checkpoint-epochs $epoch"
    done
fi

# Print and execute command
echo "Running comparison with the following settings:"
echo "  Domain: $DOMAIN"
echo "  Variable: $VAR_TARGET"
echo "  Experiment: $EXPERIMENT"
echo "  Output directory: $OUTPUT_DIR"
echo "  GAN runs: ${GAN_RUNS[@]}"
if [ ${#CHECKPOINT_EPOCHS[@]} -gt 0 ]; then
    echo "  Checkpoint epochs: ${CHECKPOINT_EPOCHS[@]}"
else
    echo "  Checkpoint epochs: final models"
fi
echo ""
echo "Command: $CMD"
echo ""

eval $CMD

# Print end time
echo ""
echo "End time: $(date)"
