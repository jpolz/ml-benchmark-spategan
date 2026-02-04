#!/bin/bash
#SBATCH --job-name=spategan_comp
#SBATCH --partition=ccgp
##SBATCH --partition=grace
##SBATCH --partition=sockdolager
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --qos=nvgpu
##SBATCH --qos=sdlgpu
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
DOMAIN="SA"
# VAR_TARGET="tasmax" # tasmax or pr
VAR_TARGET="pr" # tasmax or pr
EXPERIMENT="ESD_pseudo_reality"
DATA_PATH="/bg/fast/aihydromet/cordexbench/"
OUTPUT_DIR="./analysis/results/comparison_$(date +%Y%m%d_%H%M)"

# GAN runs to compare (modify this list as needed)
GAN_RUNS=(
    # "./runs/20251228_1013_vqzthclf" # Unet SA tasmax with orography
    # "./runs/20251223_0202_40z7snll" # Unet SA tasmax no orography
    # "./runs/20260106_2224_9jkutucc" # Unet SA tasmax with orography
    # "./runs/20260106_2224_8z16kjik" # Unet SA tasmax no orography
    # "./runs/20260106_2224_9jkutucc" # Unet SA tasmax with orography
    # "./runs/20260106_2224_8z16kjik" # Unet SA tasmax no orography λ_disc=0
    # "./runs/20260107_2332_dvpmoltj" # Unet SA tasmax with orography λ_disc=0.05
    # "./runs/20260107_2332_brjuhfpr" # Unet SA tasmax no orography λ_disc=0.001
    # "./runs/20260108_2252_hzg78mh0" # Unet SA tasmax no orography λ_disc=0.0001
    "./runs/20260127_0055_ziiiiwtb" # Unet SA pr no orography ?
    "./runs/20260127_0055_yo8vkxo6" # Unet SA pr no orography ?
    "./runs/20260127_0055_vf4l3lwh" # Unet SA pr no orography ?
)

# Optional: Checkpoint epochs to load (one per run, or leave empty for final models)
# If specified, must have same length as GAN_RUNS
CHECKPOINT_EPOCHS=(
    # 150
    # 150
    # 150
    # 150
    80
    80
    80
)

# Build the command
CMD=".venv/bin/python -m ml_benchmark_spategan.analysis.compare_models \
    --domain $DOMAIN \
    --var-target $VAR_TARGET \
    --experiment $EXPERIMENT \
    --data-path $DATA_PATH \
    --output-dir $OUTPUT_DIR"

# Add GAN runs (validate that directories exist first)
VALID_RUNS=()
for run in "${GAN_RUNS[@]}"; do
    if [ -d "$run" ]; then
        VALID_RUNS+=("$run")
    else
        echo "Warning: Run directory not found: $run"
    fi
done

# Add all valid runs as a single --gan-runs argument
if [ ${#VALID_RUNS[@]} -gt 0 ]; then
    CMD="$CMD --gan-runs ${VALID_RUNS[@]}"
fi

# Add checkpoint epochs if specified (all as a single argument)
if [ ${#CHECKPOINT_EPOCHS[@]} -gt 0 ]; then
    CMD="$CMD --checkpoint-epochs ${CHECKPOINT_EPOCHS[@]}"
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
