#!/bin/bash
#SBATCH --job-name=spategan_train
#SBATCH --partition=ccgp
##SBATCH --partition=grace
##SBATCH --partition=sockdolager
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --qos=nvgpu
##SBATCH --qos=sdlgpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Change to project root directory (where this script is located)
cd $SLURM_SUBMIT_DIR

# Configuration file (can be changed to use different configs)
CONFIG_FILE="config.yml"

echo "Using configuration: $CONFIG_FILE"
echo ""

# Run training using virtual environment
.venv/bin/python training/training.py --config $CONFIG_FILE

# Print end time
echo ""
echo "End time: $(date)"
