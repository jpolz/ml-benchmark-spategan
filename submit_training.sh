#!/bin/bash
#SBATCH --job-name=spategan_train
#SBATCH --partition=ccgp
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --qos=nvgpu
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo ""

# Change to project root directory
cd /bg/fast/env_polz-j/uvprojects/ml-benchmark-spategan

# Run training
/bg/fast/env_polz-j/uvprojects/ml-benchmark-spategan/.venv/bin/python \
    /bg/fast/env_polz-j/uvprojects/ml-benchmark-spategan/training/training_v2.py

# Print end time
echo ""
echo "End time: $(date)"
