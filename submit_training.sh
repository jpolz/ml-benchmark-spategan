#!/bin/bash

# This wrapper script runs on the login node
# It copies the config and then submits the job

# Configuration file (can be changed to use different configs)
CONFIG_FILE="config.yml"

# Create a unique temporary config file on the login node
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_CONFIG="/tmp/config_${TIMESTAMP}_$$.yml"

# Copy config to temporary location (this happens on login node)
cp $CONFIG_FILE $TEMP_CONFIG
echo "Config copied to: $TEMP_CONFIG"

# Submit the job with the temporary config path as an argument
sbatch --export=ALL,TEMP_CONFIG=$TEMP_CONFIG <<'EOFSBATCH'
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

echo "Using configuration: $TEMP_CONFIG"
echo ""

# Run training using virtual environment with the copied config
.venv/bin/python -m ml_benchmark_spategan.training.training --config $TEMP_CONFIG

# Clean up temporary config file
rm -f $TEMP_CONFIG

# Print end time
echo ""
echo "End time: $(date)"
EOFSBATCH
