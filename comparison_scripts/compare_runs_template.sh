#!/bin/bash
# Template for comparing diagnostic histories across different training runs
# Copy this file and replace dummy run IDs with your actual run IDs

# Define runs to compare with descriptions
RUNS=(
    "runs/YYYYMMDD_HHMM_xxxxxxxx"  # Description: baseline run
    "runs/YYYYMMDD_HHMM_xxxxxxxx"  # Description: experiment 1
    "runs/YYYYMMDD_HHMM_xxxxxxxx"  # Description: experiment 2
    # Add more runs as needed
)

# Run the comparison script
# Output will be saved as PNG file
.venv/bin/python analysis/compare_diagnostics.py "${RUNS[@]}" --output comparison_output.png
