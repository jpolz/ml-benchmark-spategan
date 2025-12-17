#!/bin/bash
#
# Cleanup script for integration test artifacts
# 
# This script removes:
# - Integration test run directories
# - Integration test log files
# - Integration test result directories
#
# Usage: bash cleanup_integration_tests.sh

set -e

echo "======================================================================"
echo "Cleaning up integration test artifacts"
echo "======================================================================"

# Remove integration test runs
echo ""
echo "Removing integration test run directories..."
if [ -d "runs/" ]; then
    # Find run directories created by integration tests by checking for config_integration_test.yml
    # in their config.yaml files
    count=0
    for rundir in runs/*/; do
        if [ -f "${rundir}config.yaml" ]; then
            # Check if this run was created from config_integration_test.yml
            if grep -q "config_integration_test.yml" "${rundir}config.yaml" 2>/dev/null; then
                echo "  Removing ${rundir}"
                rm -rf "${rundir}"
                ((count++))
            fi
        fi
    done
    
    if [ "$count" -gt 0 ]; then
        echo "  Removed $count integration test run directories"
    else
        echo "  No integration test run directories found"
    fi
else
    echo "  No runs/ directory found"
fi

# Remove integration test logs
echo ""
echo "Removing integration test log files..."
if [ -d "logs/" ]; then
    # Remove SLURM logs from integration tests
    count=$(find logs/ -type f \( -name "slurm_integration_test_*.err" -o -name "slurm_integration_test_*.out" -o -name "integration_test_*.log" \) | wc -l)
    if [ "$count" -gt 0 ]; then
        find logs/ -type f \( -name "slurm_integration_test_*.err" -o -name "slurm_integration_test_*.out" -o -name "integration_test_*.log" \) -delete
        echo "  Removed $count log files"
    else
        echo "  No integration test log files found"
    fi
else
    echo "  No logs/ directory found"
fi

# Remove integration test results
echo ""
echo "Removing integration test result directories..."
if [ -d "analysis/results/" ]; then
    count=$(find analysis/results/ -maxdepth 1 -type d -name "integration_test_*" | wc -l)
    if [ "$count" -gt 0 ]; then
        find analysis/results/ -maxdepth 1 -type d -name "integration_test_*" -exec rm -rf {} +
        echo "  Removed $count result directories"
    else
        echo "  No integration test result directories found"
    fi
else
    echo "  No analysis/results/ directory found"
fi

echo ""
echo "======================================================================"
echo "Cleanup complete!"
echo "======================================================================"
