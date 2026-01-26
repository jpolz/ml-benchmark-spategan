# HOWTO Guide

Quick reference for common tasks in ml-benchmark-spategan.

## Parameter Sweeps and Ablation Studies

The new YAML-based sweep framework allows flexible parameter exploration with automatic job monitoring and cleanup.

### 1. Submit a parameter sweep
```bash
# Submit sweep to CCGP partition
./submit_sweep.sh sweeps/loss_weights_sweep.yml ccgp

# Submit sweep to SDL partition
./submit_sweep.sh sweeps/discriminator_sweep.yml sdl

# Test with dry-run (no job submission)
./submit_sweep.py sweeps/test_sweep.yml ccgp --dry-run
```

Available sweep configs in `sweeps/`:
- `loss_weights_sweep.yml` - L1, MSE, GAN loss weight ablations (8 combinations)
- `discriminator_sweep.yml` - Discriminator architecture ablations (24 combinations)
- `test_sweep.yml` - Quick validation sweep (2 combinations)

### 2. Compare results after completion
```bash
# Automatically waits for jobs to finish, then compares and cleans up
./compare_sweep.sh tmp_cfg/sweep_name_*/run_manifest.txt

# Compare with a baseline run
./compare_sweep.sh tmp_cfg/sweep_name_*/run_manifest.txt --baseline runs/baseline_id

# Compare immediately without waiting
./compare_sweep.sh tmp_cfg/sweep_name_*/run_manifest.txt --no-wait

# Keep temporary config files
./compare_sweep.sh tmp_cfg/sweep_name_*/run_manifest.txt --no-cleanup
```

Results saved to `analysis/results/<sweep_name>_comparison_<timestamp>/`

### 3. Create custom sweeps
```bash
# Copy an existing sweep config
cp sweeps/loss_weights_sweep.yml sweeps/my_sweep.yml

# Edit parameters and values in the YAML file
vim sweeps/my_sweep.yml

# Submit your custom sweep
./submit_sweep.sh sweeps/my_sweep.yml ccgp
```

See [SWEEP_FRAMEWORK.md](SWEEP_FRAMEWORK.md) for detailed documentation.

---

## Training

### Run single training job
```bash
sbatch submit_training.sh
```

### Run integration test (short training)
```bash
./submit_integration_test.sh
```

---

## Analysis and Comparison

### Compare specific runs
```bash
# Edit comparison script with your run IDs
./comparison_scripts/compare_runs_SA_tasmax_hist.sh
```

### Submit comparison as batch job
```bash
sbatch submit_comparison.sh <comparison_script.sh>
```

Example:
```bash
sbatch submit_comparison.sh comparison_scripts/compare_runs_SA_tasmax_hist.sh
```

### Model selection (rank runs by composite score)
```bash
# Compare all runs and rank (lower score = better)
python evaluation/model_selection_score.py runs/*/diagnostic_history.json --compare

# Analyze single run with breakdown
python evaluation/model_selection_score.py runs/RUNID --verbose

# Custom weights: generate template, edit, then use
python evaluation/model_selection_score.py --save-weights-template weights.yaml
python evaluation/model_selection_score.py runs/* --compare --weights-config weights.yaml
```

---

## Weights & Biases Upload

Upload completed training runs to W&B for tracking and visualization:

```bash
# Sync all new runs
python sync_to_wandb.py

# Dry run (preview what would be uploaded)
python sync_to_wandb.py --dry-run

# Re-upload existing runs
python sync_to_wandb.py --force

# Sync specific runs directory
python sync_to_wandb.py --runs-dir runs/archive

# Add custom tags
python sync_to_wandb.py --tags experiment_v2,baseline
```

---

## Configuration

All training parameters set in `config.yml`. Key settings:
- Domain: SA, ALPS, or NZ
- Variable: tasmax or pr
- Experiment: ESD_pseudo_reality or Emulator_hist_future
- Loss weights: l1, mse, gan (adversarial), fss
- Learning rates, batch size, epochs, etc.

---

## Directory Structure

- `config.yml` - Main configuration file
- `runs/` - Training outputs (models, logs, diagnostics)
- `logs/` - SLURM job logs
- `analysis/results/` - Comparison outputs
- `tmp_cfg/` - Temporary configs for grid searches
- `comparison_scripts/` - Pre-configured comparison scripts
