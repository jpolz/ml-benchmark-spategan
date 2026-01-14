# HOWTO Guide

Quick reference for common tasks in ml-benchmark-spategan.

## Run and Compare Parameter Sweeps

### 1. Submit sweep jobs
```bash
./submit_disc_sweep.sh        # CCGP partition (n_critic 2,3)
./submit_disc_sweep_sdl.sh    # SDL partition (n_critic 1)
```

### 2. Monitor progress
```bash
squeue -u $USER
```

### 3. Compare all runs after completion
```bash
# Compare all sweep runs
./comparison_scripts/compare_disc_sweep.sh tmp_cfg/disc_sweep_*/run_manifest.txt

# Compare with a baseline run
./comparison_scripts/compare_disc_sweep.sh --baseline runs/YYYYMMDD_HHMM_xxxxxxxx tmp_cfg/disc_sweep_*/run_manifest.txt
```

Results saved to `analysis/results/disc_sweep_comparison_<timestamp>/`

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
