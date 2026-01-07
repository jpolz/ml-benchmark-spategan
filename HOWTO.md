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
./comparison_scripts/compare_disc_sweep.sh tmp_cfg/disc_sweep_*/run_manifest.txt
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
