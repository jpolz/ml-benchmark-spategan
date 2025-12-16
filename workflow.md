## ML Benchmark SpaGAN - Training & Analysis Infrastructure

```
INPUT CONFIG          TRAINING SCRIPT              CORE MODULES                        OUTPUTS
                                                                                                          
┌────────────┐       ┌──────────────────┐       ┌──────────────────────────┐       ┌─────────────────────┐
│config.yml  │──────>│ training_v2.py   │──────>│ ml_benchmark_spategan/   │──────>│ runs/DATE_TIME_ID/  │
│            │       │            ▲     │       │                          │       │                     │
│ • model    │       │ 1. Config  │     │       │ ├─ config/               │       │ ├─ config.yaml      │
│ • training │       │ 2. Logging │     │<─────>│ ├─ dataloader/           │       │ ├─ training.log     │
│ • data     │       │ 3. Data    │     │       │ ├─ model/spagan2d.py     │       │ ├─ checkpoints/     │
│ • logging  │       │ 4. Models  │     │       │ │   ├─ Generator         │       │ │   ├─ final_*.pt   │
└────────────┘       │ 5. Train   │     │       │ │   ├─ Discriminator     │       │ │   └─ checkpoint_* │
                     │ 6. Save    │     │       │ │   └─ train_gan_step()  │       │ ├─ sample_plots/    │
┌────────────┐       └────────────┼─────┘       │ ├─ utils/                │       │ ├─ losses.png       │
│ data/      │────────────────────┘             │ │   └─ denormalize.py    │       │ ├─ diagnostics_*.png│
│ *.nc files │                                  │ ├─ evaluation/           │       │ ├─ y_min.nc         │
└────────────┘                                  │ │   ├─ diagnostics.py    │       │ ├─ y_max.nc         │
                                                │ │   └─ indices.py        │       │ └─ x_mean/std.nc    │
                                                │ └─ visualization/        │       └──────────┬──────────┘
                                                └──────────────────────────┘                  │
                                                                                              │
                                                                                              ▼
                     ┌────────────────────────────────────────────────────────────────────────┴─────────┐
                     │                                                                                  │
                     │                        ANALYSIS MODULE                                           │
                     │                                                                                  │
┌─────────────────┐  │  ┌──────────────────┐       ┌─────────────────┐       ┌─────────────────────┐    │
│submit_          │  │  │ compare_models.py│──────>│ model_loader.py │──────>│ analysis/results/   │    │
│comparison.sh    │──┼─>│                  │       │                 │       │ comparison_YYYYMMDD/│    │
│                 │  │  │ • Load configs   │       │ • ModelWrapper  │       │                     │    │
│ • GAN runs      │  │  │ • Normalize data │  ┌───>│ • DeepESDWrapper│       │ ├─ summary.json     │    │
│ • Checkpoints   │  │  │ • Evaluate       │  │    │ • GANWrapper    │       │ ├─ psd_comparison   │    │
│ • Output dir    │  │  │ • Metrics        │  │    │   - Load UNet   │       │ ├─ prediction_*.png │    │
└─────────────────┘  │  │ • Visualize      │  │    │   - Denormalize │       │ └─ ...              │    │
                     │  └──────────┬───────┘  │    └─────────────────┘       └─────────────────────┘    │
                     │             │          │                                                         │
                     │             └──────────┴──────> data_utils.py                                    │
                     │                                  • load_cordex_data()                            │
                     │                                  • normalize_predictors()                        │
                     │                                  • prepare_torch_data()                          │
                     └──────────────────────────────────────────────────────────────────────────────────┘

```

**Training Workflow:** Load config → Setup logging → Build dataloaders → Initialize GAN models → Training loop → Save checkpoints & plots → Save normalization params

**Analysis Workflow:** Load run configs → Apply model-specific normalization → Load models & checkpoints → Evaluate metrics (RMSE, MAE, Correlation, PSD) → Generate comparison visualizations

## Key Features

### Training Infrastructure
- **Dual architecture support**: SpaGAN and Diffusion UNet (configurable via `model.architecture`)
- **Normalization methods**: standardization, minmax, minus1_to_plus1, m1p1_log_target, mp1p1_input_m1p1log_target
- **Checkpointing**: Saves model states at specified intervals and final models
- **Normalization persistence**: Saves y_min/max, x_min/max/mean/std for proper denormalization

### Analysis Module (`analysis/`)
- **Model loading**: Unified interface for DeepESD and GAN models with architecture detection
- **Checkpoint support**: Compare models at specific training epochs via `--checkpoint-epochs`
- **Model-specific normalization**: Reads normalization method from each run's config.yaml
- **Comprehensive metrics**:
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
  - Bias (mean prediction error)
  - Correlation (spatial correlation)
  - Q95 Bias (95th percentile bias)
  - Std Ratio (standard deviation ratio)
  - PSD (Power Spectral Density)
- **Visualizations**:
  - PSD comparison plots (log-log scale)
  - Prediction vs target maps (first sample + climatology)
  - Difference maps with diverging colormaps
  - Log-scale colormaps for precipitation

### SLURM Integration
- `submit_training.sh`: Batch training submission
- `submit_comparison.sh`: Batch model comparison
- Configurable GPUs, time limits, and resource allocation