## ML Benchmark SpaGAN - Training Infrastructure

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
│ data/      │────────────────────┘             │ ├─ evaluation/           │       │ └─ diagnostics_*.png│
│ *.nc files │                                  │ └─ visualization/        │       └──────────┬──────────┘
└────────────┘                                  └──────────────────────────┘                  │
                                                                                              │
                                                                   ┌────────────┐             │
                                                                   │evaluation/ │◄────────────┘
                                                                   │diagnostics │
                                                                   └────────────┘                                   
```

**Workflow:** Load config → Setup logging → Build dataloaders → Initialize GAN models → Training loop → Save checkpoints & plots → Evaluate diagnostics