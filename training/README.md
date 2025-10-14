## Evaluation Experiments

The following tables include a more detailed overview of the different evaluation experiments for the two available training experiments.

*Evaluation for the ESD pseudo-reality training experiment*

| Training Setup | Inference Set | Evaluation Type | Notes | Eval | Required |
|----------------|---------------|----------------|-------|------|----------|
| ESD “pseudo-reality”<br>Period: 1961–1980<br>Static fields: Yes/No | historical (1981–2000) | PP cross-validation | Same GCM used in training, perfectly | Error, Clim | X |
|  | historical 1981–2000 | Imperfect cross-validation | Same GCM, but imperfectly | Error, Clim |  |
|  | 2041–2060 + 2081–2100 | Extrapolation | Same GCM, but perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Extrapolation | Same GCM but imperfectly | change signal |  |


*Evaluation for the Emulator training experiment*

| Training Setup | Inference Set | Evaluation Type | Notes | Eval | Required |
|----------------|---------------|----------------|-------|------|----------|
| Emulator hist + future<br>Period: 1961–1980 + 2081–2100<br>Static fields: Yes/No | historical (1981–2000) | PP cross-validation | Same GCM used in training, perfectly | Error, Clim | X |
|  | historical 1981–2000 | Imperfect cross-validation | Same GCM, but imperfectly | Error, Clim | X |
|  | 2041–2060 + 2081–2100 | Extrapolation | Same GCM, but perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Extrapolation / Hard Transferibility | Different GCM, but perfectly | change signal | X |
|  | 2041–2060 + 2081–2100 | Extrapolation / Hard Transferibility | Different GCM, but imperfectly | change signal | X |

## Baseline Models

CORDEX-ML-Bench includes a set of ML-based baseline models built on state-of-the-art developments. This allows users to compare the performance of their models against well-established baselines. The following table provides information about these models, along with links to repositories containing their implementations.

| Model       | Description | Reference | Implementation |
|-------------|-------------|-----------|----------------|
| DeepESD     | Convolutional neural network  | [Baño-Medina et al., 2024](https://gmd.copernicus.org/articles/15/6747/2022/) | [GitHub repository]() |

## Scoreboard

To track the performance of the different models compared for RCM emulation, we maintain a scoreboard with basic evaluation results for all models. The main requirement for inclusion in this table is that your model is associated with a scientific publication and has a publicly available code repository implementing it. For more information on adding your model, please contact contact.email@email.co.

| Model               | RMSE (°C) | MAE (°C) | R²    | Training Time (hrs) | Inference Speed (samples/sec) |
|--------------------|------------|-----------|-------|-------------------|-------------------------------|
| DeepESD            | 1.12       | 0.85      | 0.91  | 5.2               | 120                           |
| LSTM               | 1.05       | 0.80      | 0.93  | 12.4              | 85                            |
| Transformer        | 0.98       | 0.75      | 0.95  | 18.7              | 65                            |
| U-Net              | 1.00       | 0.78      | 0.94  | 10.5              | 90                            |
| GraphConvNet       | 1.15       | 0.88      | 0.89  | 14.0              | 70                            |


















