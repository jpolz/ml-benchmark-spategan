## Dataset Overview

The CORDEX ML-Bench dataset is publicly available at [Zenodo](https://zenodo.org/records/15797226) as a `zip` file containing all the NetCDF files for the different training and evaluation experiments. The notebook `./data_download.ipynb` provides code for downloading this data (for any of the domains included so far). The data is around 5 GB per domain. After downloading, `./experiments.ipynb` provides a walkthrough of the data, helping users understand which data to use for training, which to use for evaluation, and what each dataset represents. We encourage users to carefully review this notebook to become familiar with the benchmark.

The CORDEX ML-Bench dataset spans three geographic domains:

<div align="center">

### New Zealand (NZ) – 0.11° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_NZ.png" alt="NZ Domain" width="300"/>
<br><br>

### Europe (ALPS) – 0.11° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_ALPS.png" alt="ALPS Domain" width="300"/>
<br><br>

### South Africa (SA) – 0.10° resolution  
<img src="https://github.com/WCRP-CORDEX/ml-benchmark/blob/main/images/image_example_SA.png" alt="SA Domain" width="300"/>
<br><br>
<br><br>
</div>

Each region includes structured training and testing data derived from dynamically downscaled Global Climate Models (GCMs), allowing systematic evaluation in both historical and future climates.

The dataset provides two core training experiments:

- **ESD Pseudo-Reality (1961–1980)**: A 20-year historical training period using a single GCM, designed to mimic ESD training. 
- **Emulator Hist+Future (1961–1980 + 2081–2100)**: A more comprehensive 40-year training period combining historical and future climates. This experiment supports evaluation of extrapolative skill, including transferability across GCMs. 

For each training setup, the dataset enables evaluation across multiple test periods and inference conditions:

- **Historical (1981–2000)**: For both perfect and imperfect inference.
- **Mid-century (2041–2060) and End-century (2081–2100)**: To assess extrapolation to future climates, including hard transferability scenarios using unseen GCMs.

## Data Structure

Each domain follows a consistent file structure, with subdirectories for training and testing data, and further divisions by period, GCM, and evaluation type. Predictors include both dynamic variables (e.g., temperature, precipitation) and optional static fields (e.g., topography).

```
Domain/
├── train/
│   ├── ESD_pseudo-reality/
│   │   ├── predictors/
│   │   └── target/
│   ├── Emulator_hist_future/
│   │   ├── predictors/
│   │   └── target/
├── test/
│   ├── historical/
│   ├── mid_century/
│   └── end_century/
```

















