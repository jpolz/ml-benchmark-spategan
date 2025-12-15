# Prediction Formatting and Templates

This folder provides utilities to prepare model predictions for CORDEX-ML-Benchmark:

- Use the provided NetCDF templates for each domain/variable to ensure the correct structure and coordinates.
- Set CORDEX ML-Benchmark-compliant global metadata attributes on prediction files prior to evaluation.

## Contents

- `format.py`: Function to set CORDEX ML-Benchmark metadata on an `xarray.Dataset` in a consistent and reproducible way.
- `templates/`: Ready-to-fill templates (NetCDF) that match the benchmark's expected schema.

## Using the templates

Use the files in `./templates/` directly as a starting point. They contain the correct dimensions, coordinates, and a placeholder field for the requested variable. Replace values with your model predictions, preserving the structure.

## Metadata formatting

The function `set_cordex_ml_benchmark_attributes` in `format.py` overwrites global attributes to comply with the CORDEX ML-Benchmark metadata specification.

### Required attributes:

- **project_id**: Should be "CORDEX"
- **activity_id**: Should be "ML-Benchmark"
- **product**: Should be "emulator-output"
- **benchmark_id**: Identifier for the benchmark version

- **institution_id**: Short acronym for the institution responsible for predictions
- **institution**: Full name of institution responsible for the emulated data sets
- **contact**: Contact information (name, email optional) of the institution or researcher that is responsible for the predictions
- **creation_date**: Date of creation (e.g., 2025-03-20)

- **emulator_id**: Short identifier for the emulation method family
- **emulator**: Description of the emulation method (architecture, key features, etc.)
- **training_id**: Short code for the specific training configuration
- **training**: Full description of training setup (predictors, predictands, domains, time periods, pre-processing, loss, etc.)

- **stochastic_output**: Does the emulator generate stochastic/probabilistic realizations? (yes or no)
- **version_realization**: Description of the realization(s) used to obtain the final prediction. If the model is deterministic, this field can be left empty. When there are M realizations and we refer to a specific run vX, we denote it as vX-rM. If we aggregate over M realizations (for example, their mean) instead of the individual realizations, we write vX-aggM
- **version_realization_info**: Optional additional information about version_realization field. For example: the realizations were aggregated using the standard mean

- **reference_url**: Reference information about the model used (e.g., DOI or publication URL)
- **reproducibility_url**: Reproducibility information about the model used (e.g., GitHub repository)