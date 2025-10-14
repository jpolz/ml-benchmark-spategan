# Prediction Formatting and Templates

This folder provides utilities to prepare model predictions for CORDEX-ML-Bench:

- Use the provided NetCDF templates for each domain/variable to ensure the correct structure and coordinates.
- Set CORDEX ESD-compliant global metadata attributes on prediction files prior to evaluation.

## Contents

- `format.py`: Function to set CORDEX ESD metadata on an `xarray.Dataset` in a consistent and reproducible way.
- `templates/`: Ready-to-fill templates (NetCDF) that match the benchmark’s expected schema.

## Using the templates

Use the files in `./templates/` directly as a starting point. They contain the correct dimensions, coordinates, and a placeholder field for the requested variable. Replace values with your model predictions, preserving the structure.

## Metadata formatting

The function `set_cordex_esd_attributes` in `format.py` overwrites global attributes to comply with the CORDEX ESD experiment design for statistical downscaling of CMIP6.

Required keys include (see `format.py` for the complete list and details):

- `project_id` ("CORDEX")
- `activity_id` ("ML-Benchmark")
- `source_type` (e.g., "ESD-Combined")
- `esd_method_id`, `esd_method_version`, `esd_method_description`
- `training_methodology`, `training_methodology_id`
- `training_experiment`, `evaluation_experiment`
- `probabilistic_output`, `realization_generation_id`
- `institution_id`, `further_info_url`