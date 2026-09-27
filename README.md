# Residential HVAC Change-Point Model

This repository prepares ecobee or EnergyPlus time-step data, aggregates it to daily observations, and fits a temperature-based HVAC change-point model. Depending on the observed heating and cooling behavior, the fitted model is classified as 3P Heating, 3P Cooling, 4P, or 5P.

## Repository contents

- `cpm_data.py`: input loading, HVAC runtime construction, and daily aggregation.
- `cpm_model.py`: change-point regression, diagnostics, and plotting.
- `run_cpm.py`: command-line workflow for one building.
- `four_parameter_CPM.ipynb`: documented, interactive example.
- `requirements.txt`: Python dependencies.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS or Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Command-line use

EnergyPlus parquet input:

```bash
python run_cpm.py path/to/building.parquet --source eplus --output-dir outputs/building_001
```

ecobee CSV or parquet input:

```bash
python run_cpm.py path/to/home.csv --source ecobee --output-dir outputs/home_001
```

Each run writes:

- `daily_cpm_data.csv`: daily temperature, temperature-difference, runtime, and mode data.
- `cpm_results.csv`: change points, slopes, RMSE, adjusted R-squared, and model type.
- `cpm_fit.png`: observations and fitted change-point curve.

## Interactive notebook

Start Jupyter from this directory:

```bash
jupyter lab four_parameter_CPM.ipynb
```

Set `INPUT_PATH` and `SOURCE_TYPE` in the configuration cell, then run the notebook from top to bottom. The notebook imports the reusable modules rather than redefining their functions.

## Input expectations

The EnergyPlus workflow expects these time-step variables:

- Date/time
- Living-zone mean air temperature
- Heating and cooling thermostat setpoints
- Outdoor dry-bulb temperature
- Cooling-coil runtime fraction
- A DX or fuel heating-coil runtime fraction
- Supplemental heating-coil runtime fraction when applicable

The exact EnergyPlus column names are centralized in `cpm_data.EPLUS_COLUMNS` so they can be adapted for another output schema.

The ecobee workflow expects calendar fields, `HvacMode`, indoor temperature, heating and cooling setpoints, outdoor temperature, and compressor/auxiliary runtime-stage columns. Temperature values follow the original notebook convention: ecobee thermostat values are stored as tenths of a degree Fahrenheit and are converted to degrees Celsius.

## Modeling conventions

- Mode `0`: HVAC off
- Mode `1`: cooling
- Mode `2`: heating
- Runtime is aggregated as hours per day.
- Days with equal heating and cooling runtime are treated as unavailable for regression.
- The default fit requires an R-squared of at least `0.1`; change it with `--minimum-r-squared`.

## Reproducibility notes

The original notebook mixed function definitions, hidden execution state, and a hard-coded local data path. This version moves reusable logic into modules, removes saved outputs, exposes configuration explicitly, and checks for required columns and failed fits. Raw building data is not included because it can be large or restricted; document the source and access conditions for any dataset used in a public study.

## License

Copyright (c) 2026, The Board of Trustees of the University of Illinois.

This software is distributed under the BSD 3-Clause License. See [LICENSE](LICENSE).

The license applies to the source code only. 

## Citation

If you use this software in research, please cite the archived software release using the metadata in [CITATION.cff](CITATION.cff). GitHub can use this file to generate formatted citations.

**Kim, You-Jeong. (2026). Residential HVAC Change-Point Model (Version 1.0.0) [Computer software].**

When a DOI or related publication becomes available, add it to `CITATION.cff` and cite it alongside the software release.

## Acknowledgments

This software was developed by You-Jeong Kim during doctoral research at the University of Illinois Urbana-Champaign.
