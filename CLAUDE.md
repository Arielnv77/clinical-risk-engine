# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML system for predicting 30-day hospital readmission risk in diabetic patients. End-to-end pipeline: data preprocessing → feature engineering → XGBoost model → SHAP explanations → FastAPI serving → Docker deployment.

**Status: early development.** The scaffold exists; the implementation does not yet.

## Stack

Python · XGBoost · SHAP · FastAPI · Docker · GitHub Actions

## Commands

No build system is configured yet. As dependencies and tooling are added, update this section with:
- `pip install -r requirements.txt` (or equivalent)
- How to run the API (`uvicorn src.api.main:app --reload`)
- How to run tests (`pytest tests/`)
- How to train the model

## Architecture

```
src/
  data/       # Data loading and cleaning (replace '?' nulls with NaN, drop weight/payer_code)
  features/   # Feature engineering pipeline
  models/     # XGBoost training, evaluation, serialization
  api/        # FastAPI app — inference endpoint + SHAP explanation endpoint
notebooks/
  01_eda.ipynb  # Exploratory analysis; establishes which features matter
data/
  raw/          # diabetic_data.csv (UCI, ~100k encounters, 50 columns) — not committed
  processed/    # Cleaned/featurized data — not committed
docker/         # Dockerfile(s)
tests/          # pytest suite
```

### Data

Source: UCI diabetes dataset (`data/raw/diabetic_data.csv`). Target variable: `readmitted` — binary collapse of `<30` vs. `>30`/`NO`.

Key feature findings from EDA (`notebooks/01_eda.ipynb`):
- **Keep:** `number_inpatient`, `number_emergency`, `time_in_hospital`, `num_medications`, `insulin`+`change` (medication adjustment during stay), `discharge_disposition_id`, `age`, `diag_1/2/3`
- **Drop:** `weight` (97% null), `payer_code` (high null, low clinical value), `encounter_id` (identifier only)
- Raw nulls are encoded as `'?'` — replace with `np.nan` before any processing

### Data flow

Raw CSV → `src/data/` (clean, encode, split) → `src/features/` (engineer features) → `src/models/` (train XGBoost, generate SHAP values) → serialized model artifact → `src/api/` (load artifact, serve predictions + explanations)

### API design intent

Two endpoints expected:
- `POST /predict` — returns readmission probability for a patient encounter
- `POST /explain` — returns SHAP values for per-patient clinical explanation

## Data handling

`data/` is gitignored. Never commit CSV files or processed data artifacts.
