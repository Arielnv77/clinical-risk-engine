# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ML system for predicting 30-day hospital readmission risk in diabetic patients. End-to-end pipeline: data preprocessing → feature engineering → XGBoost model → SHAP explanations → FastAPI serving → Docker deployment.

**Status: working end-to-end.** Training pipeline, API, dashboard, tests, CI and Docker are all implemented and verified. `data/processed/` (model + featurized CSV) is gitignored, so a fresh clone must run the training command below once before the API or Docker image will work.

## Stack

Python · XGBoost · SHAP · FastAPI · Streamlit · Docker · GitHub Actions

## Commands

- `pip install -r requirements-dev.txt` — installs runtime deps plus pytest/httpx for testing
- `python -m src.models.train` — cleans `data/raw/diabetic_data.csv`, engineers features, trains the model, and writes `data/processed/{xgb_model.json,xgb_model.pkl,diabetic_data_features.csv,model_metadata.json}`. **Must be run once before the API can start** — those artifacts are gitignored.
- `uvicorn src.api.main:app --reload` — run the API
- `streamlit run src/dashboard/app.py` — run the dashboard (expects `API_URL` env var, defaults to `http://127.0.0.1:8000`)
- `pytest tests/` — run tests. `tests/test_api.py` self-skips if the model hasn't been trained yet; the rest don't need it.
- `docker compose up` — run API + dashboard together

## Architecture

```
src/
  data/       # cleaning.py — raw CSV cleaning ('?' -> NaN, drop weight/payer_code/etc, derive readmitted_30d)
  features/   # engineer_features() — shared by training and inference, single source of truth
  models/     # train.py — reproducible training script (see Commands)
  api/        # FastAPI app — /predict + /explain, schemas.py validates categorical/ID fields against real domain values
  dashboard/  # Streamlit clinical UI, calls the API over HTTP
notebooks/    # Original exploratory work (01-04). src/ now has the reproducible equivalent of 02-03.
data/
  raw/          # diabetic_data.csv (UCI, ~100k encounters, 50 columns) — not committed
  processed/    # model + featurized CSV + metadata — not committed, regenerate with `python -m src.models.train`
docker/         # Dockerfile(s)
tests/          # pytest suite — test_feature_engineering.py, test_schemas.py, test_api.py
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
