# Clinical Risk Engine

ML system for predicting 30-day hospital readmission risk in diabetic patients.
End-to-end pipeline: data cleaning → feature engineering → XGBoost model → SHAP
explanations → FastAPI serving → Streamlit clinical dashboard → Docker deployment.

## Stack

Python · XGBoost · SHAP · FastAPI · Streamlit · Docker · GitHub Actions

## Data

Source: [UCI Diabetes 130-US hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(~100k encounters, 50 columns). Not committed to this repo — download it yourself
and place it at `data/raw/diabetic_data.csv`.

## Setup

```bash
pip install -r requirements-dev.txt
```

## Train the model

`data/processed/` (trained model + featurized CSV) is gitignored, so this must be
run once before the API or Docker image will work:

```bash
python -m src.models.train
```

This cleans the raw CSV, engineers features, trains the XGBoost classifier, and
writes `data/processed/xgb_model.json`, `xgb_model.pkl`, `diabetic_data_features.csv`,
and `model_metadata.json`.

## Run the API

```bash
uvicorn src.api.main:app --reload
```

- `POST /predict` — readmission probability + risk level for a patient encounter
- `POST /explain` — same, plus the top 5 SHAP-driven contributing factors

## Run the dashboard

```bash
API_URL=http://127.0.0.1:8000 streamlit run src/dashboard/app.py
```

## Run everything with Docker

```bash
docker compose up
```

API on `:8000`, dashboard on `:8501`.

## Tests

```bash
pytest tests/
```

`tests/test_api.py` skips automatically if the model hasn't been trained yet
(see [Train the model](#train-the-model)); the rest of the suite doesn't need it.
