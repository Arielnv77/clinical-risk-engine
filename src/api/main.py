"""
FastAPI service for the Clinical Risk Engine.
Exposes /predict and /explain endpoints for 30-day readmission risk.
"""

import sys
sys.path.append('.')

import pandas as pd
import xgboost as xgb
import shap
from fastapi import FastAPI, HTTPException

from src.api.schemas import PatientData, PredictionResponse
from src.features.engineering import engineer_features

# Clinical threshold — optimized for recall in Fase 4
THRESHOLD = 0.4

app = FastAPI(
    title="Clinical Risk Engine",
    description="30-day hospital readmission risk prediction for diabetic patients",
    version="1.0.0"
)

# Load model once at startup, not per-request
model = xgb.XGBClassifier()
model.load_model('data/processed/xgb_model.json')

# Load feature columns the model expects, in the correct order
_reference_columns = pd.read_csv('data/processed/diabetic_data_features.csv').drop(
    columns=['readmitted_30d']
).columns.tolist()


@app.get("/")
def root():
    return {"status": "Clinical Risk Engine API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predicts 30-day readmission risk for a single patient.
    """
    # Convert incoming patient data to a single-row DataFrame
    patient_dict = patient.model_dump(by_alias=True)
    df_patient = pd.DataFrame([patient_dict])

    # Apply the same feature engineering pipeline used in training
    df_features = engineer_features(df_patient)

    # Ensure exact same columns and order as training data
    df_features = df_features.reindex(columns=_reference_columns, fill_value=0)

    # Predict probability
    probability = float(model.predict_proba(df_features)[:, 1][0])
    risk_level = "high" if probability >= THRESHOLD else "low"

    return PredictionResponse(
        readmission_probability=round(probability, 4),
        risk_level=risk_level,
        threshold_used=THRESHOLD
    )