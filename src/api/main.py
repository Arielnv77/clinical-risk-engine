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

from src.api.schemas import PatientData, PredictionResponse, ExplanationResponse
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

# SHAP explainer — created once at startup
explainer = shap.TreeExplainer(model)

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
    patient_dict = patient.model_dump(by_alias=True)
    df_patient = pd.DataFrame([patient_dict])

    df_features = engineer_features(df_patient)
    df_features = df_features.reindex(columns=_reference_columns, fill_value=0)

    probability = float(model.predict_proba(df_features)[:, 1][0])
    risk_level = "high" if probability >= THRESHOLD else "low"

    return PredictionResponse(
        readmission_probability=round(probability, 4),
        risk_level=risk_level,
        threshold_used=THRESHOLD
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(patient: PatientData):
    """
    Predicts 30-day readmission risk and explains the prediction
    using SHAP values for this specific patient.
    """
    patient_dict = patient.model_dump(by_alias=True)
    df_patient = pd.DataFrame([patient_dict])
    df_features = engineer_features(df_patient)
    df_features = df_features.reindex(columns=_reference_columns, fill_value=0)

    probability = float(model.predict_proba(df_features)[:, 1][0])
    risk_level = "high" if probability >= THRESHOLD else "low"

    shap_values = explainer.shap_values(df_features)

    contributions = []
    for feature, shap_val in zip(df_features.columns, shap_values[0]):
        contributions.append({
            "feature": feature,
            "value": str(df_features[feature].iloc[0]),
            "shap_value": round(float(shap_val), 4)
        })

    top_factors = sorted(
        contributions, key=lambda x: abs(x["shap_value"]), reverse=True
    )[:5]

    return ExplanationResponse(
        readmission_probability=round(probability, 4),
        risk_level=risk_level,
        top_contributing_factors=top_factors
    )