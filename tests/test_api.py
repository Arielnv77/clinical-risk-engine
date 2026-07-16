"""
End-to-end tests for the /predict and /explain endpoints.
Requires a trained model artifact (run `python -m src.models.train` first);
skips cleanly if it isn't present, since data/processed/ is gitignored.
"""

import os

import pytest

MODEL_PRESENT = os.path.exists('data/processed/xgb_model.json') and os.path.exists(
    'data/processed/diabetic_data_features.csv'
)

if not MODEL_PRESENT:
    pytest.skip(
        "no trained model artifact found — run `python -m src.models.train` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient

from src.api.main import app
from tests.test_schemas import VALID_PATIENT

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_patient():
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["readmission_probability"] <= 1.0
    assert body["risk_level"] in {"high", "low"}


def test_predict_rejects_invalid_discharge_disposition_id():
    payload = {**VALID_PATIENT, "discharge_disposition_id": 999}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_explain_returns_top_factors():
    response = client.post("/explain", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert len(body["top_contributing_factors"]) == 5
    for factor in body["top_contributing_factors"]:
        assert "feature" in factor and "shap_value" in factor
