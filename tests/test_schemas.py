"""
Tests for PatientData validation. No model artifact required — these run
everywhere, including CI, since they only exercise pydantic validation.
"""

import sys
sys.path.append('.')

import pytest
from pydantic import ValidationError

from src.api.schemas import PatientData

VALID_PATIENT = {
    "race": "Caucasian", "gender": "Female", "age": "[50-60)",
    "admission_type_id": 1, "discharge_disposition_id": 1, "admission_source_id": 7,
    "time_in_hospital": 3, "num_lab_procedures": 45, "num_procedures": 1,
    "num_medications": 15, "number_outpatient": 0, "number_emergency": 0,
    "number_inpatient": 0, "diag_1": "250.01", "diag_2": "428", "diag_3": "401",
    "number_diagnoses": 5, "change": "No", "diabetesMed": "Yes",
    "metformin": "No", "repaglinide": "No", "nateglinide": "No",
    "chlorpropamide": "No", "glimepiride": "No", "acetohexamide": "No",
    "glipizide": "No", "glyburide": "No", "tolbutamide": "No",
    "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
    "miglitol": "No", "troglitazone": "No", "tolazamide": "No",
    "examide": "No", "citoglipton": "No", "insulin": "Steady",
    "glyburide-metformin": "No", "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No",
}


def test_valid_patient_is_accepted():
    patient = PatientData(**VALID_PATIENT)
    assert patient.insulin == "Steady"


def test_out_of_range_discharge_disposition_id_rejected():
    payload = {**VALID_PATIENT, "discharge_disposition_id": 999}
    with pytest.raises(ValidationError):
        PatientData(**payload)


def test_out_of_range_admission_type_id_rejected():
    payload = {**VALID_PATIENT, "admission_type_id": 42}
    with pytest.raises(ValidationError):
        PatientData(**payload)


def test_invalid_medication_value_rejected():
    payload = {**VALID_PATIENT, "insulin": "Maybe"}
    with pytest.raises(ValidationError):
        PatientData(**payload)


def test_invalid_race_rejected():
    payload = {**VALID_PATIENT, "race": "NotARace"}
    with pytest.raises(ValidationError):
        PatientData(**payload)
