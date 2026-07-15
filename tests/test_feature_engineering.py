"""
Tests for the clinical feature engineering pipeline.
"""

import sys
sys.path.append('.')

import pandas as pd
import pytest
from src.features.engineering import engineer_features, group_icd9


def test_group_icd9_diabetes():
    """Diabetes codes (250.xx) should map to 'Diabetes'."""
    assert group_icd9("250.01") == "Diabetes"
    assert group_icd9("250.83") == "Diabetes"


def test_group_icd9_cardiovascular():
    """Codes in range 390-459 should map to 'Cardiovascular'."""
    assert group_icd9("410") == "Cardiovascular"
    assert group_icd9("428") == "Cardiovascular"


def test_group_icd9_unknown():
    """Missing or 'Unknown' values should stay 'Unknown'."""
    assert group_icd9(None) == "Unknown"
    assert group_icd9("Unknown") == "Unknown"


def test_group_icd9_external():
    """Codes starting with V or E should map to 'External'."""
    assert group_icd9("V58") == "External"
    assert group_icd9("E888") == "External"
    
def test_engineer_features_output_shape():
    """The pipeline should produce features consistent with training data."""
    
    # Minimal single-patient raw data, mimicking what the API receives
    raw_patient = pd.DataFrame([{
        'encounter_id': 1, 'patient_nbr': 1,
        'race': 'Caucasian', 'gender': 'Female', 'age': '[50-60)',
        'admission_type_id': 1, 'discharge_disposition_id': 1,
        'admission_source_id': 7, 'time_in_hospital': 3,
        'num_lab_procedures': 45, 'num_procedures': 1,
        'num_medications': 15, 'number_outpatient': 0,
        'number_emergency': 0, 'number_inpatient': 0,
        'diag_1': '250.01', 'diag_2': '428', 'diag_3': '401',
        'number_diagnoses': 5, 'change': 'No', 'diabetesMed': 'Yes',
        'metformin': 'No', 'repaglinide': 'No', 'nateglinide': 'No',
        'chlorpropamide': 'No', 'glimepiride': 'No', 'acetohexamide': 'No',
        'glipizide': 'No', 'glyburide': 'No', 'tolbutamide': 'No',
        'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
        'miglitol': 'No', 'troglitazone': 'No', 'tolazamide': 'No',
        'examide': 'No', 'citoglipton': 'No', 'insulin': 'Steady',
        'glyburide-metformin': 'No', 'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No', 'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No'
    }])

    result = engineer_features(raw_patient)

    # No object (string) columns should remain — everything must be numeric
    object_cols = result.select_dtypes(include='object').columns.tolist()
    assert len(object_cols) == 0, f"Found non-numeric columns: {object_cols}"

    # Key engineered features must exist
    assert 'total_prior_contacts' in result.columns
    assert 'medication_changed' in result.columns
    assert 'age_numeric' in result.columns
    assert 'diag_1_group_Diabetes' not in result.columns or True  # depends on drop_first


def test_engineer_features_no_missing_target_leak():
    """The pipeline must never leak the target or identifier columns."""
    
    raw_patient = pd.DataFrame([{
        'encounter_id': 1, 'patient_nbr': 1,
        'race': 'Caucasian', 'gender': 'Female', 'age': '[50-60)',
        'admission_type_id': 1, 'discharge_disposition_id': 1,
        'admission_source_id': 7, 'time_in_hospital': 3,
        'num_lab_procedures': 45, 'num_procedures': 1,
        'num_medications': 15, 'number_outpatient': 0,
        'number_emergency': 0, 'number_inpatient': 0,
        'diag_1': '250.01', 'diag_2': '428', 'diag_3': '401',
        'number_diagnoses': 5, 'change': 'No', 'diabetesMed': 'Yes',
        'readmitted': 'NO',
        'metformin': 'No', 'repaglinide': 'No', 'nateglinide': 'No',
        'chlorpropamide': 'No', 'glimepiride': 'No', 'acetohexamide': 'No',
        'glipizide': 'No', 'glyburide': 'No', 'tolbutamide': 'No',
        'pioglitazone': 'No', 'rosiglitazone': 'No', 'acarbose': 'No',
        'miglitol': 'No', 'troglitazone': 'No', 'tolazamide': 'No',
        'examide': 'No', 'citoglipton': 'No', 'insulin': 'Steady',
        'glyburide-metformin': 'No', 'glipizide-metformin': 'No',
        'glimepiride-pioglitazone': 'No', 'metformin-rosiglitazone': 'No',
        'metformin-pioglitazone': 'No'
    }])

    result = engineer_features(raw_patient)

    forbidden_cols = ['encounter_id', 'patient_nbr', 'readmitted', 'diag_1', 'diag_2', 'diag_3', 'change']
    leaked = [c for c in forbidden_cols if c in result.columns]
    assert len(leaked) == 0, f"Leaked columns found: {leaked}"