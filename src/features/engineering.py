"""
Clinical feature engineering pipeline.
Transforms raw patient data into model-ready features.
Used by both training notebooks and the FastAPI service.
"""

import pandas as pd
import numpy as np


DIABETES_MEDS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide',
    'citoglipton', 'insulin', 'glyburide-metformin',
    'glipizide-metformin', 'glimepiride-pioglitazone',
    'metformin-rosiglitazone', 'metformin-pioglitazone'
]

AGE_MAPPING = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}

MED_MAPPING = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}


def group_icd9(code):
    """Groups ICD-9 codes into clinical categories."""
    if pd.isna(code) or code == 'Unknown':
        return 'Unknown'
    
    code = str(code).strip()
    
    if code.startswith('V') or code.startswith('E'):
        return 'External'
    
    try:
        num = float(code)
    except ValueError:
        return 'Other'
    
    if 250 <= num < 251:
        return 'Diabetes'
    elif 390 <= num < 460:
        return 'Cardiovascular'
    elif 460 <= num < 520:
        return 'Respiratory'
    elif 520 <= num < 580:
        return 'Digestive'
    elif 580 <= num < 630:
        return 'Genitourinary'
    elif 140 <= num < 240:
        return 'Cancer'
    elif 800 <= num < 1000:
        return 'Injury'
    else:
        return 'Other'
    
    
    
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw patient data into model-ready features.
    Must produce identical output whether called on training data
    or a single new patient at inference time.
    """
    df = df.copy()

    # ICD-9 grouping
    df['diag_1_group'] = df['diag_1'].apply(group_icd9)
    df['diag_2_group'] = df['diag_2'].apply(group_icd9)
    df['diag_3_group'] = df['diag_3'].apply(group_icd9)

    # Healthcare utilization score
    df['total_prior_contacts'] = (
        df['number_inpatient'] +
        df['number_emergency'] +
        df['number_outpatient']
    )

    # Medication change flag
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)

    # Active diabetes medications count
    df['num_diabetes_meds'] = (df[DIABETES_MEDS] != 'No').sum(axis=1)

    # Age midpoint
    df['age_numeric'] = df['age'].map(AGE_MAPPING)

    # One-hot encoding of categoricals
    categorical_cols = [
        'race', 'gender', 'age',
        'diag_1_group', 'diag_2_group', 'diag_3_group'
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Medication ordinal encoding
    for col in DIABETES_MEDS:
        df[col] = df[col].map(MED_MAPPING)

    # diabetesMed binary encoding
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # Drop identifiers and replaced originals
    cols_to_drop = [
        'encounter_id', 'patient_nbr', 'readmitted',
        'diag_1', 'diag_2', 'diag_3', 'change'
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Clean column names for XGBoost compatibility
    df.columns = (
        df.columns
        .str.replace('[', '', regex=False)
        .str.replace(']', '', regex=False)
        .str.replace('<', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(' ', '_', regex=False)
    )

    return df
