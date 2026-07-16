"""
Raw data cleaning for the diabetic readmission dataset.
Mirrors the decisions made in notebooks/01_eda.ipynb so training is
reproducible from a script instead of only from notebook cells.
"""

import numpy as np
import pandas as pd

# Dropped for high null rate or low clinical value (see notebooks/01_eda.ipynb)
COLS_TO_DROP = ['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code']


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the raw UCI diabetes CSV into a clean training frame:
    - '?' -> NaN
    - drops high-null / low-value columns
    - imputes race (mode) and diag_1/2/3 (Unknown)
    - derives the binary readmitted_30d target
    """
    df = df.copy()
    df = df.replace('?', np.nan)

    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

    df['race'] = df['race'].fillna(df['race'].mode()[0])
    for col in ['diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].fillna('Unknown')

    df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)

    return df
