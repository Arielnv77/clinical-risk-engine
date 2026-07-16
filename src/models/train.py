"""
Trains the 30-day readmission XGBoost model end-to-end, from raw CSV to
serialized artifact. Reproduces what notebooks/01-03 did by hand, as a
script that CI/Docker/a fresh clone can actually run.

Usage (from repo root):
    python -m src.models.train
"""

import sys
sys.path.append('.')

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.cleaning import clean_raw_data
from src.features.engineering import engineer_features

RAW_PATH = 'data/raw/diabetic_data.csv'
FEATURES_PATH = 'data/processed/diabetic_data_features.csv'
MODEL_PATH = 'data/processed/xgb_model.json'
MODEL_PICKLE_PATH = 'data/processed/xgb_model.pkl'
METADATA_PATH = 'data/processed/model_metadata.json'

# Matches the final configuration chosen in notebooks/03_modeling.ipynb
MODEL_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='auc',
    early_stopping_rounds=20,
)

# Clinical threshold — kept at 0.4 to match the currently deployed API.
# The exploratory notebook (03_modeling.ipynb) also evaluated 0.3 for higher
# recall; that tradeoff was deliberately not adopted here.
THRESHOLD = 0.4


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    raw = pd.read_csv(RAW_PATH)

    print("Cleaning...")
    clean = clean_raw_data(raw)

    print("Engineering features...")
    features = engineer_features(clean)

    X = features.drop(columns=['readmitted_30d'])
    y = features['readmitted_30d']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Train: {X_train.shape} — negatives={neg}, positives={pos}, scale_pos_weight={scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, **MODEL_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\nAUC-ROC: {auc:.4f}")
    print(classification_report(y_test, (y_prob >= THRESHOLD).astype(int), target_names=['No readmit', 'Readmit <30d']))

    print(f"\nSaving feature reference to {FEATURES_PATH} ...")
    features.to_csv(FEATURES_PATH, index=False)

    print(f"Saving model to {MODEL_PATH} ...")
    model.save_model(MODEL_PATH)

    print(f"Saving model pickle to {MODEL_PICKLE_PATH} ...")
    with open(MODEL_PICKLE_PATH, 'wb') as f:
        pickle.dump(model, f)

    metadata = {
        'threshold': THRESHOLD,
        'auc_roc': round(float(auc), 4),
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'model_params': MODEL_PARAMS,
    }
    print(f"Saving model metadata to {METADATA_PATH} ...")
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Done.")


if __name__ == '__main__':
    main()
