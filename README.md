# Clinical Risk Engine

**30-day hospital readmission risk prediction for diabetic patients — a clinical decision support system.**

[![Tests](https://github.com/Arielnv77/clinical-risk-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/Arielnv77/clinical-risk-engine/actions)

---

## Overview

Clinical Risk Engine predicts whether a diabetic patient is likely to be readmitted to the hospital within 30 days of discharge, and explains *why* — at both a population level and a per-patient level — using SHAP.

The system is designed as **clinical decision support**, not autonomous decision-making: it surfaces risk and reasoning for a clinician to act on, consistent with the EU AI Act's framing of high-risk clinical AI systems and MDR expectations around human oversight.

Raw patient data → Feature engineering → XGBoost → Calibrated threshold → SHAP explanation
↓
FastAPI (/predict, /explain) → Streamlit dashboard
↓
Dockerized, tested, CI/CD via GitHub Actions

---

## Why this problem

Hospitals are frequently penalized — financially and in care quality metrics — for early readmissions. A clinician deciding whether a patient is ready for discharge has to synthesize dozens of variables under time pressure. This system surfaces that risk automatically, with an explanation a clinician can act on, not just a score.

---

## Dataset

**[Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)**, UCI ML Repository — 101,766 hospital encounters across 130 US hospitals.

Chosen over MIMIC-III for this project: no credentialing process required, single-file structure, and sufficient clinical richness (diagnoses, medications, utilization history) to demonstrate the full engineering pipeline without the multi-week setup cost of a critical-care relational database. See [dataset rationale](#) in project notes for the full comparison.

---

## Clinical EDA — key findings

Before touching the data, hypotheses were formed based on clinical reasoning, then verified against the data:

| Feature | Hypothesis | Result |
|---|---|---|
| `number_inpatient` (prior admissions) | Strong predictor | **Confirmed — strongest signal** (2.3x higher in readmitted group) |
| `discharge_disposition_id` (discharge destination) | Non-home discharge → higher risk | **Confirmed — strong signal** (home discharge: 83% vs 67% between groups) |
| `num_medications` | Moderate predictor | Confirmed — weak-moderate |
| `time_in_hospital` | Weak standalone signal | Confirmed weak alone, but significant via SHAP interaction effects |

**Data integrity decisions:**
- Removed patients with `discharge_disposition_id` indicating death/hospice (verified against official `IDs_mapping.csv`, not assumed) — a deceased patient's "not readmitted" label would be clinically false.
- Deduplicated to one encounter per patient (first admission) — prevents data leakage from the same patient appearing in both train and test.
- Final dataset: 69,990 unique patients, 9% positive class (readmitted <30 days).

---

## Feature engineering

- **ICD-9 diagnosis grouping**: ~900 raw diagnosis codes collapsed into 9 clinical categories (Diabetes, Cardiovascular, Respiratory, etc.) — resolves data sparsity where individual codes had too few examples for the model to learn from.
- **`total_prior_contacts`**: combined inpatient + emergency + outpatient visit history into a single healthcare-utilization score — one of the strongest predictors in the model.
- **`medication_changed`**, **`num_diabetes_meds`**, **`age_numeric`**: additional clinically-motivated features, with honest reporting of which ones showed weak standalone signal.

Full pipeline is implemented as a single reusable function (`src/features/engineering.py`), shared between training notebooks and the production API — eliminating training-serving skew.

---

## Model

**XGBoost classifier**, with class imbalance handled via `scale_pos_weight` rather than synthetic oversampling (SMOTE), to avoid generating synthetic patients in a clinical context.

**Threshold optimization**: rather than the default 0.5, the decision threshold was set to **0.4** based on the cost asymmetry in this domain — a missed high-risk patient (false negative) has real clinical cost, while a false alarm mainly costs clinician attention. This was a deliberate trade-off, not a default.

| Metric | Value |
|---|---|
| AUC-ROC | 0.68 |
| Recall (readmit class) | 85% |
| Precision (readmit class) | 11% |

**Hyperparameter tuning**: `RandomizedSearchCV` with stratified 5-fold cross-validation was run across 20 parameter combinations. The improvement over the baseline configuration was marginal (<0.001 AUC), indicating the model was near its performance ceiling given the current feature set — additional gains would likely require richer data sources (lab trends, clinical notes) rather than further tuning.

---

## Explainability — SHAP

- **Global**: confirms `discharge_disposition_id` and `number_inpatient` as dominant predictors — consistent with manual EDA, validating the modeling pipeline independently.
- **Local**: every prediction is explained per-patient, surfacing the top contributing factors and their direction (increases/decreases risk) — exposed live via the `/explain` API endpoint and the dashboard.

---

## API

Built with **FastAPI**, serving two endpoints:

- `POST /predict` — returns readmission probability and risk classification
- `POST /explain` — returns probability + top 5 SHAP-based contributing factors for that specific patient

Interactive documentation auto-generated at `/docs`.

---

## Dashboard

A **Streamlit** clinical interface consumes the API over HTTP (not a direct model import — the API is the single source of truth, so any other client could integrate the same way). Displays risk level, probability, and a visual breakdown of contributing factors per patient.

---

## Infrastructure

- **Docker Compose** orchestrates the API and dashboard as separate containers on an internal network.
- **GitHub Actions** runs the full test suite on every push to `main`, executing in a clean environment to catch issues that local development might mask.
- **pytest** covers the feature engineering pipeline, including checks that no target or identifier columns leak into the model input.

---

## Limitations

This model is **not production-ready for real clinical deployment**. AUC 0.68 with 11% precision at the chosen threshold would generate a high false-alarm rate at hospital scale. This is disclosed deliberately: the project prioritizes methodological rigor (threshold justification, leakage prevention, explainability, reproducibility) over inflating performance metrics. Production use would require:
- Richer data (lab result trends, clinical notes, temporal vitals)
- Human-in-the-loop validation before any clinical action
- Formal regulatory validation under MDR / AI Act high-risk system requirements

---

## Running the project

**With Docker (recommended):**
```bash
docker compose up --build
```
- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`

**Locally:**
```bash
conda create -n clinical-risk python=3.11
conda activate clinical-risk
pip install -r requirements.txt

uvicorn src.api.main:app --reload          # terminal 1
streamlit run src/dashboard/app.py         # terminal 2
```

**Tests:**
```bash
pytest tests/ -v
```

---

## Stack

Python · pandas · scikit-learn · XGBoost · SHAP · FastAPI · Pydantic · Streamlit · Docker · GitHub Actions

---

## Author

Ariel Núñez Valencia — AI Engineering & Data Science, UAX Madrid
