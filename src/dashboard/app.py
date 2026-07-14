"""
Streamlit dashboard for the Clinical Risk Engine.
Provides a clinical interface to the FastAPI /explain endpoint.
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Clinical Risk Engine", layout="centered")

st.title("Clinical Risk Engine")
st.caption("30-day hospital readmission risk — clinical decision support")

st.divider()

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.selectbox("Age range", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ])
    time_in_hospital = st.number_input("Days in hospital", min_value=1, max_value=14, value=3)
    num_medications = st.number_input("Number of medications", min_value=0, max_value=50, value=15)

with col2:
    discharge_disposition_id = st.number_input("Discharge disposition ID", min_value=1, max_value=30, value=1)
    number_inpatient = st.number_input("Prior inpatient visits", min_value=0, max_value=20, value=0)
    number_emergency = st.number_input("Prior emergency visits", min_value=0, max_value=20, value=0)
    number_outpatient = st.number_input("Prior outpatient visits", min_value=0, max_value=20, value=0)
    diag_1 = st.text_input("Primary diagnosis (ICD-9)", value="250.01")
    

st.divider()

diabetes_med = st.selectbox("On diabetes medication?", ["Yes", "No"])
change = st.selectbox("Medication changed during stay?", ["No", "Ch"])
insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])

st.divider()

if st.button("Predict Readmission Risk", type="primary"):
    
    # Build the payload matching the API's expected schema
    payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": 1,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": 7,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": 45,
        "num_procedures": 1,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": "428",
        "diag_3": "401",
        "number_diagnoses": 5,
        "change": change,
        "diabetesMed": diabetes_med,
        "metformin": "No", "repaglinide": "No", "nateglinide": "No",
        "chlorpropamide": "No", "glimepiride": "No", "acetohexamide": "No",
        "glipizide": "No", "glyburide": "No", "tolbutamide": "No",
        "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
        "miglitol": "No", "troglitazone": "No", "tolazamide": "No",
        "examide": "No", "citoglipton": "No", "insulin": insulin,
        "glyburide-metformin": "No", "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No", "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No"
    }

    with st.spinner("Analyzing patient risk..."):
        response = requests.post(f"{API_URL}/explain", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Result")

        prob = result["readmission_probability"]
        risk = result["risk_level"]

        if risk == "high":
            st.error(f"**High Risk** — {prob*100:.1f}% probability of readmission within 30 days")
        else:
            st.success(f"**Low Risk** — {prob*100:.1f}% probability of readmission within 30 days")

        st.subheader("Key Contributing Factors")
        for factor in result["top_contributing_factors"]:
            direction = "increases" if factor["shap_value"] > 0 else "decreases"
            st.write(f"- **{factor['feature']}** = {factor['value']} → {direction} risk ({factor['shap_value']:+.3f})")

    else:
        st.error(f"API error: {response.status_code} — {response.text}")