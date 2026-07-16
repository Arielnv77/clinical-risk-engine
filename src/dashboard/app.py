"""
Streamlit dashboard for the Clinical Risk Engine.
Provides a clinical interface to the FastAPI /explain endpoint.
"""

import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Clinical Risk Engine",
    page_icon="⌗",
    layout="wide"
)

# ---- Custom styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.main-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 700;
    font-size: 2.1rem;
    color: #16292B;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

.main-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #5B7370;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

.risk-card {
    border-radius: 6px;
    padding: 1.8rem 2rem;
    margin: 1rem 0 1.5rem 0;
    border-left: 6px solid;
    background-color: #FFFFFF;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

.risk-card.high { border-left-color: #B5432E; background-color: #FBF2EF; }
.risk-card.low  { border-left-color: #4F7A5C; background-color: #F2F7F3; }

.risk-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    opacity: 0.75;
}

.risk-card.high .risk-label { color: #B5432E; }
.risk-card.low .risk-label  { color: #4F7A5C; }

.risk-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    line-height: 1.1;
    color: #16292B;
    margin: 0.2rem 0;
}

.risk-caption {
    font-size: 0.9rem;
    color: #4A5A58;
}

.factor-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid #E2E8E6;
}

.factor-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #16292B;
    width: 220px;
    flex-shrink: 0;
}

.factor-bar-bg {
    flex-grow: 1;
    height: 8px;
    background-color: #E2E8E6;
    border-radius: 4px;
    overflow: hidden;
}

.factor-bar-fill {
    height: 100%;
    border-radius: 4px;
}

.factor-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    width: 70px;
    text-align: right;
    flex-shrink: 0;
}

section[data-testid="stSidebar"] {
    background-color: #EDF2F1;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="main-title">Clinical Risk Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">30-day readmission risk · clinical decision support</p>', unsafe_allow_html=True)


# ---- Sidebar form ----
with st.sidebar:
    st.markdown("### Patient Data")

    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.selectbox("Age range", [
        "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
        "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
    ], index=5)

    st.markdown("---")

    discharge_disposition_id = st.number_input("Discharge disposition ID", min_value=1, max_value=30, value=1)
    time_in_hospital = st.number_input("Days in hospital", min_value=1, max_value=14, value=3)
    num_medications = st.number_input("Number of medications", min_value=0, max_value=50, value=15)

    st.markdown("---")

    number_inpatient = st.number_input("Prior inpatient visits", min_value=0, max_value=20, value=0)
    number_emergency = st.number_input("Prior emergency visits", min_value=0, max_value=20, value=0)
    number_outpatient = st.number_input("Prior outpatient visits", min_value=0, max_value=20, value=0)

    st.markdown("---")

    diagnosis_options = {
        "Diabetes": "250.00",
        "Cardiovascular disease": "410",
        "Respiratory disease": "480",
        "Digestive disease": "530",
        "Genitourinary disease": "590",
        "Cancer": "199",
        "Injury": "800",
        "External cause / outpatient": "V58",
        "Other": "999"
    }
    diagnosis_label = st.selectbox("Primary diagnosis category", list(diagnosis_options.keys()))
    diag_1 = diagnosis_options[diagnosis_label]

    diabetes_med = st.selectbox("On diabetes medication?", ["Yes", "No"])
    change = st.selectbox("Medication changed during stay?", ["No", "Ch"])
    insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])

    st.markdown("---")

    with st.expander("Additional clinical data"):
        diag_2_label = st.selectbox("Secondary diagnosis category", list(diagnosis_options.keys()), index=1, key="diag_2")
        diag_2 = diagnosis_options[diag_2_label]
        diag_3_label = st.selectbox("Tertiary diagnosis category", list(diagnosis_options.keys()), index=1, key="diag_3")
        diag_3 = diagnosis_options[diag_3_label]
        number_diagnoses = st.number_input("Number of diagnoses on record", min_value=1, max_value=16, value=5)

        admission_type_options = {
            "Emergency": 1, "Urgent": 2, "Elective": 3, "Newborn": 4,
            "Trauma Center": 7
        }
        admission_type_label = st.selectbox("Admission type", list(admission_type_options.keys()))
        admission_type_id = admission_type_options[admission_type_label]

        admission_source_options = {
            "Emergency Room": 7, "Physician Referral": 1, "Clinic Referral": 2,
            "Transfer from a hospital": 4, "Transfer from a Skilled Nursing Facility (SNF)": 5,
            "Transfer from another health care facility": 6,
        }
        admission_source_label = st.selectbox("Admission source", list(admission_source_options.keys()))
        admission_source_id = admission_source_options[admission_source_label]

        num_lab_procedures = st.number_input("Number of lab procedures", min_value=1, max_value=132, value=45)
        num_procedures = st.number_input("Number of non-lab procedures", min_value=0, max_value=6, value=1)

        other_meds = st.multiselect(
            "Other active diabetes medications",
            ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
             "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
             "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
             "examide", "citoglipton", "glyburide-metformin", "glipizide-metformin",
             "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
        )

    st.markdown("---")
    predict_clicked = st.button("Predict Readmission Risk", type="primary", use_container_width=True)


# ---- Main panel ----
if not predict_clicked:
    st.info("Fill in patient data on the left and click **Predict Readmission Risk** to see the assessment.")


if predict_clicked:
    other_med_names = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
    ]
    other_med_status = {name: ("Steady" if name in other_meds else "No") for name in other_med_names}

    payload = {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "change": change,
        "diabetesMed": diabetes_med,
        "insulin": insulin,
        **other_med_status,
    }

    try:
        with st.spinner("Analyzing patient risk..."):
            response = requests.post(f"{API_URL}/explain", json=payload, timeout=10)
    except requests.exceptions.ConnectionError:
        st.error(f"Could not reach the API at {API_URL}. Is it running?")
        response = None
    except requests.exceptions.Timeout:
        st.error("The API took too long to respond. Please try again.")
        response = None

    if response is not None and response.status_code == 200:
        result = response.json()
        prob = result["readmission_probability"]
        risk = result["risk_level"]
        risk_class = "high" if risk == "high" else "low"
        risk_text = "High Risk" if risk == "high" else "Low Risk"

        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div class="risk-label">{risk_text} · 30-day readmission</div>
            <div class="risk-number">{prob*100:.1f}%</div>
            <div class="risk-caption">Probability this patient is readmitted within 30 days of discharge</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Key Contributing Factors")
        st.caption("SHAP values — how much each factor pushes risk up or down for this patient")

        factors = result["top_contributing_factors"]
        max_abs = max(abs(f["shap_value"]) for f in factors)

        for f in factors:
            val = f["shap_value"]
            pct = min(abs(val) / max_abs * 100, 100)
            color = "#B5432E" if val > 0 else "#4F7A5C"
            sign = "+" if val > 0 else ""

            st.markdown(f"""
            <div class="factor-row">
                <div class="factor-name">{f['feature']} = {f['value']}</div>
                <div class="factor-bar-bg">
                    <div class="factor-bar-fill" style="width:{pct}%; background-color:{color};"></div>
                </div>
                <div class="factor-value" style="color:{color};">{sign}{val:.3f}</div>
            </div>
            """, unsafe_allow_html=True)

    elif response is not None:
        st.error(f"API error: {response.status_code} — {response.text}")