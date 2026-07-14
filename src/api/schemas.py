"""
Pydantic schemas defining the expected shape of API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PatientData(BaseModel):
    """Raw patient data as expected from the client, before feature engineering."""
    
    race: str = Field(..., example="Caucasian")
    gender: str = Field(..., example="Female")
    age: str = Field(..., example="[50-60)")
    admission_type_id: int = Field(..., example=1)
    discharge_disposition_id: int = Field(..., example=1)
    admission_source_id: int = Field(..., example=7)
    time_in_hospital: int = Field(..., example=3)
    num_lab_procedures: int = Field(..., example=45)
    num_procedures: int = Field(..., example=1)
    num_medications: int = Field(..., example=15)
    number_outpatient: int = Field(..., example=0)
    number_emergency: int = Field(..., example=0)
    number_inpatient: int = Field(..., example=0)
    diag_1: str = Field(..., example="250.01")
    diag_2: str = Field(..., example="428")
    diag_3: str = Field(..., example="401")
    number_diagnoses: int = Field(..., example=5)
    change: str = Field(..., example="No")
    diabetesMed: str = Field(..., example="Yes")
    metformin: str = Field(..., example="No")
    repaglinide: str = Field(..., example="No")
    nateglinide: str = Field(..., example="No")
    chlorpropamide: str = Field(..., example="No")
    glimepiride: str = Field(..., example="No")
    acetohexamide: str = Field(..., example="No")
    glipizide: str = Field(..., example="No")
    glyburide: str = Field(..., example="No")
    tolbutamide: str = Field(..., example="No")
    pioglitazone: str = Field(..., example="No")
    rosiglitazone: str = Field(..., example="No")
    acarbose: str = Field(..., example="No")
    miglitol: str = Field(..., example="No")
    troglitazone: str = Field(..., example="No")
    tolazamide: str = Field(..., example="No")
    examide: str = Field(..., example="No")
    citoglipton: str = Field(..., example="No")
    insulin: str = Field(..., example="Steady")
    glyburide_metformin: str = Field(..., alias="glyburide-metformin", example="No")
    glipizide_metformin: str = Field(..., alias="glipizide-metformin", example="No")
    glimepiride_pioglitazone: str = Field(..., alias="glimepiride-pioglitazone", example="No")
    metformin_rosiglitazone: str = Field(..., alias="metformin-rosiglitazone", example="No")
    metformin_pioglitazone: str = Field(..., alias="metformin-pioglitazone", example="No")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""
    
    readmission_probability: float = Field(..., example=0.42)
    risk_level: str = Field(..., example="high")
    threshold_used: float = Field(..., example=0.4)


class FeatureContribution(BaseModel):
    """A single feature's contribution to a prediction."""
    
    feature: str
    value: str
    shap_value: float


class ExplanationResponse(BaseModel):
    """Response returned by the /explain endpoint."""
    
    readmission_probability: float = Field(..., example=0.42)
    risk_level: str = Field(..., example="high")
    top_contributing_factors: list[FeatureContribution]