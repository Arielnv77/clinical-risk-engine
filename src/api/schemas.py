"""
Pydantic schemas defining the expected shape of API requests and responses.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Valid codes per the UCI IDS_mapping.csv — anything outside these sets is not
# a real clinical code and must be rejected rather than fed to the model.
VALID_ADMISSION_TYPE_IDS = frozenset(range(1, 9))
VALID_DISCHARGE_DISPOSITION_IDS = frozenset(range(1, 31))
VALID_ADMISSION_SOURCE_IDS = frozenset(set(range(1, 27)) - {16})

Race = Literal["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?"]
Gender = Literal["Female", "Male", "Unknown/Invalid"]
AgeRange = Literal[
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"
]
MedicationStatus = Literal["No", "Steady", "Up", "Down"]
YesNo = Literal["Yes", "No"]
ChangeFlag = Literal["No", "Ch"]


class PatientData(BaseModel):
    """Raw patient data as expected from the client, before feature engineering."""

    race: Race = Field(..., example="Caucasian")
    gender: Gender = Field(..., example="Female")
    age: AgeRange = Field(..., example="[50-60)")
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
    change: ChangeFlag = Field(..., example="No")
    diabetesMed: YesNo = Field(..., example="Yes")
    metformin: MedicationStatus = Field(..., example="No")
    repaglinide: MedicationStatus = Field(..., example="No")
    nateglinide: MedicationStatus = Field(..., example="No")
    chlorpropamide: MedicationStatus = Field(..., example="No")
    glimepiride: MedicationStatus = Field(..., example="No")
    acetohexamide: MedicationStatus = Field(..., example="No")
    glipizide: MedicationStatus = Field(..., example="No")
    glyburide: MedicationStatus = Field(..., example="No")
    tolbutamide: MedicationStatus = Field(..., example="No")
    pioglitazone: MedicationStatus = Field(..., example="No")
    rosiglitazone: MedicationStatus = Field(..., example="No")
    acarbose: MedicationStatus = Field(..., example="No")
    miglitol: MedicationStatus = Field(..., example="No")
    troglitazone: MedicationStatus = Field(..., example="No")
    tolazamide: MedicationStatus = Field(..., example="No")
    examide: MedicationStatus = Field(..., example="No")
    citoglipton: MedicationStatus = Field(..., example="No")
    insulin: MedicationStatus = Field(..., example="Steady")
    glyburide_metformin: MedicationStatus = Field(..., alias="glyburide-metformin", example="No")
    glipizide_metformin: MedicationStatus = Field(..., alias="glipizide-metformin", example="No")
    glimepiride_pioglitazone: MedicationStatus = Field(..., alias="glimepiride-pioglitazone", example="No")
    metformin_rosiglitazone: MedicationStatus = Field(..., alias="metformin-rosiglitazone", example="No")
    metformin_pioglitazone: MedicationStatus = Field(..., alias="metformin-pioglitazone", example="No")

    @field_validator("admission_type_id")
    @classmethod
    def _validate_admission_type_id(cls, v: int) -> int:
        if v not in VALID_ADMISSION_TYPE_IDS:
            raise ValueError(f"admission_type_id must be one of {sorted(VALID_ADMISSION_TYPE_IDS)}")
        return v

    @field_validator("discharge_disposition_id")
    @classmethod
    def _validate_discharge_disposition_id(cls, v: int) -> int:
        if v not in VALID_DISCHARGE_DISPOSITION_IDS:
            raise ValueError(f"discharge_disposition_id must be one of {sorted(VALID_DISCHARGE_DISPOSITION_IDS)}")
        return v

    @field_validator("admission_source_id")
    @classmethod
    def _validate_admission_source_id(cls, v: int) -> int:
        if v not in VALID_ADMISSION_SOURCE_IDS:
            raise ValueError(f"admission_source_id must be one of {sorted(VALID_ADMISSION_SOURCE_IDS)}")
        return v

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