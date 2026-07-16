"""
Pydantic schemas defining the expected shape of API requests and responses.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    model_config = ConfigDict(populate_by_name=True)

    race: Race = Field(..., json_schema_extra={"example": "Caucasian"})
    gender: Gender = Field(..., json_schema_extra={"example": "Female"})
    age: AgeRange = Field(..., json_schema_extra={"example": "[50-60)"})
    admission_type_id: int = Field(..., json_schema_extra={"example": 1})
    discharge_disposition_id: int = Field(..., json_schema_extra={"example": 1})
    admission_source_id: int = Field(..., json_schema_extra={"example": 7})
    time_in_hospital: int = Field(..., json_schema_extra={"example": 3})
    num_lab_procedures: int = Field(..., json_schema_extra={"example": 45})
    num_procedures: int = Field(..., json_schema_extra={"example": 1})
    num_medications: int = Field(..., json_schema_extra={"example": 15})
    number_outpatient: int = Field(..., json_schema_extra={"example": 0})
    number_emergency: int = Field(..., json_schema_extra={"example": 0})
    number_inpatient: int = Field(..., json_schema_extra={"example": 0})
    diag_1: str = Field(..., json_schema_extra={"example": "250.01"})
    diag_2: str = Field(..., json_schema_extra={"example": "428"})
    diag_3: str = Field(..., json_schema_extra={"example": "401"})
    number_diagnoses: int = Field(..., json_schema_extra={"example": 5})
    change: ChangeFlag = Field(..., json_schema_extra={"example": "No"})
    diabetesMed: YesNo = Field(..., json_schema_extra={"example": "Yes"})
    metformin: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    repaglinide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    nateglinide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    chlorpropamide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    glimepiride: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    acetohexamide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    glipizide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    glyburide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    tolbutamide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    pioglitazone: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    rosiglitazone: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    acarbose: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    miglitol: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    troglitazone: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    tolazamide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    examide: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    citoglipton: MedicationStatus = Field(..., json_schema_extra={"example": "No"})
    insulin: MedicationStatus = Field(..., json_schema_extra={"example": "Steady"})
    glyburide_metformin: MedicationStatus = Field(
        ..., alias="glyburide-metformin", json_schema_extra={"example": "No"}
    )
    glipizide_metformin: MedicationStatus = Field(
        ..., alias="glipizide-metformin", json_schema_extra={"example": "No"}
    )
    glimepiride_pioglitazone: MedicationStatus = Field(
        ..., alias="glimepiride-pioglitazone", json_schema_extra={"example": "No"}
    )
    metformin_rosiglitazone: MedicationStatus = Field(
        ..., alias="metformin-rosiglitazone", json_schema_extra={"example": "No"}
    )
    metformin_pioglitazone: MedicationStatus = Field(
        ..., alias="metformin-pioglitazone", json_schema_extra={"example": "No"}
    )

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


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""

    readmission_probability: float = Field(..., json_schema_extra={"example": 0.42})
    risk_level: str = Field(..., json_schema_extra={"example": "high"})
    threshold_used: float = Field(..., json_schema_extra={"example": 0.4})


class FeatureContribution(BaseModel):
    """A single feature's contribution to a prediction."""

    feature: str
    value: str
    shap_value: float


class ExplanationResponse(BaseModel):
    """Response returned by the /explain endpoint."""

    readmission_probability: float = Field(..., json_schema_extra={"example": 0.42})
    risk_level: str = Field(..., json_schema_extra={"example": "high"})
    top_contributing_factors: list[FeatureContribution]
