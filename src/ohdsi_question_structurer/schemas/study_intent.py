from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class AnalyticsType(str, Enum):
    CHARACTERIZATION = "characterization"
    EFFECT_ESTIMATION = "effect_estimation"
    PATIENT_LEVEL_PREDICTION = "patient_level_prediction"

class TemplateName(str, Enum):
    PATIENT_CHARACTERIZATION = "patient_characterization"
    TREATMENT_PATTERNS = "treatment_patterns"
    OUTCOME_INCIDENCE = "outcome_incidence"
    SELF_CONTROLLED_CASE_SERIES = "self_controlled_case_series"
    COHORT_METHOD = "cohort_method"
    PATIENT_LEVEL_PREDICTION_TEMPLATE = "patient_level_prediction"

class Parameters(BaseModel):
    target_cohort: Optional[str] = Field(None, description="Short, descriptive string identifying the target cohort.")
    treatment_cohorts: Optional[List[str]] = Field(None, description="List of strings identifying treatment cohorts.")
    outcome_cohort: Optional[str] = Field(None, description="Short, descriptive string identifying the outcome cohort.")
    time_at_risk: Optional[str] = Field(None, description="Description of the time at risk (e.g., 'On treatment').")
    comparator_cohort: Optional[str] = Field(None, description="Short, descriptive string identifying the comparator cohort.")
    nesting_cohort: Optional[str] = Field(None, description="Short, descriptive string identifying the nesting cohort.")

class Analysis(BaseModel):
    analytics_type: AnalyticsType
    template_name: TemplateName
    parameters: Parameters

class StudyIntent(BaseModel):
    analyses: List[Analysis]