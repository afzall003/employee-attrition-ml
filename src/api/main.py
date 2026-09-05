"""
Production API for the Employee Attrition ML project.

This API exposes the validated calibrated Random Forest production model
through a controlled decision-support interface.

IMPORTANT:
    This system provides decision-support signals only.

    It must NOT be used to make automatic employment decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.models.predict import (
    PRODUCTION_THRESHOLD,
    STABLE_FEATURES,
    load_metadata,
    load_model,
    predict_employee,
)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "employee_attrition_production_calibrated.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "employee_attrition_production_calibrated_metadata.json"
)


# ============================================================================
# APPLICATION
# ============================================================================

app = FastAPI(
    title="Employee Attrition Production API",
    description=(
        "Production-grade decision-support API for the validated and "
        "calibrated Employee Attrition Random Forest model.\n\n"
        "## Purpose\n"
        "The API estimates employee attrition probability using the "
        "validated production feature set.\n\n"
        "## Important Governance Restriction\n"
        "This API provides decision-support signals only. It does not make "
        "employment decisions and must not be used for automatic hiring, "
        "promotion, termination, retention, or other employment decisions. "
        "Human review is required.\n\n"
        "## Model\n"
        "Random Forest classifier with probability calibration using "
        "Sigmoid / Platt scaling.\n\n"
        "## Production Threshold\n"
        f"The active decision-support threshold is {PRODUCTION_THRESHOLD:.2f}."
    ),
    version="1.2.0",
    contact={
        "name": "Employee Attrition ML API",
    },
    license_info={
        "name": "Internal / Controlled Use",
    },
    openapi_tags=[
        {
            "name": "System",
            "description": "API health and service information.",
        },
        {
            "name": "Model",
            "description": "Production model contract and predictions.",
        },
        {
            "name": "Governance",
            "description": "Model governance and human-review restrictions.",
        },
    ],
)


# ============================================================================
# INPUT SCHEMA
# ============================================================================


class EmployeeInput(BaseModel):
    """
    Production feature contract for one employee.

    The schema intentionally contains only the ten validated stable
    production features.

    Categorical values such as Job_Role and Overtime are accepted as
    strings without restricting them to the categories currently observed
    in the training dataset. The underlying OneHotEncoder uses
    handle_unknown='ignore', allowing unseen categories to be processed
    safely by the production model.
    """

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "example": {
                "Work_Life_Balance": 2,
                "Job_Satisfaction": 2,
                "Distance_From_Home": 15,
                "Average_Hours_Worked_Per_Week": 48,
                "Years_Since_Last_Promotion": 1,
                "Work_Environment_Satisfaction": 2,
                "Job_Role": "Executive",
                "Age": 31,
                "Overtime": "Yes",
                "Absenteeism": 8,
            }
        },
    )

    Work_Life_Balance: float = Field(
        ...,
        description="Employee work-life balance rating.",
        examples=[2],
    )

    Job_Satisfaction: float = Field(
        ...,
        description="Employee job satisfaction rating.",
        examples=[2],
    )

    Distance_From_Home: float = Field(
        ...,
        description="Distance between the employee's home and workplace.",
        examples=[15],
    )

    Average_Hours_Worked_Per_Week: float = Field(
        ...,
        description="Average number of hours worked per week.",
        examples=[48],
    )

    Years_Since_Last_Promotion: float = Field(
        ...,
        description="Number of years since the employee's last promotion.",
        examples=[1],
    )

    Work_Environment_Satisfaction: float = Field(
        ...,
        description="Employee satisfaction with the work environment.",
        examples=[2],
    )

    Job_Role: str = Field(
        ...,
        description=(
            "Employee job role. Unseen categorical values are supported "
            "by the production preprocessing pipeline."
        ),
        examples=["Executive"],
    )

    Age: float = Field(
        ...,
        description="Employee age.",
        examples=[31],
    )

    Overtime: str = Field(
        ...,
        description="Whether the employee works overtime.",
        examples=["Yes"],
    )

    Absenteeism: float = Field(
        ...,
        description="Employee absenteeism measure.",
        examples=[8],
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class PredictionResponse(BaseModel):
    """Decision-support prediction response."""

    attrition_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated estimated probability of employee attrition, "
            "between 0 and 1."
        ),
        examples=[0.3587634701618515],
    )

    production_threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Production decision-support threshold used to determine "
            "the attrition risk flag."
        ),
        examples=[0.25],
    )

    attrition_risk_flag: bool = Field(
        ...,
        description=(
            "Decision-support flag indicating whether the predicted "
            "attrition probability meets or exceeds the production threshold."
        ),
        examples=[True],
    )

    decision_support_only: bool = Field(
        ...,
        description=(
            "Always true. The prediction is intended only as a "
            "decision-support signal."
        ),
        examples=[True],
    )

    automatic_employment_decisions: bool = Field(
        ...,
        description=(
            "Always false. The API does not make automatic employment decisions."
        ),
        examples=[False],
    )


class HealthResponse(BaseModel):
    """API health response."""

    status: str = Field(
        ...,
        description="Current API health status.",
        examples=["healthy"],
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the production model can be loaded.",
        examples=[True],
    )

    model_type: str = Field(
        ...,
        description="Runtime type of the loaded model.",
        examples=["CalibratedClassifierCV"],
    )


class ContractResponse(BaseModel):
    """Production model contract response."""

    model: str = Field(
        ...,
        description="Production model family.",
        examples=["Random Forest"],
    )

    calibration: str = Field(
        ...,
        description="Probability calibration method.",
        examples=["Sigmoid / Platt"],
    )

    threshold: float = Field(
        ...,
        description="Active production decision-support threshold.",
        examples=[0.25],
    )

    feature_count: int = Field(
        ...,
        description="Number of validated production features.",
        examples=[10],
    )

    features: list[str] = Field(
        ...,
        description="Validated production feature names.",
    )

    decision_support_only: bool = Field(
        ...,
        description="Whether the model is restricted to decision support.",
        examples=[True],
    )

    automatic_employment_decisions: bool = Field(
        ...,
        description="Whether the API can make automatic employment decisions.",
        examples=[False],
    )


class GovernanceResponse(BaseModel):
    """Production governance response."""

    decision_support_only: bool = Field(
        ...,
        description="Whether predictions are restricted to decision support.",
        examples=[True],
    )

    automatic_employment_decisions: bool = Field(
        ...,
        description="Whether automatic employment decisions are permitted.",
        examples=[False],
    )

    employment_decision_authority: Literal["human"] = Field(
        ...,
        description="Authority responsible for employment decisions.",
        examples=["human"],
    )

    model_output: str = Field(
        ...,
        description="Type of output produced by the model.",
        examples=["Risk probability and decision-support flag only."],
    )

    required_human_review: bool = Field(
        ...,
        description="Whether human review is required.",
        examples=[True],
    )

    message: str = Field(
        ...,
        description="Explicit governance restriction.",
        examples=[
            "Predictions must not be used for automatic employment decisions."
        ],
    )


# ============================================================================
# MODEL ACCESS
# ============================================================================


def get_model():
    """
    Load the production model.

    The underlying predict.py module remains the single source of truth
    for model loading.
    """

    try:
        return load_model()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Production model unavailable: {exc}",
        ) from exc


def get_production_metadata() -> dict:
    """Load production metadata."""

    try:
        return load_metadata()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Production metadata unavailable: {exc}",
        ) from exc


# ============================================================================
# ROOT
# ============================================================================


@app.get(
    "/",
    tags=["System"],
    summary="Get API service information",
)
def root() -> dict[str, Any]:
    """
    Return basic information about the production API.
    """

    return {
        "service": "Employee Attrition Production API",
        "status": "running",
        "api_version": "1.2.0",
        "model": "Random Forest",
        "calibration": "Sigmoid / Platt",
        "threshold": PRODUCTION_THRESHOLD,
        "decision_support_only": True,
        "automatic_employment_decisions": False,
        "documentation": "/docs",
        "openapi": "/openapi.json",
    }


# ============================================================================
# HEALTH
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API and model health",
    responses={
        503: {
            "description": "Production model is unavailable.",
        }
    },
)
def health() -> HealthResponse:
    """
    Verify that the production model can be loaded.

    Returns HTTP 200 when the model is available and HTTP 503 when
    the production model cannot be loaded.
    """

    try:
        model = get_model()

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=type(model).__name__,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {exc}",
        ) from exc


# ============================================================================
# PRODUCTION CONTRACT
# ============================================================================


@app.get(
    "/contract",
    response_model=ContractResponse,
    tags=["Model"],
    summary="Get the active production model contract",
)
def contract() -> ContractResponse:
    """
    Return the active production model contract.

    The contract identifies the production model, calibration method,
    threshold, and validated feature set.
    """

    metadata = get_production_metadata()

    return ContractResponse(
        model=metadata.get(
            "model_type",
            "Random Forest",
        ),
        calibration=metadata.get(
            "calibration",
            {},
        ).get(
            "name",
            "Sigmoid / Platt",
        ),
        threshold=float(
            metadata.get(
                "production_threshold",
                PRODUCTION_THRESHOLD,
            )
        ),
        feature_count=len(STABLE_FEATURES),
        features=STABLE_FEATURES,
        decision_support_only=True,
        automatic_employment_decisions=False,
    )


# ============================================================================
# PREDICTION
# ============================================================================


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Model"],
    summary="Generate an employee attrition risk signal",
    description=(
        "Generate a calibrated employee attrition probability and "
        "decision-support risk flag.\n\n"
        "This endpoint does not make or recommend an automatic "
        "employment decision. Human review is required."
    ),
    responses={
        422: {
            "description": (
                "Invalid request data or invalid prediction input."
            )
        },
        500: {
            "description": "Unexpected prediction failure."
        },
        503: {
            "description": "Production model is unavailable."
        },
    },
)
def predict(
    employee: EmployeeInput,
) -> PredictionResponse:
    """
    Generate an attrition probability and decision-support flag.

    No employment decision is made by this endpoint.
    """

    try:
        employee_data = employee.model_dump(
            exclude_none=True
        )

        prediction = predict_employee(
            employee_data
        )

        return PredictionResponse(
            attrition_probability=prediction[
                "attrition_probability"
            ],
            production_threshold=prediction[
                "production_threshold"
            ],
            attrition_risk_flag=prediction[
                "attrition_risk_flag"
            ],
            decision_support_only=True,
            automatic_employment_decisions=False,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ============================================================================
# GOVERNANCE
# ============================================================================


@app.get(
    "/governance",
    response_model=GovernanceResponse,
    tags=["Governance"],
    summary="Get model governance restrictions",
)
def governance() -> GovernanceResponse:
    """
    Explicitly expose the production governance restrictions.

    The endpoint documents that predictions are decision-support signals
    only and that employment decisions remain under human authority.
    """

    return GovernanceResponse(
        decision_support_only=True,
        automatic_employment_decisions=False,
        employment_decision_authority="human",
        model_output=(
            "Risk probability and decision-support flag only."
        ),
        required_human_review=True,
        message=(
            "Predictions must not be used for automatic employment "
            "decisions."
        ),
    )


# ============================================================================
# LOCAL EXECUTION
# ============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )