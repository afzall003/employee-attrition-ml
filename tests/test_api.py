"""
Automated API validation tests for the Employee Attrition ML project.

These tests validate the production FastAPI decision-support interface.

IMPORTANT:
    The API provides decision-support signals only.
    It must not be used to make automatic employment decisions.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


# ============================================================================
# TEST CLIENT
# ============================================================================

client = TestClient(app)


# ============================================================================
# VALID TEST EMPLOYEE
# ============================================================================

VALID_EMPLOYEE = {
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


# ============================================================================
# HEALTH
# ============================================================================


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_type"] == "CalibratedClassifierCV"


# ============================================================================
# CONTRACT
# ============================================================================


def test_contract_endpoint():
    response = client.get("/contract")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest"
    assert data["calibration"] == "Platt scaling"
    assert data["threshold"] == 0.25
    assert data["feature_count"] == 10

    expected_features = [
        "Work_Life_Balance",
        "Job_Satisfaction",
        "Distance_From_Home",
        "Average_Hours_Worked_Per_Week",
        "Years_Since_Last_Promotion",
        "Work_Environment_Satisfaction",
        "Job_Role",
        "Age",
        "Overtime",
        "Absenteeism",
    ]

    assert data["features"] == expected_features

    assert data["decision_support_only"] is True
    assert data["automatic_employment_decisions"] is False


# ============================================================================
# PREDICTION
# ============================================================================


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=VALID_EMPLOYEE,
    )

    assert response.status_code == 200

    data = response.json()

    assert "attrition_probability" in data
    assert "production_threshold" in data
    assert "attrition_risk_flag" in data

    assert 0.0 <= data["attrition_probability"] <= 1.0
    assert data["production_threshold"] == 0.25

    expected_flag = (
        data["attrition_probability"] >= 0.25
    )

    assert data["attrition_risk_flag"] == expected_flag

    assert data["decision_support_only"] is True
    assert data["automatic_employment_decisions"] is False


# ============================================================================
# MISSING FEATURE
# ============================================================================


def test_missing_required_feature():
    payload = VALID_EMPLOYEE.copy()

    del payload["Job_Role"]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data

    detail = data["detail"]

    assert any(
        item.get("loc") == ["body", "Job_Role"]
        for item in detail
    )


# ============================================================================
# INVALID NUMERIC TYPE
# ============================================================================


def test_invalid_numeric_type():
    payload = VALID_EMPLOYEE.copy()

    payload["Age"] = "thirty-one"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


# ============================================================================
# UNKNOWN CATEGORY
# ============================================================================


def test_unknown_category_is_handled():
    """
    Unknown categorical values should be handled safely.

    The production preprocessing contract uses:

        OneHotEncoder(handle_unknown="ignore")

    Therefore an unseen Job_Role should not cause an API failure.
    """

    payload = VALID_EMPLOYEE.copy()

    payload["Job_Role"] = "Future Job Role"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "attrition_probability" in data
    assert "production_threshold" in data
    assert "attrition_risk_flag" in data

    assert 0.0 <= data["attrition_probability"] <= 1.0
    assert data["production_threshold"] == 0.25

    expected_flag = (
        data["attrition_probability"] >= 0.25
    )

    assert data["attrition_risk_flag"] == expected_flag

    assert data["decision_support_only"] is True
    assert data["automatic_employment_decisions"] is False

# ============================================================================
# EXTRA FIELDS
# ============================================================================


def test_extra_fields_are_ignored():
    payload = VALID_EMPLOYEE.copy()

    payload["Unexpected_Field"] = "ignored"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert 0.0 <= data["attrition_probability"] <= 1.0


# ============================================================================
# GOVERNANCE
# ============================================================================


def test_governance_endpoint():
    response = client.get("/governance")

    assert response.status_code == 200

    data = response.json()

    assert data["decision_support_only"] is True
    assert data["automatic_employment_decisions"] is False
    assert data["employment_decision_authority"] == "human"
    assert data["required_human_review"] is True


# ============================================================================
# ROOT
# ============================================================================


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "Employee Attrition Production API"
    assert data["status"] == "running"
    assert data["model"] == "Random Forest"
    assert data["calibration"] == "Sigmoid / Platt"
    assert data["threshold"] == 0.25
    assert data["decision_support_only"] is True
    assert data["automatic_employment_decisions"] is False