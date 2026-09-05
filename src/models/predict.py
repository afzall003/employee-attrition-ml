"""
Production inference for the Employee Attrition ML project.

This module loads the validated calibrated Random Forest artifact and
performs production inference using the exact 10-feature production
contract.

IMPORTANT:
    This model is for decision support only.

    It must NOT be used to make automatic employment decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


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
# PRODUCTION CONTRACT
# ============================================================================

PRODUCTION_THRESHOLD = 0.25

STABLE_FEATURES = [
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


# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model():
    """Load the calibrated production model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_metadata() -> dict:
    """Load production model metadata."""

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Production metadata not found: {METADATA_PATH}"
        )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


# ============================================================================
# INPUT VALIDATION
# ============================================================================


def validate_features(df: pd.DataFrame) -> None:
    """Validate inference input against the production feature contract."""

    missing = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing production features: "
            + ", ".join(missing)
        )


# ============================================================================
# PREDICTION
# ============================================================================


def predict_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate calibrated attrition probabilities and decision-support flags.

    Returns a copy of the input dataframe with:

        attrition_probability
        attrition_risk_flag
        production_threshold
    """

    validate_features(df)

    model = load_model()

    X = df[STABLE_FEATURES].copy()

    probabilities = model.predict_proba(X)

    # CalibratedClassifierCV follows sklearn's binary class ordering.
    # Class 1 corresponds to the positive attrition class.
    positive_index = list(model.classes_).index(1)

    positive_probability = probabilities[
        :,
        positive_index,
    ]

    result = df.copy()

    result["attrition_probability"] = positive_probability

    result["production_threshold"] = PRODUCTION_THRESHOLD

    result["attrition_risk_flag"] = (
        positive_probability >= PRODUCTION_THRESHOLD
    )

    return result


# ============================================================================
# SINGLE EMPLOYEE PREDICTION
# ============================================================================


def predict_employee(
    employee: dict,
) -> dict:
    """
    Predict attrition probability for one employee.

    Example:

        employee = {
            "Work_Life_Balance": 2,
            "Job_Satisfaction": 2,
            "Distance_From_Home": 15,
            "Average_Hours_Worked_Per_Week": 48,
            "Years_Since_Last_Promotion": 1,
            "Work_Environment_Satisfaction": 2,
            "Job_Role": "Sales Executive",
            "Age": 31,
            "Overtime": "Yes",
            "Absenteeism": 8,
        }
    """

    df = pd.DataFrame(
        [employee]
    )

    result = predict_dataframe(df)

    row = result.iloc[0]

    return {
        "attrition_probability": float(
            row["attrition_probability"]
        ),
        "production_threshold": PRODUCTION_THRESHOLD,
        "attrition_risk_flag": bool(
            row["attrition_risk_flag"]
        ),
    }


# ============================================================================
# CSV INFERENCE
# ============================================================================


def predict_csv(
    input_path: Path,
    output_path: Path,
) -> None:
    """Run inference on a CSV file."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    result = predict_dataframe(df)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("EMPLOYEE ATTRITION — PRODUCTION INFERENCE")
    print("=" * 72)

    print()
    print(f"Input rows:           {len(df)}")
    print(
        f"Threshold:            {PRODUCTION_THRESHOLD:.2f}"
    )

    print(
        f"Flagged employees:    "
        f"{result['attrition_risk_flag'].sum()}"
    )

    print(
        f"Flagged percentage:   "
        f"{result['attrition_risk_flag'].mean() * 100:.2f}%"
    )

    print()
    print(f"Output:               {output_path}")

    print()
    print("IMPORTANT:")
    print(
        "Predictions are decision-support signals only."
    )
    print(
        "They must not be used for automatic employment decisions."
    )


# ============================================================================
# CLI
# ============================================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Employee Attrition calibrated production inference"
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV containing the 10 production features.",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV path.",
    )

    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = (
                PROJECT_ROOT
                / "reports"
                / "production_predictions.csv"
            )

        predict_csv(
            input_path=input_path,
            output_path=output_path,
        )

    else:
        print()
        print("=" * 72)
        print("EMPLOYEE ATTRITION — PRODUCTION MODEL")
        print("=" * 72)

        model = load_model()
        metadata = load_metadata()

        print()
        print("[MODEL]")
        print(
            f"Artifact:             {MODEL_PATH}"
        )
        print(
            f"Model:                {metadata['model_type']}"
        )
        print(
            f"Calibration:          "
            f"{metadata['calibration']['name']}"
        )
        print(
            f"Threshold:            "
            f"{metadata['production_threshold']:.2f}"
        )

        print()
        print("[FEATURE CONTRACT]")

        for feature in STABLE_FEATURES:
            print(f"- {feature}")

        print()
        print("[STATUS]")
        print("Production artifact loaded successfully.")
        print(
            "Decision-support mode: ENABLED"
        )
        print(
            "Automatic employment decisions: PROHIBITED"
        )

        print()
        print("Model type:")
        print(type(model))


if __name__ == "__main__":
    main()