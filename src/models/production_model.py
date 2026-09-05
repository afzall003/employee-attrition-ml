"""
Production model training for the Employee Attrition ML project.

This module trains the exact model configuration validated by the
calibrated production candidate validation pipeline:

    Model:
        Random Forest

    Stable features:
        Work_Life_Balance
        Job_Satisfaction
        Distance_From_Home
        Average_Hours_Worked_Per_Week
        Years_Since_Last_Promotion
        Work_Environment_Satisfaction
        Job_Role
        Age
        Overtime
        Absenteeism

    Calibration:
        Sigmoid / Platt scaling

    Production threshold:
        0.25

The saved artifact contains the complete preprocessing + calibrated model
pipeline required for production inference.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "employee_attrition_production_calibrated.joblib"

METADATA_PATH = (
    MODEL_DIR
    / "employee_attrition_production_calibrated_metadata.json"
)


# ============================================================================
# CANONICAL MODEL CONTRACT
# ============================================================================

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

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

NUMERICAL_FEATURES = [
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Distance_From_Home",
    "Average_Hours_Worked_Per_Week",
    "Years_Since_Last_Promotion",
    "Age",
    "Absenteeism",
]

CATEGORICAL_FEATURES = [
    "Work_Environment_Satisfaction",
    "Job_Role",
    "Overtime",
]


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 400,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================================
# HELPERS
# ============================================================================


def calculate_sha256(path: Path) -> str:
    """Return SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the dataset against the production feature contract."""

    required_columns = (
        [IDENTIFIER_COLUMN]
        + STABLE_FEATURES
        + [TARGET_COLUMN]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Dataset is missing required production columns: "
            + ", ".join(missing)
        )

    if df[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError(
            f"{IDENTIFIER_COLUMN} contains duplicate values."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} contains missing values."
        )

    invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - {
        POSITIVE_CLASS,
        NEGATIVE_CLASS,
    }

    if invalid_targets:
        raise ValueError(
            "Unexpected target values found: "
            + ", ".join(map(str, invalid_targets))
        )

    missing_feature_values = df[STABLE_FEATURES].isna().sum().sum()

    if missing_feature_values:
        print(
            "WARNING: Stable features contain missing values. "
            "Pipeline imputers will handle them."
        )


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing used by the production model."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_base_model() -> Pipeline:
    """Build the uncalibrated Random Forest pipeline."""

    preprocessor = build_preprocessor()

    classifier = RandomForestClassifier(
        **RF_PARAMS
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )


def build_calibrated_model() -> CalibratedClassifierCV:
    """
    Build the calibrated production model.

    Sigmoid calibration is Platt scaling.
    """

    base_pipeline = build_base_model()

    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipeline,
        method="sigmoid",
        cv=5,
    )

    return calibrated_model


# ============================================================================
# TRAINING
# ============================================================================


def train_production_model() -> None:
    """Train and save the calibrated production candidate."""

    print()
    print("=" * 72)
    print("EMPLOYEE ATTRITION — PRODUCTION MODEL TRAINING")
    print("=" * 72)

    print()
    print("[DATASET]")
    print(f"Path:                 {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    validate_dataset(df)

    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print(f"Target prevalence:    {(df[TARGET_COLUMN] == POSITIVE_CLASS).mean():.2%}")

    print()
    print("[PRODUCTION CONTRACT]")
    print("Model:                 Random Forest")
    print("Features:              10 stable features")
    print("Calibration:           Sigmoid / Platt")
    print(f"Threshold:             {PRODUCTION_THRESHOLD:.2f}")

    print()
    print("[FEATURES]")

    for feature in STABLE_FEATURES:
        print(f"- {feature}")

    X = df[STABLE_FEATURES].copy()

    y = (
        df[TARGET_COLUMN]
        .eq(POSITIVE_CLASS)
        .astype(int)
    )

    print()
    print("[TRAINING]")
    print("Training on the complete canonical dataset...")
    print("Random Forest: 400 estimators")
    print("Calibration: Sigmoid / Platt")
    print("Calibration CV: 5 folds")

    model = build_calibrated_model()

    model.fit(X, y)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    dataset_sha256 = calculate_sha256(DATA_PATH)

    metadata = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "model_type": "Random Forest",

        "calibration": {
            "method": "sigmoid",
            "name": "Platt scaling",
            "cv": 5,
        },

        "production_threshold": PRODUCTION_THRESHOLD,

        "target": {
            "column": TARGET_COLUMN,
            "positive_class": POSITIVE_CLASS,
            "negative_class": NEGATIVE_CLASS,
        },

        "identifier": IDENTIFIER_COLUMN,

        "features": {
            "count": len(STABLE_FEATURES),
            "stable_features": STABLE_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },

        "random_forest": RF_PARAMS,

        "training_dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "sha256": dataset_sha256,
            "target_prevalence": float(y.mean()),
        },

        "artifact": {
            "path": str(MODEL_PATH),
            "format": "joblib",
        },

        "governance": {
            "decision_support_only": True,
            "automatic_employment_decisions": False,
            "threshold_requires_business_approval": True,
            "intervention_capacity_requires_approval": True,
            "post_deployment_monitoring_required": True,
            "threshold_reassessment_required": True,
        },
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print("[ARTIFACT]")
    print(f"Model:                {MODEL_PATH}")
    print(f"Metadata:             {METADATA_PATH}")

    print()
    print("[MODEL INSPECTION]")

    print(
        "Loaded model type:",
        type(model),
    )

    print(
        "Classes:",
        list(model.classes_),
    )

    print(
        "Stable feature count:",
        len(STABLE_FEATURES),
    )

    print()
    print("=" * 72)
    print("PRODUCTION MODEL TRAINING COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    train_production_model()