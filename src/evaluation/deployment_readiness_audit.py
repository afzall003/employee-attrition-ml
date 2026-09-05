"""
Deployment Readiness Audit
--------------------------

Audits whether the selected employee attrition model configuration is
internally consistent and suitable for controlled deployment.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

Selected model:
    Random Forest

Stable feature set:
    10 features

Decision threshold:
    0.44
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "deployment_readiness_audit"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CANONICAL SPECIFICATION
# ============================================================

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

THRESHOLD = 0.44

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
    "Work_Environment_Satisfaction",
]

CATEGORICAL_FEATURES = [
    "Job_Role",
    "Overtime",
]

EXPECTED_TARGET_VALUES = {"Yes", "No"}

EXPECTED_CANONICAL_SHA256 = (
    "9b294a270e34d159ce21e7f2c4d0be394d53f83736bc7d413296be4cf2768ed6"
)


# ============================================================
# HELPERS
# ============================================================

def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def print_check(name: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{status} {name}")


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-safe values."""
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    return value


# ============================================================
# MODEL CONSTRUCTION
# ============================================================

def build_model() -> Pipeline:
    """
    Construct the selected Random Forest model using the
    configuration established during stable model optimization.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
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

    preprocessor = ColumnTransformer(
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

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# DATASET AUDIT
# ============================================================

def audit_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Audit canonical dataset against the established specification."""

    checks: dict[str, bool] = {}

    checks["file_exists"] = DATASET_PATH.exists()
    print_check("file_exists", checks["file_exists"])

    checks["expected_rows"] = len(df) == EXPECTED_ROWS
    print_check("expected_rows", checks["expected_rows"])

    checks["expected_columns"] = len(df.columns) == EXPECTED_COLUMNS
    print_check("expected_columns", checks["expected_columns"])

    checks["target_exists"] = TARGET_COLUMN in df.columns
    print_check("target_exists", checks["target_exists"])

    checks["identifier_exists"] = IDENTIFIER_COLUMN in df.columns
    print_check("identifier_exists", checks["identifier_exists"])

    checks["stable_features_exist"] = all(
        feature in df.columns for feature in STABLE_FEATURES
    )
    print_check("stable_features_exist", checks["stable_features_exist"])

    target_values = set(
        df[TARGET_COLUMN].dropna().astype(str).unique()
    )

    checks["target_values_valid"] = (
        target_values == EXPECTED_TARGET_VALUES
    )
    print_check("target_values_valid", checks["target_values_valid"])

    checks["no_missing_cells"] = int(df.isna().sum().sum()) == 0
    print_check("no_missing_cells", checks["no_missing_cells"])

    checks["identifier_unique"] = (
        df[IDENTIFIER_COLUMN].nunique() == len(df)
    )
    print_check("identifier_unique", checks["identifier_unique"])

    feature_count = len(STABLE_FEATURES)

    checks["stable_feature_count"] = feature_count == 10
    print_check("stable_feature_count", checks["stable_feature_count"])

    return checks


# ============================================================
# FEATURE SCHEMA AUDIT
# ============================================================

def audit_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Verify the exact production feature schema."""

    checks: dict[str, bool] = {}

    actual_features = [
        column
        for column in df.columns
        if column not in {TARGET_COLUMN, IDENTIFIER_COLUMN}
    ]

    checks["model_feature_count"] = len(actual_features) == 24
    print_check(
        "model_feature_count_24",
        checks["model_feature_count"],
    )

    checks["stable_features_exact"] = set(STABLE_FEATURES).issubset(
        set(actual_features)
    )
    print_check(
        "stable_features_present_in_model_schema",
        checks["stable_features_exact"],
    )

    checks["numerical_feature_count"] = (
        len(NUMERICAL_FEATURES) == 8
    )
    print_check(
        "numerical_feature_count_8",
        checks["numerical_feature_count"],
    )

    checks["categorical_feature_count"] = (
        len(CATEGORICAL_FEATURES) == 2
    )
    print_check(
        "categorical_feature_count_2",
        checks["categorical_feature_count"],
    )

    checks["feature_partition_complete"] = (
        set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        == set(STABLE_FEATURES)
    )
    print_check(
        "feature_partition_complete",
        checks["feature_partition_complete"],
    )

    checks["no_identifier_in_features"] = (
        IDENTIFIER_COLUMN not in STABLE_FEATURES
    )
    print_check(
        "identifier_excluded",
        checks["no_identifier_in_features"],
    )

    checks["no_target_in_features"] = (
        TARGET_COLUMN not in STABLE_FEATURES
    )
    print_check(
        "target_excluded",
        checks["no_target_in_features"],
    )

    return checks


# ============================================================
# MODEL AUDIT
# ============================================================

def audit_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Fit and audit the selected production model."""

    checks: dict[str, bool] = {}

    model.fit(X, y)

    classifier = model.named_steps["model"]

    checks["model_is_random_forest"] = isinstance(
        classifier,
        RandomForestClassifier,
    )
    print_check(
        "model_is_random_forest",
        checks["model_is_random_forest"],
    )

    checks["n_estimators_400"] = classifier.n_estimators == 400
    print_check(
        "n_estimators_400",
        checks["n_estimators_400"],
    )

    checks["max_features_sqrt"] = classifier.max_features == "sqrt"
    print_check(
        "max_features_sqrt",
        checks["max_features_sqrt"],
    )

    checks["min_samples_leaf_10"] = classifier.min_samples_leaf == 10
    print_check(
        "min_samples_leaf_10",
        checks["min_samples_leaf_10"],
    )

    checks["class_weight_balanced"] = (
        classifier.class_weight == "balanced"
    )
    print_check(
        "class_weight_balanced",
        checks["class_weight_balanced"],
    )

    checks["threshold_valid"] = 0.0 < THRESHOLD < 1.0
    print_check(
        "threshold_valid",
        checks["threshold_valid"],
    )

    probabilities = model.predict_proba(X)[:, 1]

    checks["probabilities_finite"] = bool(
        np.isfinite(probabilities).all()
    )
    print_check(
        "probabilities_finite",
        checks["probabilities_finite"],
    )

    checks["probabilities_in_range"] = bool(
        ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    )
    print_check(
        "probabilities_in_range",
        checks["probabilities_in_range"],
    )

    predictions = (probabilities >= THRESHOLD).astype(int)

    checks["predictions_binary"] = set(
        np.unique(predictions)
    ).issubset({0, 1})

    print_check(
        "predictions_binary",
        checks["predictions_binary"],
    )

    return checks


# ============================================================
# PRODUCTION BEHAVIOR AUDIT
# ============================================================

def audit_prediction_behavior(
    model: Pipeline,
    X: pd.DataFrame,
) -> dict[str, Any]:
    """Measure prediction behavior using the selected threshold."""

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    positive_rate = float(predictions.mean())

    return {
        "threshold": THRESHOLD,
        "probability_min": float(probabilities.min()),
        "probability_max": float(probabilities.max()),
        "probability_mean": float(probabilities.mean()),
        "predicted_positive_count": int(predictions.sum()),
        "predicted_negative_count": int((predictions == 0).sum()),
        "predicted_positive_rate": positive_rate,
        "flagged_per_1000": positive_rate * 1000.0,
    }


# ============================================================
# DEPLOYMENT DIAGNOSTICS
# ============================================================

def generate_diagnostics(
    dataset_checks: dict[str, bool],
    schema_checks: dict[str, bool],
    model_checks: dict[str, bool],
    behavior: dict[str, Any],
) -> list[str]:

    flags: list[str] = []

    if not all(dataset_checks.values()):
        flags.append(
            "Canonical dataset validation contains one or more failures."
        )

    if not all(schema_checks.values()):
        flags.append(
            "Stable feature schema validation contains one or more failures."
        )

    if not all(model_checks.values()):
        flags.append(
            "Selected model configuration validation contains one or more failures."
        )

    if behavior["predicted_positive_rate"] > 0.50:
        flags.append(
            "The selected threshold flags more than half of observations; "
            "intervention capacity should be explicitly reviewed."
        )

    if behavior["predicted_positive_rate"] > 0.472:
        flags.append(
            "Predicted-positive rate is more than twice the observed "
            "23.60% attrition prevalence."
        )

    if not flags:
        flags.append(
            "No structural deployment-readiness failures were detected."
        )

    return flags


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Running deployment readiness audit...")
    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    print()
    print("=" * 64)
    print("EMPLOYEE ATTRITION — DEPLOYMENT READINESS AUDIT")
    print("=" * 64)

    # --------------------------------------------------------
    # HASH
    # --------------------------------------------------------

    sha256 = calculate_sha256(DATASET_PATH)

    print()
    print("[CANONICAL DATASET]")
    print(f"Path:                 {DATASET_PATH}")
    print(f"SHA-256:              {sha256}")

    hash_matches = sha256 == EXPECTED_CANONICAL_SHA256

    print_check(
        "canonical_sha256",
        hash_matches,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print()
    print("[DATASET VALIDATION]")

    dataset_checks = audit_dataset(df)

    # --------------------------------------------------------
    # FEATURE SCHEMA
    # --------------------------------------------------------

    print()
    print("[FEATURE SCHEMA VALIDATION]")

    schema_checks = audit_feature_schema(df)

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    y = (
        df[TARGET_COLUMN]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    X = df[STABLE_FEATURES].copy()

    print()
    print("[MODEL INPUT]")

    print(f"Stable features:      {len(STABLE_FEATURES)}")
    print(f"Numerical features:   {len(NUMERICAL_FEATURES)}")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Target prevalence:    {y.mean() * 100:.2f}%")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print("[MODEL CONFIGURATION]")

    model = build_model()

    model_checks = audit_model(
        model=model,
        X=X,
        y=y,
    )

    # --------------------------------------------------------
    # PREDICTION BEHAVIOR
    # --------------------------------------------------------

    behavior = audit_prediction_behavior(
        model=model,
        X=X,
    )

    print()
    print("[PREDICTION BEHAVIOR]")

    print(
        f"Threshold:            {behavior['threshold']:.2f}"
    )
    print(
        f"Probability range:    "
        f"{behavior['probability_min']:.4f} - "
        f"{behavior['probability_max']:.4f}"
    )
    print(
        f"Mean probability:     "
        f"{behavior['probability_mean']:.4f}"
    )
    print(
        f"Predicted positive:   "
        f"{behavior['predicted_positive_rate'] * 100:.2f}%"
    )
    print(
        f"Flagged per 1000:     "
        f"{behavior['flagged_per_1000']:.1f}"
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    diagnostics = generate_diagnostics(
        dataset_checks=dataset_checks,
        schema_checks=schema_checks,
        model_checks=model_checks,
        behavior=behavior,
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in diagnostics:
        print(f"- {flag}")

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    structural_pass = (
        hash_matches
        and all(dataset_checks.values())
        and all(schema_checks.values())
        and all(model_checks.values())
    )

    if structural_pass and behavior["predicted_positive_rate"] <= 0.50:
        status = "PASS"
    elif structural_pass:
        status = "CONDITIONAL PASS"
    else:
        status = "FAIL"

    print()
    print("[OVERALL STATUS]")
    print(
        f"DEPLOYMENT READINESS STATUS: {status}"
    )

    # --------------------------------------------------------
    # DIAGNOSIS
    # --------------------------------------------------------

    if status == "PASS":
        diagnosis = (
            "The selected canonical dataset, stable feature schema, "
            "Random Forest configuration, threshold, and prediction "
            "pipeline passed structural deployment-readiness checks."
        )
    elif status == "CONDITIONAL PASS":
        diagnosis = (
            "The selected model configuration is structurally reproducible, "
            "but the operating threshold produces a high intervention volume. "
            "Deployment requires explicit business-capacity review."
        )
    else:
        diagnosis = (
            "One or more structural deployment-readiness checks failed. "
            "The model should not proceed to deployment packaging until "
            "the identified failures are resolved."
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = {
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(df),
            "columns": len(df.columns),
            "sha256": sha256,
            "expected_sha256": EXPECTED_CANONICAL_SHA256,
            "hash_matches": hash_matches,
        },
        "stable_features": STABLE_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": {
            "column": TARGET_COLUMN,
            "prevalence": float(y.mean()),
            "positive_count": int(y.sum()),
            "negative_count": int((y == 0).sum()),
        },
        "model": {
            "name": "Random Forest",
            "n_estimators": 400,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "random_state": 42,
            "threshold": THRESHOLD,
        },
        "checks": {
            "dataset": dataset_checks,
            "schema": schema_checks,
            "model": model_checks,
        },
        "prediction_behavior": behavior,
        "diagnostic_flags": diagnostics,
        "status": status,
        "diagnosis": diagnosis,
    }

    json_path = REPORT_DIR / "deployment_readiness_audit_report.json"
    summary_path = REPORT_DIR / "deployment_readiness_audit_summary.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(report),
            handle,
            indent=2,
        )

    summary_lines = [
        "EMPLOYEE ATTRITION — DEPLOYMENT READINESS AUDIT",
        "",
        f"Dataset: {DATASET_PATH.name}",
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Stable features: {len(STABLE_FEATURES)}",
        "Model: Random Forest",
        f"Threshold: {THRESHOLD:.2f}",
        f"Predicted positive rate: "
        f"{behavior['predicted_positive_rate'] * 100:.2f}%",
        f"Flagged per 1000: "
        f"{behavior['flagged_per_1000']:.1f}",
        "",
        f"Status: {status}",
        "",
        "Diagnostic flags:",
    ]

    summary_lines.extend(
        f"- {flag}"
        for flag in diagnostics
    )

    summary_lines.extend(
        [
            "",
            "Diagnosis:",
            diagnosis,
        ]
    )

    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    print()
    print("[OUTPUT]")
    print(f"JSON report:       {json_path}")
    print(f"Summary report:    {summary_path}")

    print()
    print("=" * 64)
    print("DEPLOYMENT READINESS AUDIT COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()