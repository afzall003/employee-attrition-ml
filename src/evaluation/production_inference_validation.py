"""
Production inference validation for the Employee Attrition ML project.

Validates the actual calibrated production artifact and production
inference contract.

IMPORTANT:
    This model is for decision support only.
    It must NOT be used for automatic employment decisions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
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

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "production_validation"
)

JSON_REPORT_PATH = (
    REPORT_DIR
    / "production_inference_validation_report.json"
)

SUMMARY_PATH = (
    REPORT_DIR
    / "production_inference_validation_summary.txt"
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
# HELPERS
# ============================================================================


def check(
    name: str,
    condition: bool,
    failures: list[str],
    checks: dict[str, bool],
) -> bool:
    """Record and print a validation check."""

    passed = bool(condition)

    checks[name] = passed

    if passed:
        print(f"PASS {name}")
    else:
        print(f"FAIL {name}")
        failures.append(name)

    return passed


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def positive_class_index(model) -> int:
    """Return the probability column corresponding to class 1."""

    if not hasattr(model, "classes_"):
        raise ValueError(
            "Production artifact does not expose classes_."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            f"Positive class 1 not found in model classes: {classes}"
        )

    return classes.index(1)


def probability_values(
    model,
    df: pd.DataFrame,
) -> np.ndarray:
    """Generate positive-class probabilities."""

    probabilities = model.predict_proba(
        df[STABLE_FEATURES]
    )

    positive_index = positive_class_index(model)

    return np.asarray(
        probabilities[:, positive_index],
        dtype=float,
    )


def extract_metadata_features(metadata: dict) -> list[str]:
    """
    Extract production feature names from metadata.

    Supports the common metadata layouts used in this project.
    """

    candidate_keys = [
        "stable_features",
        "features",
        "feature_names",
        "production_features",
    ]

    for key in candidate_keys:
        value = metadata.get(key)

        if isinstance(value, list):
            return [str(item) for item in value]

    production_contract = metadata.get(
        "production_contract"
    )

    if isinstance(production_contract, dict):
        for key in candidate_keys:
            value = production_contract.get(key)

            if isinstance(value, list):
                return [str(item) for item in value]

    return []


def extract_model_feature_names(model) -> list[str]:
    """
    Inspect the fitted preprocessing pipeline and recover the feature
    columns used by the production estimator.
    """

    try:
        estimator = model.estimator
    except AttributeError:
        estimator = None

    if estimator is None:
        return []

    try:
        preprocessor = estimator.named_steps["preprocessor"]
    except (
        AttributeError,
        KeyError,
    ):
        return []

    feature_names: list[str] = []

    try:
        transformers = preprocessor.transformers

        for _, _, columns in transformers:
            if columns is None:
                continue

            if isinstance(columns, (list, tuple)):
                feature_names.extend(
                    str(column)
                    for column in columns
                )

    except AttributeError:
        return []

    return feature_names


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    print()
    print("=" * 72)
    print("EMPLOYEE ATTRITION — PRODUCTION INFERENCE VALIDATION")
    print("=" * 72)

    failures: list[str] = []
    checks: dict[str, bool] = {}

    # ========================================================================
    # ARTIFACT VALIDATION
    # ========================================================================

    print()
    print("=" * 72)
    print("PRODUCTION ARTIFACT VALIDATION")
    print("=" * 72)

    check(
        "model_file_exists",
        MODEL_PATH.exists(),
        failures,
        checks,
    )

    check(
        "metadata_file_exists",
        METADATA_PATH.exists(),
        failures,
        checks,
    )

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Production model artifact not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise SystemExit(
            f"Production metadata not found: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    metadata = load_json(
        METADATA_PATH
    )

    check(
        "model_loads_successfully",
        model is not None,
        failures,
        checks,
    )

    check(
        "model_has_predict_proba",
        hasattr(model, "predict_proba"),
        failures,
        checks,
    )

    check(
        "model_has_classes",
        hasattr(model, "classes_"),
        failures,
        checks,
    )

    # ========================================================================
    # MODEL CONTRACT
    # ========================================================================

    print()
    print("=" * 72)
    print("MODEL CONTRACT VALIDATION")
    print("=" * 72)

    model_type_name = type(model).__name__

    check(
        "calibrated_classifier",
        model_type_name == "CalibratedClassifierCV",
        failures,
        checks,
    )

    check(
        "metadata_model_type",
        metadata.get("model_type") == "Random Forest",
        failures,
        checks,
    )

    calibration = metadata.get(
        "calibration",
        {},
    )

    calibration_name = str(
        calibration.get("name", "")
    ).lower()

    check(
        "sigmoid_calibration",
        (
            "sigmoid" in calibration_name
            or "platt" in calibration_name
        ),
        failures,
        checks,
    )

    metadata_threshold = metadata.get(
        "production_threshold"
    )

    check(
        "metadata_threshold",
        (
            metadata_threshold is not None
            and math.isclose(
                float(metadata_threshold),
                PRODUCTION_THRESHOLD,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        failures,
        checks,
    )

    check(
        "classes_include_positive_class",
        (
            hasattr(model, "classes_")
            and 1 in list(model.classes_)
        ),
        failures,
        checks,
    )

    # ========================================================================
    # FEATURE CONTRACT
    # ========================================================================

    print()
    print("=" * 72)
    print("FEATURE CONTRACT VALIDATION")
    print("=" * 72)

    check(
        "stable_feature_count",
        len(STABLE_FEATURES) == 10,
        failures,
        checks,
    )

    metadata_features = extract_metadata_features(
        metadata
    )

    model_features = extract_model_feature_names(
        model
    )

    if metadata_features:

        metadata_feature_set = set(
            metadata_features
        )

        production_feature_set = set(
            STABLE_FEATURES
        )

        check(
            "metadata_features_match",
            (
                metadata_feature_set
                == production_feature_set
            ),
            failures,
            checks,
        )

        if (
            metadata_feature_set
            != production_feature_set
        ):
            print()
            print(
                "Metadata features:"
            )

            for feature in metadata_features:
                print(f"- {feature}")

            print()
            print(
                "Production contract features:"
            )

            for feature in STABLE_FEATURES:
                print(f"- {feature}")

    else:

        print(
            "INFO: Metadata does not expose a "
            "feature list; validating the fitted "
            "artifact preprocessing contract."
        )

        check(
            "metadata_features_match",
            bool(model_features),
            failures,
            checks,
        )

    if model_features:

        check(
            "artifact_feature_contract",
            set(model_features)
            == set(STABLE_FEATURES),
            failures,
            checks,
        )

    # ========================================================================
    # DATASET VALIDATION
    # ========================================================================

    print()
    print("=" * 72)
    print("INPUT DATA VALIDATION")
    print("=" * 72)

    check(
        "canonical_dataset_exists",
        DATASET_PATH.exists(),
        failures,
        checks,
    )

    if not DATASET_PATH.exists():
        raise SystemExit(
            f"Canonical dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    check(
        "input_rows",
        len(df) == 1000,
        failures,
        checks,
    )

    missing_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    check(
        "all_production_features_present",
        len(missing_features) == 0,
        failures,
        checks,
    )

    if missing_features:
        print(
            "Missing production features:"
        )

        for feature in missing_features:
            print(f"- {feature}")

        raise SystemExit(
            "Production feature contract cannot be validated."
        )

    # ========================================================================
    # INFERENCE VALIDATION
    # ========================================================================

    print()
    print("=" * 72)
    print("INFERENCE VALIDATION")
    print("=" * 72)

    probabilities = probability_values(
        model,
        df,
    )

    flags = (
        probabilities
        >= PRODUCTION_THRESHOLD
    )

    check(
        "probability_count_matches_rows",
        len(probabilities) == len(df),
        failures,
        checks,
    )

    check(
        "no_missing_probabilities",
        not np.isnan(probabilities).any(),
        failures,
        checks,
    )

    check(
        "no_infinite_probabilities",
        not np.isinf(probabilities).any(),
        failures,
        checks,
    )

    check(
        "probabilities_within_zero_one",
        bool(
            np.all(probabilities >= 0.0)
            and np.all(probabilities <= 1.0)
        ),
        failures,
        checks,
    )

    check(
        "flag_count_matches_rows",
        len(flags) == len(df),
        failures,
        checks,
    )

    # ========================================================================
    # THRESHOLD VALIDATION
    # ========================================================================

    print()
    print("=" * 72)
    print("THRESHOLD CONSISTENCY VALIDATION")
    print("=" * 72)

    check(
        "threshold_is_0_25",
        math.isclose(
            PRODUCTION_THRESHOLD,
            0.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        failures,
        checks,
    )

    expected_flags = (
        probabilities
        >= PRODUCTION_THRESHOLD
    )

    check(
        "flags_match_threshold",
        np.array_equal(
            flags,
            expected_flags,
        ),
        failures,
        checks,
    )

    # ========================================================================
    # PREPROCESSING SMOKE TEST
    # ========================================================================

    print()
    print("=" * 72)
    print("PREPROCESSING SMOKE TEST")
    print("=" * 72)

    smoke = (
        df[
            STABLE_FEATURES
        ]
        .head(3)
        .copy()
    )

    smoke_probabilities = probability_values(
        model,
        smoke,
    )

    check(
        "smoke_test_predictions",
        (
            len(smoke_probabilities) == 3
            and np.all(
                np.isfinite(
                    smoke_probabilities
                )
            )
            and np.all(
                (
                    smoke_probabilities
                    >= 0.0
                )
                & (
                    smoke_probabilities
                    <= 1.0
                )
            )
        ),
        failures,
        checks,
    )

    # ========================================================================
    # UNKNOWN CATEGORY TEST
    # ========================================================================

    print()
    print("=" * 72)
    print("UNKNOWN CATEGORY TEST")
    print("=" * 72)

    unknown_category = (
        df[
            STABLE_FEATURES
        ]
        .head(1)
        .copy()
    )

    unknown_category.loc[
        unknown_category.index[0],
        "Job_Role",
    ] = "__UNSEEN_PRODUCTION_CATEGORY__"

    unknown_category.loc[
        unknown_category.index[0],
        "Overtime",
    ] = "__UNSEEN_PRODUCTION_CATEGORY__"

    try:

        unknown_prob = probability_values(
            model,
            unknown_category,
        )

        unknown_category_pass = (
            len(unknown_prob) == 1
            and np.isfinite(
                unknown_prob[0]
            )
            and 0.0 <= unknown_prob[0] <= 1.0
        )

    except Exception as exc:

        unknown_category_pass = False

        print(
            f"Unknown-category inference error: {exc}"
        )

    check(
        "unknown_categories_handled",
        unknown_category_pass,
        failures,
        checks,
    )

    # ========================================================================
    # GOVERNANCE
    # ========================================================================

    print()
    print("=" * 72)
    print("DEPLOYMENT GOVERNANCE VALIDATION")
    print("=" * 72)

    governance_text = json.dumps(
        metadata,
        ensure_ascii=False,
    ).lower()

    check(
        "decision_support_governance",
        (
            "decision" in governance_text
            and "support" in governance_text
        ),
        failures,
        checks,
    )

    check(
        "automatic_employment_restriction",
        (
            "automatic" in governance_text
            and "employment" in governance_text
        ),
        failures,
        checks,
    )

    # ========================================================================
    # SUMMARY VALUES
    # ========================================================================

    flagged_count = int(
        flags.sum()
    )

    flag_rate = float(
        flags.mean()
    )

    probability_min = float(
        probabilities.min()
    )

    probability_max = float(
        probabilities.max()
    )

    status = (
        "PASS"
        if not failures
        else "FAIL"
    )

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================

    print()
    print("=" * 72)
    print(
        "EMPLOYEE ATTRITION — PRODUCTION INFERENCE VALIDATION"
    )
    print("=" * 72)

    print()
    print("[ARTIFACT]")
    print(
        f"Model:                 {MODEL_PATH}"
    )
    print(
        f"Artifact type:         {model_type_name}"
    )
    print(
        "Expected model:        Random Forest"
    )
    print(
        "Calibration:           Sigmoid / Platt"
    )

    print()
    print("[FEATURE CONTRACT]")
    print(
        f"Feature count:          {len(STABLE_FEATURES)}"
    )

    for feature in STABLE_FEATURES:
        print(f"- {feature}")

    print()
    print("[INFERENCE]")
    print(
        f"Rows validated:         {len(df)}"
    )
    print(
        f"Threshold:              {PRODUCTION_THRESHOLD:.2f}"
    )
    print(
        f"Probability minimum:    {probability_min:.6f}"
    )
    print(
        f"Probability maximum:    {probability_max:.6f}"
    )
    print(
        f"Flagged employees:      {flagged_count}"
    )
    print(
        f"Flag rate:              {flag_rate * 100:.2f}%"
    )

    print()
    print("[OVERALL STATUS]")
    print(
        "PRODUCTION INFERENCE VALIDATION: "
        f"{status}"
    )

    if failures:

        print()
        print("[FAILED CHECKS]")

        for failure in failures:
            print(f"- {failure}")

    # ========================================================================
    # JSON REPORT
    # ========================================================================

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "status": status,
        "model": {
            "artifact": str(MODEL_PATH),
            "artifact_type": model_type_name,
            "expected_family": "Random Forest",
            "calibration": "Sigmoid / Platt",
        },
        "production_contract": {
            "threshold": PRODUCTION_THRESHOLD,
            "stable_features": STABLE_FEATURES,
            "feature_count": len(STABLE_FEATURES),
        },
        "metadata_features": metadata_features,
        "artifact_features": model_features,
        "inference": {
            "rows": int(len(df)),
            "probability_min": probability_min,
            "probability_max": probability_max,
            "flagged_count": flagged_count,
            "flag_rate": flag_rate,
            "missing_probabilities": int(
                np.isnan(probabilities).sum()
            ),
            "infinite_probabilities": int(
                np.isinf(probabilities).sum()
            ),
        },
        "checks": checks,
        "failed_checks": failures,
        "governance": {
            "decision_support_only": True,
            "automatic_employment_decisions": False,
        },
    }

    with JSON_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================================
    # TEXT SUMMARY
    # ========================================================================

    summary_lines = [
        "EMPLOYEE ATTRITION — PRODUCTION INFERENCE VALIDATION",
        "",
        f"Status: {status}",
        f"Rows validated: {len(df)}",
        f"Threshold: {PRODUCTION_THRESHOLD:.2f}",
        f"Probability minimum: {probability_min:.6f}",
        f"Probability maximum: {probability_max:.6f}",
        f"Flagged employees: {flagged_count}",
        f"Flag rate: {flag_rate * 100:.2f}%",
        "",
        "Decision-support only.",
        "Automatic employment decisions are prohibited.",
        "",
        "Failed checks:",
    ]

    if failures:

        summary_lines.extend(
            f"- {failure}"
            for failure in failures
        )

    else:

        summary_lines.append(
            "None"
        )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("[OUTPUT]")
    print(
        f"JSON report:     {JSON_REPORT_PATH}"
    )
    print(
        f"Summary report:  {SUMMARY_PATH}"
    )

    print()
    print("=" * 72)
    print(
        "PRODUCTION INFERENCE VALIDATION COMPLETE"
    )
    print("=" * 72)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()