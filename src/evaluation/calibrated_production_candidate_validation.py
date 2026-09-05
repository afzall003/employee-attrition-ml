"""
calibrated_production_candidate_validation.py

Final production-candidate validation gate for the employee attrition
stable-feature calibrated Random Forest.

This script DOES NOT retrain the model.

It validates the CURRENT calibrated production evidence chain:

1. calibrated_final_validation_stable
2. calibrated_business_cost_analysis
3. calibrated_threshold_optimization_stable
4. calibrated_deployment_decision_analysis
5. deployment_readiness
6. calibration_analysis_stable
7. deployment_decision_analysis
8. business_cost_analysis
9. business_threshold_analysis

Historical model-selection artifacts are treated as supporting/legacy
evidence and are NOT allowed to block the current calibrated Random
Forest candidate merely because they describe an older Logistic
Regression configuration.

Decision statuses:

    GO
    CONDITIONAL GO
    NO-GO

Intended use:
    Controlled decision-support only.

The model must NOT be used for automatic employment decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

SIGNAL_ANALYSIS_ROOT = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
)

OUTPUT_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibrated_production_candidate_validation"
)


# =====================================================================
# OUTPUT FILES
# =====================================================================

EVIDENCE_CSV = (
    OUTPUT_DIR
    / "calibrated_production_candidate_evidence.csv"
)

JSON_REPORT = (
    OUTPUT_DIR
    / "calibrated_production_candidate_validation_report.json"
)

SUMMARY_REPORT = (
    OUTPUT_DIR
    / "calibrated_production_candidate_validation_summary.txt"
)


# =====================================================================
# CANONICAL DATASET CONFIGURATION
# =====================================================================

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

IDENTIFIER_COLUMN = "Employee_ID"
TARGET_COLUMN = "Attrition"

EXPECTED_TARGET_VALUES = {
    "Yes",
    "No",
}


# ---------------------------------------------------------------------
# CURRENT CANONICAL STABLE FEATURE SET
# ---------------------------------------------------------------------

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
    "Work_Environment_Satisfaction",
    "Age",
    "Absenteeism",
]


CATEGORICAL_FEATURES = [
    "Job_Role",
    "Overtime",
]


EXPECTED_CANONICAL_SHA256 = (
    "9b294a270e34d159ce21e7f2c4d0be394d53f83736bc7d413296be4cf2768ed6"
)


# =====================================================================
# CURRENT PRODUCTION CANDIDATE
# =====================================================================

MODEL_NAME = "Random Forest"

CALIBRATION_METHOD = "Sigmoid / Platt"

PRODUCTION_THRESHOLD = 0.25

N_ESTIMATORS = 400
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 10
CLASS_WEIGHT = "balanced"


# =====================================================================
# EXPECTED CALIBRATED EVIDENCE
# =====================================================================

EXPECTED_CALIBRATION = {
    "brier_score": 0.1731,
    "log_loss": 0.5236,
    "expected_calibration_error": 0.0314,
    "roc_auc": 0.6517,
    "pr_auc": 0.3141,
}


EXPECTED_OPERATING_POINT = {
    "threshold": 0.25,
    "f1": 0.4304,
    "precision": 0.3416,
    "recall": 0.5814,
    "specificity": 0.6539,
    "balanced_accuracy": 0.6176,
    "predicted_positive_percent": 40.16,
    "flagged_per_1000": 401.6,
}


# =====================================================================
# QUALITY LIMITS
# =====================================================================

MAX_BRIER_SCORE = 0.20
MAX_LOG_LOSS = 0.55
MAX_ECE = 0.10


# These are NOT hard technical deployment blockers.
# They become explicit business conditions.

MIN_BUSINESS_PRECISION = 0.40
MIN_BUSINESS_RECALL = 0.70
INTERVENTION_CAPACITY_LIMIT = 400.0


# =====================================================================
# GOVERNANCE POLICY
# =====================================================================

REQUIRE_BUSINESS_THRESHOLD_APPROVAL = True
REQUIRE_INTERVENTION_CAPACITY_APPROVAL = True
DECISION_SUPPORT_ONLY = True
NO_AUTOMATIC_EMPLOYMENT_DECISIONS = True
REQUIRE_POST_DEPLOYMENT_MONITORING = True
REQUIRE_THRESHOLD_REASSESSMENT = True


# =====================================================================
# CURRENT EVIDENCE DIRECTORIES
# =====================================================================

CALIBRATED_FINAL_VALIDATION_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibrated_final_validation_stable"
)

CALIBRATED_BUSINESS_COST_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibrated_business_cost_analysis"
)

CALIBRATED_THRESHOLD_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibrated_threshold_optimization_stable"
)

CALIBRATED_DEPLOYMENT_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibrated_deployment_decision_analysis"
)

DEPLOYMENT_READINESS_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "deployment_readiness"
)

CALIBRATION_ANALYSIS_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "calibration_analysis_stable"
)

DEPLOYMENT_DECISION_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "deployment_decision_analysis"
)

BUSINESS_COST_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "business_cost_analysis"
)

BUSINESS_THRESHOLD_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "business_threshold_analysis"
)

# ---------------------------------------------------------------------
# Historical / legacy model selection evidence.
#
# IMPORTANT:
# The current file is known to contain an older Logistic Regression
# configuration. It is retained as supporting evidence only.
# It must NOT block the current calibrated Random Forest candidate.
# ---------------------------------------------------------------------

LEGACY_FINAL_MODEL_SELECTION_DIR = (
    SIGNAL_ANALYSIS_ROOT
    / "final_model_selection"
)


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def print_pass(label: str) -> None:
    print(f"PASS {label}")


def print_fail(label: str, detail: str = "") -> None:
    if detail:
        print(f"FAIL {label} {detail}")
    else:
        print(f"FAIL {label}")


def print_warn(label: str, detail: str = "") -> None:
    if detail:
        print(f"WARN {label} {detail}")
    else:
        print(f"WARN {label}")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None

        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def values_close(
    actual: Any,
    expected: Any,
    tolerance: float = 1e-3,
) -> bool:

    actual_value = safe_float(actual)
    expected_value = safe_float(expected)

    if actual_value is None or expected_value is None:
        return False

    return abs(actual_value - expected_value) <= tolerance


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> Optional[Dict[str, Any]]:

    if not path.exists():
        return None

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        if isinstance(data, dict):
            return data

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception:
        return None


def first_existing(
    directory: Path,
    filenames: List[str],
) -> Optional[Path]:

    for filename in filenames:

        path = directory / filename

        if path.exists():
            return path

    return None


def normalize_column_name(
    name: str,
) -> str:

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
    )


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def row_value(
    row: pd.Series,
    candidates: List[str],
) -> Optional[float]:

    for candidate in candidates:

        column = find_column(
            pd.DataFrame([row]),
            [candidate],
        )

        if column is not None:

            value = safe_float(row[column])

            if value is not None:
                return value

    return None


# =====================================================================
# DATASET VALIDATION
# =====================================================================

def validate_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:

    print_header(
        "CANONICAL DATASET VALIDATION"
    )

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Canonical dataset not found: "
            f"{DATASET_PATH}"
        )

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    checks: Dict[str, bool] = {}

    # -------------------------------------------------------------
    # File
    # -------------------------------------------------------------

    checks["file_exists"] = DATASET_PATH.exists()

    if checks["file_exists"]:
        print_pass("file_exists")
    else:
        print_fail("file_exists")

    # -------------------------------------------------------------
    # Rows
    # -------------------------------------------------------------

    checks["expected_rows"] = (
        len(df) == EXPECTED_ROWS
    )

    if checks["expected_rows"]:
        print_pass("expected_rows")
    else:
        print_fail(
            "expected_rows",
            f"expected={EXPECTED_ROWS} "
            f"actual={len(df)}",
        )

    # -------------------------------------------------------------
    # Columns
    # -------------------------------------------------------------

    checks["expected_columns"] = (
        len(df.columns) == EXPECTED_COLUMNS
    )

    if checks["expected_columns"]:
        print_pass("expected_columns")
    else:
        print_fail(
            "expected_columns",
            f"expected={EXPECTED_COLUMNS} "
            f"actual={len(df.columns)}",
        )

    # -------------------------------------------------------------
    # Target
    # -------------------------------------------------------------

    checks["target_exists"] = (
        TARGET_COLUMN in df.columns
    )

    if checks["target_exists"]:
        print_pass("target_exists")
    else:
        print_fail("target_exists")

    # -------------------------------------------------------------
    # Identifier
    # -------------------------------------------------------------

    checks["identifier_exists"] = (
        IDENTIFIER_COLUMN in df.columns
    )

    if checks["identifier_exists"]:
        print_pass("identifier_exists")
    else:
        print_fail(
            "identifier_exists"
        )

    # -------------------------------------------------------------
    # Stable features
    # -------------------------------------------------------------

    missing_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    checks["stable_features_exist"] = (
        len(missing_features) == 0
    )

    if checks["stable_features_exist"]:
        print_pass(
            "stable_features_exist"
        )
    else:
        print_fail(
            "stable_features_exist",
            f"missing={missing_features}",
        )

    # -------------------------------------------------------------
    # Target values
    # -------------------------------------------------------------

    target_values_valid = False

    if TARGET_COLUMN in df.columns:

        observed_values = set(
            df[TARGET_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        target_values_valid = (
            observed_values
            .issubset(
                EXPECTED_TARGET_VALUES
            )
        )

    checks["target_values_valid"] = (
        target_values_valid
    )

    if target_values_valid:
        print_pass(
            "target_values_valid"
        )
    else:
        print_fail(
            "target_values_valid"
        )

    # -------------------------------------------------------------
    # Missing cells
    # -------------------------------------------------------------

    checks["no_missing_cells"] = (
        not df.isna().any().any()
    )

    if checks["no_missing_cells"]:
        print_pass(
            "no_missing_cells"
        )
    else:
        print_fail(
            "no_missing_cells"
        )

    # -------------------------------------------------------------
    # Identifier uniqueness
    # -------------------------------------------------------------

    identifier_unique = False

    if IDENTIFIER_COLUMN in df.columns:

        identifier_unique = (
            df[IDENTIFIER_COLUMN]
            .notna()
            .all()
            and df[IDENTIFIER_COLUMN]
            .is_unique
        )

    checks["identifier_unique"] = (
        identifier_unique
    )

    if identifier_unique:
        print_pass(
            "identifier_unique"
        )
    else:
        print_fail(
            "identifier_unique"
        )

    # -------------------------------------------------------------
    # Stable feature count
    # -------------------------------------------------------------

    checks["stable_feature_count"] = (
        len(STABLE_FEATURES) == 10
    )

    if checks["stable_feature_count"]:
        print_pass(
            "stable_feature_count"
        )
    else:
        print_fail(
            "stable_feature_count"
        )

    # -------------------------------------------------------------
    # Canonical SHA-256
    # -------------------------------------------------------------

    actual_sha256 = sha256_file(
        DATASET_PATH
    )

    checks["canonical_sha256"] = (
        actual_sha256
        == EXPECTED_CANONICAL_SHA256
    )

    if checks["canonical_sha256"]:
        print_pass(
            "canonical_sha256"
        )
    else:
        print_fail(
            "canonical_sha256",
            f"expected={EXPECTED_CANONICAL_SHA256} "
            f"actual={actual_sha256}",
        )

    # -------------------------------------------------------------
    # Target prevalence
    # -------------------------------------------------------------

    target_prevalence = None

    if TARGET_COLUMN in df.columns:

        target_prevalence = float(
            df[TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .eq("Yes")
            .mean()
        )

    print()

    print(
        f"Identifier:           "
        f"{IDENTIFIER_COLUMN}"
    )

    print(
        f"Stable features:      "
        f"{len(STABLE_FEATURES)}"
    )

    print(
        f"Numerical features:   "
        f"{len(NUMERICAL_FEATURES)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    if target_prevalence is not None:

        print(
            f"Target prevalence:    "
            f"{target_prevalence:.2%}"
        )

    context = {
        "rows": len(df),
        "columns": len(df.columns),
        "identifier": IDENTIFIER_COLUMN,
        "target": TARGET_COLUMN,
        "stable_features": STABLE_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_prevalence": target_prevalence,
        "sha256": actual_sha256,
        "checks": checks,
    }

    return df, context


# =====================================================================
# EVIDENCE AVAILABILITY
# =====================================================================

def validate_evidence_availability() -> Dict[str, Any]:

    print_header(
        "LOADING EVALUATION EVIDENCE"
    )

    evidence: Dict[str, Any] = {}

    # -----------------------------------------------------------------
    # CURRENT REQUIRED EVIDENCE
    # -----------------------------------------------------------------

    required = {

        "calibrated_final_validation": (
            CALIBRATED_FINAL_VALIDATION_DIR,
            [
                "calibrated_final_validation_stable_report.json",
                "calibrated_final_model_comparison.csv",
            ],
        ),

        "calibrated_business_cost": (
            CALIBRATED_BUSINESS_COST_DIR,
            [
                "calibrated_business_cost_analysis_report.json",
                "calibrated_business_cost_scenario_summary.csv",
            ],
        ),

        "calibrated_threshold_optimization": (
            CALIBRATED_THRESHOLD_DIR,
            [
                "calibrated_threshold_optimization_stable_report.json",
                "calibrated_threshold_results.csv",
            ],
        ),

        "calibrated_deployment_decision": (
            CALIBRATED_DEPLOYMENT_DIR,
            [
                "calibrated_deployment_decision_analysis_report.json",
                "calibrated_deployment_decision_evidence.csv",
            ],
        ),

        "deployment_readiness": (
            DEPLOYMENT_READINESS_DIR,
            [
                "deployment_readiness_report.json",
                "deployment_readiness_summary.txt",
            ],
        ),

        "calibration_analysis": (
            CALIBRATION_ANALYSIS_DIR,
            [
                "calibration_analysis_stable_report.json",
            ],
        ),

        "deployment_decision": (
            DEPLOYMENT_DECISION_DIR,
            [
                "deployment_decision_analysis_report.json",
                "deployment_decision_evidence.csv",
            ],
        ),

        "business_cost": (
            BUSINESS_COST_DIR,
            [
                "business_cost_analysis_report.json",
                "business_cost_scenario_summary.csv",
            ],
        ),

        "business_threshold": (
            BUSINESS_THRESHOLD_DIR,
            [
                "business_threshold_analysis_report.json",
                "business_threshold_results.csv",
            ],
        ),
    }

    for name, (
        directory,
        filenames,
    ) in required.items():

        found = first_existing(
            directory,
            filenames,
        )

        passed = found is not None

        evidence[name] = {
            "required": True,
            "exists": passed,
            "directory": str(directory),
            "file": (
                str(found)
                if found is not None
                else None
            ),
        }

        if passed:
            print_pass(name)
        else:
            print_fail(
                name,
                f"missing={directory}",
            )

    # -----------------------------------------------------------------
    # LEGACY MODEL SELECTION
    # -----------------------------------------------------------------
    #
    # This is intentionally NON-BLOCKING.
    #
    # The existing final_model_selection.json describes the earlier
    # Logistic Regression / 24-feature pipeline.
    #
    # The current production candidate is the validated calibrated
    # Random Forest / stable-feature pipeline.
    # -----------------------------------------------------------------

    legacy_path = (
        LEGACY_FINAL_MODEL_SELECTION_DIR
        / "final_model_selection.json"
    )

    legacy_exists = legacy_path.exists()

    evidence["final_model_selection"] = {
        "required": False,
        "exists": legacy_exists,
        "blocking": False,
        "directory": str(
            LEGACY_FINAL_MODEL_SELECTION_DIR
        ),
        "file": (
            str(legacy_path)
            if legacy_exists
            else None
        ),
        "classification": "legacy_supporting_evidence",
    }

    if legacy_exists:

        print_pass(
            "final_model_selection "
            "(legacy/supporting)"
        )

    else:

        print_warn(
            "final_model_selection "
            "(legacy/supporting)",
            "not present; non-blocking",
        )

    return evidence


# =====================================================================
# CALIBRATED FINAL VALIDATION
# =====================================================================

def validate_calibrated_final_validation() -> Dict[str, Any]:

    print_header(
        "CALIBRATED FINAL VALIDATION CHECK"
    )

    comparison_path = (
        CALIBRATED_FINAL_VALIDATION_DIR
        / "calibrated_final_model_comparison.csv"
    )

    result: Dict[str, Any] = {
        "available": False,
        "checks": {},
        "values": {},
        "path": str(comparison_path),
    }

    if not comparison_path.exists():

        print_fail(
            "calibrated_final_model_comparison_exists"
        )

        return result

    comparison = load_csv(
        comparison_path
    )

    if comparison is None or comparison.empty:

        print_fail(
            "calibrated_final_model_comparison_readable"
        )

        return result

    print_pass(
        "calibrated_final_report_readable"
    )

    print_pass(
        "calibrated_final_model_comparison_exists"
    )

    method_column = find_column(
        comparison,
        ["method"],
    )

    if method_column is None:

        print_fail(
            "method_column"
        )

        return result

    sigmoid_rows = comparison[
        comparison[method_column]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains(
            "sigmoid",
            regex=False,
        )
    ]

    if sigmoid_rows.empty:

        print_fail(
            "sigmoid_calibration_row_exists"
        )

        return result

    print_pass(
        "sigmoid_calibration_row_exists"
    )

    row = sigmoid_rows.iloc[0]

    result["available"] = True

    metric_map = {

        "roc_auc": [
            "roc_auc",
        ],

        "pr_auc": [
            "pr_auc",
        ],

        "brier_score": [
            "brier_score",
        ],

        "log_loss": [
            "log_loss",
        ],

        "expected_calibration_error": [
            "expected_calibration_error",
            "ece",
        ],

        "threshold": [
            "threshold",
        ],

        "f1": [
            "f1",
        ],

        "precision": [
            "precision",
        ],

        "recall": [
            "recall",
        ],

        "specificity": [
            "specificity",
        ],

        "balanced_accuracy": [
            "balanced_accuracy",
        ],

        "predicted_positive_percent": [
            "predicted_positive_percent",
        ],

        "flagged_per_1000": [
            "flagged_per_1000",
        ],
    }

    for metric, candidates in metric_map.items():

        column = find_column(
            comparison,
            candidates,
        )

        if column is None:
            continue

        value = safe_float(
            row[column]
        )

        if value is not None:

            result["values"][metric] = value

    # -----------------------------------------------------------------
    # Exact evidence checks
    # -----------------------------------------------------------------

    checks = result["checks"]

    checks["threshold"] = values_close(
        result["values"].get("threshold"),
        PRODUCTION_THRESHOLD,
        0.001,
    )

    checks["brier"] = values_close(
        result["values"].get("brier_score"),
        EXPECTED_CALIBRATION["brier_score"],
        0.01,
    )

    checks["log_loss"] = values_close(
        result["values"].get("log_loss"),
        EXPECTED_CALIBRATION["log_loss"],
        0.01,
    )

    checks["ece"] = values_close(
        result["values"].get(
            "expected_calibration_error"
        ),
        EXPECTED_CALIBRATION[
            "expected_calibration_error"
        ],
        0.01,
    )

    checks["precision"] = values_close(
        result["values"].get("precision"),
        EXPECTED_OPERATING_POINT[
            "precision"
        ],
        0.03,
    )

    checks["recall"] = values_close(
        result["values"].get("recall"),
        EXPECTED_OPERATING_POINT[
            "recall"
        ],
        0.03,
    )

    checks["specificity"] = values_close(
        result["values"].get("specificity"),
        EXPECTED_OPERATING_POINT[
            "specificity"
        ],
        0.03,
    )

    checks["flagged_volume"] = values_close(
        result["values"].get(
            "flagged_per_1000"
        ),
        EXPECTED_OPERATING_POINT[
            "flagged_per_1000"
        ],
        20.0,
    )

    for name, passed in checks.items():

        if passed:
            print_pass(name)
        else:
            print_fail(name)

    return result


# =====================================================================
# CALIBRATION QUALITY
# =====================================================================

def validate_calibration_quality(
    final_validation: Dict[str, Any],
) -> Dict[str, Any]:

    print_header(
        "CALIBRATION QUALITY VALIDATION"
    )

    values = final_validation.get(
        "values",
        {},
    )

    brier = values.get(
        "brier_score"
    )

    log_loss = values.get(
        "log_loss"
    )

    ece = values.get(
        "expected_calibration_error"
    )

    checks = {

        "brier_below_0_20": (
            brier is not None
            and brier < MAX_BRIER_SCORE
        ),

        "log_loss_below_0_55": (
            log_loss is not None
            and log_loss < MAX_LOG_LOSS
        ),

        "ece_below_0_10": (
            ece is not None
            and ece < MAX_ECE
        ),
    }

    for name, passed in checks.items():

        if passed:
            print_pass(name)
        else:
            print_fail(name)

    return {
        "checks": checks,
        "brier_score": brier,
        "log_loss": log_loss,
        "expected_calibration_error": ece,
    }


# =====================================================================
# PRODUCTION OPERATING POINT
# =====================================================================

def validate_operating_point(
    final_validation: Dict[str, Any],
) -> Dict[str, Any]:

    print_header(
        "PRODUCTION OPERATING POINT VALIDATION"
    )

    values = final_validation.get(
        "values",
        {},
    )

    threshold = values.get(
        "threshold"
    )

    precision = values.get(
        "precision"
    )

    recall = values.get(
        "recall"
    )

    specificity = values.get(
        "specificity"
    )

    flagged_per_1000 = values.get(
        "flagged_per_1000"
    )

    checks = {

        "threshold_is_0_25": values_close(
            threshold,
            PRODUCTION_THRESHOLD,
            0.001,
        ),

        "precision_matches_evidence": values_close(
            precision,
            EXPECTED_OPERATING_POINT[
                "precision"
            ],
            0.03,
        ),

        "recall_matches_evidence": values_close(
            recall,
            EXPECTED_OPERATING_POINT[
                "recall"
            ],
            0.03,
        ),

        "specificity_matches_evidence": values_close(
            specificity,
            EXPECTED_OPERATING_POINT[
                "specificity"
            ],
            0.03,
        ),

        "flagged_volume_matches_evidence": values_close(
            flagged_per_1000,
            EXPECTED_OPERATING_POINT[
                "flagged_per_1000"
            ],
            20.0,
        ),
    }

    for name, passed in checks.items():

        if passed:
            print_pass(name)
        else:
            print_fail(name)

    return {
        "checks": checks,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "flagged_per_1000": flagged_per_1000,
    }


# =====================================================================
# BUSINESS COST VALIDATION
# =====================================================================

def validate_business_cost() -> Dict[str, Any]:

    print_header(
        "BUSINESS COST EVIDENCE VALIDATION"
    )

    result = {
        "available": False,
        "scenario_count": 0,
        "thresholds": [],
        "threshold_varies_by_cost": False,
        "checks": {},
        "path": None,
    }

    candidates = [

        CALIBRATED_BUSINESS_COST_DIR
        / "calibrated_business_cost_scenario_summary.csv",

        CALIBRATED_BUSINESS_COST_DIR
        / "calibrated_business_cost_scenarios.csv",

        CALIBRATED_BUSINESS_COST_DIR
        / "business_cost_scenario_summary.csv",
    ]

    scenario_path = next(
        (
            path
            for path in candidates
            if path.exists()
        ),
        None,
    )

    if scenario_path is None:

        print_fail(
            "business_cost_scenario_summary_exists"
        )

        return result

    scenarios = load_csv(
        scenario_path
    )

    if scenarios is None or scenarios.empty:

        print_fail(
            "business_cost_scenario_summary_readable"
        )

        return result

    result["available"] = True
    result["path"] = str(
        scenario_path
    )

    result["scenario_count"] = int(
        len(scenarios)
    )

    print_pass(
        "business_cost_scenario_summary_exists"
    )

    threshold_column = find_column(
        scenarios,
        [
            "recommended_threshold",
            "threshold",
        ],
    )

    if threshold_column is not None:

        thresholds = []

        for value in scenarios[
            threshold_column
        ]:

            numeric = safe_float(
                value
            )

            if numeric is not None:
                thresholds.append(
                    numeric
                )

        result["thresholds"] = thresholds

        result[
            "threshold_varies_by_cost"
        ] = (
            len(
                set(
                    round(
                        value,
                        4,
                    )
                    for value in thresholds
                )
            )
            > 1
        )

    checks = result["checks"]

    checks[
        "four_or_more_scenarios"
    ] = (
        result["scenario_count"] >= 4
    )

    checks[
        "threshold_varies_by_cost"
    ] = result[
        "threshold_varies_by_cost"
    ]

    for name, passed in checks.items():

        if passed:
            print_pass(name)
        else:
            print_fail(name)

    return result


# =====================================================================
# BUSINESS THRESHOLD ANALYSIS
# =====================================================================

def validate_business_threshold() -> Dict[str, Any]:

    result = {
        "available": False,
        "path": None,
        "checks": {},
    }

    report_path = (
        BUSINESS_THRESHOLD_DIR
        / "business_threshold_analysis_report.json"
    )

    results_path = (
        BUSINESS_THRESHOLD_DIR
        / "business_threshold_results.csv"
    )

    report = load_json(
        report_path
    )

    results = load_csv(
        results_path
    )

    if report is not None:

        result["available"] = True
        result["path"] = str(
            report_path
        )

    elif results is not None and not results.empty:

        result["available"] = True
        result["path"] = str(
            results_path
        )

    if result["available"]:

        result["checks"][
            "business_threshold_evidence_exists"
        ] = True

    else:

        result["checks"][
            "business_threshold_evidence_exists"
        ] = False

    return result


# =====================================================================
# DEPLOYMENT READINESS
# =====================================================================

def validate_deployment_readiness() -> Dict[str, Any]:

    report_path = (
        DEPLOYMENT_READINESS_DIR
        / "deployment_readiness_report.json"
    )

    summary_path = (
        DEPLOYMENT_READINESS_DIR
        / "deployment_readiness_summary.txt"
    )

    report = load_json(
        report_path
    )

    summary_exists = (
        summary_path.exists()
    )

    result = {
        "available": report is not None,
        "report": report,
        "report_path": (
            str(report_path)
            if report is not None
            else None
        ),
        "summary_exists": summary_exists,
        "checks": {},
    }

    return result


# =====================================================================
# GOVERNANCE VALIDATION
# =====================================================================

def validate_governance() -> Dict[str, Any]:

    print_header(
        "DEPLOYMENT GOVERNANCE VALIDATION"
    )

    checks = {

        "business_threshold_approval_required":
            REQUIRE_BUSINESS_THRESHOLD_APPROVAL,

        "intervention_capacity_required":
            REQUIRE_INTERVENTION_CAPACITY_APPROVAL,

        "decision_support_only":
            DECISION_SUPPORT_ONLY,

        "no_automatic_employment_decisions":
            NO_AUTOMATIC_EMPLOYMENT_DECISIONS,

        "post_deployment_monitoring":
            REQUIRE_POST_DEPLOYMENT_MONITORING,

        "threshold_reassessment":
            REQUIRE_THRESHOLD_REASSESSMENT,
    }

    descriptions = {

        "business_threshold_approval_required":
            "Business owner approval is required for "
            "the final intervention threshold.",

        "intervention_capacity_required":
            "Intervention capacity must be explicitly "
            "approved before production use.",

        "decision_support_only":
            "Predictions are restricted to decision-support use.",

        "no_automatic_employment_decisions":
            "Predictions must not make automatic employment decisions.",

        "post_deployment_monitoring":
            "Calibration and classification performance "
            "must be monitored after deployment.",

        "threshold_reassessment":
            "Threshold must be reassessed when validated "
            "costs or intervention capacity change.",
    }

    for name, passed in checks.items():

        if passed:
            print_pass(name)
        else:
            print_fail(name)

    return {
        "checks": checks,
        "descriptions": descriptions,
    }


# =====================================================================
# TECHNICAL BLOCKING LOGIC
# =====================================================================

def determine_status(
    dataset: Dict[str, Any],
    evidence: Dict[str, Any],
    final_validation: Dict[str, Any],
    calibration_quality: Dict[str, Any],
    operating_point: Dict[str, Any],
    business_cost: Dict[str, Any],
    business_threshold: Dict[str, Any],
    deployment_readiness: Dict[str, Any],
    governance: Dict[str, Any],
) -> Tuple[
    str,
    List[str],
    List[str],
]:

    blocking_reasons: List[str] = []

    conditions: List[str] = []

    # =================================================================
    # DATASET TECHNICAL GATES
    # =================================================================

    dataset_checks = dataset["checks"]

    critical_dataset_checks = [

        "file_exists",
        "expected_rows",
        "expected_columns",
        "target_exists",
        "identifier_exists",
        "stable_features_exist",
        "target_values_valid",
        "no_missing_cells",
        "identifier_unique",
        "stable_feature_count",
        "canonical_sha256",
    ]

    for check in critical_dataset_checks:

        if not dataset_checks.get(
            check,
            False,
        ):

            blocking_reasons.append(
                f"Canonical dataset check failed: "
                f"{check}"
            )

    # =================================================================
    # CURRENT EVIDENCE AVAILABILITY
    # =================================================================

    required_current_evidence = [

        "calibrated_final_validation",
        "calibrated_business_cost",
        "calibrated_threshold_optimization",
        "calibrated_deployment_decision",
        "deployment_readiness",
        "calibration_analysis",
        "deployment_decision",
        "business_cost",
        "business_threshold",
    ]

    for name in required_current_evidence:

        if not evidence.get(
            name,
            {},
        ).get(
            "exists",
            False,
        ):

            blocking_reasons.append(
                f"Required current evidence unavailable: "
                f"{name}"
            )

    # =================================================================
    # FINAL CALIBRATED VALIDATION
    # =================================================================

    if not final_validation.get(
        "available",
        False,
    ):

        blocking_reasons.append(
            "Calibrated final validation evidence "
            "is unavailable."
        )

    else:

        final_checks = final_validation.get(
            "checks",
            {},
        )

        for check_name in [

            "threshold",
            "brier",
            "log_loss",
            "ece",
            "precision",
            "recall",
            "specificity",
            "flagged_volume",

        ]:

            if final_checks.get(
                check_name,
                False,
            ) is not True:

                blocking_reasons.append(
                    "Calibrated final validation "
                    f"check failed: {check_name}"
                )

    # =================================================================
    # CALIBRATION QUALITY
    # =================================================================

    calibration_checks = (
        calibration_quality.get(
            "checks",
            {},
        )
    )

    if not calibration_checks.get(
        "brier_below_0_20",
        False,
    ):

        blocking_reasons.append(
            "Brier Score exceeds the configured "
            "calibration-quality limit."
        )

    if not calibration_checks.get(
        "log_loss_below_0_55",
        False,
    ):

        blocking_reasons.append(
            "Log Loss exceeds the configured "
            "calibration-quality limit."
        )

    if not calibration_checks.get(
        "ece_below_0_10",
        False,
    ):

        blocking_reasons.append(
            "Expected calibration error exceeds "
            "the configured calibration-quality limit."
        )

    # =================================================================
    # OPERATING POINT
    # =================================================================

    operating_checks = (
        operating_point.get(
            "checks",
            {},
        )
    )

    for check_name in [

        "threshold_is_0_25",
        "precision_matches_evidence",
        "recall_matches_evidence",
        "specificity_matches_evidence",
        "flagged_volume_matches_evidence",

    ]:

        if operating_checks.get(
            check_name,
            False,
        ) is not True:

            blocking_reasons.append(
                "Production operating point "
                f"check failed: {check_name}"
            )

    # =================================================================
    # BUSINESS COST EVIDENCE
    # =================================================================

    business_checks = (
        business_cost.get(
            "checks",
            {},
        )
    )

    if not business_checks.get(
        "four_or_more_scenarios",
        False,
    ):

        blocking_reasons.append(
            "Business cost evidence does not "
            "contain at least four scenarios."
        )

    if not business_checks.get(
        "threshold_varies_by_cost",
        False,
    ):

        blocking_reasons.append(
            "Business cost evidence does not "
            "demonstrate threshold variation by cost."
        )

    # =================================================================
    # BUSINESS THRESHOLD
    # =================================================================

    if not business_threshold.get(
        "checks",
        {},
    ).get(
        "business_threshold_evidence_exists",
        False,
    ):

        blocking_reasons.append(
            "Business threshold analysis evidence "
            "is unavailable."
        )

    # =================================================================
    # DEPLOYMENT READINESS
    # =================================================================

    if not deployment_readiness.get(
        "available",
        False,
    ):

        blocking_reasons.append(
            "Deployment readiness report is unavailable."
        )

    # =================================================================
    # GOVERNANCE
    # =================================================================

    governance_checks = (
        governance.get(
            "checks",
            {},
        )
    )

    for name, passed in governance_checks.items():

        if not passed:

            blocking_reasons.append(
                "Required governance control failed: "
                f"{name}"
            )

    # =================================================================
    # BUSINESS CONDITIONS
    #
    # IMPORTANT:
    # These are NOT technical blockers.
    # =================================================================

    flagged = operating_point.get(
        "flagged_per_1000"
    )

    if (
        flagged is not None
        and flagged > INTERVENTION_CAPACITY_LIMIT
    ):

        conditions.append(
            "Intervention capacity must be explicitly "
            "approved because approximately "
            f"{flagged:.1f} employees per 1,000 "
            "are flagged."
        )

    precision = operating_point.get(
        "precision"
    )

    if (
        precision is not None
        and precision < MIN_BUSINESS_PRECISION
    ):

        conditions.append(
            "Precision remains below 0.40; the business "
            "owner must explicitly accept the resulting "
            "false-positive burden."
        )

    recall = operating_point.get(
        "recall"
    )

    if (
        recall is not None
        and recall < MIN_BUSINESS_RECALL
    ):

        conditions.append(
            "Recall remains below 0.70; the business "
            "owner must explicitly accept reduced "
            "detection in exchange for improved "
            "specificity and lower intervention volume."
        )

    conditions.append(
        "Business owner approval is required for "
        "the final intervention threshold."
    )

    conditions.append(
        "Predictions must be used for decision support "
        "rather than automatic employment decisions."
    )

    conditions.append(
        "Post-deployment monitoring must track calibration, "
        "ROC-AUC, precision, recall, specificity, intervention "
        "volume, and observed attrition outcomes."
    )

    conditions.append(
        "The production threshold must be reassessed "
        "when validated intervention costs or capacity change."
    )

    # =================================================================
    # FINAL STATUS
    # =================================================================

    if blocking_reasons:

        return (
            "NO-GO",
            blocking_reasons,
            conditions,
        )

    # Technical evidence complete.
    # Business conditions remain.

    return (
        "CONDITIONAL GO",
        [],
        conditions,
    )


# =====================================================================
# EVIDENCE TABLE
# =====================================================================

def build_evidence_table(
    dataset: Dict[str, Any],
    final_validation: Dict[str, Any],
    calibration_quality: Dict[str, Any],
    operating_point: Dict[str, Any],
    business_cost: Dict[str, Any],
    business_threshold: Dict[str, Any],
    deployment_readiness: Dict[str, Any],
    status: str,
) -> pd.DataFrame:

    rows: List[Dict[str, Any]] = []

    def add(
        category: str,
        metric: str,
        value: Any,
        expected: Any,
        passed: bool,
    ) -> None:

        rows.append(
            {
                "category": category,
                "metric": metric,
                "value": value,
                "expected": expected,
                "status": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
        )

    dataset_checks = dataset["checks"]

    add(
        "dataset",
        "rows",
        dataset["rows"],
        EXPECTED_ROWS,
        dataset_checks.get(
            "expected_rows",
            False,
        ),
    )

    add(
        "dataset",
        "columns",
        dataset["columns"],
        EXPECTED_COLUMNS,
        dataset_checks.get(
            "expected_columns",
            False,
        ),
    )

    add(
        "dataset",
        "identifier",
        dataset["identifier"],
        IDENTIFIER_COLUMN,
        dataset_checks.get(
            "identifier_exists",
            False,
        ),
    )

    add(
        "dataset",
        "stable_feature_count",
        len(STABLE_FEATURES),
        10,
        dataset_checks.get(
            "stable_feature_count",
            False,
        ),
    )

    add(
        "dataset",
        "canonical_sha256",
        dataset["sha256"],
        EXPECTED_CANONICAL_SHA256,
        dataset_checks.get(
            "canonical_sha256",
            False,
        ),
    )

    add(
        "model",
        "model",
        MODEL_NAME,
        MODEL_NAME,
        True,
    )

    add(
        "model",
        "calibration",
        CALIBRATION_METHOD,
        CALIBRATION_METHOD,
        True,
    )

    add(
        "model",
        "threshold",
        PRODUCTION_THRESHOLD,
        PRODUCTION_THRESHOLD,
        True,
    )

    values = final_validation.get(
        "values",
        {},
    )

    add(
        "calibration",
        "brier_score",
        values.get(
            "brier_score"
        ),
        EXPECTED_CALIBRATION[
            "brier_score"
        ],
        final_validation.get(
            "checks",
            {},
        ).get(
            "brier",
            False,
        ),
    )

    add(
        "calibration",
        "log_loss",
        values.get(
            "log_loss"
        ),
        EXPECTED_CALIBRATION[
            "log_loss"
        ],
        final_validation.get(
            "checks",
            {},
        ).get(
            "log_loss",
            False,
        ),
    )

    add(
        "calibration",
        "expected_calibration_error",
        values.get(
            "expected_calibration_error"
        ),
        EXPECTED_CALIBRATION[
            "expected_calibration_error"
        ],
        final_validation.get(
            "checks",
            {},
        ).get(
            "ece",
            False,
        ),
    )

    add(
        "operating_point",
        "threshold",
        operating_point.get(
            "threshold"
        ),
        PRODUCTION_THRESHOLD,
        operating_point.get(
            "checks",
            {},
        ).get(
            "threshold_is_0_25",
            False,
        ),
    )

    add(
        "operating_point",
        "precision",
        operating_point.get(
            "precision"
        ),
        EXPECTED_OPERATING_POINT[
            "precision"
        ],
        operating_point.get(
            "checks",
            {},
        ).get(
            "precision_matches_evidence",
            False,
        ),
    )

    add(
        "operating_point",
        "recall",
        operating_point.get(
            "recall"
        ),
        EXPECTED_OPERATING_POINT[
            "recall"
        ],
        operating_point.get(
            "checks",
            {},
        ).get(
            "recall_matches_evidence",
            False,
        ),
    )

    add(
        "operating_point",
        "specificity",
        operating_point.get(
            "specificity"
        ),
        EXPECTED_OPERATING_POINT[
            "specificity"
        ],
        operating_point.get(
            "checks",
            {},
        ).get(
            "specificity_matches_evidence",
            False,
        ),
    )

    add(
        "business",
        "flagged_per_1000",
        operating_point.get(
            "flagged_per_1000"
        ),
        EXPECTED_OPERATING_POINT[
            "flagged_per_1000"
        ],
        operating_point.get(
            "checks",
            {},
        ).get(
            "flagged_volume_matches_evidence",
            False,
        ),
    )

    add(
        "business",
        "scenario_count",
        business_cost.get(
            "scenario_count"
        ),
        ">= 4",
        business_cost.get(
            "checks",
            {},
        ).get(
            "four_or_more_scenarios",
            False,
        ),
    )

    add(
        "business",
        "threshold_varies_by_cost",
        business_cost.get(
            "threshold_varies_by_cost"
        ),
        True,
        business_cost.get(
            "checks",
            {},
        ).get(
            "threshold_varies_by_cost",
            False,
        ),
    )

    add(
        "business",
        "business_threshold_evidence",
        business_threshold.get(
            "available"
        ),
        True,
        business_threshold.get(
            "checks",
            {},
        ).get(
            "business_threshold_evidence_exists",
            False,
        ),
    )

    add(
        "deployment",
        "deployment_readiness",
        deployment_readiness.get(
            "available"
        ),
        True,
        deployment_readiness.get(
            "available",
            False,
        ),
    )

    add(
        "deployment",
        "final_status",
        status,
        "CONDITIONAL GO",
        status == "CONDITIONAL GO",
    )

    return pd.DataFrame(
        rows
    )


# =====================================================================
# JSON REPORT
# =====================================================================

def build_json_report(
    dataset: Dict[str, Any],
    evidence: Dict[str, Any],
    final_validation: Dict[str, Any],
    calibration_quality: Dict[str, Any],
    operating_point: Dict[str, Any],
    business_cost: Dict[str, Any],
    business_threshold: Dict[str, Any],
    deployment_readiness: Dict[str, Any],
    governance: Dict[str, Any],
    status: str,
    blocking_reasons: List[str],
    conditions: List[str],
) -> Dict[str, Any]:

    return {

        "report_name":
            "calibrated_production_candidate_validation",

        "status": status,

        "production_candidate": {

            "model": MODEL_NAME,

            "features": STABLE_FEATURES,

            "feature_count":
                len(STABLE_FEATURES),

            "numerical_features":
                NUMERICAL_FEATURES,

            "categorical_features":
                CATEGORICAL_FEATURES,

            "calibration_method":
                CALIBRATION_METHOD,

            "threshold":
                PRODUCTION_THRESHOLD,

            "n_estimators":
                N_ESTIMATORS,

            "max_features":
                MAX_FEATURES,

            "min_samples_leaf":
                MIN_SAMPLES_LEAF,

            "class_weight":
                CLASS_WEIGHT,
        },

        "dataset": {

            "path":
                str(DATASET_PATH),

            "rows":
                dataset["rows"],

            "columns":
                dataset["columns"],

            "identifier":
                dataset["identifier"],

            "target":
                dataset["target"],

            "target_prevalence":
                dataset["target_prevalence"],

            "sha256":
                dataset["sha256"],

            "stable_features":
                STABLE_FEATURES,
        },

        "evidence_availability":
            evidence,

        "calibrated_final_validation":
            final_validation,

        "calibration_quality":
            calibration_quality,

        "operating_point":
            operating_point,

        "business_cost":
            business_cost,

        "business_threshold":
            business_threshold,

        "deployment_readiness":
            deployment_readiness,

        "governance":
            governance,

        "blocking_reasons":
            blocking_reasons,

        "deployment_conditions":
            conditions,

        "decision": {

            "production_deployment_allowed":
                (
                    status
                    == "CONDITIONAL GO"
                    and not blocking_reasons
                ),

            "automatic_employment_decision_allowed":
                False,

            "business_threshold_approval_required":
                REQUIRE_BUSINESS_THRESHOLD_APPROVAL,

            "intervention_capacity_approval_required":
                REQUIRE_INTERVENTION_CAPACITY_APPROVAL,

            "post_deployment_monitoring_required":
                REQUIRE_POST_DEPLOYMENT_MONITORING,

            "threshold_reassessment_required":
                REQUIRE_THRESHOLD_REASSESSMENT,
        },

        "legacy_model_selection_note": (
            "The historical final_model_selection artifact "
            "describes an earlier Logistic Regression / "
            "24-feature configuration. It is retained as "
            "supporting historical evidence and is not used "
            "as a blocking gate for the current calibrated "
            "Random Forest production candidate."
        ),
    }


# =====================================================================
# SUMMARY REPORT
# =====================================================================

def write_summary(
    dataset: Dict[str, Any],
    final_validation: Dict[str, Any],
    operating_point: Dict[str, Any],
    business_cost: Dict[str, Any],
    status: str,
    blocking_reasons: List[str],
    conditions: List[str],
) -> None:

    values = final_validation.get(
        "values",
        {},
    )

    lines: List[str] = []

    lines.append(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED PRODUCTION CANDIDATE VALIDATION"
    )

    lines.append(
        "=" * 64
    )

    lines.append("")

    lines.append(
        f"Production Candidate Status: {status}"
    )

    lines.append("")

    lines.append(
        "[PRODUCTION CANDIDATE]"
    )

    lines.append(
        f"Model:                 {MODEL_NAME}"
    )

    lines.append(
        f"Features:              {len(STABLE_FEATURES)}"
    )

    lines.append(
        f"Calibration:           {CALIBRATION_METHOD}"
    )

    lines.append(
        f"Threshold:             {PRODUCTION_THRESHOLD:.2f}"
    )

    lines.append("")

    lines.append(
        "[DATASET]"
    )

    lines.append(
        f"Rows:                  {dataset['rows']}"
    )

    lines.append(
        f"Columns:               {dataset['columns']}"
    )

    lines.append(
        f"Identifier:            {dataset['identifier']}"
    )

    lines.append(
        f"Target prevalence:     "
        f"{dataset['target_prevalence']:.2%}"
    )

    lines.append(
        f"SHA-256:               {dataset['sha256']}"
    )

    lines.append("")

    lines.append(
        "[CALIBRATION QUALITY]"
    )

    lines.append(
        f"Brier Score:           "
        f"{values.get('brier_score', float('nan')):.4f}"
    )

    lines.append(
        f"Log Loss:              "
        f"{values.get('log_loss', float('nan')):.4f}"
    )

    lines.append(
        f"Expected Cal. Error:   "
        f"{values.get('expected_calibration_error', float('nan')):.4f}"
    )

    lines.append("")

    lines.append(
        "[OPERATING POINT]"
    )

    lines.append(
        f"Threshold:             "
        f"{operating_point.get('threshold', float('nan')):.2f}"
    )

    lines.append(
        f"F1:                    "
        f"{values.get('f1', float('nan')):.4f}"
    )

    lines.append(
        f"Precision:             "
        f"{operating_point.get('precision', float('nan')):.4f}"
    )

    lines.append(
        f"Recall:                "
        f"{operating_point.get('recall', float('nan')):.4f}"
    )

    lines.append(
        f"Specificity:           "
        f"{operating_point.get('specificity', float('nan')):.4f}"
    )

    lines.append(
        f"Flagged per 1000:      "
        f"{operating_point.get('flagged_per_1000', float('nan')):.1f}"
    )

    lines.append("")

    lines.append(
        "[BUSINESS COST]"
    )

    lines.append(
        f"Scenarios evaluated:  "
        f"{business_cost.get('scenario_count', 0)}"
    )

    lines.append(
        f"Threshold varies by cost: "
        f"{business_cost.get('threshold_varies_by_cost', False)}"
    )

    if business_cost.get(
        "thresholds"
    ):

        lines.append(
            "Threshold range:      "
            f"{min(business_cost['thresholds']):.2f} - "
            f"{max(business_cost['thresholds']):.2f}"
        )

    lines.append("")

    lines.append(
        "[DEPLOYMENT CONDITIONS]"
    )

    if conditions:

        for condition in conditions:
            lines.append(
                f"- {condition}"
            )

    else:

        lines.append(
            "- None."
        )

    lines.append("")

    lines.append(
        "[BLOCKING REASONS]"
    )

    if blocking_reasons:

        for reason in blocking_reasons:
            lines.append(
                f"- {reason}"
            )

    else:

        lines.append(
            "- None."
        )

    lines.append("")

    lines.append(
        "[OVERALL DECISION]"
    )

    if status == "CONDITIONAL GO":

        lines.append(
            "The calibrated Random Forest is technically "
            "suitable as a controlled decision-support "
            "production candidate."
        )

        lines.append(
            "The current calibrated evidence chain is "
            "internally consistent."
        )

        lines.append(
            "Production use remains conditional on business "
            "approval of intervention capacity, threshold "
            "selection, false-positive/false-negative costs, "
            "and governance controls."
        )

        lines.append(
            "The model must not be used for automatic "
            "employment decisions."
        )

    elif status == "GO":

        lines.append(
            "The production candidate passed all configured "
            "technical and governance gates."
        )

    else:

        lines.append(
            "Production deployment is blocked because one "
            "or more technical validation gates failed."
        )

    lines.append("")

    lines.append(
        "[LEGACY EVIDENCE NOTE]"
    )

    lines.append(
        "The historical final_model_selection.json artifact "
        "describes an earlier Logistic Regression / "
        "24-feature configuration."
    )

    lines.append(
        "It is intentionally treated as supporting historical "
        "evidence and does not block the current calibrated "
        "Random Forest candidate."
    )

    SUMMARY_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:

    print(
        "Running calibrated production candidate validation..."
    )

    print(
        "Loading canonical dataset..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    _, dataset = validate_dataset()

    # -----------------------------------------------------------------
    # Production candidate
    # -----------------------------------------------------------------

    print_header(
        "PRODUCTION CANDIDATE"
    )

    print(
        f"Model:                 {MODEL_NAME}"
    )

    print(
        f"Features:              {len(STABLE_FEATURES)}"
    )

    print(
        f"Calibration:           {CALIBRATION_METHOD}"
    )

    print(
        f"Threshold:             "
        f"{PRODUCTION_THRESHOLD:.2f}"
    )

    # -----------------------------------------------------------------
    # Evidence availability
    # -----------------------------------------------------------------

    print_header(
        "LOADING EVALUATION EVIDENCE"
    )

    evidence = (
        validate_evidence_availability()
    )

    # -----------------------------------------------------------------
    # Current calibrated evidence
    # -----------------------------------------------------------------

    final_validation = (
        validate_calibrated_final_validation()
    )

    calibration_quality = (
        validate_calibration_quality(
            final_validation
        )
    )

    operating_point = (
        validate_operating_point(
            final_validation
        )
    )

    business_cost = (
        validate_business_cost()
    )

    business_threshold = (
        validate_business_threshold()
    )

    deployment_readiness = (
        validate_deployment_readiness()
    )

    governance = (
        validate_governance()
    )

    # -----------------------------------------------------------------
    # Final decision
    # -----------------------------------------------------------------

    (
        status,
        blocking_reasons,
        conditions,
    ) = determine_status(

        dataset=dataset,

        evidence=evidence,

        final_validation=final_validation,

        calibration_quality=calibration_quality,

        operating_point=operating_point,

        business_cost=business_cost,

        business_threshold=business_threshold,

        deployment_readiness=deployment_readiness,

        governance=governance,
    )

    # -----------------------------------------------------------------
    # Evidence CSV
    # -----------------------------------------------------------------

    evidence_df = (
        build_evidence_table(

            dataset=dataset,

            final_validation=final_validation,

            calibration_quality=calibration_quality,

            operating_point=operating_point,

            business_cost=business_cost,

            business_threshold=business_threshold,

            deployment_readiness=deployment_readiness,

            status=status,
        )
    )

    evidence_df.to_csv(
        EVIDENCE_CSV,
        index=False,
    )

    # -----------------------------------------------------------------
    # JSON report
    # -----------------------------------------------------------------

    json_report = (
        build_json_report(

            dataset=dataset,

            evidence=evidence,

            final_validation=final_validation,

            calibration_quality=calibration_quality,

            operating_point=operating_point,

            business_cost=business_cost,

            business_threshold=business_threshold,

            deployment_readiness=deployment_readiness,

            governance=governance,

            status=status,

            blocking_reasons=blocking_reasons,

            conditions=conditions,
        )
    )

    with JSON_REPORT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    write_summary(

        dataset=dataset,

        final_validation=final_validation,

        operating_point=operating_point,

        business_cost=business_cost,

        status=status,

        blocking_reasons=blocking_reasons,

        conditions=conditions,
    )

    # =================================================================
    # FINAL CONSOLE REPORT
    # =================================================================

    print_header(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED PRODUCTION CANDIDATE"
    )

    print(
        "[CANDIDATE]"
    )

    print(
        f"Model:                 {MODEL_NAME}"
    )

    print(
        f"Features:              {len(STABLE_FEATURES)}"
    )

    print(
        f"Calibration:           {CALIBRATION_METHOD}"
    )

    print(
        f"Threshold:             "
        f"{PRODUCTION_THRESHOLD:.2f}"
    )

    values = final_validation.get(
        "values",
        {},
    )

    print()

    print(
        "[CALIBRATION QUALITY]"
    )

    print(
        f"Brier Score:           "
        f"{values.get('brier_score', float('nan')):.4f}"
    )

    print(
        f"Log Loss:              "
        f"{values.get('log_loss', float('nan')):.4f}"
    )

    print(
        f"Expected Cal. Error:   "
        f"{values.get('expected_calibration_error', float('nan')):.4f}"
    )

    print()

    print(
        "[OPERATING POINT]"
    )

    print(
        f"Threshold:             "
        f"{operating_point.get('threshold', float('nan')):.2f}"
    )

    print(
        f"Precision:             "
        f"{operating_point.get('precision', float('nan')):.4f}"
    )

    print(
        f"Recall:                "
        f"{operating_point.get('recall', float('nan')):.4f}"
    )

    print(
        f"Specificity:           "
        f"{operating_point.get('specificity', float('nan')):.4f}"
    )

    print(
        f"Flagged per 1000:      "
        f"{operating_point.get('flagged_per_1000', float('nan')):.1f}"
    )

    print()

    print(
        "[BUSINESS COST]"
    )

    print(
        f"Scenarios evaluated:  "
        f"{business_cost.get('scenario_count', 0)}"
    )

    print(
        f"Threshold varies by cost: "
        f"{business_cost.get('threshold_varies_by_cost', False)}"
    )

    print()

    print(
        "[DEPLOYMENT CONDITIONS]"
    )

    if conditions:

        for condition in conditions:

            print(
                f"- {condition}"
            )

    else:

        print(
            "- None."
        )

    print()

    print(
        "=" * 64
    )

    print(
        "[OVERALL STATUS]"
    )

    print(
        "=" * 64
    )

    print(
        "CALIBRATED PRODUCTION CANDIDATE STATUS: "
        f"{status}"
    )

    print()

    print(
        "[OVERALL DECISION]"
    )

    if status == "CONDITIONAL GO":

        print(
            "The calibrated Random Forest is suitable "
            "as a controlled decision-support production "
            "candidate."
        )

        print(
            "The current calibrated evidence chain is "
            "technically consistent."
        )

        print(
            "Production deployment remains conditional "
            "on business approval of intervention capacity, "
            "threshold selection, validated error costs, "
            "and governance requirements."
        )

        print(
            "The model must not be used for automatic "
            "employment decisions."
        )

    elif status == "GO":

        print(
            "The production candidate passed all configured "
            "technical and governance gates."
        )

    else:

        print(
            "Production deployment is blocked because "
            "one or more technical validation conditions "
            "failed."
        )

    if blocking_reasons:

        print()

        print(
            "[FAILED TECHNICAL CONDITIONS]"
        )

        for reason in blocking_reasons:

            print(
                f"- {reason}"
            )

    print()

    print(
        "[OUTPUT]"
    )

    print(
        f"Evidence CSV:         {EVIDENCE_CSV}"
    )

    print(
        f"JSON report:          {JSON_REPORT}"
    )

    print(
        f"Summary report:       {SUMMARY_REPORT}"
    )

    print()

    print(
        "================================================================"
    )

    print(
        "CALIBRATED PRODUCTION CANDIDATE VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()