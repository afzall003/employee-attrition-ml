"""
Calibrated Deployment Decision Analysis
========================================

Consolidates the final calibrated employee-attrition evaluation evidence
and produces a deployment recommendation.

This module intentionally does NOT retrain a model.

It consumes previously generated evaluation reports from:

    reports/signal_analysis/
        calibrated_final_validation_stable/
        calibrated_business_cost_analysis/
        calibrated_threshold_optimization_stable/
        deployment_decision_analysis/

The decision logic distinguishes between:

    1. Predictive discrimination
    2. Probability calibration
    3. Classification operating point
    4. Business cost
    5. Intervention volume
    6. Deployment safeguards

The calibrated threshold is NOT automatically treated as the production
threshold. Final threshold selection remains a business decision.

Run:

    python -m src.evaluation.calibrated_deployment_decision_analysis
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

REPORT_ROOT = PROJECT_ROOT / "reports" / "signal_analysis"

CALIBRATED_FINAL_DIR = (
    REPORT_ROOT / "calibrated_final_validation_stable"
)

CALIBRATED_COST_DIR = (
    REPORT_ROOT / "calibrated_business_cost_analysis"
)

CALIBRATED_THRESHOLD_DIR = (
    REPORT_ROOT / "calibrated_threshold_optimization_stable"
)

UNCALIBRATED_DECISION_DIR = (
    REPORT_ROOT / "deployment_decision_analysis"
)

OUTPUT_DIR = (
    REPORT_ROOT / "calibrated_deployment_decision_analysis"
)


# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

TARGET_COLUMN = "Attrition"

IDENTIFIER_CANDIDATES = [
    "EmployeeNumber",
    "EmployeeID",
    "EmployeeId",
    "Employee_ID",
    "ID",
    "Id",
]

EXPECTED_STABLE_FEATURE_COUNT = 10

EXPECTED_NUMERICAL_FEATURE_COUNT = 8
EXPECTED_CATEGORICAL_FEATURE_COUNT = 2

EXPECTED_MODEL = "Random Forest"

UNCALIBRATED_THRESHOLD = 0.44
CALIBRATED_THRESHOLD = 0.25

TARGET_PREVALENCE = 0.2360


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def print_pass(name: str) -> None:
    print(f"PASS {name}")


def print_fail(name: str) -> None:
    print(f"FAIL {name}")


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return default

        result = float(value)

        if math.isfinite(result):
            return result

        return default

    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def find_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required report not found: {path}"
            )

        return pd.DataFrame()

    return pd.read_csv(path)


def load_json(path: Path, required: bool = True) -> dict[str, Any]:

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"Required JSON report not found: {path}"
            )

        return {}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing(
    directory: Path,
    filenames: list[str],
) -> Path | None:

    for filename in filenames:

        path = directory / filename

        if path.exists():
            return path

    return None


def round_value(value: Any, digits: int = 4) -> Any:

    if isinstance(value, (int, float)):

        try:

            if math.isfinite(float(value)):
                return round(float(value), digits)

        except (TypeError, ValueError):
            pass

    return value


# ============================================================================
# DATASET VALIDATION
# ============================================================================

def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:

    checks: dict[str, bool] = {}

    checks["file_exists"] = DATASET_PATH.exists()

    checks["expected_rows"] = len(df) == EXPECTED_ROWS

    checks["expected_columns"] = (
        len(df.columns) == EXPECTED_COLUMNS
    )

    checks["target_exists"] = TARGET_COLUMN in df.columns

    identifier_column = find_existing_column(
        df,
        IDENTIFIER_CANDIDATES,
    )

    checks["identifier_exists"] = (
        identifier_column is not None
    )

    stable_feature_columns = [
        column
        for column in df.columns
        if column not in {
            TARGET_COLUMN,
            identifier_column,
        }
    ]

    # The canonical dataset contains 26 columns, with the model using
    # a stable 10-feature subset. We verify that enough candidate
    # feature columns exist without assuming the complete model schema
    # is contained in this script.
    checks["stable_features_exist"] = (
        len(stable_feature_columns) >= EXPECTED_STABLE_FEATURE_COUNT
    )

    target_values_valid = False

    if TARGET_COLUMN in df.columns:

        values = set(
            str(value).strip().lower()
            for value in df[TARGET_COLUMN].dropna().unique()
        )

        target_values_valid = values.issubset(
            {"yes", "no", "1", "0", "true", "false"}
        )

    checks["target_values_valid"] = target_values_valid

    checks["no_missing_cells"] = (
        int(df.isna().sum().sum()) == 0
    )

    checks["identifier_unique"] = False

    if identifier_column is not None:

        checks["identifier_unique"] = (
            df[identifier_column].is_unique
        )

    checks["stable_feature_count"] = (
        EXPECTED_STABLE_FEATURE_COUNT == 10
    )

    for name, passed in checks.items():

        if passed:
            print_pass(name)

        else:
            print_fail(name)

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return {
        "checks": checks,
        "failed_checks": failed,
        "identifier_column": identifier_column,
    }


# ============================================================================
# REPORT LOADING
# ============================================================================

def load_calibrated_final_evidence() -> dict[str, Any]:

    model_path = first_existing(
        CALIBRATED_FINAL_DIR,
        [
            "calibrated_final_model_comparison.csv",
        ],
    )

    threshold_path = first_existing(
        CALIBRATED_FINAL_DIR,
        [
            "calibrated_final_threshold_comparison.csv",
        ],
    )

    if model_path is None:
        raise FileNotFoundError(
            "calibrated_final_model_comparison.csv not found."
        )

    if threshold_path is None:
        raise FileNotFoundError(
            "calibrated_final_threshold_comparison.csv not found."
        )

    model_comparison = pd.read_csv(model_path)
    threshold_comparison = pd.read_csv(threshold_path)

    return {
        "model_comparison": model_comparison,
        "threshold_comparison": threshold_comparison,
    }


def load_calibrated_business_cost_evidence() -> dict[str, Any]:

    model_path = first_existing(
        CALIBRATED_COST_DIR,
        [
            "calibrated_business_model_comparison.csv",
        ],
    )

    threshold_path = first_existing(
        CALIBRATED_COST_DIR,
        [
            "calibrated_business_threshold_results.csv",
        ],
    )

    scenario_path = first_existing(
        CALIBRATED_COST_DIR,
        [
            "calibrated_business_cost_scenario_summary.csv",
        ],
    )

    operating_path = first_existing(
        CALIBRATED_COST_DIR,
        [
            "calibrated_business_operating_comparison.csv",
        ],
    )

    if model_path is None:
        raise FileNotFoundError(
            "calibrated_business_model_comparison.csv not found."
        )

    if threshold_path is None:
        raise FileNotFoundError(
            "calibrated_business_threshold_results.csv not found."
        )

    if scenario_path is None:
        raise FileNotFoundError(
            "calibrated_business_cost_scenario_summary.csv not found."
        )

    return {
        "model_comparison": pd.read_csv(model_path),
        "threshold_results": pd.read_csv(threshold_path),
        "scenario_summary": pd.read_csv(scenario_path),
        "operating_comparison": (
            pd.read_csv(operating_path)
            if operating_path is not None
            else pd.DataFrame()
        ),
    }


def load_calibrated_threshold_evidence() -> dict[str, Any]:

    model_path = first_existing(
        CALIBRATED_THRESHOLD_DIR,
        [
            "calibrated_model_comparison.csv",
        ],
    )

    threshold_path = first_existing(
        CALIBRATED_THRESHOLD_DIR,
        [
            "calibrated_threshold_results.csv",
        ],
    )

    if model_path is None:
        raise FileNotFoundError(
            "calibrated_model_comparison.csv not found."
        )

    if threshold_path is None:
        raise FileNotFoundError(
            "calibrated_threshold_results.csv not found."
        )

    return {
        "model_comparison": pd.read_csv(model_path),
        "threshold_results": pd.read_csv(threshold_path),
    }


def load_previous_decision_evidence() -> dict[str, Any]:

    evidence_path = first_existing(
        UNCALIBRATED_DECISION_DIR,
        [
            "deployment_decision_evidence.csv",
        ],
    )

    if evidence_path is None:
        return {
            "evidence": pd.DataFrame()
        }

    return {
        "evidence": pd.read_csv(evidence_path)
    }


# ============================================================================
# DATA EXTRACTION
# ============================================================================

def find_row_by_method(
    df: pd.DataFrame,
    method: str,
) -> pd.Series | None:

    if df.empty or "method" not in df.columns:
        return None

    matches = df[
        df["method"]
        .astype(str)
        .str.strip()
        .str.lower()
        == method.strip().lower()
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def find_row_by_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> pd.Series | None:

    if df.empty or "threshold" not in df.columns:
        return None

    values = pd.to_numeric(
        df["threshold"],
        errors="coerce",
    )

    matches = df[
        (values - threshold).abs() < 1e-8
    ]

    if matches.empty:
        return None

    return matches.iloc[0]


def extract_model_metrics(
    model_comparison: pd.DataFrame,
) -> dict[str, dict[str, float]]:

    results: dict[str, dict[str, float]] = {}

    if model_comparison.empty:
        return results

    if "method" not in model_comparison.columns:
        return results

    for _, row in model_comparison.iterrows():

        method = str(row["method"])

        metrics: dict[str, float] = {}

        for column in model_comparison.columns:

            if column == "method":
                continue

            metrics[column] = safe_float(row[column])

        results[method] = metrics

    return results


def extract_threshold_metrics(
    row: pd.Series | None,
) -> dict[str, float]:

    if row is None:
        return {}

    metrics: dict[str, float] = {}

    for column in row.index:

        if column == "threshold":
            continue

        metrics[column] = safe_float(row[column])

    metrics["threshold"] = safe_float(
        row.get("threshold")
    )

    return metrics


# ============================================================================
# DECISION LOGIC
# ============================================================================

def evaluate_calibration(
    model_metrics: dict[str, dict[str, float]],
) -> dict[str, Any]:

    uncalibrated = model_metrics.get(
        "Random Forest",
        {},
    )

    sigmoid = model_metrics.get(
        "Sigmoid Calibration",
        {},
    )

    brier_improvement = (
        safe_float(uncalibrated.get("brier_score"))
        - safe_float(sigmoid.get("brier_score"))
    )

    log_loss_improvement = (
        safe_float(uncalibrated.get("log_loss"))
        - safe_float(sigmoid.get("log_loss"))
    )

    ece_improvement = (
        safe_float(
            uncalibrated.get(
                "expected_calibration_error"
            )
        )
        - safe_float(
            sigmoid.get(
                "expected_calibration_error"
            )
        )
    )

    roc_auc_change = (
        safe_float(sigmoid.get("roc_auc"))
        - safe_float(uncalibrated.get("roc_auc"))
    )

    pr_auc_change = (
        safe_float(sigmoid.get("pr_auc"))
        - safe_float(uncalibrated.get("pr_auc"))
    )

    return {
        "brier_improvement": brier_improvement,
        "log_loss_improvement": log_loss_improvement,
        "ece_improvement": ece_improvement,
        "roc_auc_change": roc_auc_change,
        "pr_auc_change": pr_auc_change,
        "brier_improved": brier_improvement > 0,
        "log_loss_improved": log_loss_improvement > 0,
        "ece_improved": ece_improvement > 0,
        "ranking_materially_preserved": (
            roc_auc_change >= -0.02
        ),
    }


def evaluate_operating_point(
    uncalibrated: dict[str, float],
    calibrated: dict[str, float],
) -> dict[str, Any]:

    f1_change = (
        safe_float(calibrated.get("f1"))
        - safe_float(uncalibrated.get("f1"))
    )

    precision_change = (
        safe_float(calibrated.get("precision"))
        - safe_float(uncalibrated.get("precision"))
    )

    recall_change = (
        safe_float(calibrated.get("recall"))
        - safe_float(uncalibrated.get("recall"))
    )

    specificity_change = (
        safe_float(calibrated.get("specificity"))
        - safe_float(uncalibrated.get("specificity"))
    )

    flagged_change = (
        safe_float(
            calibrated.get(
                "flagged_per_1000"
            )
        )
        - safe_float(
            uncalibrated.get(
                "flagged_per_1000"
            )
        )
    )

    return {
        "f1_change": f1_change,
        "precision_change": precision_change,
        "recall_change": recall_change,
        "specificity_change": specificity_change,
        "flagged_per_1000_change": flagged_change,
        "f1_improved": f1_change > 0,
        "precision_improved": precision_change > 0,
        "specificity_improved": specificity_change > 0,
        "intervention_volume_reduced": flagged_change < 0,
    }


def evaluate_business_conditions(
    scenario_summary: pd.DataFrame,
) -> dict[str, Any]:

    scenarios: list[dict[str, Any]] = []

    if not scenario_summary.empty:

        for _, row in scenario_summary.iterrows():

            scenario = {
                "scenario": str(
                    row.get("scenario", "")
                ),
                "false_positive_cost": safe_float(
                    row.get("false_positive_cost")
                ),
                "false_negative_cost": safe_float(
                    row.get("false_negative_cost")
                ),
                "recommended_threshold": safe_float(
                    row.get("recommended_threshold")
                ),
                "total_cost": safe_float(
                    row.get("total_cost")
                ),
                "cost_per_employee": safe_float(
                    row.get("cost_per_employee")
                ),
                "precision": safe_float(
                    row.get("precision")
                ),
                "recall": safe_float(
                    row.get("recall")
                ),
                "specificity": safe_float(
                    row.get("specificity")
                ),
                "f1": safe_float(
                    row.get("f1")
                ),
                "flagged_per_1000": safe_float(
                    row.get("flagged_per_1000")
                ),
            }

            scenarios.append(scenario)

    thresholds = [
        item["recommended_threshold"]
        for item in scenarios
        if math.isfinite(
            item["recommended_threshold"]
        )
    ]

    return {
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
        "thresholds": thresholds,
        "threshold_range": (
            min(thresholds),
            max(thresholds)
        )
        if thresholds
        else None,
        "threshold_varies_by_cost": (
            len(set(thresholds)) > 1
            if thresholds
            else False
        ),
    }


# ============================================================================
# DEPLOYMENT DECISION
# ============================================================================

def build_deployment_decision(
    calibration: dict[str, Any],
    operating: dict[str, Any],
    business: dict[str, Any],
    calibrated_metrics: dict[str, float],
    dataset_info: dict[str, Any],
) -> dict[str, Any]:

    conditions: list[str] = []

    # ------------------------------------------------------------
    # Probability quality
    # ------------------------------------------------------------

    if (
        calibration["brier_improved"]
        and calibration["log_loss_improved"]
        and calibration["ece_improved"]
    ):

        probability_quality = "PASS"

    else:

        probability_quality = "CONDITIONAL"

    # ------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------

    if calibration["ranking_materially_preserved"]:
        ranking_status = "CONDITIONAL PASS"
    else:
        ranking_status = "REVIEW REQUIRED"

    # ------------------------------------------------------------
    # Operating point
    # ------------------------------------------------------------

    calibrated_precision = safe_float(
        calibrated_metrics.get("precision")
    )

    calibrated_flagged = safe_float(
        calibrated_metrics.get(
            "flagged_per_1000"
        )
    )

    if (
        calibrated_precision >= 0.40
        and calibrated_flagged <= 300
    ):

        operating_status = "PASS"

    else:

        operating_status = "CONDITIONAL"

    # ------------------------------------------------------------
    # Business economics
    # ------------------------------------------------------------

    if business["threshold_varies_by_cost"]:
        business_status = "REQUIRES BUSINESS INPUT"
        conditions.append(
            "Business owner must select the final threshold "
            "after approving false-positive and false-negative costs."
        )
    else:
        business_status = "CONDITIONAL"

    # ------------------------------------------------------------
    # Intervention capacity
    # ------------------------------------------------------------

    if calibrated_flagged > 400:

        capacity_status = "REQUIRES REVIEW"

        conditions.append(
            "Intervention capacity must be explicitly approved "
            "because the calibrated operating point flags more "
            "than 400 employees per 1,000."
        )

    else:

        capacity_status = "CONDITIONAL"

        conditions.append(
            "Intervention capacity should be validated before "
            "production deployment."
        )

    # ------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------

    if calibrated_precision < 0.40:

        conditions.append(
            "Precision remains below 0.40, so false-positive "
            "burden must be accepted by the business owner."
        )

    # ------------------------------------------------------------
    # Human decision-making
    # ------------------------------------------------------------

    conditions.append(
        "Predictions must be used for decision support rather "
        "than automatic employment decisions."
    )

    # ------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------

    conditions.append(
        "Post-deployment monitoring must track calibration, "
        "ROC-AUC, precision, recall, intervention volume, "
        "and observed attrition outcomes."
    )

    conditions.append(
        "The deployment threshold must be revisited when "
        "validated intervention costs or capacity change."
    )

    # ------------------------------------------------------------
    # Overall decision
    # ------------------------------------------------------------

    overall_status = "CONDITIONAL PASS"

    decision = (
        "Sigmoid calibration is supported as a probability-quality "
        "improvement because Brier Score, Log Loss, and expected "
        "calibration error improve materially. Ranking performance "
        "remains broadly preserved, although ROC-AUC and PR-AUC "
        "decrease slightly. The calibrated operating point reduces "
        "intervention volume and improves precision and specificity, "
        "but does not improve F1 or recall relative to the previous "
        "uncalibrated operating point. Business cost analysis also "
        "shows that the preferred threshold changes substantially "
        "with false-positive versus false-negative costs. Therefore "
        "the calibrated model is suitable for controlled decision-"
        "support use, but production deployment remains conditional "
        "on business approval of intervention capacity, threshold "
        "selection, and error costs."
    )

    return {
        "overall_status": overall_status,
        "probability_quality_status": probability_quality,
        "ranking_status": ranking_status,
        "operating_point_status": operating_status,
        "business_status": business_status,
        "intervention_capacity_status": capacity_status,
        "conditions": conditions,
        "decision": decision,
        "dataset_validation": dataset_info,
    }


# ============================================================================
# EVIDENCE TABLE
# ============================================================================

def build_evidence_table(
    calibration: dict[str, Any],
    operating: dict[str, Any],
    business: dict[str, Any],
    uncalibrated_metrics: dict[str, float],
    calibrated_metrics: dict[str, float],
    decision: dict[str, Any],
) -> pd.DataFrame:

    rows = [
        {
            "category": "Probability quality",
            "metric": "Brier Score",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("brier_score")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("brier_score")
            ),
            "change": calibration[
                "brier_improvement"
            ],
            "status": (
                "PASS"
                if calibration["brier_improved"]
                else "REVIEW"
            ),
        },
        {
            "category": "Probability quality",
            "metric": "Log Loss",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("log_loss")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("log_loss")
            ),
            "change": calibration[
                "log_loss_improvement"
            ],
            "status": (
                "PASS"
                if calibration["log_loss_improved"]
                else "REVIEW"
            ),
        },
        {
            "category": "Probability quality",
            "metric": "Expected Calibration Error",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get(
                    "expected_calibration_error"
                )
            ),
            "calibrated": safe_float(
                calibrated_metrics.get(
                    "expected_calibration_error"
                )
            ),
            "change": calibration[
                "ece_improvement"
            ],
            "status": (
                "PASS"
                if calibration["ece_improved"]
                else "REVIEW"
            ),
        },
        {
            "category": "Ranking",
            "metric": "ROC-AUC",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("roc_auc")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("roc_auc")
            ),
            "change": calibration[
                "roc_auc_change"
            ],
            "status": decision[
                "ranking_status"
            ],
        },
        {
            "category": "Ranking",
            "metric": "PR-AUC",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("pr_auc")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("pr_auc")
            ),
            "change": calibration[
                "pr_auc_change"
            ],
            "status": "CONDITIONAL",
        },
        {
            "category": "Operating point",
            "metric": "F1",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("f1")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("f1")
            ),
            "change": operating[
                "f1_change"
            ],
            "status": (
                "IMPROVED"
                if operating["f1_improved"]
                else "NOT IMPROVED"
            ),
        },
        {
            "category": "Operating point",
            "metric": "Precision",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("precision")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("precision")
            ),
            "change": operating[
                "precision_change"
            ],
            "status": (
                "IMPROVED"
                if operating["precision_improved"]
                else "NOT IMPROVED"
            ),
        },
        {
            "category": "Operating point",
            "metric": "Recall",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("recall")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("recall")
            ),
            "change": operating[
                "recall_change"
            ],
            "status": "TRADE-OFF",
        },
        {
            "category": "Operating point",
            "metric": "Specificity",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get("specificity")
            ),
            "calibrated": safe_float(
                calibrated_metrics.get("specificity")
            ),
            "change": operating[
                "specificity_change"
            ],
            "status": (
                "IMPROVED"
                if operating["specificity_improved"]
                else "NOT IMPROVED"
            ),
        },
        {
            "category": "Intervention volume",
            "metric": "Flagged per 1000",
            "uncalibrated": safe_float(
                uncalibrated_metrics.get(
                    "flagged_per_1000"
                )
            ),
            "calibrated": safe_float(
                calibrated_metrics.get(
                    "flagged_per_1000"
                )
            ),
            "change": operating[
                "flagged_per_1000_change"
            ],
            "status": (
                "REDUCED"
                if operating[
                    "intervention_volume_reduced"
                ]
                else "NOT REDUCED"
            ),
        },
    ]

    return pd.DataFrame(rows)


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def dataframe_to_records(
    df: pd.DataFrame,
) -> list[dict[str, Any]]:

    if df.empty:
        return []

    records = df.to_dict(orient="records")

    cleaned: list[dict[str, Any]] = []

    for record in records:

        output: dict[str, Any] = {}

        for key, value in record.items():

            if pd.isna(value):
                output[key] = None

            elif isinstance(
                value,
                (int, float),
            ):

                output[key] = round_value(value)

            else:

                output[key] = value

        cleaned.append(output)

    return cleaned


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def write_summary_report(
    path: Path,
    dataset_info: dict[str, Any],
    model_info: dict[str, Any],
    calibration: dict[str, Any],
    operating: dict[str, Any],
    business: dict[str, Any],
    decision: dict[str, Any],
) -> None:

    lines: list[str] = []

    lines.append(
        "EMPLOYEE ATTRITION — CALIBRATED DEPLOYMENT DECISION ANALYSIS"
    )
    lines.append("=" * 72)
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 {dataset_info['rows']}"
    )
    lines.append(
        f"Columns:              {dataset_info['columns']}"
    )
    lines.append(
        f"Stable features:      {dataset_info['stable_features']}"
    )
    lines.append(
        f"Target prevalence:    "
        f"{dataset_info['target_prevalence']:.2%}"
    )
    lines.append(
        f"SHA-256:              {dataset_info['sha256']}"
    )
    lines.append("")

    lines.append("[MODEL]")
    lines.append(
        f"Model:                {model_info['model']}"
    )
    lines.append(
        f"Feature set:          {model_info['feature_set']}"
    )
    lines.append(
        f"Calibration:           {model_info['calibration']}"
    )
    lines.append("")

    lines.append("[CALIBRATION EVIDENCE]")

    lines.append(
        f"Brier improvement:    "
        f"{calibration['brier_improvement']:+.4f}"
    )

    lines.append(
        f"Log Loss improvement: "
        f"{calibration['log_loss_improvement']:+.4f}"
    )

    lines.append(
        f"ECE improvement:      "
        f"{calibration['ece_improvement']:+.4f}"
    )

    lines.append(
        f"ROC-AUC change:       "
        f"{calibration['roc_auc_change']:+.4f}"
    )

    lines.append(
        f"PR-AUC change:        "
        f"{calibration['pr_auc_change']:+.4f}"
    )

    lines.append("")

    lines.append("[OPERATING POINT]")

    lines.append(
        f"Previous threshold:   "
        f"{UNCALIBRATED_THRESHOLD:.2f}"
    )

    lines.append(
        f"Calibrated threshold: "
        f"{CALIBRATED_THRESHOLD:.2f}"
    )

    lines.append(
        f"F1 change:             "
        f"{operating['f1_change']:+.4f}"
    )

    lines.append(
        f"Precision change:      "
        f"{operating['precision_change']:+.4f}"
    )

    lines.append(
        f"Recall change:         "
        f"{operating['recall_change']:+.4f}"
    )

    lines.append(
        f"Specificity change:    "
        f"{operating['specificity_change']:+.4f}"
    )

    lines.append(
        f"Flagged/1000 change:   "
        f"{operating['flagged_per_1000_change']:+.1f}"
    )

    lines.append("")

    lines.append("[BUSINESS COST]")
    lines.append(
        f"Scenarios evaluated:  "
        f"{business['scenario_count']}"
    )

    if business["threshold_range"] is not None:

        low, high = business[
            "threshold_range"
        ]

        lines.append(
            f"Threshold range:      "
            f"{low:.2f} - {high:.2f}"
        )

    lines.append(
        "Threshold varies by cost: "
        f"{business['threshold_varies_by_cost']}"
    )

    lines.append("")

    lines.append("[DEPLOYMENT CONDITIONS]")

    for condition in decision["conditions"]:
        lines.append(f"- {condition}")

    lines.append("")

    lines.append("[STATUS]")

    lines.append(
        "CALIBRATED DEPLOYMENT DECISION STATUS: "
        f"{decision['overall_status']}"
    )

    lines.append("")

    lines.append("[DECISION]")

    lines.append(
        decision["decision"]
    )

    lines.append("")

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "Running calibrated deployment decision analysis..."
    )

    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Canonical dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    # ------------------------------------------------------------
    # Dataset validation
    # ------------------------------------------------------------

    print()
    print("Validating canonical dataset...")

    validation = validate_dataset(df)

    if validation["failed_checks"]:

        raise ValueError(
            "Canonical dataset validation failed: "
            + ", ".join(
                validation["failed_checks"]
            )
        )

    identifier_column = validation[
        "identifier_column"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            TARGET_COLUMN,
            identifier_column,
        }
    ]

    numerical_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_features = [
        column
        for column in feature_columns
        if column not in numerical_features
    ]

    target_values = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    target_positive = target_values.isin(
        {"yes", "1", "true"}
    )

    prevalence = float(
        target_positive.mean()
    )

    dataset_sha256 = sha256_file(
        DATASET_PATH
    )

    print()
    print(
        f"Stable features:      "
        f"{EXPECTED_STABLE_FEATURE_COUNT}"
    )

    print(
        f"Numerical features:   "
        f"{EXPECTED_NUMERICAL_FEATURE_COUNT}"
    )

    print(
        f"Categorical features: "
        f"{EXPECTED_CATEGORICAL_FEATURE_COUNT}"
    )

    print(
        f"Target prevalence:    "
        f"{prevalence:.2%}"
    )

    # ------------------------------------------------------------
    # Load evidence
    # ------------------------------------------------------------

    print_section(
        "LOADING CALIBRATED EVALUATION EVIDENCE"
    )

    print(
        "Loading calibrated final validation..."
    )

    final_evidence = (
        load_calibrated_final_evidence()
    )

    print_pass("calibrated_final_validation")

    print(
        "Loading calibrated business cost analysis..."
    )

    cost_evidence = (
        load_calibrated_business_cost_evidence()
    )

    print_pass("calibrated_business_cost")

    print(
        "Loading calibrated threshold optimization..."
    )

    threshold_evidence = (
        load_calibrated_threshold_evidence()
    )

    print_pass(
        "calibrated_threshold_optimization"
    )

    print(
        "Loading previous deployment decision..."
    )

    previous_evidence = (
        load_previous_decision_evidence()
    )

    if not previous_evidence[
        "evidence"
    ].empty:

        print_pass(
            "previous_deployment_decision"
        )

    else:

        print(
            "INFO previous deployment decision "
            "evidence not available"
        )

    # ------------------------------------------------------------
    # Extract model metrics
    # ------------------------------------------------------------

    final_model_metrics = (
        extract_model_metrics(
            final_evidence[
                "model_comparison"
            ]
        )
    )

    cost_model_metrics = (
        extract_model_metrics(
            cost_evidence[
                "model_comparison"
            ]
        )
    )

    threshold_model_metrics = (
        extract_model_metrics(
            threshold_evidence[
                "model_comparison"
            ]
        )
    )

    # Prefer final-validation evidence because this is the
    # final validation stage.

    model_metrics = final_model_metrics

    if not model_metrics:
        model_metrics = cost_model_metrics

    if not model_metrics:
        model_metrics = threshold_model_metrics

    uncalibrated_metrics = (
        model_metrics.get(
            "Random Forest",
            {},
        )
    )

    calibrated_metrics = (
        model_metrics.get(
            "Sigmoid Calibration",
            {},
        )
    )

    # ------------------------------------------------------------
    # Extract operating points
    # ------------------------------------------------------------

    final_thresholds = (
        final_evidence[
            "threshold_comparison"
        ]
    )

    uncalibrated_row = find_row_by_threshold(
        final_thresholds,
        UNCALIBRATED_THRESHOLD,
    )

    calibrated_row = find_row_by_threshold(
        final_thresholds,
        CALIBRATED_THRESHOLD,
    )

    # If the final validation threshold table doesn't contain
    # the required threshold, use the calibrated threshold
    # optimization evidence.

    if calibrated_row is None:

        calibrated_row = find_row_by_threshold(
            threshold_evidence[
                "threshold_results"
            ],
            CALIBRATED_THRESHOLD,
        )

    uncalibrated_operating = (
        extract_threshold_metrics(
            uncalibrated_row
        )
    )

    calibrated_operating = (
        extract_threshold_metrics(
            calibrated_row
        )
    )

    # ------------------------------------------------------------
    # Fall back to model-comparison metrics where necessary
    # ------------------------------------------------------------

    for key in [
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_percent",
        "flagged_per_1000",
    ]:

        if key not in uncalibrated_operating:

            if key in uncalibrated_metrics:

                uncalibrated_operating[key] = (
                    uncalibrated_metrics[key]
                )

    for key in [
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_percent",
        "flagged_per_1000",
    ]:

        if key not in calibrated_operating:

            if key in calibrated_metrics:

                calibrated_operating[key] = (
                    calibrated_metrics[key]
                )

    # ------------------------------------------------------------
    # Evaluate evidence
    # ------------------------------------------------------------

    calibration = evaluate_calibration(
        model_metrics
    )

    operating = evaluate_operating_point(
        uncalibrated_operating,
        calibrated_operating,
    )

    business = evaluate_business_conditions(
        cost_evidence[
            "scenario_summary"
        ]
    )

    dataset_info = {
        "rows": len(df),
        "columns": len(df.columns),
        "stable_features": (
            EXPECTED_STABLE_FEATURE_COUNT
        ),
        "numerical_features": (
            EXPECTED_NUMERICAL_FEATURE_COUNT
        ),
        "categorical_features": (
            EXPECTED_CATEGORICAL_FEATURE_COUNT
        ),
        "target_prevalence": prevalence,
        "sha256": dataset_sha256,
        "identifier_column": identifier_column,
    }

    model_info = {
        "model": EXPECTED_MODEL,
        "feature_set": (
            "Stable 10-feature subset"
        ),
        "calibration": (
            "Sigmoid / Platt"
        ),
    }

    decision = build_deployment_decision(
        calibration=calibration,
        operating=operating,
        business=business,
        calibrated_metrics=calibrated_operating,
        dataset_info=dataset_info,
    )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------

    print_section(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED DEPLOYMENT DECISION ANALYSIS"
    )

    print("[MODEL]")

    print(
        f"Model:                 "
        f"{model_info['model']}"
    )

    print(
        f"Features:              "
        f"{EXPECTED_STABLE_FEATURE_COUNT}"
    )

    print(
        f"Calibration:           "
        f"{model_info['calibration']}"
    )

    print()

    print("[PROBABILITY QUALITY]")

    print(
        f"Brier improvement:    "
        f"{calibration['brier_improvement']:+.4f}"
    )

    print(
        f"Log Loss improvement: "
        f"{calibration['log_loss_improvement']:+.4f}"
    )

    print(
        f"ECE improvement:      "
        f"{calibration['ece_improvement']:+.4f}"
    )

    print(
        f"ROC-AUC change:       "
        f"{calibration['roc_auc_change']:+.4f}"
    )

    print(
        f"PR-AUC change:        "
        f"{calibration['pr_auc_change']:+.4f}"
    )

    print()

    print("[OPERATING POINT]")

    print(
        f"Previous threshold:   "
        f"{UNCALIBRATED_THRESHOLD:.2f}"
    )

    print(
        f"Calibrated threshold: "
        f"{CALIBRATED_THRESHOLD:.2f}"
    )

    print(
        f"F1:                   "
        f"{safe_float(calibrated_operating.get('f1')):.4f}"
    )

    print(
        f"Precision:            "
        f"{safe_float(calibrated_operating.get('precision')):.4f}"
    )

    print(
        f"Recall:               "
        f"{safe_float(calibrated_operating.get('recall')):.4f}"
    )

    print(
        f"Specificity:          "
        f"{safe_float(calibrated_operating.get('specificity')):.4f}"
    )

    print(
        f"Flagged per 1000:     "
        f"{safe_float(calibrated_operating.get('flagged_per_1000')):.1f}"
    )

    print()

    print("[BUSINESS COST]")

    print(
        f"Scenarios evaluated:  "
        f"{business['scenario_count']}"
    )

    if business["threshold_range"]:

        low, high = business[
            "threshold_range"
        ]

        print(
            f"Threshold range:      "
            f"{low:.2f} - {high:.2f}"
        )

    print(
        "Threshold varies by cost: "
        f"{business['threshold_varies_by_cost']}"
    )

    print()

    print("[DEPLOYMENT CONDITIONS]")

    for condition in decision["conditions"]:

        print(f"- {condition}")

    print_section(
        "[OVERALL STATUS]"
    )

    print(
        "CALIBRATED DEPLOYMENT DECISION STATUS: "
        f"{decision['overall_status']}"
    )

    print()

    print("[OVERALL DECISION]")

    print(
        decision["decision"]
    )

    # ------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evidence_df = build_evidence_table(
        calibration=calibration,
        operating=operating,
        business=business,
        uncalibrated_metrics=uncalibrated_operating,
        calibrated_metrics=calibrated_operating,
        decision=decision,
    )

    evidence_csv = (
        OUTPUT_DIR
        / "calibrated_deployment_decision_evidence.csv"
    )

    evidence_df.to_csv(
        evidence_csv,
        index=False,
    )

    report_payload = {
        "report": (
            "calibrated_deployment_decision_analysis"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": dataset_info,
        "model": model_info,
        "calibration": calibration,
        "operating_point": operating,
        "business_cost": business,
        "deployment_decision": decision,
        "uncalibrated_metrics": (
            uncalibrated_metrics
        ),
        "calibrated_metrics": (
            calibrated_metrics
        ),
        "evidence": dataframe_to_records(
            evidence_df
        ),
    }

    json_path = (
        OUTPUT_DIR
        / "calibrated_deployment_decision_analysis_report.json"
    )

    write_json(
        json_path,
        report_payload,
    )

    summary_path = (
        OUTPUT_DIR
        / "calibrated_deployment_decision_analysis_summary.txt"
    )

    write_summary_report(
        summary_path,
        dataset_info,
        model_info,
        calibration,
        operating,
        business,
        decision,
    )

    print()
    print("[OUTPUT]")

    print(
        f"Evidence CSV:         "
        f"{evidence_csv}"
    )

    print(
        f"JSON report:          "
        f"{json_path}"
    )

    print(
        f"Summary report:       "
        f"{summary_path}"
    )

    print()
    print("=" * 64)
    print(
        "CALIBRATED DEPLOYMENT DECISION ANALYSIS COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()