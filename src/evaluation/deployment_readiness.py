"""
Deployment Readiness Assessment
===============================

Evaluates whether the currently selected employee-attrition model has
sufficient technical and governance evidence to proceed toward deployment.

This module is intentionally evidence-driven. It consumes the existing
artifacts produced by the project's evaluation pipeline and does not
retrain the model.

Run:
    python -m src.evaluation.deployment_readiness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_ROOT = PROJECT_ROOT / "reports" / "signal_analysis"

OUTPUT_DIR = REPORT_ROOT / "deployment_readiness"

OUTPUT_JSON = OUTPUT_DIR / "deployment_readiness_report.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "deployment_readiness_summary.txt"


# ============================================================================
# EXISTING EVIDENCE PATHS
# ============================================================================

FINAL_MODEL_DIR = REPORT_ROOT / "final_model_selection"
FINAL_MODEL_JSON = FINAL_MODEL_DIR / "final_model_selection.json"
FINAL_MODEL_DATASET = FINAL_MODEL_DIR / "dataset_summary.csv"
FINAL_MODEL_THRESHOLD = FINAL_MODEL_DIR / "threshold_comparison.csv"

BUSINESS_THRESHOLD_DIR = REPORT_ROOT / "business_threshold_analysis"
BUSINESS_THRESHOLD_JSON = (
    BUSINESS_THRESHOLD_DIR / "business_threshold_analysis_report.json"
)
BUSINESS_THRESHOLD_RESULTS = (
    BUSINESS_THRESHOLD_DIR / "business_threshold_results.csv"
)
BUSINESS_THRESHOLD_COMPARISON = (
    BUSINESS_THRESHOLD_DIR / "business_threshold_comparison.csv"
)

CALIBRATED_FINAL_DIR = REPORT_ROOT / "calibrated_final_validation_stable"
CALIBRATED_FINAL_JSON = (
    CALIBRATED_FINAL_DIR / "calibrated_final_validation_stable_report.json"
)
CALIBRATED_FINAL_MODEL_COMPARISON = (
    CALIBRATED_FINAL_DIR / "calibrated_final_model_comparison.csv"
)
CALIBRATED_FINAL_THRESHOLD_COMPARISON = (
    CALIBRATED_FINAL_DIR / "calibrated_final_threshold_comparison.csv"
)

CALIBRATED_BUSINESS_DIR = REPORT_ROOT / "calibrated_business_cost_analysis"
CALIBRATED_BUSINESS_JSON = (
    CALIBRATED_BUSINESS_DIR / "calibrated_business_cost_analysis_report.json"
)
CALIBRATED_BUSINESS_SCENARIOS = (
    CALIBRATED_BUSINESS_DIR / "calibrated_business_cost_scenario_summary.csv"
)

CALIBRATED_DEPLOYMENT_DIR = (
    REPORT_ROOT / "calibrated_deployment_decision_analysis"
)
CALIBRATED_DEPLOYMENT_JSON = (
    CALIBRATED_DEPLOYMENT_DIR
    / "calibrated_deployment_decision_analysis_report.json"
)

PRODUCTION_CANDIDATE_DIR = (
    REPORT_ROOT / "calibrated_production_candidate_validation"
)
PRODUCTION_CANDIDATE_JSON = (
    PRODUCTION_CANDIDATE_DIR
    / "calibrated_production_candidate_validation_report.json"
)


# ============================================================================
# HELPERS
# ============================================================================


def print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def print_check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"

    if detail:
        print(f"{status} {name} {detail}")
    else:
        print(f"{status} {name}")


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            value = json.load(f)

        if isinstance(value, dict):
            return value

    except (OSError, json.JSONDecodeError):
        return None

    return None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def recursive_find(
    obj: Any,
    keys: tuple[str, ...],
) -> list[Any]:
    """
    Recursively find values associated with any of the supplied keys.

    This allows the readiness audit to tolerate small differences in JSON
    nesting without inventing evidence.
    """
    found: list[Any] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            normalized = str(key).strip().lower()

            if normalized in keys:
                found.append(value)

            found.extend(recursive_find(value, keys))

    elif isinstance(obj, list):
        for item in obj:
            found.extend(recursive_find(item, keys))

    return found


def first_scalar(
    obj: Any,
    keys: tuple[str, ...],
) -> Any | None:
    values = recursive_find(obj, keys)

    for value in values:
        if isinstance(value, (str, int, float, bool)):
            return value

    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "pass", "passed", "yes", "go"}:
            return True

        if normalized in {"false", "fail", "failed", "no", "no-go"}:
            return False

    return None


def locate_value(
    data: dict[str, Any] | None,
    keys: tuple[str, ...],
) -> Any | None:
    if not data:
        return None

    return first_scalar(data, keys)


# ============================================================================
# EVIDENCE LOADING
# ============================================================================


def load_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}

    evidence["final_model"] = safe_load_json(FINAL_MODEL_JSON)
    evidence["business_threshold"] = safe_load_json(BUSINESS_THRESHOLD_JSON)
    evidence["calibrated_final"] = safe_load_json(CALIBRATED_FINAL_JSON)
    evidence["calibrated_business"] = safe_load_json(CALIBRATED_BUSINESS_JSON)
    evidence["calibrated_deployment"] = safe_load_json(
        CALIBRATED_DEPLOYMENT_JSON
    )
    evidence["production_candidate"] = safe_load_json(
        PRODUCTION_CANDIDATE_JSON
    )

    evidence["final_model_dataset"] = safe_read_csv(FINAL_MODEL_DATASET)
    evidence["final_model_threshold"] = safe_read_csv(
        FINAL_MODEL_THRESHOLD
    )
    evidence["business_threshold_results"] = safe_read_csv(
        BUSINESS_THRESHOLD_RESULTS
    )
    evidence["business_threshold_comparison"] = safe_read_csv(
        BUSINESS_THRESHOLD_COMPARISON
    )
    evidence["calibrated_final_model_comparison"] = safe_read_csv(
        CALIBRATED_FINAL_MODEL_COMPARISON
    )
    evidence["calibrated_final_threshold_comparison"] = safe_read_csv(
        CALIBRATED_FINAL_THRESHOLD_COMPARISON
    )
    evidence["calibrated_business_scenarios"] = safe_read_csv(
        CALIBRATED_BUSINESS_SCENARIOS
    )

    return evidence


# ============================================================================
# EVIDENCE CHECKS
# ============================================================================


def check_final_model_selection(evidence: dict[str, Any]) -> tuple[bool, str]:
    data = evidence["final_model"]

    if data is None:
        return False, f"missing={FINAL_MODEL_JSON}"

    return True, ""


def check_business_threshold(evidence: dict[str, Any]) -> tuple[bool, str]:
    data = evidence["business_threshold"]

    if data is None:
        return False, f"missing={BUSINESS_THRESHOLD_JSON}"

    csv = evidence["business_threshold_results"]

    if csv is None or csv.empty:
        return False, f"missing={BUSINESS_THRESHOLD_RESULTS}"

    return True, ""


def check_calibrated_final(evidence: dict[str, Any]) -> tuple[bool, str]:
    data = evidence["calibrated_final"]

    if data is None:
        return False, f"missing={CALIBRATED_FINAL_JSON}"

    comparison = evidence["calibrated_final_model_comparison"]

    if comparison is None or comparison.empty:
        return False, f"missing={CALIBRATED_FINAL_MODEL_COMPARISON}"

    return True, ""


def check_calibrated_business(evidence: dict[str, Any]) -> tuple[bool, str]:
    data = evidence["calibrated_business"]

    if data is None:
        return False, f"missing={CALIBRATED_BUSINESS_JSON}"

    scenarios = evidence["calibrated_business_scenarios"]

    if scenarios is None or scenarios.empty:
        return False, f"missing={CALIBRATED_BUSINESS_SCENARIOS}"

    return True, ""


def check_calibrated_deployment(
    evidence: dict[str, Any],
) -> tuple[bool, str]:
    data = evidence["calibrated_deployment"]

    if data is None:
        return False, f"missing={CALIBRATED_DEPLOYMENT_JSON}"

    return True, ""


# ============================================================================
# METRIC EXTRACTION
# ============================================================================


def extract_calibrated_metrics(
    evidence: dict[str, Any],
) -> dict[str, float | None]:
    data = evidence["calibrated_final"]

    metrics: dict[str, float | None] = {
        "brier_score": None,
        "log_loss": None,
        "expected_calibration_error": None,
        "roc_auc": None,
        "pr_auc": None,
        "threshold": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "balanced_accuracy": None,
        "predicted_positive_percent": None,
        "flagged_per_1000": None,
    }

    if data:
        aliases = {
            "brier_score": (
                "brier_score",
                "brier",
            ),
            "log_loss": (
                "log_loss",
            ),
            "expected_calibration_error": (
                "expected_calibration_error",
                "ece",
            ),
            "roc_auc": (
                "roc_auc",
                "roc-auc",
            ),
            "pr_auc": (
                "pr_auc",
                "pr-auc",
            ),
            "threshold": (
                "threshold",
            ),
            "f1": (
                "f1",
            ),
            "precision": (
                "precision",
            ),
            "recall": (
                "recall",
            ),
            "specificity": (
                "specificity",
            ),
            "balanced_accuracy": (
                "balanced_accuracy",
            ),
            "predicted_positive_percent": (
                "predicted_positive_percent",
            ),
            "flagged_per_1000": (
                "flagged_per_1000",
            ),
        }

        for metric_name, keys in aliases.items():
            metrics[metric_name] = as_float(
                locate_value(data, keys)
            )

    comparison = evidence["calibrated_final_model_comparison"]

    if comparison is not None and not comparison.empty:
        calibrated_rows = comparison[
            comparison.astype(str)
            .apply(
                lambda col: col.str.contains(
                    "Sigmoid",
                    case=False,
                    na=False,
                )
            )
            .any(axis=1)
        ]

        if not calibrated_rows.empty:
            row = calibrated_rows.iloc[0]

            column_aliases = {
                "brier_score": ["brier_score"],
                "log_loss": ["log_loss"],
                "expected_calibration_error": [
                    "expected_calibration_error"
                ],
                "roc_auc": ["roc_auc"],
                "pr_auc": ["pr_auc"],
                "threshold": ["threshold"],
                "f1": ["f1"],
                "precision": ["precision"],
                "recall": ["recall"],
                "specificity": ["specificity"],
                "balanced_accuracy": ["balanced_accuracy"],
                "predicted_positive_percent": [
                    "predicted_positive_percent"
                ],
                "flagged_per_1000": ["flagged_per_1000"],
            }

            for metric_name, columns in column_aliases.items():
                for column in columns:
                    if column in comparison.columns:
                        value = as_float(row[column])

                        if value is not None:
                            metrics[metric_name] = value
                            break

    return metrics


# ============================================================================
# READINESS LOGIC
# ============================================================================


def build_readiness(
    evidence: dict[str, Any],
    metrics: dict[str, float | None],
) -> tuple[dict[str, Any], list[str], list[str]]:
    checks: list[dict[str, Any]] = []

    blocking_failures: list[str] = []
    conditions: list[str] = []

    # ---------------------------------------------------------------------
    # Technical evidence
    # ---------------------------------------------------------------------

    technical_checks = [
        (
            "final_model_selection",
            *check_final_model_selection(evidence),
        ),
        (
            "business_threshold",
            *check_business_threshold(evidence),
        ),
        (
            "calibrated_final_validation",
            *check_calibrated_final(evidence),
        ),
        (
            "calibrated_business_cost",
            *check_calibrated_business(evidence),
        ),
        (
            "calibrated_deployment_decision",
            *check_calibrated_deployment(evidence),
        ),
    ]

    for name, passed, detail in technical_checks:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "detail": detail,
                "blocking": True,
            }
        )

        if not passed:
            blocking_failures.append(name)

    # ---------------------------------------------------------------------
    # Calibration quality
    # ---------------------------------------------------------------------

    brier = metrics["brier_score"]
    log_loss = metrics["log_loss"]
    ece = metrics["expected_calibration_error"]

    brier_pass = brier is not None and brier < 0.20
    log_loss_pass = log_loss is not None and log_loss < 0.55
    ece_pass = ece is not None and ece < 0.10

    checks.extend(
        [
            {
                "name": "brier_below_0_20",
                "passed": brier_pass,
                "detail": "" if brier_pass else "calibration quality threshold not met",
                "blocking": True,
            },
            {
                "name": "log_loss_below_0_55",
                "passed": log_loss_pass,
                "detail": "" if log_loss_pass else "log-loss threshold not met",
                "blocking": True,
            },
            {
                "name": "ece_below_0_10",
                "passed": ece_pass,
                "detail": "" if ece_pass else "ECE threshold not met",
                "blocking": True,
            },
        ]
    )

    if not brier_pass:
        blocking_failures.append("brier_below_0_20")

    if not log_loss_pass:
        blocking_failures.append("log_loss_below_0_55")

    if not ece_pass:
        blocking_failures.append("ece_below_0_10")

    # ---------------------------------------------------------------------
    # Operating-point conditions
    #
    # These are intentionally NON-BLOCKING because the project evidence
    # already establishes that threshold choice is a business decision.
    # ---------------------------------------------------------------------

    precision = metrics["precision"]
    recall = metrics["recall"]
    flagged = metrics["flagged_per_1000"]

    if precision is not None and precision < 0.40:
        conditions.append(
            "Precision remains below 0.40; false-positive burden "
            "requires explicit business acceptance."
        )

    if recall is not None and recall < 0.70:
        conditions.append(
            "Recall remains below 0.70; reduced detection must be "
            "explicitly accepted by the business owner."
        )

    if flagged is not None and flagged > 400:
        conditions.append(
            f"Intervention capacity must be explicitly approved for "
            f"approximately {flagged:.1f} flags per 1,000 employees."
        )

    # ---------------------------------------------------------------------
    # Governance conditions
    # ---------------------------------------------------------------------

    governance_conditions = [
        "Business owner approval is required for the final intervention threshold.",
        "Predictions must be used for decision support rather than automatic employment decisions.",
        "Post-deployment monitoring must track calibration and classification performance.",
        "Threshold must be reassessed when validated intervention costs or capacity change.",
    ]

    conditions.extend(governance_conditions)

    # ---------------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------------

    if blocking_failures:
        status = "NO-GO"
        decision = (
            "Deployment readiness is blocked because one or more "
            "technical evidence or calibration-quality requirements failed."
        )
    else:
        status = "CONDITIONAL PASS"
        decision = (
            "The technical evidence supports controlled deployment "
            "readiness, subject to business approval of threshold, "
            "intervention capacity, error costs, and governance conditions."
        )

    report = {
        "assessment": "employee_attrition_deployment_readiness",
        "status": status,
        "decision": decision,
        "model": {
            "name": "Random Forest",
            "feature_count": 10,
            "calibration": "Sigmoid / Platt",
            "threshold": metrics["threshold"],
        },
        "metrics": metrics,
        "technical_checks": checks,
        "blocking_failures": blocking_failures,
        "deployment_conditions": conditions,
        "evidence_paths": {
            "final_model_selection": str(FINAL_MODEL_JSON),
            "business_threshold": str(BUSINESS_THRESHOLD_JSON),
            "calibrated_final_validation": str(CALIBRATED_FINAL_JSON),
            "calibrated_business_cost": str(CALIBRATED_BUSINESS_JSON),
            "calibrated_deployment_decision": str(
                CALIBRATED_DEPLOYMENT_JSON
            ),
        },
    }

    return report, blocking_failures, conditions


# ============================================================================
# SUMMARY
# ============================================================================


def build_summary(
    report: dict[str, Any],
) -> str:
    metrics = report["metrics"]

    lines: list[str] = []

    lines.append("EMPLOYEE ATTRITION — DEPLOYMENT READINESS")
    lines.append("=" * 64)
    lines.append("")
    lines.append("[STATUS]")
    lines.append(
        f"Deployment readiness status: {report['status']}"
    )
    lines.append("")

    lines.append("[MODEL]")
    lines.append(
        f"Model:                 {report['model']['name']}"
    )
    lines.append(
        f"Features:              {report['model']['feature_count']}"
    )
    lines.append(
        f"Calibration:           {report['model']['calibration']}"
    )

    threshold = metrics.get("threshold")
    if threshold is not None:
        lines.append(
            f"Threshold:             {threshold:.2f}"
        )

    lines.append("")

    lines.append("[CALIBRATION QUALITY]")

    brier = metrics.get("brier_score")
    log_loss = metrics.get("log_loss")
    ece = metrics.get("expected_calibration_error")

    lines.append(
        f"Brier Score:           "
        f"{brier:.4f}" if brier is not None
        else "Brier Score:           unavailable"
    )

    lines.append(
        f"Log Loss:              "
        f"{log_loss:.4f}" if log_loss is not None
        else "Log Loss:              unavailable"
    )

    lines.append(
        f"Expected Cal. Error:   "
        f"{ece:.4f}" if ece is not None
        else "Expected Cal. Error:   unavailable"
    )

    lines.append("")

    lines.append("[OPERATING POINT]")

    for label, key, fmt in [
        ("Precision", "precision", ".4f"),
        ("Recall", "recall", ".4f"),
        ("Specificity", "specificity", ".4f"),
        ("F1", "f1", ".4f"),
        ("Flagged per 1000", "flagged_per_1000", ".1f"),
    ]:
        value = metrics.get(key)

        if value is None:
            lines.append(f"{label + ':':22} unavailable")
        else:
            lines.append(
                f"{label + ':':22} {value:{fmt}}"
            )

    lines.append("")

    lines.append("[TECHNICAL CHECKS]")

    for check in report["technical_checks"]:
        status = "PASS" if check["passed"] else "FAIL"

        line = f"- {status} {check['name']}"

        if check.get("detail"):
            line += f": {check['detail']}"

        lines.append(line)

    lines.append("")

    lines.append("[DEPLOYMENT CONDITIONS]")

    for condition in report["deployment_conditions"]:
        lines.append(f"- {condition}")

    lines.append("")

    lines.append("[BLOCKING FAILURES]")

    if report["blocking_failures"]:
        for failure in report["blocking_failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("- None")

    lines.append("")

    lines.append("[OVERALL DECISION]")
    lines.append(report["decision"])

    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    print("Running deployment readiness analysis...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_header("LOADING DEPLOYMENT EVIDENCE")

    evidence = load_evidence()

    print_check(
        "final_model_selection",
        evidence["final_model"] is not None,
    )

    print_check(
        "business_threshold",
        evidence["business_threshold"] is not None,
    )

    print_check(
        "calibrated_final_validation",
        evidence["calibrated_final"] is not None,
    )

    print_check(
        "calibrated_business_cost",
        evidence["calibrated_business"] is not None,
    )

    print_check(
        "calibrated_deployment_decision",
        evidence["calibrated_deployment"] is not None,
    )

    metrics = extract_calibrated_metrics(evidence)

    report, blocking_failures, _ = build_readiness(
        evidence,
        metrics,
    )

    print_header(
        "EMPLOYEE ATTRITION — DEPLOYMENT READINESS"
    )

    print("[MODEL]")
    print(
        f"Model:                 "
        f"{report['model']['name']}"
    )
    print(
        f"Features:              "
        f"{report['model']['feature_count']}"
    )
    print(
        f"Calibration:           "
        f"{report['model']['calibration']}"
    )

    if metrics["threshold"] is not None:
        print(
            f"Threshold:             "
            f"{metrics['threshold']:.2f}"
        )

    print()
    print("[CALIBRATION QUALITY]")

    if metrics["brier_score"] is not None:
        print(
            f"Brier Score:           "
            f"{metrics['brier_score']:.4f}"
        )

    if metrics["log_loss"] is not None:
        print(
            f"Log Loss:              "
            f"{metrics['log_loss']:.4f}"
        )

    if metrics["expected_calibration_error"] is not None:
        print(
            f"Expected Cal. Error:   "
            f"{metrics['expected_calibration_error']:.4f}"
        )

    print()
    print("[OPERATING POINT]")

    if metrics["precision"] is not None:
        print(
            f"Precision:             "
            f"{metrics['precision']:.4f}"
        )

    if metrics["recall"] is not None:
        print(
            f"Recall:                "
            f"{metrics['recall']:.4f}"
        )

    if metrics["specificity"] is not None:
        print(
            f"Specificity:           "
            f"{metrics['specificity']:.4f}"
        )

    if metrics["f1"] is not None:
        print(
            f"F1:                    "
            f"{metrics['f1']:.4f}"
        )

    if metrics["flagged_per_1000"] is not None:
        print(
            f"Flagged per 1000:      "
            f"{metrics['flagged_per_1000']:.1f}"
        )

    print()
    print("[DEPLOYMENT CONDITIONS]")

    for condition in report["deployment_conditions"]:
        print(f"- {condition}")

    print()
    print("=" * 64)
    print("[OVERALL STATUS]")
    print("=" * 64)

    print(
        f"DEPLOYMENT READINESS STATUS: "
        f"{report['status']}"
    )

    print()
    print("[OVERALL DECISION]")
    print(report["decision"])

    if blocking_failures:
        print()
        print("[FAILED TECHNICAL CONDITIONS]")

        for failure in blocking_failures:
            print(f"- {failure}")

    # ---------------------------------------------------------------------
    # Write JSON
    # ---------------------------------------------------------------------

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------------
    # Write summary
    # ---------------------------------------------------------------------

    summary = build_summary(report)

    with OUTPUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)

    print()
    print("[OUTPUT]")
    print(
        f"JSON report:          {OUTPUT_JSON}"
    )
    print(
        f"Summary report:       {OUTPUT_SUMMARY}"
    )

    print()
    print("=" * 64)
    print("DEPLOYMENT READINESS ANALYSIS COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()