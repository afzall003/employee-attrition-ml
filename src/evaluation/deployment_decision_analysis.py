"""
Deployment Decision Analysis
============================

Final decision layer for the employee attrition ML pipeline.

This module consolidates evidence from:

1. Final stable model selection
2. Final stable model validation
3. Business threshold analysis
4. Business cost analysis
5. Deployment readiness audit

It does NOT retrain the model.

Selected model:
    Random Forest

Stable feature set:
    10 validated features

Selected threshold:
    0.44

Decision philosophy:
    The model is evaluated for controlled decision-support deployment.
    A "CONDITIONAL PASS" is returned when predictive performance is
    useful but business capacity, precision, or threshold economics
    still require explicit review.

Outputs:
    deployment_decision_analysis_report.json
    deployment_decision_analysis_summary.txt
    deployment_decision_evidence.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

SIGNAL_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
)

REPORT_DIR = (
    SIGNAL_ANALYSIS_DIR
    / "deployment_decision_analysis"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "Random Forest"

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

THRESHOLD = 0.44

N_ESTIMATORS = 400
MAX_FEATURES = "sqrt"
MIN_SAMPLES_LEAF = 10
CLASS_WEIGHT = "balanced"

TARGET_PREVALENCE = 0.236


# ============================================================
# EXPECTED FINAL VALIDATION RESULTS
# ============================================================

# These values come from the completed evaluation stages.

FINAL_ROC_AUC = 0.6550
FINAL_PR_AUC = 0.3213

FINAL_F1 = 0.4463
FINAL_PRECISION = 0.3196
FINAL_RECALL = 0.7390
FINAL_SPECIFICITY = 0.5141
FINAL_BALANCED_ACCURACY = 0.6266
FINAL_ACCURACY = 0.5672

FINAL_PREDICTED_POSITIVE = 54.56

REPEATED_ROC_AUC = 0.6565
REPEATED_ROC_AUC_STD = 0.0260
REPEATED_ROC_AUC_MIN = 0.6184
REPEATED_ROC_AUC_MAX = 0.7098

BUSINESS_ROC_AUC = 0.6620
BUSINESS_PR_AUC = 0.3258

F1_DEFAULT = 0.4137
F1_IMPROVEMENT = 0.0325


# ============================================================
# DECISION THRESHOLDS
# ============================================================

MIN_USEFUL_ROC_AUC = 0.60
MAX_ACCEPTABLE_ROC_AUC_STD = 0.05

MIN_RECALL_FOR_DETECTION = 0.70

MAX_COMFORTABLE_FLAG_RATE = 50.0
MAX_COMFORTABLE_PRECISION = 0.40

MAX_CONDITIONAL_FLAG_RATE = 60.0


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path) -> Dict:
    """Load JSON report if it exists."""

    if not path.exists():
        return {}

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def find_report(
    directory: Path,
    filename: str,
) -> Path | None:
    """Return report path if available."""

    path = directory / filename

    if path.exists():
        return path

    return None


def add_evidence(
    rows: List[Dict],
    category: str,
    metric: str,
    value,
    assessment: str,
    impact: str,
) -> None:

    rows.append(
        {
            "category": category,
            "metric": metric,
            "value": value,
            "assessment": assessment,
            "impact": impact,
        }
    )


# ============================================================
# SOURCE REPORT LOADING
# ============================================================

def load_source_reports() -> Dict:

    reports = {}

    report_locations = {
        "final_validation": (
            SIGNAL_ANALYSIS_DIR
            / "final_validation_stable"
            / "final_validation_stable_report.json"
        ),
        "final_model_selection": (
            SIGNAL_ANALYSIS_DIR
            / "final_model_selection_stable"
            / "final_model_selection_stable_report.json"
        ),
        "deployment_readiness": (
            SIGNAL_ANALYSIS_DIR
            / "deployment_readiness_audit"
            / "deployment_readiness_audit_report.json"
        ),
        "business_threshold": (
            SIGNAL_ANALYSIS_DIR
            / "business_threshold_analysis"
            / "business_threshold_analysis_report.json"
        ),
        "business_cost": (
            SIGNAL_ANALYSIS_DIR
            / "business_cost_analysis"
            / "business_cost_analysis_report.json"
        ),
    }

    for name, path in report_locations.items():

        reports[name] = {
            "path": str(path),
            "exists": path.exists(),
            "data": load_json(path),
        }

    return reports


# ============================================================
# DECISION ANALYSIS
# ============================================================

def build_decision() -> Dict:

    evidence = []

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    add_evidence(
        evidence,
        "Predictive performance",
        "Repeated ROC-AUC",
        REPEATED_ROC_AUC,
        (
            "Useful predictive separation"
            if REPEATED_ROC_AUC >= MIN_USEFUL_ROC_AUC
            else "Weak predictive separation"
        ),
        "positive"
        if REPEATED_ROC_AUC >= MIN_USEFUL_ROC_AUC
        else "negative",
    )

    add_evidence(
        evidence,
        "Predictive stability",
        "ROC-AUC standard deviation",
        REPEATED_ROC_AUC_STD,
        (
            "Reasonably stable"
            if REPEATED_ROC_AUC_STD
            <= MAX_ACCEPTABLE_ROC_AUC_STD
            else "High variability"
        ),
        "positive"
        if REPEATED_ROC_AUC_STD
        <= MAX_ACCEPTABLE_ROC_AUC_STD
        else "negative",
    )

    add_evidence(
        evidence,
        "Predictive performance",
        "OOF PR-AUC",
        FINAL_PR_AUC,
        "Useful ranking information",
        "positive",
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    add_evidence(
        evidence,
        "Threshold",
        "Selected threshold",
        THRESHOLD,
        "Validated operating threshold",
        "positive",
    )

    add_evidence(
        evidence,
        "Threshold",
        "F1 improvement over 0.50",
        F1_IMPROVEMENT,
        "Threshold improves F1",
        "positive",
    )

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision_assessment = (
        "Precision remains below 0.40; false-positive "
        "burden is substantial."
        if FINAL_PRECISION < MAX_COMFORTABLE_PRECISION
        else "Precision is acceptable."
    )

    add_evidence(
        evidence,
        "Operational risk",
        "Precision",
        FINAL_PRECISION,
        precision_assessment,
        "negative"
        if FINAL_PRECISION < MAX_COMFORTABLE_PRECISION
        else "positive",
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    add_evidence(
        evidence,
        "Detection",
        "Recall",
        FINAL_RECALL,
        (
            "Detection-oriented operating point"
            if FINAL_RECALL >= MIN_RECALL_FOR_DETECTION
            else "Detection target not achieved"
        ),
        "positive"
        if FINAL_RECALL >= MIN_RECALL_FOR_DETECTION
        else "negative",
    )

    # --------------------------------------------------------
    # SPECIFICITY
    # --------------------------------------------------------

    add_evidence(
        evidence,
        "Operational risk",
        "Specificity",
        FINAL_SPECIFICITY,
        (
            "High false-positive rate"
            if FINAL_SPECIFICITY < 0.60
            else "Reasonable specificity"
        ),
        "negative"
        if FINAL_SPECIFICITY < 0.60
        else "positive",
    )

    # --------------------------------------------------------
    # INTERVENTION VOLUME
    # --------------------------------------------------------

    flag_rate_ratio = (
        FINAL_PREDICTED_POSITIVE
        / (TARGET_PREVALENCE * 100)
    )

    add_evidence(
        evidence,
        "Business capacity",
        "Predicted positive rate (%)",
        FINAL_PREDICTED_POSITIVE,
        (
            "More than half of employees would be flagged"
            if FINAL_PREDICTED_POSITIVE > 50
            else "Intervention volume below 50%"
        ),
        "negative"
        if FINAL_PREDICTED_POSITIVE > 50
        else "positive",
    )

    add_evidence(
        evidence,
        "Business capacity",
        "Flagged / prevalence ratio",
        round(flag_rate_ratio, 2),
        (
            "More than twice observed attrition prevalence"
            if flag_rate_ratio > 2
            else "Within two times observed prevalence"
        ),
        "negative"
        if flag_rate_ratio > 2
        else "positive",
    )

    # --------------------------------------------------------
    # DEPLOYMENT READINESS
    # --------------------------------------------------------

    add_evidence(
        evidence,
        "Deployment readiness",
        "Structural reproducibility",
        "PASS",
        "Model configuration is reproducible.",
        "positive",
    )

    add_evidence(
        evidence,
        "Deployment readiness",
        "Business capacity review",
        "REQUIRED",
        (
            "Intervention volume requires explicit "
            "business-capacity review."
        ),
        "negative",
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    critical_positive_conditions = (
        REPEATED_ROC_AUC >= MIN_USEFUL_ROC_AUC
        and REPEATED_ROC_AUC_STD
        <= MAX_ACCEPTABLE_ROC_AUC_STD
        and FINAL_RECALL >= MIN_RECALL_FOR_DETECTION
    )

    critical_business_risks = (
        FINAL_PRECISION < MAX_COMFORTABLE_PRECISION
        or FINAL_PREDICTED_POSITIVE
        > MAX_COMFORTABLE_FLAG_RATE
        or FINAL_SPECIFICITY < 0.60
    )

    if not critical_positive_conditions:
        status = "NOT READY"

        decision = (
            "The model does not currently demonstrate "
            "sufficient predictive performance or stability "
            "for deployment."
        )

    elif critical_business_risks:
        status = "CONDITIONAL PASS"

        decision = (
            "The model demonstrates useful and reasonably "
            "stable predictive separation, but deployment "
            "should remain conditional because the selected "
            "operating point creates a high intervention "
            "volume and a substantial false-positive burden."
        )

    else:
        status = "PASS"

        decision = (
            "The model demonstrates useful predictive "
            "performance and acceptable operational "
            "characteristics for deployment."
        )

    # --------------------------------------------------------
    # DEPLOYMENT CONDITIONS
    # --------------------------------------------------------

    conditions = [
        (
            "Business owner must approve intervention "
            "capacity before production use."
        ),
        (
            "The 0.44 threshold should not be treated as "
            "permanent; threshold selection should be "
            "revisited using observed intervention costs."
        ),
        (
            "Predictions should be used for decision support "
            "rather than automatic employment decisions."
        ),
        (
            "Model performance should be monitored after "
            "deployment using observed attrition outcomes."
        ),
        (
            "Precision, recall, intervention volume and "
            "calibration should be reviewed periodically."
        ),
    ]

    return {
        "status": status,
        "decision": decision,
        "evidence": evidence,
        "conditions": conditions,
        "flag_rate_ratio": flag_rate_ratio,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Running deployment decision analysis..."
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        "Loading canonical dataset..."
    )

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    # --------------------------------------------------------
    # SOURCE REPORTS
    # --------------------------------------------------------

    print()
    print(
        "Loading completed evaluation reports..."
    )

    source_reports = load_source_reports()

    for name, report in source_reports.items():

        print(
            f"{'PASS' if report['exists'] else 'WARN'} "
            f"{name}"
        )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    result = build_decision()

    evidence_df = pd.DataFrame(
        result["evidence"]
    )

    evidence_path = (
        REPORT_DIR
        / "deployment_decision_evidence.csv"
    )

    evidence_df.to_csv(
        evidence_path,
        index=False,
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = {
        "analysis": (
            "deployment_decision_analysis"
        ),
        "dataset": {
            "path": str(DATA_PATH),
            "rows": len(df),
            "columns": len(df.columns),
            "target_prevalence": TARGET_PREVALENCE,
        },
        "model": {
            "name": MODEL_NAME,
            "feature_count": len(STABLE_FEATURES),
            "stable_features": STABLE_FEATURES,
            "n_estimators": N_ESTIMATORS,
            "max_features": MAX_FEATURES,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "class_weight": CLASS_WEIGHT,
        },
        "operating_point": {
            "threshold": THRESHOLD,
            "roc_auc": FINAL_ROC_AUC,
            "pr_auc": FINAL_PR_AUC,
            "f1": FINAL_F1,
            "precision": FINAL_PRECISION,
            "recall": FINAL_RECALL,
            "specificity": FINAL_SPECIFICITY,
            "balanced_accuracy": FINAL_BALANCED_ACCURACY,
            "accuracy": FINAL_ACCURACY,
            "predicted_positive_percent": (
                FINAL_PREDICTED_POSITIVE
            ),
        },
        "stability": {
            "roc_auc_mean": REPEATED_ROC_AUC,
            "roc_auc_std": REPEATED_ROC_AUC_STD,
            "roc_auc_min": REPEATED_ROC_AUC_MIN,
            "roc_auc_max": REPEATED_ROC_AUC_MAX,
        },
        "business": {
            "flagged_per_1000": (
                FINAL_PREDICTED_POSITIVE * 10
            ),
            "flag_prevalence_ratio": (
                result["flag_rate_ratio"]
            ),
            "f1_default_threshold": F1_DEFAULT,
            "f1_improvement": F1_IMPROVEMENT,
        },
        "decision": {
            "status": result["status"],
            "decision": result["decision"],
            "conditions": result["conditions"],
        },
        "source_reports": source_reports,
    }

    json_path = (
        REPORT_DIR
        / "deployment_decision_analysis_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "deployment_decision_analysis_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "DEPLOYMENT DECISION ANALYSIS\n"
        )

        file.write("=" * 64 + "\n\n")

        file.write("[MODEL]\n")
        file.write(
            f"Model:                 {MODEL_NAME}\n"
        )
        file.write(
            f"Features:              "
            f"{len(STABLE_FEATURES)}\n"
        )
        file.write(
            f"Threshold:             "
            f"{THRESHOLD:.2f}\n\n"
        )

        file.write(
            "[PREDICTIVE PERFORMANCE]\n"
        )
        file.write(
            f"ROC-AUC:               "
            f"{FINAL_ROC_AUC:.4f}\n"
        )
        file.write(
            f"PR-AUC:                "
            f"{FINAL_PR_AUC:.4f}\n"
        )
        file.write(
            f"F1:                    "
            f"{FINAL_F1:.4f}\n"
        )
        file.write(
            f"Precision:             "
            f"{FINAL_PRECISION:.4f}\n"
        )
        file.write(
            f"Recall:                "
            f"{FINAL_RECALL:.4f}\n"
        )
        file.write(
            f"Specificity:           "
            f"{FINAL_SPECIFICITY:.4f}\n\n"
        )

        file.write(
            "[STABILITY]\n"
        )
        file.write(
            f"ROC-AUC mean:          "
            f"{REPEATED_ROC_AUC:.4f}\n"
        )
        file.write(
            f"ROC-AUC std:           "
            f"{REPEATED_ROC_AUC_STD:.4f}\n"
        )
        file.write(
            f"ROC-AUC min:           "
            f"{REPEATED_ROC_AUC_MIN:.4f}\n"
        )
        file.write(
            f"ROC-AUC max:           "
            f"{REPEATED_ROC_AUC_MAX:.4f}\n\n"
        )

        file.write(
            "[BUSINESS IMPACT]\n"
        )
        file.write(
            f"Predicted positive:    "
            f"{FINAL_PREDICTED_POSITIVE:.2f}%\n"
        )
        file.write(
            f"Flagged per 1000:      "
            f"{FINAL_PREDICTED_POSITIVE * 10:.1f}\n"
        )
        file.write(
            f"Flag / prevalence:     "
            f"{result['flag_rate_ratio']:.2f}x\n"
        )
        file.write(
            f"F1 improvement:        "
            f"{F1_IMPROVEMENT:.4f}\n\n"
        )

        file.write(
            "[DEPLOYMENT DECISION]\n"
        )
        file.write(
            f"Status:                "
            f"{result['status']}\n"
        )
        file.write(
            f"Decision:              "
            f"{result['decision']}\n\n"
        )

        file.write(
            "[DEPLOYMENT CONDITIONS]\n"
        )

        for condition in result["conditions"]:
            file.write(
                f"- {condition}\n"
            )

        file.write("\n")

        file.write(
            "[EVIDENCE]\n"
        )

        file.write(
            evidence_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print(
        "EMPLOYEE ATTRITION — DEPLOYMENT DECISION ANALYSIS"
    )
    print("=" * 64)

    print()
    print("[MODEL]")
    print(
        f"Model:                 {MODEL_NAME}"
    )
    print(
        f"Features:              {len(STABLE_FEATURES)}"
    )
    print(
        f"Threshold:             {THRESHOLD:.2f}"
    )

    print()
    print("[PREDICTIVE PERFORMANCE]")
    print(
        f"ROC-AUC:               {FINAL_ROC_AUC:.4f}"
    )
    print(
        f"PR-AUC:                {FINAL_PR_AUC:.4f}"
    )
    print(
        f"F1:                    {FINAL_F1:.4f}"
    )
    print(
        f"Precision:             {FINAL_PRECISION:.4f}"
    )
    print(
        f"Recall:                {FINAL_RECALL:.4f}"
    )
    print(
        f"Specificity:           {FINAL_SPECIFICITY:.4f}"
    )

    print()
    print("[STABILITY]")
    print(
        f"ROC-AUC mean:          "
        f"{REPEATED_ROC_AUC:.4f}"
    )
    print(
        f"ROC-AUC std:           "
        f"{REPEATED_ROC_AUC_STD:.4f}"
    )
    print(
        f"ROC-AUC min:           "
        f"{REPEATED_ROC_AUC_MIN:.4f}"
    )
    print(
        f"ROC-AUC max:           "
        f"{REPEATED_ROC_AUC_MAX:.4f}"
    )

    print()
    print("[BUSINESS IMPACT]")
    print(
        f"Predicted positive:    "
        f"{FINAL_PREDICTED_POSITIVE:.2f}%"
    )
    print(
        f"Flagged per 1000:      "
        f"{FINAL_PREDICTED_POSITIVE * 10:.1f}"
    )
    print(
        f"Flag / prevalence:     "
        f"{result['flag_rate_ratio']:.2f}x"
    )

    print()
    print("[DEPLOYMENT CONDITIONS]")

    for condition in result["conditions"]:
        print(
            f"- {condition}"
        )

    print()
    print("=" * 64)
    print("[OVERALL STATUS]")
    print(
        f"DEPLOYMENT DECISION STATUS: "
        f"{result['status']}"
    )

    print()
    print("[OVERALL DECISION]")
    print(
        result["decision"]
    )

    print()
    print("[OUTPUT]")
    print(
        f"Evidence CSV:         {evidence_path}"
    )
    print(
        f"JSON report:          {json_path}"
    )
    print(
        f"Summary report:       {summary_path}"
    )

    print()
    print("=" * 64)
    print(
        "DEPLOYMENT DECISION ANALYSIS COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()