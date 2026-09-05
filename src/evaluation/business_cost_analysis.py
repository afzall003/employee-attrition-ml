"""
Business cost analysis for the final stable-feature employee attrition model.

Purpose
-------
Evaluate the operational cost of different classification thresholds under
multiple false-positive / false-negative cost scenarios.

The analysis uses:
    - canonical dataset
    - stable 10-feature subset
    - selected Random Forest configuration
    - repeated stratified out-of-fold predictions

No threshold is selected solely from F1. Instead, business cost scenarios
are evaluated so that the final operating point can be reviewed explicitly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "business_cost_analysis"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CANONICAL DATASET SPECIFICATION
# ============================================================

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26
TARGET = "Attrition"
IDENTIFIER = "Employee_ID"

EXPECTED_SHA256 = (
    "9b294a270e34d159ce21e7f2c4d0be394d53f83736bc7d413296be4cf2768ed6"
)


# ============================================================
# FINAL STABLE FEATURE SET
# ============================================================

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


# ============================================================
# FINAL MODEL CONFIGURATION
# ============================================================

MODEL_CONFIG = {
    "n_estimators": 400,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
}


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42


# ============================================================
# THRESHOLDS
# ============================================================

CANDIDATE_THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.44,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


# ============================================================
# BUSINESS COST SCENARIOS
# ============================================================
#
# Cost interpretation:
#
# FP = intervention made for an employee who does not leave.
# FN = employee leaves but was not flagged.
#
# The absolute values are relative cost units. They are intentionally
# configurable rather than pretending that a universal business cost
# exists.
#

COST_SCENARIOS = {
    "balanced_1_to_1": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 1.0,
    },
    "moderate_detection_2_to_1": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 2.0,
    },
    "high_detection_5_to_1": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
    },
    "very_high_detection_10_to_1": {
        "false_positive_cost": 1.0,
        "false_negative_cost": 10.0,
    },
}


# ============================================================
# HELPERS
# ============================================================


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """Validate the canonical dataset before model evaluation."""

    checks = {
        "file_exists": DATA_PATH.exists(),
        "expected_rows": len(df) == EXPECTED_ROWS,
        "expected_columns": len(df.columns) == EXPECTED_COLUMNS,
        "target_exists": TARGET in df.columns,
        "identifier_exists": IDENTIFIER in df.columns,
        "stable_features_exist": all(
            feature in df.columns for feature in STABLE_FEATURES
        ),
        "target_values_valid": set(df[TARGET].dropna().unique()).issubset(
            {0, 1, "Yes", "No", True, False}
        ),
        "no_missing_cells": not df.isnull().any().any(),
        "identifier_unique": (
            df[IDENTIFIER].is_unique if IDENTIFIER in df.columns else False
        ),
    }

    return checks


def normalize_target(series: pd.Series) -> pd.Series:
    """Convert common binary target representations to integer 0/1."""

    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int)

        if not set(values.unique()).issubset({0, 1}):
            raise ValueError(
                "Numeric target contains values other than 0 and 1."
            )

        return values

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if normalized.isnull().any():
        raise ValueError(
            "Unable to normalize all target values to binary 0/1."
        )

    return normalized.astype(int)


def make_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build preprocessing pipeline from stable feature schema."""

    numerical_features = X.select_dtypes(
        include=["number", "int64", "float64"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in X.columns
        if column not in numerical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                "passthrough",
                numerical_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return (
        preprocessor,
        numerical_features,
        categorical_features,
    )


def build_model(X: pd.DataFrame) -> Pipeline:
    """Build the final Random Forest model pipeline."""

    preprocessor, _, _ = make_preprocessor(X)

    model = RandomForestClassifier(**MODEL_CONFIG)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================


def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """
    Generate repeated out-of-fold predictions.

    Each observation can receive multiple predictions because the validation
    design is repeated. Predictions are averaged per observation so that the
    final threshold analysis uses one probability estimate per row.
    """

    print("Generating repeated out-of-fold predictions...")
    print(f"Folds per repeat:      {N_SPLITS}")
    print(f"Repeats:               {N_REPEATS}")
    print(f"Total validation:      {N_SPLITS * N_REPEATS}")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    prediction_sum = np.zeros(len(X), dtype=float)
    prediction_count = np.zeros(len(X), dtype=int)

    for split_number, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(
            f"Validation split "
            f"{split_number}/{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]

        model = build_model(X_train)

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_valid)[:, 1]

        prediction_sum[valid_idx] += probabilities
        prediction_count[valid_idx] += 1

    if np.any(prediction_count == 0):
        raise RuntimeError(
            "Some observations did not receive an out-of-fold prediction."
        )

    return prediction_sum / prediction_count


# ============================================================
# THRESHOLD METRICS
# ============================================================


def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate classification metrics for one threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    predicted_positive_rate = predictions.mean()

    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "predicted_positive_percent": predicted_positive_rate * 100,
        "flagged_per_1000": predicted_positive_rate * 1000,
        "false_positives_per_1000": (
            fp / len(y_true) * 1000
        ),
        "missed_attrition_per_1000": (
            fn / len(y_true) * 1000
        ),
    }


# ============================================================
# BUSINESS COST CALCULATION
# ============================================================


def add_business_costs(
    threshold_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate total and normalized business costs for each scenario.

    Cost = FP * false_positive_cost
         + FN * false_negative_cost
    """

    result = threshold_df.copy()

    for scenario_name, scenario in COST_SCENARIOS.items():

        fp_cost = scenario["false_positive_cost"]
        fn_cost = scenario["false_negative_cost"]

        total_cost = (
            result["fp"] * fp_cost
            + result["fn"] * fn_cost
        )

        normalized_cost = (
            total_cost / len(y_global)
        )

        result[f"{scenario_name}_total_cost"] = total_cost
        result[f"{scenario_name}_cost_per_employee"] = (
            normalized_cost
        )

    return result


# ============================================================
# SCENARIO SUMMARY
# ============================================================


def determine_best_thresholds(
    threshold_df: pd.DataFrame,
) -> pd.DataFrame:
    """Determine the minimum-cost threshold for each scenario."""

    rows = []

    for scenario_name, scenario in COST_SCENARIOS.items():

        cost_column = f"{scenario_name}_total_cost"

        best_row = threshold_df.loc[
            threshold_df[cost_column].idxmin()
        ]

        rows.append(
            {
                "scenario": scenario_name,
                "false_positive_cost": scenario[
                    "false_positive_cost"
                ],
                "false_negative_cost": scenario[
                    "false_negative_cost"
                ],
                "recommended_threshold": best_row[
                    "threshold"
                ],
                "total_cost": best_row[
                    cost_column
                ],
                "cost_per_employee": best_row[
                    f"{scenario_name}_cost_per_employee"
                ],
                "f1": best_row["f1"],
                "precision": best_row["precision"],
                "recall": best_row["recall"],
                "specificity": best_row["specificity"],
                "predicted_positive_percent": best_row[
                    "predicted_positive_percent"
                ],
                "flagged_per_1000": best_row[
                    "flagged_per_1000"
                ],
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN ANALYSIS
# ============================================================


def main() -> None:

    print("Running business cost analysis...")
    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    print()
    print("Validating canonical dataset...")

    checks = validate_dataset(df)

    for check_name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{status} "
            f"{check_name}"
        )

    if not all(checks.values()):
        raise ValueError(
            "Canonical dataset validation failed."
        )

    actual_sha256 = sha256_file(DATA_PATH)

    if actual_sha256 != EXPECTED_SHA256:
        raise ValueError(
            "Canonical dataset SHA-256 does not match "
            "the established canonical hash."
        )

    print("PASS canonical_sha256")

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    y = normalize_target(df[TARGET])

    X = df[STABLE_FEATURES].copy()

    numerical_features = X.select_dtypes(
        include=["number", "int64", "float64"]
    ).columns.tolist()

    categorical_features = [
        feature
        for feature in X.columns
        if feature not in numerical_features
    ]

    print()
    print("Stable features:", len(STABLE_FEATURES))
    print(
        "Numerical features:",
        len(numerical_features),
    )
    print(
        "Categorical features:",
        len(categorical_features),
    )

    prevalence = y.mean()

    print(
        f"Target prevalence:    {prevalence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # OOF PREDICTIONS
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("REPEATED OUT-OF-FOLD BUSINESS COST VALIDATION")
    print("=" * 64)

    global y_global
    y_global = y.to_numpy()

    probabilities = generate_oof_predictions(
        X,
        y,
    )

    # --------------------------------------------------------
    # RANKING METRICS
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print()
    print("Calculating ranking metrics...")
    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )
    print(
        f"PR-AUC:  {pr_auc:.4f}"
    )

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS
    # --------------------------------------------------------

    print()
    print("Evaluating candidate thresholds...")

    threshold_rows = []

    for threshold in CANDIDATE_THRESHOLDS:

        metrics = calculate_threshold_metrics(
            y.to_numpy(),
            probabilities,
            threshold,
        )

        threshold_rows.append(metrics)

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_df = add_business_costs(
        threshold_df
    )

    scenario_df = determine_best_thresholds(
        threshold_df
    )

    # --------------------------------------------------------
    # F1-OPTIMAL THRESHOLD
    # --------------------------------------------------------

    f1_best = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("EMPLOYEE ATTRITION — BUSINESS COST ANALYSIS")
    print("=" * 64)

    print()
    print("[DATASET]")
    print(
        f"Rows:                 {len(df)}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )
    print(
        f"Features:             {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    {prevalence * 100:.2f}%"
    )

    print()
    print("[MODEL]")
    print(
        "Model:                 Random Forest"
    )
    print(
        "Feature set:           Stable 10-feature subset"
    )
    print(
        "Validation:            5-fold × 5-repeat"
    )

    print()
    print("[OUT-OF-FOLD RANKING]")
    print(
        f"ROC-AUC:               {roc_auc:.4f}"
    )
    print(
        f"PR-AUC:                {pr_auc:.4f}"
    )

    print()
    print("[THRESHOLD METRICS]")

    display_columns = [
        "threshold",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_percent",
        "flagged_per_1000",
        "false_positives_per_1000",
        "missed_attrition_per_1000",
    ]

    print(
        threshold_df[display_columns].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "f1": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "specificity": "{:.4f}".format,
                "balanced_accuracy": "{:.4f}".format,
                "predicted_positive_percent": "{:.2f}".format,
                "flagged_per_1000": "{:.1f}".format,
                "false_positives_per_1000": "{:.1f}".format,
                "missed_attrition_per_1000": "{:.1f}".format,
            },
        )
    )

    print()
    print("[F1-OPTIMAL REFERENCE]")
    print(
        f"Threshold:            {f1_best['threshold']:.2f}"
    )
    print(
        f"F1:                   {f1_best['f1']:.4f}"
    )
    print(
        f"Precision:            {f1_best['precision']:.4f}"
    )
    print(
        f"Recall:               {f1_best['recall']:.4f}"
    )
    print(
        f"Specificity:          {f1_best['specificity']:.4f}"
    )
    print(
        f"Flagged per 1000:     "
        f"{f1_best['flagged_per_1000']:.1f}"
    )

    print()
    print("[BUSINESS COST SCENARIOS]")

    print(
        scenario_df[
            [
                "scenario",
                "false_positive_cost",
                "false_negative_cost",
                "recommended_threshold",
                "total_cost",
                "cost_per_employee",
                "precision",
                "recall",
                "specificity",
                "flagged_per_1000",
            ]
        ].to_string(
            index=False,
            formatters={
                "recommended_threshold": "{:.2f}".format,
                "total_cost": "{:.2f}".format,
                "cost_per_employee": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "specificity": "{:.4f}".format,
                "flagged_per_1000": "{:.1f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # DIAGNOSTIC FLAGS
    # --------------------------------------------------------

    flags = []

    f1_default = threshold_df.loc[
        threshold_df["threshold"] == 0.50,
        "f1",
    ].iloc[0]

    f1_improvement = (
        f1_best["f1"] - f1_default
    )

    if abs(f1_best["threshold"] - 0.50) >= 0.05:
        flags.append(
            "The F1-optimal threshold differs materially "
            "from the default 0.50 operating point."
        )

    if f1_improvement > 0:
        flags.append(
            "F1 optimization improves classification "
            f"performance by {f1_improvement:.4f} "
            "relative to threshold 0.50."
        )

    if f1_best["predicted_positive_percent"] > (
        prevalence * 100 * 2
    ):
        flags.append(
            "The F1-optimal threshold flags more than "
            "twice the observed attrition prevalence."
        )

    if f1_best["precision"] < 0.40:
        flags.append(
            "Precision remains below 0.40 at the "
            "F1-optimal threshold, indicating a "
            "substantial false-positive burden."
        )

    if f1_best["recall"] >= 0.70:
        flags.append(
            "The F1-optimal threshold prioritizes "
            "detection with recall of at least 0.70."
        )

    if f1_best["specificity"] < 0.60:
        flags.append(
            "Specificity is below 0.60 at the "
            "F1-optimal threshold."
        )

    if roc_auc >= 0.60:
        flags.append(
            "Out-of-fold ROC-AUC indicates useful "
            "ranking information before thresholding."
        )

    # Check whether cost-optimal thresholds differ
    unique_business_thresholds = (
        scenario_df["recommended_threshold"]
        .nunique()
    )

    if unique_business_thresholds > 1:
        flags.append(
            "The cost-optimal threshold changes across "
            "business cost scenarios, confirming that "
            "threshold selection depends on intervention "
            "economics."
        )

    # --------------------------------------------------------
    # DIAGNOSTIC OUTPUT
    # --------------------------------------------------------

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            print(f"- {flag}")
    else:
        print("- No major business-threshold diagnostic flags.")

    # --------------------------------------------------------
    # OVERALL DIAGNOSIS
    # --------------------------------------------------------

    best_cost_thresholds = (
        scenario_df["recommended_threshold"]
        .tolist()
    )

    if unique_business_thresholds > 1:

        diagnosis = (
            "Business cost analysis shows that the optimal "
            "operating threshold depends materially on the "
            "relative cost of false positives and false "
            "negatives. The F1-optimal threshold should "
            "therefore be treated as a reference rather than "
            "an automatic deployment decision. A final "
            "threshold should be selected using the organization's "
            "intervention capacity and validated error costs."
        )

    else:

        diagnosis = (
            "Business cost analysis produces a consistent "
            "minimum-cost threshold across the evaluated "
            "cost scenarios. The result should still be "
            "reviewed against real intervention capacity "
            "and validated business error costs before deployment."
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # --------------------------------------------------------
    # SAVE CSV REPORTS
    # --------------------------------------------------------

    threshold_csv = (
        REPORT_DIR
        / "business_cost_threshold_results.csv"
    )

    scenario_csv = (
        REPORT_DIR
        / "business_cost_scenario_summary.csv"
    )

    threshold_df.to_csv(
        threshold_csv,
        index=False,
    )

    scenario_df.to_csv(
        scenario_csv,
        index=False,
    )

    # --------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------

    json_report = {
        "dataset": {
            "path": str(DATA_PATH),
            "sha256": actual_sha256,
            "rows": len(df),
            "columns": len(df.columns),
            "target": TARGET,
            "target_prevalence": float(prevalence),
        },
        "stable_features": STABLE_FEATURES,
        "model": {
            "name": "Random Forest",
            "configuration": MODEL_CONFIG,
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
        },
        "ranking_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
        },
        "f1_optimal_reference": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.floating, float)
                )
                else int(value)
                if isinstance(
                    value,
                    (np.integer, int)
                )
                else value
            )
            for key, value in f1_best.items()
        },
        "candidate_thresholds": threshold_df.to_dict(
            orient="records"
        ),
        "business_cost_scenarios": scenario_df.to_dict(
            orient="records"
        ),
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    json_path = (
        REPORT_DIR
        / "business_cost_analysis_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY TXT
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "business_cost_analysis_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — BUSINESS COST ANALYSIS\n"
        )
        file.write("=" * 64 + "\n\n")

        file.write("[DATASET]\n")
        file.write(
            f"Rows: {len(df)}\n"
        )
        file.write(
            f"Columns: {len(df.columns)}\n"
        )
        file.write(
            f"Target prevalence: {prevalence * 100:.2f}%\n"
        )
        file.write(
            f"SHA-256: {actual_sha256}\n\n"
        )

        file.write("[MODEL]\n")
        file.write(
            "Random Forest\n"
        )
        file.write(
            "Stable 10-feature subset\n"
        )
        file.write(
            "5-fold × 5-repeat validation\n\n"
        )

        file.write("[RANKING]\n")
        file.write(
            f"ROC-AUC: {roc_auc:.4f}\n"
        )
        file.write(
            f"PR-AUC: {pr_auc:.4f}\n\n"
        )

        file.write("[F1-OPTIMAL REFERENCE]\n")
        file.write(
            f"Threshold: {f1_best['threshold']:.2f}\n"
        )
        file.write(
            f"F1: {f1_best['f1']:.4f}\n"
        )
        file.write(
            f"Precision: {f1_best['precision']:.4f}\n"
        )
        file.write(
            f"Recall: {f1_best['recall']:.4f}\n"
        )
        file.write(
            f"Specificity: {f1_best['specificity']:.4f}\n"
        )
        file.write(
            f"Flagged per 1000: "
            f"{f1_best['flagged_per_1000']:.1f}\n\n"
        )

        file.write("[BUSINESS COST SCENARIOS]\n")

        for _, row in scenario_df.iterrows():

            file.write(
                f"\n{row['scenario']}\n"
            )

            file.write(
                f"  FP cost: "
                f"{row['false_positive_cost']}\n"
            )

            file.write(
                f"  FN cost: "
                f"{row['false_negative_cost']}\n"
            )

            file.write(
                f"  Recommended threshold: "
                f"{row['recommended_threshold']:.2f}\n"
            )

            file.write(
                f"  Total cost: "
                f"{row['total_cost']:.2f}\n"
            )

            file.write(
                f"  Cost per employee: "
                f"{row['cost_per_employee']:.4f}\n"
            )

            file.write(
                f"  Precision: "
                f"{row['precision']:.4f}\n"
            )

            file.write(
                f"  Recall: "
                f"{row['recall']:.4f}\n"
            )

            file.write(
                f"  Specificity: "
                f"{row['specificity']:.4f}\n"
            )

            file.write(
                f"  Flagged per 1000: "
                f"{row['flagged_per_1000']:.1f}\n"
            )

        file.write(
            "\n[DIAGNOSTIC FLAGS]\n"
        )

        for flag in flags:
            file.write(
                f"- {flag}\n"
            )

        file.write(
            "\n[OVERALL DIAGNOSIS]\n"
        )
        file.write(
            diagnosis + "\n"
        )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("[OUTPUT]")
    print(
        f"Threshold CSV:        {threshold_csv}"
    )
    print(
        f"Scenario CSV:         {scenario_csv}"
    )
    print(
        f"JSON report:          {json_path}"
    )
    print(
        f"Summary report:       {summary_path}"
    )

    print()
    print("=" * 64)
    print("BUSINESS COST ANALYSIS COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()