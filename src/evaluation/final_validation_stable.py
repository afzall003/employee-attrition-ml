"""
Final validation of the stable-feature employee attrition model.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

Stable feature set:
    1. Work_Life_Balance
    2. Job_Satisfaction
    3. Distance_From_Home
    4. Average_Hours_Worked_Per_Week
    5. Years_Since_Last_Promotion
    6. Work_Environment_Satisfaction
    7. Job_Role
    8. Age
    9. Overtime
    10. Absenteeism

Selected model:
    Random Forest

Selected threshold:
    0.44

Validation:
    5-fold stratified CV x 5 repeats = 25 validation evaluations

Purpose:
    Perform the final validation of the selected stable-feature model
    before final model training / artifact generation.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_attrition_dataset_v2.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "final_validation_stable"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Attrition"

IDENTIFIER_COLUMNS = [
    "Employee_ID",
]

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

SELECTED_THRESHOLD = 0.44

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# Optimized Random Forest parameters obtained from
# model_optimization_stable.py
RF_PARAMS = {
    "class_weight": "balanced",
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "n_estimators": 400,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, (np.floating, np.integer)):
        return float(value)

    if pd.isna(value):
        return None

    return float(value)


def load_dataset() -> pd.DataFrame:
    print("Loading canonical dataset...")
    print(f"Dataset: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    return df


def validate_dataset(df: pd.DataFrame) -> dict[str, Any]:
    print()
    print("Validating canonical dataset...")

    checks: dict[str, Any] = {}

    checks["file_exists"] = DATA_PATH.exists()
    checks["expected_rows"] = len(df) == 1000
    checks["expected_columns"] = len(df.columns) == 26
    checks["target_exists"] = TARGET in df.columns

    missing_stable = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    checks["stable_features_exist"] = len(missing_stable) == 0

    if TARGET in df.columns:
        target_values = set(
            df[TARGET]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        checks["target_values_valid"] = target_values.issubset(
            {"Yes", "No", "0", "1"}
        )
    else:
        checks["target_values_valid"] = False

    checks["no_missing_cells"] = int(df.isna().sum().sum()) == 0

    if "Employee_ID" in df.columns:
        checks["identifier_unique"] = (
            df["Employee_ID"].nunique() == len(df)
        )
    else:
        checks["identifier_unique"] = False

    checks["all_pass"] = all(checks.values())

    print()
    for name, status in checks.items():
        if name == "all_pass":
            continue

        print(
            f"{'PASS' if status else 'FAIL':4} "
            f"{name}"
        )

    if not checks["all_pass"]:
        raise ValueError(
            "Canonical dataset validation failed. "
            "Final validation cannot continue."
        )

    return checks


def prepare_target(series: pd.Series) -> pd.Series:
    mapping = {
        "No": 0,
        "Yes": 1,
        "0": 0,
        "1": 1,
        0: 0,
        1: 1,
    }

    y = series.map(mapping)

    if y.isna().any():
        invalid = series[y.isna()].unique().tolist()

        raise ValueError(
            f"Unexpected target values found: {invalid}"
        )

    return y.astype(int)


def build_pipeline(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(**RF_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }

    return metrics


# ============================================================
# VALIDATION
# ============================================================

def run_repeated_validation(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:

    print()
    print("Generating repeated stratified validation...")
    print(
        f"Folds per repeat:      {N_SPLITS}"
    )
    print(
        f"Repeats:               {N_REPEATS}"
    )
    print(
        f"Total validation:      "
        f"{N_SPLITS * N_REPEATS}"
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    results = []

    all_true = []
    all_probabilities = []

    split_number = 0

    for train_idx, validation_idx in splitter.split(X, y):

        split_number += 1

        print(
            f"Validation split "
            f"{split_number}/{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[validation_idx]

        pipeline = build_pipeline(
            numerical_features,
            categorical_features,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probabilities = pipeline.predict_proba(
            X_valid
        )[:, 1]

        metrics = calculate_metrics(
            y_valid.to_numpy(),
            probabilities,
            SELECTED_THRESHOLD,
        )

        metrics["split"] = split_number
        metrics["train_rows"] = len(train_idx)
        metrics["validation_rows"] = len(validation_idx)

        metrics["train_prevalence"] = float(
            y_train.mean()
        )

        metrics["validation_prevalence"] = float(
            y_valid.mean()
        )

        results.append(metrics)

        all_true.extend(
            y_valid.to_numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.tolist()
        )

    results_df = pd.DataFrame(results)

    return (
        results_df,
        np.asarray(all_true),
        np.asarray(all_probabilities),
    )


# ============================================================
# OOF / AGGREGATED METRICS
# ============================================================

def calculate_aggregate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    return calculate_metrics(
        y_true,
        probabilities,
        SELECTED_THRESHOLD,
    )


def calculate_default_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:

    return calculate_metrics(
        y_true,
        probabilities,
        0.50,
    )


# ============================================================
# DIAGNOSTIC ANALYSIS
# ============================================================

def generate_diagnostics(
    split_df: pd.DataFrame,
    aggregate: dict[str, float],
    default_metrics: dict[str, float],
) -> list[str]:

    flags: list[str] = []

    if aggregate["roc_auc"] < 0.60:
        flags.append(
            "Aggregated out-of-fold ROC-AUC is below 0.60, "
            "indicating weak ranking performance."
        )
    elif aggregate["roc_auc"] < 0.70:
        flags.append(
            "Aggregated out-of-fold ROC-AUC indicates "
            "modest predictive separation."
        )
    else:
        flags.append(
            "Aggregated out-of-fold ROC-AUC indicates "
            "strong predictive separation."
        )

    roc_std = split_df["roc_auc"].std()

    if roc_std >= 0.05:
        flags.append(
            f"ROC-AUC remains split-sensitive "
            f"(std={roc_std:.4f})."
        )
    else:
        flags.append(
            f"ROC-AUC variability across validation splits "
            f"is moderate (std={roc_std:.4f})."
        )

    f1_improvement = (
        aggregate["f1"]
        - default_metrics["f1"]
    )

    if f1_improvement > 0:
        flags.append(
            "Threshold 0.44 improves F1 relative to "
            "the default 0.50 threshold."
        )
    else:
        flags.append(
            "Threshold 0.44 does not improve F1 relative "
            "to the default 0.50 threshold."
        )

    predicted_rate = aggregate[
        "predicted_positive_rate"
    ]

    prevalence = (
        y_global.mean()
    )

    if predicted_rate > 2 * prevalence:
        flags.append(
            "The optimized threshold predicts attrition "
            "for more than twice the observed prevalence; "
            "business capacity should be reviewed."
        )

    if aggregate["precision"] < 0.40:
        flags.append(
            "Precision remains below 0.40, so a substantial "
            "fraction of flagged employees are expected "
            "to be false positives."
        )

    if aggregate["recall"] >= 0.70:
        flags.append(
            "Recall is at least 0.70, indicating that the "
            "selected operating point prioritizes detection."
        )

    if aggregate["specificity"] < 0.60:
        flags.append(
            "Specificity is below 0.60, indicating a relatively "
            "high false-positive rate at threshold 0.44."
        )

    return flags


# Global target prevalence used only for diagnostics.
y_global = pd.Series(dtype=int)


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(
    df: pd.DataFrame,
    validation_checks: dict[str, Any],
    split_df: pd.DataFrame,
    aggregate: dict[str, float],
    default_metrics: dict[str, float],
    diagnostics: list[str],
    fingerprint: str,
) -> dict[str, Any]:

    prevalence = float(
        df[TARGET].map(
            {
                "No": 0,
                "Yes": 1,
                "0": 0,
                "1": 1,
            }
        ).mean()
    )

    model_features = [
        column
        for column in STABLE_FEATURES
        if column not in IDENTIFIER_COLUMNS
    ]

    numerical_features = [
        feature
        for feature in model_features
        if pd.api.types.is_numeric_dtype(
            df[feature]
        )
    ]

    categorical_features = [
        feature
        for feature in model_features
        if feature not in numerical_features
    ]

    f1_improvement = (
        aggregate["f1"]
        - default_metrics["f1"]
    )

    if (
        aggregate["roc_auc"] >= 0.60
        and aggregate["f1"] > default_metrics["f1"]
        and aggregate["precision"] > 0
        and aggregate["recall"] > 0
    ):
        overall_status = "CONDITIONAL PASS"
    else:
        overall_status = "FAIL"

    report = {
        "project": "employee_attrition_ml",
        "evaluation": "final_validation_stable",
        "dataset": {
            "path": str(
                DATA_PATH.relative_to(PROJECT_ROOT)
            ),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target": TARGET,
            "target_prevalence": prevalence,
            "sha256": fingerprint,
        },
        "dataset_validation": validation_checks,
        "feature_set": {
            "feature_count": len(model_features),
            "features": model_features,
            "numerical_features": numerical_features,
            "categorical_features": categorical_features,
        },
        "model": {
            "name": "Random Forest",
            "parameters": RF_PARAMS,
        },
        "validation_design": {
            "method": "Repeated Stratified K-Fold",
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
            "random_state": RANDOM_STATE,
        },
        "threshold": {
            "selected": SELECTED_THRESHOLD,
            "default": 0.50,
            "source": (
                "stable_threshold_optimization"
            ),
        },
        "aggregate_metrics": aggregate,
        "default_threshold_metrics": default_metrics,
        "threshold_comparison": {
            "f1_improvement": f1_improvement,
            "precision_change": (
                aggregate["precision"]
                - default_metrics["precision"]
            ),
            "recall_change": (
                aggregate["recall"]
                - default_metrics["recall"]
            ),
            "specificity_change": (
                aggregate["specificity"]
                - default_metrics["specificity"]
            ),
            "balanced_accuracy_change": (
                aggregate["balanced_accuracy"]
                - default_metrics["balanced_accuracy"]
            ),
        },
        "split_statistics": {
            "roc_auc_mean": safe_float(
                split_df["roc_auc"].mean()
            ),
            "roc_auc_std": safe_float(
                split_df["roc_auc"].std()
            ),
            "roc_auc_min": safe_float(
                split_df["roc_auc"].min()
            ),
            "roc_auc_max": safe_float(
                split_df["roc_auc"].max()
            ),
            "pr_auc_mean": safe_float(
                split_df["pr_auc"].mean()
            ),
            "f1_mean": safe_float(
                split_df["f1"].mean()
            ),
            "precision_mean": safe_float(
                split_df["precision"].mean()
            ),
            "recall_mean": safe_float(
                split_df["recall"].mean()
            ),
            "specificity_mean": safe_float(
                split_df["specificity"].mean()
            ),
        },
        "diagnostic_flags": diagnostics,
        "overall_status": overall_status,
    }

    return report


def save_reports(
    report: dict[str, Any],
    split_df: pd.DataFrame,
    aggregate: dict[str, float],
    default_metrics: dict[str, float],
) -> None:

    json_path = (
        REPORT_DIR
        / "final_validation_stable_report.json"
    )

    summary_path = (
        REPORT_DIR
        / "final_validation_stable_summary.txt"
    )

    split_path = (
        REPORT_DIR
        / "split_performance.csv"
    )

    comparison_path = (
        REPORT_DIR
        / "threshold_comparison.csv"
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

    split_df.to_csv(
        split_path,
        index=False,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "threshold": 0.50,
                **default_metrics,
            },
            {
                "threshold": SELECTED_THRESHOLD,
                **aggregate,
            },
        ]
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "FINAL STABLE MODEL VALIDATION\n"
        )
        file.write("=" * 64 + "\n\n")

        file.write("[DATASET]\n")
        file.write(
            f"Rows:                 "
            f"{report['dataset']['rows']}\n"
        )
        file.write(
            f"Columns:              "
            f"{report['dataset']['columns']}\n"
        )
        file.write(
            f"Target prevalence:    "
            f"{report['dataset']['target_prevalence']:.2%}\n"
        )

        file.write("\n[STABLE FEATURE SET]\n")
        file.write(
            f"Feature count:         "
            f"{report['feature_set']['feature_count']}\n"
        )

        for index, feature in enumerate(
            report["feature_set"]["features"],
            start=1,
        ):
            file.write(
                f"{index:2}. {feature}\n"
            )

        file.write("\n[MODEL]\n")
        file.write("Model:                 Random Forest\n")
        file.write(
            f"Threshold:            "
            f"{SELECTED_THRESHOLD:.2f}\n"
        )

        file.write(
            "\n[OUT-OF-FOLD PERFORMANCE — "
            "THRESHOLD 0.44]\n"
        )

        file.write(
            f"ROC-AUC:              "
            f"{aggregate['roc_auc']:.4f}\n"
        )
        file.write(
            f"PR-AUC:               "
            f"{aggregate['pr_auc']:.4f}\n"
        )
        file.write(
            f"F1:                   "
            f"{aggregate['f1']:.4f}\n"
        )
        file.write(
            f"Precision:            "
            f"{aggregate['precision']:.4f}\n"
        )
        file.write(
            f"Recall:               "
            f"{aggregate['recall']:.4f}\n"
        )
        file.write(
            f"Specificity:          "
            f"{aggregate['specificity']:.4f}\n"
        )
        file.write(
            f"Balanced Accuracy:    "
            f"{aggregate['balanced_accuracy']:.4f}\n"
        )
        file.write(
            f"Accuracy:             "
            f"{aggregate['accuracy']:.4f}\n"
        )
        file.write(
            f"Predicted Positive:   "
            f"{aggregate['predicted_positive_rate']:.2%}\n"
        )

        file.write(
            "\n[DEFAULT THRESHOLD — 0.50]\n"
        )
        file.write(
            f"F1:                   "
            f"{default_metrics['f1']:.4f}\n"
        )
        file.write(
            f"Precision:            "
            f"{default_metrics['precision']:.4f}\n"
        )
        file.write(
            f"Recall:               "
            f"{default_metrics['recall']:.4f}\n"
        )
        file.write(
            f"Specificity:          "
            f"{default_metrics['specificity']:.4f}\n"
        )
        file.write(
            f"Balanced Accuracy:    "
            f"{default_metrics['balanced_accuracy']:.4f}\n"
        )

        file.write(
            "\n[THRESHOLD IMPROVEMENT]\n"
        )
        file.write(
            f"F1 improvement:       "
            f"{aggregate['f1'] - default_metrics['f1']:+.4f}\n"
        )

        file.write(
            "\n[SPLIT STATISTICS]\n"
        )
        file.write(
            f"ROC-AUC mean:         "
            f"{split_df['roc_auc'].mean():.4f}\n"
        )
        file.write(
            f"ROC-AUC std:          "
            f"{split_df['roc_auc'].std():.4f}\n"
        )
        file.write(
            f"ROC-AUC min:          "
            f"{split_df['roc_auc'].min():.4f}\n"
        )
        file.write(
            f"ROC-AUC max:          "
            f"{split_df['roc_auc'].max():.4f}\n"
        )
        file.write(
            f"PR-AUC mean:          "
            f"{split_df['pr_auc'].mean():.4f}\n"
        )

        file.write(
            "\n[DIAGNOSTIC FLAGS]\n"
        )

        for flag in report["diagnostic_flags"]:
            file.write(f"- {flag}\n")

        file.write(
            "\n[OVERALL STATUS]\n"
        )
        file.write(
            f"{report['overall_status']}\n"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Running final stable-feature model validation..."
    )

    df = load_dataset()

    validation_checks = validate_dataset(df)

    fingerprint = sha256_file(DATA_PATH)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = prepare_target(df[TARGET])

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    model_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in IDENTIFIER_COLUMNS
    ]

    X = df[model_features].copy()

    numerical_features = [
        feature
        for feature in model_features
        if pd.api.types.is_numeric_dtype(
            X[feature]
        )
    ]

    categorical_features = [
        feature
        for feature in model_features
        if feature not in numerical_features
    ]

    print()
    print(
        f"Stable features:       "
        f"{len(model_features)}"
    )
    print(
        f"Numerical features:    "
        f"{len(numerical_features)}"
    )
    print(
        f"Categorical features:  "
        f"{len(categorical_features)}"
    )

    print()
    print(
        f"Target prevalence:     "
        f"{y.mean():.2%}"
    )

    # Set global prevalence for diagnostics.
    global y_global
    y_global = y

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print_header(
        "FINAL REPEATED OUT-OF-FOLD VALIDATION"
    )

    (
        split_df,
        oof_y,
        oof_probabilities,
    ) = run_repeated_validation(
        X,
        y,
        numerical_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Aggregate OOF metrics
    # --------------------------------------------------------

    print()
    print(
        "Calculating aggregate out-of-fold metrics..."
    )

    aggregate = calculate_aggregate_metrics(
        oof_y,
        oof_probabilities,
    )

    default_metrics = calculate_default_threshold_metrics(
        oof_y,
        oof_probabilities,
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = generate_diagnostics(
        split_df,
        aggregate,
        default_metrics,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = generate_report(
        df=df,
        validation_checks=validation_checks,
        split_df=split_df,
        aggregate=aggregate,
        default_metrics=default_metrics,
        diagnostics=diagnostics,
        fingerprint=fingerprint,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print_header(
        "EMPLOYEE ATTRITION — FINAL STABLE MODEL VALIDATION"
    )

    print()
    print("[DATASET]")
    print(
        f"Rows:                 {len(df)}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )
    print(
        f"Features:             {len(model_features)}"
    )
    print(
        f"Target prevalence:    {y.mean():.2%}"
    )

    print()
    print("[FINAL MODEL]")
    print("Model:                 Random Forest")
    print(
        f"Features:             {len(model_features)}"
    )
    print(
        f"Threshold:            {SELECTED_THRESHOLD:.2f}"
    )

    print()
    print("[OUT-OF-FOLD PERFORMANCE]")

    print(
        f"ROC-AUC:              "
        f"{aggregate['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:               "
        f"{aggregate['pr_auc']:.4f}"
    )
    print(
        f"F1:                   "
        f"{aggregate['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{aggregate['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{aggregate['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{aggregate['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{aggregate['balanced_accuracy']:.4f}"
    )
    print(
        f"Accuracy:             "
        f"{aggregate['accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{aggregate['predicted_positive_rate']:.2%}"
    )

    print()
    print("[DEFAULT THRESHOLD — 0.50]")

    print(
        f"F1:                   "
        f"{default_metrics['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{default_metrics['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{default_metrics['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{default_metrics['specificity']:.4f}"
    )

    print()
    print("[THRESHOLD COMPARISON]")

    print(
        f"F1 change:             "
        f"{aggregate['f1'] - default_metrics['f1']:+.4f}"
    )
    print(
        f"Precision change:     "
        f"{aggregate['precision'] - default_metrics['precision']:+.4f}"
    )
    print(
        f"Recall change:        "
        f"{aggregate['recall'] - default_metrics['recall']:+.4f}"
    )

    print()
    print("[SPLIT STABILITY]")

    print(
        f"ROC-AUC mean:         "
        f"{split_df['roc_auc'].mean():.4f}"
    )
    print(
        f"ROC-AUC std:          "
        f"{split_df['roc_auc'].std():.4f}"
    )
    print(
        f"ROC-AUC min:          "
        f"{split_df['roc_auc'].min():.4f}"
    )
    print(
        f"ROC-AUC max:          "
        f"{split_df['roc_auc'].max():.4f}"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in diagnostics:
        print(f"- {flag}")

    print()
    print("[OVERALL STATUS]")
    print(
        f"FINAL VALIDATION STATUS: "
        f"{report['overall_status']}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_reports(
        report,
        split_df,
        aggregate,
        default_metrics,
    )

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {REPORT_DIR}"
    )
    print(
        f"JSON report:          "
        f"{REPORT_DIR / 'final_validation_stable_report.json'}"
    )
    print(
        f"Summary report:       "
        f"{REPORT_DIR / 'final_validation_stable_summary.txt'}"
    )
    print(
        f"Split performance:    "
        f"{REPORT_DIR / 'split_performance.csv'}"
    )
    print(
        f"Threshold comparison: "
        f"{REPORT_DIR / 'threshold_comparison.csv'}"
    )

    print()
    print(
        "=" * 64
    )
    print(
        "FINAL STABLE MODEL VALIDATION COMPLETE"
    )
    print(
        "=" * 64
    )


if __name__ == "__main__":
    main()