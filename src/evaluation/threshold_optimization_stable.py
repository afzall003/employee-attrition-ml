"""
Stable-feature threshold optimization for the employee attrition project.

Purpose
-------
Determine an operating classification threshold for the optimized
Random Forest model using the canonical dataset and the validated
10-feature stable feature set.

The threshold is selected using out-of-fold predictions so that the
threshold-selection process does not use predictions from models that
were trained on the same observations.

Usage
-----
python -m src.evaluation.threshold_optimization_stable
"""

from __future__ import annotations

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
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "threshold_optimization_stable"
)


# ============================================================
# DATASET SPECIFICATION
# ============================================================

TARGET_COLUMN = "Attrition"

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


# ============================================================
# MODEL SPECIFICATION
# ============================================================

RANDOM_FOREST_PARAMS = {
    "class_weight": "balanced",
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "n_estimators": 400,
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# VALIDATION SPECIFICATION
# ============================================================

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# Dense threshold grid.
THRESHOLDS = np.round(
    np.arange(0.05, 0.951, 0.01),
    2,
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def json_safe(value: Any) -> Any:
    """
    Convert NumPy / pandas values into JSON-safe Python values.
    """

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.ndarray):
        return [
            json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def encode_target(series: pd.Series) -> pd.Series:
    """
    Encode semantic Attrition labels into binary values.
    """

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    mapping = {
        "No": 0,
        "Yes": 1,
        "no": 0,
        "yes": 1,
        "N": 0,
        "Y": 1,
        "0": 0,
        "1": 1,
    }

    encoded = series.astype(str).str.strip().map(mapping)

    if encoded.isna().any():
        unknown = sorted(
            series[encoded.isna()]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown target labels encountered: {unknown}"
        )

    return encoded.astype(int)


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """
    Calculate classification metrics for a probability threshold.
    """

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

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


# ============================================================
# PIPELINE
# ============================================================

def build_model(
    numerical_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """
    Build the optimized stable-feature Random Forest pipeline.
    """

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
                "encoder",
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
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        **RANDOM_FOREST_PARAMS
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the canonical dataset.
    """

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    missing_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Stable features missing from dataset: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    y = encode_target(df[TARGET_COLUMN])

    X = df[STABLE_FEATURES].copy()

    return X, y


# ============================================================
# FEATURE TYPES
# ============================================================

def determine_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Determine numerical and categorical stable features.
    """

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    return (
        numerical_features,
        categorical_features,
    )


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================

def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[np.ndarray, RepeatedStratifiedKFold]:
    """
    Generate repeated out-of-fold probabilities.

    Each observation receives multiple validation predictions across
    repeated folds. These predictions are averaged to produce a robust
    out-of-fold probability estimate.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    all_probabilities = []
    all_targets = []

    total_splits = N_SPLITS * N_REPEATS

    print()
    print("Generating repeated out-of-fold predictions...")
    print(f"Folds per repeat:      {N_SPLITS}")
    print(f"Repeats:               {N_REPEATS}")
    print(f"Total validation:      {total_splits}")

    for split_number, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(
            f"Validation split "
            f"{split_number}/{total_splits}"
        )

        model = build_model(
            numerical_features,
            categorical_features,
        )

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model.fit(
            X_train,
            y_train,
        )

        probabilities = model.predict_proba(
            X_valid
        )[:, 1]

        all_probabilities.extend(
            probabilities.tolist()
        )

        all_targets.extend(
            y_valid.tolist()
        )

    return (
        np.asarray(all_targets),
        np.asarray(all_probabilities),
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate all candidate thresholds.
    """

    rows = []

    for threshold in THRESHOLDS:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================
# THRESHOLD SELECTION
# ============================================================

def select_threshold(
    threshold_results: pd.DataFrame,
) -> dict[str, Any]:
    """
    Select an operating threshold.

    Primary objective:
        maximize F1

    Tie-breaking:
        balanced accuracy
        recall
        threshold closest to 0.50

    The threshold is selected from out-of-fold predictions rather than
    from the final fitted model.
    """

    ranked = threshold_results.copy()

    ranked["distance_from_default"] = (
        (ranked["threshold"] - 0.50)
        .abs()
    )

    ranked = ranked.sort_values(
        by=[
            "f1",
            "balanced_accuracy",
            "recall",
            "distance_from_default",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    selected = ranked.iloc[0].to_dict()

    return json_safe(selected)


# ============================================================
# DEFAULT VS OPTIMIZED THRESHOLD
# ============================================================

def compare_thresholds(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
) -> pd.DataFrame:
    """
    Compare default 0.50 threshold with optimized threshold.
    """

    comparison = threshold_results[
        threshold_results["threshold"].isin(
            [
                0.50,
                selected_threshold,
            ]
        )
    ].copy()

    comparison["threshold_type"] = np.where(
        comparison["threshold"] == 0.50,
        "default_0.50",
        "optimized",
    )

    return comparison


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_diagnostic_flags(
    threshold_results: pd.DataFrame,
    selected_threshold: dict[str, Any],
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[str]:
    """
    Generate diagnostic observations.
    """

    flags: list[str] = []

    selected = float(
        selected_threshold["threshold"]
    )

    default_row = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            0.50,
        )
    ].iloc[0]

    optimized_row = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            selected,
        )
    ].iloc[0]

    if selected < 0.40:
        flags.append(
            "The selected threshold is substantially below "
            "the default 0.50 threshold, indicating that "
            "higher recall is favored for attrition detection."
        )

    elif selected > 0.60:
        flags.append(
            "The selected threshold is substantially above "
            "the default 0.50 threshold, indicating that "
            "higher precision is favored over sensitivity."
        )

    else:
        flags.append(
            "The selected threshold remains reasonably close "
            "to the default 0.50 operating point."
        )

    if (
        optimized_row["f1"]
        > default_row["f1"] + 0.02
    ):
        flags.append(
            "Threshold optimization materially improves "
            "F1 relative to the default 0.50 threshold."
        )

    elif (
        optimized_row["f1"]
        > default_row["f1"]
    ):
        flags.append(
            "Threshold optimization improves F1 relative "
            "to the default 0.50 threshold."
        )

    else:
        flags.append(
            "Threshold optimization does not improve F1 "
            "relative to the default 0.50 threshold."
        )

    if optimized_row[
        "predicted_positive_rate"
    ] > 0.50:
        flags.append(
            "The optimized threshold predicts attrition for "
            "more than half of observations; this operating "
            "point should be reviewed against business capacity."
        )

    prevalence = float(
        np.mean(y_true)
    )

    predicted_rate = float(
        optimized_row["predicted_positive_rate"]
    )

    if predicted_rate > prevalence * 2:
        flags.append(
            "The optimized predicted-positive rate is more than "
            "twice the observed attrition prevalence."
        )

    if (
        roc_auc_score(
            y_true,
            probabilities,
        )
        < 0.60
    ):
        flags.append(
            "Out-of-fold ROC-AUC remains below 0.60; "
            "threshold optimization cannot compensate for "
            "weak ranking performance."
        )

    else:
        flags.append(
            "Out-of-fold ROC-AUC indicates useful ranking "
            "information before thresholding."
        )

    return flags


# ============================================================
# DIAGNOSIS
# ============================================================

def generate_overall_diagnosis(
    selected_threshold: dict[str, Any],
    threshold_results: pd.DataFrame,
) -> str:
    """
    Generate overall threshold diagnosis.
    """

    selected = float(
        selected_threshold["threshold"]
    )

    default = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            0.50,
        )
    ].iloc[0]

    optimized = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            selected,
        )
    ].iloc[0]

    f1_delta = (
        optimized["f1"]
        - default["f1"]
    )

    recall = optimized["recall"]
    precision = optimized["precision"]

    if f1_delta >= 0.02:
        return (
            f"The optimized threshold of {selected:.2f} "
            f"provides a meaningful improvement over the "
            f"default 0.50 threshold, with F1 increasing by "
            f"{f1_delta:.4f}. The selected operating point "
            f"provides precision of {precision:.4f} and recall "
            f"of {recall:.4f}. It is recommended for final "
            f"validation, subject to business review of the "
            f"precision-recall trade-off."
        )

    if f1_delta > 0:
        return (
            f"The optimized threshold of {selected:.2f} "
            f"provides a modest improvement over the default "
            f"0.50 threshold. The threshold should be carried "
            f"forward to final validation, with particular "
            f"attention to the precision-recall trade-off."
        )

    return (
        "Threshold optimization does not provide a meaningful "
        "improvement over the default 0.50 operating point. "
        "The default threshold should remain the baseline "
        "candidate unless business costs justify a different "
        "operating point."
    )


# ============================================================
# REPORT WRITING
# ============================================================

def save_reports(
    threshold_results: pd.DataFrame,
    comparison: pd.DataFrame,
    selected_threshold: dict[str, Any],
    flags: list[str],
    diagnosis: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    numerical_features: list[str],
    categorical_features: list[str],
) -> None:
    """
    Save CSV, JSON, and human-readable summary reports.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_csv = (
        REPORT_DIR
        / "threshold_results.csv"
    )

    comparison_csv = (
        REPORT_DIR
        / "threshold_comparison.csv"
    )

    json_path = (
        REPORT_DIR
        / "threshold_optimization_stable_report.json"
    )

    summary_path = (
        REPORT_DIR
        / "threshold_optimization_stable_summary.txt"
    )

    threshold_results.to_csv(
        threshold_csv,
        index=False,
    )

    comparison.to_csv(
        comparison_csv,
        index=False,
    )

    oof_roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    oof_pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(y_true)),
            "target": TARGET_COLUMN,
            "target_prevalence": float(
                np.mean(y_true)
            ),
        },
        "stable_features": STABLE_FEATURES,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "model": {
            "name": "Random Forest",
            "parameters": RANDOM_FOREST_PARAMS,
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
            "method": (
                "Repeated stratified out-of-fold "
                "probability generation"
            ),
        },
        "ranking_performance": {
            "roc_auc": float(oof_roc_auc),
            "pr_auc": float(oof_pr_auc),
        },
        "threshold_selection": {
            "objective": "maximize_f1",
            "selected": selected_threshold,
        },
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
        "threshold_results": (
            threshold_results.to_dict(
                orient="records"
            )
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(report),
            file,
            indent=2,
        )

    default_row = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            0.50,
        )
    ].iloc[0]

    selected = float(
        selected_threshold["threshold"]
    )

    optimized_row = threshold_results[
        np.isclose(
            threshold_results["threshold"],
            selected,
        )
    ].iloc[0]

    lines = []

    lines.append(
        "=" * 64
    )
    lines.append(
        "EMPLOYEE ATTRITION — STABLE THRESHOLD OPTIMIZATION"
    )
    lines.append(
        "=" * 64
    )
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Path:                 {DATA_PATH.relative_to(PROJECT_ROOT)}"
    )
    lines.append(
        f"Rows:                 {len(y_true)}"
    )
    lines.append(
        f"Target prevalence:    {np.mean(y_true) * 100:.2f}%"
    )
    lines.append("")

    lines.append("[STABLE FEATURES]")
    for index, feature in enumerate(
        STABLE_FEATURES,
        start=1,
    ):
        lines.append(
            f"{index:2d}. {feature}"
        )
    lines.append("")

    lines.append("[MODEL]")
    lines.append(
        "Random Forest"
    )

    for key, value in RANDOM_FOREST_PARAMS.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append("[VALIDATION DESIGN]")
    lines.append(
        f"Folds per repeat:      {N_SPLITS}"
    )
    lines.append(
        f"Repeats:               {N_REPEATS}"
    )
    lines.append(
        f"Total validation:      {N_SPLITS * N_REPEATS}"
    )
    lines.append(
        "Threshold selection:   Out-of-fold predictions"
    )
    lines.append("")

    lines.append("[RANKING PERFORMANCE]")
    lines.append(
        f"ROC-AUC:               {oof_roc_auc:.4f}"
    )
    lines.append(
        f"PR-AUC:                {oof_pr_auc:.4f}"
    )
    lines.append("")

    lines.append("[THRESHOLD SELECTION]")
    lines.append(
        f"Default threshold:     0.50"
    )
    lines.append(
        f"Selected threshold:    {selected:.2f}"
    )
    lines.append(
        "Selection objective:   Maximize F1"
    )
    lines.append("")

    lines.append("[DEFAULT 0.50]")
    lines.append(
        f"Accuracy:              {default_row['accuracy']:.4f}"
    )
    lines.append(
        f"Balanced Accuracy:     {default_row['balanced_accuracy']:.4f}"
    )
    lines.append(
        f"Precision:             {default_row['precision']:.4f}"
    )
    lines.append(
        f"Recall:                {default_row['recall']:.4f}"
    )
    lines.append(
        f"Specificity:           {default_row['specificity']:.4f}"
    )
    lines.append(
        f"F1:                    {default_row['f1']:.4f}"
    )
    lines.append(
        f"Predicted Positive:    "
        f"{default_row['predicted_positive_rate'] * 100:.2f}%"
    )
    lines.append("")

    lines.append(
        f"[OPTIMIZED {selected:.2f}]"
    )
    lines.append(
        f"Accuracy:              {optimized_row['accuracy']:.4f}"
    )
    lines.append(
        f"Balanced Accuracy:     "
        f"{optimized_row['balanced_accuracy']:.4f}"
    )
    lines.append(
        f"Precision:             {optimized_row['precision']:.4f}"
    )
    lines.append(
        f"Recall:                {optimized_row['recall']:.4f}"
    )
    lines.append(
        f"Specificity:           "
        f"{optimized_row['specificity']:.4f}"
    )
    lines.append(
        f"F1:                    {optimized_row['f1']:.4f}"
    )
    lines.append(
        f"Predicted Positive:    "
        f"{optimized_row['predicted_positive_rate'] * 100:.2f}%"
    )
    lines.append("")

    lines.append("[DIAGNOSTIC FLAGS]")
    for flag in flags:
        lines.append(
            f"- {flag}"
        )
    lines.append("")

    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(
        diagnosis
    )
    lines.append("")

    lines.append("[OUTPUT]")
    lines.append(
        f"Reports:              {REPORT_DIR}"
    )
    lines.append(
        f"Threshold CSV:        {threshold_csv}"
    )
    lines.append(
        f"Comparison CSV:       {comparison_csv}"
    )
    lines.append(
        f"JSON report:          {json_path}"
    )
    lines.append(
        f"Summary report:       {summary_path}"
    )
    lines.append("")

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "Running stable-feature threshold optimization..."
    )

    X, y = load_dataset()

    numerical_features, categorical_features = (
        determine_feature_types(X)
    )

    print()
    print(
        f"Stable features:       {len(STABLE_FEATURES)}"
    )
    print(
        f"Numerical features:    {len(numerical_features)}"
    )
    print(
        f"Categorical features:  {len(categorical_features)}"
    )

    print()
    print("=" * 64)
    print(
        "OUT-OF-FOLD THRESHOLD OPTIMIZATION"
    )
    print("=" * 64)

    y_oof, probabilities = generate_oof_predictions(
        X,
        y,
        numerical_features,
        categorical_features,
    )

    print()
    print(
        "Calculating ranking metrics..."
    )

    oof_roc_auc = roc_auc_score(
        y_oof,
        probabilities,
    )

    oof_pr_auc = average_precision_score(
        y_oof,
        probabilities,
    )

    print(
        f"ROC-AUC: {oof_roc_auc:.4f}"
    )
    print(
        f"PR-AUC:  {oof_pr_auc:.4f}"
    )

    print()
    print(
        "Evaluating candidate thresholds..."
    )

    threshold_results = evaluate_thresholds(
        y_oof,
        probabilities,
    )

    selected_threshold = select_threshold(
        threshold_results
    )

    print()
    print(
        f"Selected threshold: "
        f"{selected_threshold['threshold']:.2f}"
    )
    print(
        f"F1:                  "
        f"{selected_threshold['f1']:.4f}"
    )
    print(
        f"Precision:           "
        f"{selected_threshold['precision']:.4f}"
    )
    print(
        f"Recall:              "
        f"{selected_threshold['recall']:.4f}"
    )
    print(
        f"Specificity:         "
        f"{selected_threshold['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:   "
        f"{selected_threshold['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:  "
        f"{selected_threshold['predicted_positive_rate'] * 100:.2f}%"
    )

    comparison = compare_thresholds(
        threshold_results,
        float(
            selected_threshold["threshold"]
        ),
    )

    flags = generate_diagnostic_flags(
        threshold_results,
        selected_threshold,
        y_oof,
        probabilities,
    )

    diagnosis = generate_overall_diagnosis(
        selected_threshold,
        threshold_results,
    )

    print()
    print(
        "Generating diagnostic flags..."
    )

    print()
    print("=" * 64)
    print(
        "EMPLOYEE ATTRITION — STABLE THRESHOLD OPTIMIZATION"
    )
    print("=" * 64)

    print()
    print("[DATASET]")
    print(
        f"Rows:                 {len(y_oof)}"
    )
    print(
        f"Features:             {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    {np.mean(y_oof) * 100:.2f}%"
    )

    print()
    print("[SELECTED THRESHOLD]")
    print(
        f"Threshold:            "
        f"{selected_threshold['threshold']:.2f}"
    )
    print(
        f"F1:                   "
        f"{selected_threshold['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{selected_threshold['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{selected_threshold['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{selected_threshold['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{selected_threshold['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{selected_threshold['predicted_positive_rate'] * 100:.2f}%"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(
            f"- {flag}"
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    save_reports(
        threshold_results,
        comparison,
        selected_threshold,
        flags,
        diagnosis,
        y_oof,
        probabilities,
        numerical_features,
        categorical_features,
    )

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {REPORT_DIR}"
    )
    print(
        f"Threshold CSV:        "
        f"{REPORT_DIR / 'threshold_results.csv'}"
    )
    print(
        f"Comparison CSV:       "
        f"{REPORT_DIR / 'threshold_comparison.csv'}"
    )
    print(
        f"JSON report:          "
        f"{REPORT_DIR / 'threshold_optimization_stable_report.json'}"
    )
    print(
        f"Summary report:       "
        f"{REPORT_DIR / 'threshold_optimization_stable_summary.txt'}"
    )

    print()
    print("=" * 64)
    print(
        "STABLE THRESHOLD OPTIMIZATION COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()