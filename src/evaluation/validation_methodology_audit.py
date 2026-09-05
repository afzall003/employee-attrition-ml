"""
Validation Methodology Audit
============================

Purpose
-------
Diagnose whether the observed holdout generalization gap is related to
the validation methodology, fixed train/holdout composition, or sampling
variability.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

This script does NOT:
- modify the dataset
- retrain or overwrite the final model
- tune the final model
- retune the untouched holdout
- select a new production threshold

It only performs diagnostic validation analysis.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "final_model_selection"
    / "final_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "validation_methodology_audit"
)

JSON_REPORT = REPORT_DIR / "validation_methodology_audit_report.json"
SUMMARY_REPORT = (
    REPORT_DIR / "validation_methodology_audit_summary.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
OPERATING_THRESHOLD = 0.15

CV_FOLDS = 5
CV_REPEATS = 5

MIN_DISTRIBUTION_DELTA = 0.10


# ============================================================
# HELPERS
# ============================================================

def json_safe(value: Any) -> Any:
    """Convert numpy/pandas objects into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:

    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if "Attrition" not in df.columns:
        raise ValueError(
            "Target column 'Attrition' not found."
        )

    y = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Attrition."
        )

    X = df.drop(columns=["Attrition"])

    return X, y.astype(int)


def identify_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:

    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical = [
        column
        for column in X.columns
        if column not in numerical
    ]

    return numerical, categorical


def build_models(
    X: pd.DataFrame,
) -> dict[str, Pipeline]:

    numerical, categorical = identify_columns(X)

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
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical,
            ),
        ]
    )

    models = {

        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.01,
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=42
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# FIXED HOLDOUT RECONSTRUCTION
# ============================================================

def reconstruct_fixed_holdout(
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=HOLDOUT_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return {
        "X_train": X_train,
        "X_holdout": X_holdout,
        "y_train": y_train,
        "y_holdout": y_holdout,
    }


# ============================================================
# BASIC SPLIT ANALYSIS
# ============================================================

def analyze_split(
    split: dict[str, Any],
) -> dict[str, Any]:

    y_train = split["y_train"]
    y_holdout = split["y_holdout"]

    return {
        "training_rows": int(len(y_train)),
        "holdout_rows": int(len(y_holdout)),
        "training_positive": int(y_train.sum()),
        "training_negative": int((y_train == 0).sum()),
        "holdout_positive": int(y_holdout.sum()),
        "holdout_negative": int((y_holdout == 0).sum()),
        "training_prevalence": float(y_train.mean()),
        "holdout_prevalence": float(y_holdout.mean()),
        "prevalence_delta_pp": float(
            (y_holdout.mean() - y_train.mean()) * 100
        ),
    }


# ============================================================
# FEATURE DISTRIBUTION COMPARISON
# ============================================================

def compare_feature_distributions(
    split: dict[str, Any],
) -> list[dict[str, Any]]:

    X_train = split["X_train"]
    X_holdout = split["X_holdout"]

    numerical = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    results = []

    for feature in numerical:

        train_values = X_train[feature].dropna()
        holdout_values = X_holdout[feature].dropna()

        if len(train_values) == 0 or len(holdout_values) == 0:
            continue

        statistic, p_value = ks_2samp(
            train_values,
            holdout_values,
        )

        train_mean = float(train_values.mean())
        holdout_mean = float(holdout_values.mean())

        pooled_std = np.sqrt(
            (
                train_values.var()
                + holdout_values.var()
            )
            / 2
        )

        if pooled_std == 0:
            standardized_difference = 0.0
        else:
            standardized_difference = (
                holdout_mean - train_mean
            ) / pooled_std

        results.append(
            {
                "feature": feature,
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "train_mean": train_mean,
                "holdout_mean": holdout_mean,
                "standardized_mean_difference":
                    float(standardized_difference),
                "absolute_standardized_difference":
                    float(abs(standardized_difference)),
            }
        )

    results.sort(
        key=lambda x: x[
            "absolute_standardized_difference"
        ],
        reverse=True,
    )

    return results


# ============================================================
# MODEL EVALUATION
# ============================================================

def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:

    predictions = (
        probabilities >= OPERATING_THRESHOLD
    ).astype(int)

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
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
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
    }


def evaluate_fixed_holdout(
    models: dict[str, Pipeline],
    split: dict[str, Any],
) -> dict[str, Any]:

    X_train = split["X_train"]
    X_holdout = split["X_holdout"]
    y_train = split["y_train"]
    y_holdout = split["y_holdout"]

    results = {}

    for name, model in models.items():

        print(f"Evaluating fixed holdout: {name}")

        estimator = clone(model)

        estimator.fit(
            X_train,
            y_train,
        )

        probabilities = estimator.predict_proba(
            X_holdout
        )[:, 1]

        results[name] = calculate_metrics(
            y_holdout,
            probabilities,
        )

    return results


# ============================================================
# STRATIFIED CV
# ============================================================

def evaluate_repeated_cv(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, Any]:

    print()
    print(
        "Running repeated stratified cross-validation..."
    )

    cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    results = {}

    for name, model in models.items():

        print(
            f"Evaluating repeated CV: {name}"
        )

        fold_metrics = []

        for fold_index, (
            train_index,
            validation_index,
        ) in enumerate(cv.split(X, y), start=1):

            estimator = clone(model)

            X_train = X.iloc[train_index]
            X_validation = X.iloc[validation_index]

            y_train = y.iloc[train_index]
            y_validation = y.iloc[validation_index]

            estimator.fit(
                X_train,
                y_train,
            )

            probabilities = estimator.predict_proba(
                X_validation
            )[:, 1]

            metrics = calculate_metrics(
                y_validation,
                probabilities,
            )

            metrics["fold"] = fold_index

            fold_metrics.append(metrics)

        frame = pd.DataFrame(fold_metrics)

        results[name] = {
            "roc_auc_mean":
                float(frame["roc_auc"].mean()),
            "roc_auc_std":
                float(frame["roc_auc"].std()),
            "roc_auc_min":
                float(frame["roc_auc"].min()),
            "roc_auc_max":
                float(frame["roc_auc"].max()),

            "pr_auc_mean":
                float(frame["pr_auc"].mean()),
            "pr_auc_std":
                float(frame["pr_auc"].std()),
            "pr_auc_min":
                float(frame["pr_auc"].min()),
            "pr_auc_max":
                float(frame["pr_auc"].max()),

            "f1_mean":
                float(frame["f1"].mean()),
            "f1_std":
                float(frame["f1"].std()),

            "fold_metrics":
                fold_metrics,
        }

    return results


# ============================================================
# REPEATED CV VS HOLDOUT
# ============================================================

def compare_cv_and_holdout(
    holdout_results: dict[str, Any],
    cv_results: dict[str, Any],
) -> list[dict[str, Any]]:

    comparison = []

    for model_name in holdout_results:

        holdout = holdout_results[model_name]
        cv = cv_results[model_name]

        comparison.append(
            {
                "model": model_name,
                "holdout_roc_auc":
                    holdout["roc_auc"],
                "cv_roc_auc_mean":
                    cv["roc_auc_mean"],
                "roc_auc_delta":
                    holdout["roc_auc"]
                    - cv["roc_auc_mean"],

                "holdout_pr_auc":
                    holdout["pr_auc"],
                "cv_pr_auc_mean":
                    cv["pr_auc_mean"],
                "pr_auc_delta":
                    holdout["pr_auc"]
                    - cv["pr_auc_mean"],

                "holdout_f1":
                    holdout["f1"],
                "cv_f1_mean":
                    cv["f1_mean"],
                "f1_delta":
                    holdout["f1"]
                    - cv["f1_mean"],
            }
        )

    return comparison


# ============================================================
# SOURCE-CODE SPLIT INSPECTION
# ============================================================

def inspect_split_references() -> dict[str, Any]:

    source_files = [
        PROJECT_ROOT
        / "src"
        / "evaluation"
        / "final_model_selection.py",

        PROJECT_ROOT
        / "src"
        / "evaluation"
        / "final_validation.py",

        PROJECT_ROOT
        / "src"
        / "evaluation"
        / "generalization_diagnosis.py",

        PROJECT_ROOT
        / "src"
        / "evaluation"
        / "feature_stability.py",
    ]

    references = []

    patterns = [
        r"train_test_split",
        r"random_state\s*=\s*\d+",
        r"test_size\s*=\s*[\d\.]+",
        r"StratifiedKFold",
        r"RepeatedStratifiedKFold",
        r"KFold",
    ]

    for path in source_files:

        if not path.exists():
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        matches = []

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                text,
            ):

                start = max(
                    0,
                    match.start() - 100,
                )

                end = min(
                    len(text),
                    match.end() + 150,
                )

                matches.append(
                    text[start:end]
                    .replace("\n", " ")
                    .strip()
                )

        references.append(
            {
                "file": str(
                    path.relative_to(PROJECT_ROOT)
                ),
                "matches": matches,
            }
        )

    return references


# ============================================================
# MODEL ARTIFACT CHECK
# ============================================================

def inspect_model_artifact() -> dict[str, Any]:

    result = {
        "exists": MODEL_PATH.exists(),
        "path": str(
            MODEL_PATH.relative_to(PROJECT_ROOT)
        ),
    }

    if not MODEL_PATH.exists():
        return result

    try:
        import joblib

        model = joblib.load(MODEL_PATH)

        result["type"] = type(model).__name__

        if hasattr(model, "steps"):
            result["pipeline_steps"] = [
                name
                for name, _ in model.steps
            ]

        if hasattr(
            model,
            "named_steps",
        ):

            classifier = (
                model.named_steps.get(
                    "classifier"
                )
            )

            if classifier is not None:
                result[
                    "estimator"
                ] = type(classifier).__name__

                if hasattr(
                    classifier,
                    "get_params",
                ):
                    result[
                        "parameters"
                    ] = {
                        str(k): json_safe(v)
                        for k, v in classifier
                        .get_params()
                        .items()
                    }

    except Exception as exc:

        result["load_error"] = str(exc)

    return result


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_flags(
    split_analysis: dict[str, Any],
    distributions: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
) -> list[str]:

    flags = []

    if abs(
        split_analysis["prevalence_delta_pp"]
    ) > 2:

        flags.append(
            "Training and holdout target prevalence "
            "differ by more than 2 percentage points."
        )

    materially_shifted = [
        item
        for item in distributions
        if item[
            "absolute_standardized_difference"
        ] >= MIN_DISTRIBUTION_DELTA
    ]

    if materially_shifted:

        flags.append(
            f"{len(materially_shifted)} numerical features "
            "show a standardized train/holdout distribution "
            "difference of at least 0.10."
        )

    for item in comparison:

        if item["roc_auc_delta"] <= -0.05:

            flags.append(
                f"{item['model']} shows a holdout ROC-AUC "
                "at least 0.05 below its repeated-CV mean."
            )

    if all(
        item["holdout_roc_auc"] < 0.60
        for item in comparison
    ):

        flags.append(
            "All evaluated model families have holdout "
            "ROC-AUC below 0.60."
        )

    if any(
        item["cv_roc_auc_mean"] >= 0.60
        and item["holdout_roc_auc"] < 0.60
        for item in comparison
    ):

        flags.append(
            "Cross-validation indicates useful signal that "
            "does not consistently reproduce on the fixed "
            "holdout."
        )

    if not flags:

        flags.append(
            "No major validation-methodology anomaly was "
            "detected by this diagnostic."
        )

    return flags


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    dataset_info: dict[str, Any],
    split_analysis: dict[str, Any],
    distributions: list[dict[str, Any]],
    holdout_results: dict[str, Any],
    cv_results: dict[str, Any],
    comparison: list[dict[str, Any]],
    flags: list[str],
    artifact_info: dict[str, Any],
) -> str:

    lines = []

    lines.append("=" * 60)
    lines.append(
        "EMPLOYEE ATTRITION — VALIDATION METHODOLOGY AUDIT"
    )
    lines.append("=" * 60)
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Dataset:              {dataset_info['path']}"
    )
    lines.append(
        f"Rows:                 {dataset_info['rows']}"
    )
    lines.append(
        f"Features:             {dataset_info['features']}"
    )
    lines.append(
        f"Target prevalence:    "
        f"{dataset_info['prevalence']:.2%}"
    )

    lines.append("")

    lines.append("[FIXED HOLDOUT]")
    lines.append(
        f"Random state:         {RANDOM_STATE}"
    )
    lines.append(
        f"Holdout fraction:     {HOLDOUT_SIZE:.0%}"
    )
    lines.append(
        f"Training rows:        "
        f"{split_analysis['training_rows']}"
    )
    lines.append(
        f"Holdout rows:         "
        f"{split_analysis['holdout_rows']}"
    )
    lines.append(
        f"Training prevalence:  "
        f"{split_analysis['training_prevalence']:.2%}"
    )
    lines.append(
        f"Holdout prevalence:   "
        f"{split_analysis['holdout_prevalence']:.2%}"
    )
    lines.append(
        f"Difference:            "
        f"{split_analysis['prevalence_delta_pp']:+.2f} pp"
    )

    lines.append("")

    lines.append("[TOP TRAIN/HOLDOUT DISTRIBUTION DIFFERENCES]")

    for item in distributions[:10]:

        lines.append(
            f"{item['feature']:<35} "
            f"KS={item['ks_statistic']:.4f} "
            f"SMD={item['standardized_mean_difference']:+.4f}"
        )

    lines.append("")

    lines.append("[MODEL COMPARISON]")

    for item in comparison:

        lines.append(
            f"{item['model']:<25} "
            f"Holdout ROC-AUC="
            f"{item['holdout_roc_auc']:.4f}  "
            f"CV mean="
            f"{item['cv_roc_auc_mean']:.4f}  "
            f"Delta="
            f"{item['roc_auc_delta']:+.4f}"
        )

    lines.append("")

    lines.append("[REPEATED CV PERFORMANCE]")

    for name, result in cv_results.items():

        lines.append(
            f"{name:<25} "
            f"ROC-AUC="
            f"{result['roc_auc_mean']:.4f} "
            f"+/- "
            f"{result['roc_auc_std']:.4f} "
            f"range="
            f"{result['roc_auc_min']:.4f}-"
            f"{result['roc_auc_max']:.4f}"
        )

    lines.append("")

    lines.append("[FINAL MODEL ARTIFACT]")

    lines.append(
        f"Exists:               "
        f"{artifact_info.get('exists')}"
    )

    if artifact_info.get("type"):
        lines.append(
            f"Type:                 "
            f"{artifact_info['type']}"
        )

    if artifact_info.get("estimator"):
        lines.append(
            f"Estimator:            "
            f"{artifact_info['estimator']}"
        )

    lines.append("")

    lines.append("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        lines.append(f"- {flag}")

    lines.append("")

    lines.append("[OVERALL DIAGNOSIS]")

    max_shift = (
        max(
            (
                x[
                    "absolute_standardized_difference"
                ]
                for x in distributions
            ),
            default=0.0,
        )
    )

    mean_holdout_gap = np.mean(
        [
            x["roc_auc_delta"]
            for x in comparison
        ]
    )

    if (
        mean_holdout_gap <= -0.05
        and max_shift < 0.20
    ):

        lines.append(
            "The fixed holdout underperforms repeated "
            "cross-validation despite limited broad feature "
            "distribution shift. This is consistent with "
            "sampling variability or an unusually difficult "
            "holdout rather than a simple global dataset shift."
        )

    elif max_shift >= 0.20:

        lines.append(
            "The fixed holdout contains meaningful numerical "
            "distribution differences relative to training. "
            "Holdout composition may therefore contribute to "
            "the observed generalization gap."
        )

    else:

        lines.append(
            "The validation methodology does not show a single "
            "dominant failure mode. The remaining performance "
            "gap should be interpreted together with repeated "
            "cross-validation and feature stability results."
        )

    lines.append("")

    lines.append(
        "This audit is diagnostic only. The untouched holdout "
        "must not be used for model or threshold tuning."
    )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Running validation methodology audit...")
    print()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_dataset()

    print("Dataset loaded successfully.")

    print(
        f"Rows:                 {len(X)}"
    )

    print(
        f"Features:             {X.shape[1]}"
    )

    print(
        f"Target prevalence:    {y.mean():.2%}"
    )

    dataset_info = {
        "path": str(
            DATASET_PATH.relative_to(PROJECT_ROOT)
        ),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "prevalence": float(y.mean()),
    }

    print()
    print("Reconstructing fixed 80/20 stratified split...")

    split = reconstruct_fixed_holdout(
        X,
        y,
    )

    split_analysis = analyze_split(
        split
    )

    print(
        f"Training rows:        "
        f"{split_analysis['training_rows']}"
    )

    print(
        f"Holdout rows:         "
        f"{split_analysis['holdout_rows']}"
    )

    print()
    print(
        "Comparing train/holdout numerical distributions..."
    )

    distributions = compare_feature_distributions(
        split
    )

    print()
    print(
        "Building diagnostic model families..."
    )

    models = build_models(X)

    print()
    print(
        "Evaluating fixed holdout..."
    )

    holdout_results = evaluate_fixed_holdout(
        models,
        split,
    )

    cv_results = evaluate_repeated_cv(
        models,
        X,
        y,
    )

    comparison = compare_cv_and_holdout(
        holdout_results,
        cv_results,
    )

    print()
    print(
        "Inspecting validation references in source code..."
    )

    split_references = inspect_split_references()

    print()
    print(
        "Inspecting final model artifact..."
    )

    artifact_info = inspect_model_artifact()

    flags = generate_flags(
        split_analysis,
        distributions,
        comparison,
    )

    summary = build_summary(
        dataset_info,
        split_analysis,
        distributions,
        holdout_results,
        cv_results,
        comparison,
        flags,
        artifact_info,
    )

    report = {
        "timestamp": datetime.now().isoformat(),

        "dataset": dataset_info,

        "configuration": {
            "random_state": RANDOM_STATE,
            "holdout_size": HOLDOUT_SIZE,
            "operating_threshold":
                OPERATING_THRESHOLD,
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
        },

        "fixed_holdout": split_analysis,

        "feature_distributions":
            distributions,

        "holdout_results":
            holdout_results,

        "repeated_cv_results":
            cv_results,

        "cv_vs_holdout":
            comparison,

        "source_code_split_references":
            split_references,

        "model_artifact":
            artifact_info,

        "diagnostic_flags":
            flags,

        "diagnostic_only": True,
    }

    JSON_REPORT.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    SUMMARY_REPORT.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — VALIDATION METHODOLOGY AUDIT"
    )
    print("=" * 60)

    print()
    print("[FIXED HOLDOUT]")

    print(
        f"Training prevalence:  "
        f"{split_analysis['training_prevalence']:.2%}"
    )

    print(
        f"Holdout prevalence:   "
        f"{split_analysis['holdout_prevalence']:.2%}"
    )

    print(
        f"Difference:            "
        f"{split_analysis['prevalence_delta_pp']:+.2f} pp"
    )

    print()
    print("[CV VS HOLDOUT]")

    for item in comparison:

        print(
            f"{item['model']:<25} "
            f"Holdout={item['holdout_roc_auc']:.4f} "
            f"CV={item['cv_roc_auc_mean']:.4f} "
            f"Delta={item['roc_auc_delta']:+.4f}"
        )

    print()
    print("[TOP DISTRIBUTION SHIFTS]")

    for item in distributions[:10]:

        print(
            f"{item['feature']:<35} "
            f"KS={item['ks_statistic']:.4f} "
            f"SMD={item['standardized_mean_difference']:+.4f}"
        )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(f"- {flag}")

    print()
    print("[OVERALL DIAGNOSIS]")

    max_shift = max(
        (
            x[
                "absolute_standardized_difference"
            ]
            for x in distributions
        ),
        default=0.0,
    )

    mean_gap = np.mean(
        [
            x["roc_auc_delta"]
            for x in comparison
        ]
    )

    if (
        mean_gap <= -0.05
        and max_shift < 0.20
    ):

        print(
            "The fixed holdout appears harder than the "
            "average cross-validation split. The evidence "
            "is consistent with sampling variability or "
            "holdout difficulty."
        )

    elif max_shift >= 0.20:

        print(
            "The holdout shows meaningful feature-distribution "
            "differences from training. Holdout composition "
            "may contribute to the observed gap."
        )

    else:

        print(
            "No single validation-methodology failure mode "
            "dominates the current evidence."
        )

    print()
    print("[OUTPUT]")
    print(
        f"JSON report:       {JSON_REPORT}"
    )
    print(
        f"Summary report:    {SUMMARY_REPORT}"
    )

    print()
    print("=" * 60)
    print(
        "VALIDATION METHODOLOGY AUDIT COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()