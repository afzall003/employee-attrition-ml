"""
Employee Attrition — Generalization Diagnosis

Purpose
-------
Diagnose why the optimized model's cross-validation performance does not
generalize well to the untouched holdout set.

This script:
    1. Loads the V2 dataset.
    2. Recreates the 80/20 stratified train/holdout partition.
    3. Loads the existing final_model.joblib artifact.
    4. Generates train/holdout probabilities.
    5. Compares CV-style performance against holdout performance.
    6. Measures numerical feature distribution drift.
    7. Measures categorical distribution drift.
    8. Examines probability distributions.
    9. Evaluates several operating thresholds on the holdout.
   10. Evaluates probability calibration.
   11. Identifies the strongest features available from the model.
   12. Produces JSON and TXT diagnostic reports.

IMPORTANT
---------
This script is diagnostic only.

It does NOT:
    - modify the dataset
    - modify the final model
    - retrain the final model
    - optimize a new threshold
    - overwrite final_model.joblib

Run:
    python -m src.evaluation.generalization_diagnosis
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from scipy.stats import (
    chi2_contingency,
    ks_2samp,
)

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

warnings.filterwarnings("ignore")


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

MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "final_model_selection"
    / "final_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "generalization_diagnosis"
)

JSON_PATH = OUTPUT_DIR / "generalization_diagnosis_report.json"
SUMMARY_PATH = OUTPUT_DIR / "generalization_diagnosis_summary.txt"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

OPERATING_THRESHOLD = 0.15

THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
]

DRIFT_WARNING_THRESHOLD = 0.10
DRIFT_STRONG_THRESHOLD = 0.20


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value: Any) -> float | None:
    """Convert a value to a JSON-safe float."""
    if value is None:
        return None

    try:
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    """Recursively convert NumPy/Pandas objects to JSON-safe values."""

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
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def print_section(title: str) -> None:
    """Print a formatted section header."""

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the V2 employee attrition dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if "Attrition" not in df.columns:
        raise ValueError(
            "Target column 'Attrition' was not found."
        )

    return df


def load_model() -> Any:
    """Load the existing final model artifact."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model artifact not found:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# TARGET PREPARATION
# ============================================================

def prepare_target(df: pd.DataFrame) -> np.ndarray:
    """
    Convert Attrition to binary.

    Yes -> 1
    No  -> 0
    """

    target = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
            }
        )
    )

    if target.isna().any():
        invalid = (
            df.loc[target.isna(), "Attrition"]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unexpected Attrition values found: {invalid}"
        )

    return target.to_numpy(dtype=int)


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the raw dataframe for the model.

    Attrition is removed because it is the target.

    Employee_ID is removed because it is an identifier rather than
    a predictive employee attribute.
    """

    X = df.drop(
        columns=["Attrition"],
        errors="ignore",
    ).copy()

    if "Employee_ID" in X.columns:
        X = X.drop(columns=["Employee_ID"])

    return X


# ============================================================
# MODEL PROBABILITIES
# ============================================================

def generate_probabilities(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate positive-class probabilities."""

    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "Loaded model does not expose predict_proba()."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError(
            "Unexpected probability output shape."
        )

    if probabilities.shape[1] < 2:
        raise ValueError(
            "Model does not appear to provide binary-class probabilities."
        )

    return probabilities[:, 1]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate classification and ranking metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )
    except ValueError:
        pr_auc = np.nan

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

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    predicted_positive_rate = (
        predictions.mean()
        * 100.0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": safe_float(threshold),
        "roc_auc": safe_float(roc_auc),
        "pr_auc": safe_float(pr_auc),
        "precision": safe_float(precision),
        "recall": safe_float(recall),
        "f1": safe_float(f1),
        "accuracy": safe_float(accuracy),
        "predicted_positive_rate": safe_float(
            predicted_positive_rate
        ),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


# ============================================================
# PROBABILITY DIAGNOSTICS
# ============================================================

def probability_statistics(
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Calculate probability distribution statistics."""

    percentiles = {
        "p01": 1,
        "p05": 5,
        "p10": 10,
        "p25": 25,
        "p50": 50,
        "p75": 75,
        "p90": 90,
        "p95": 95,
        "p99": 99,
    }

    result = {
        "minimum": safe_float(np.min(probabilities)),
        "maximum": safe_float(np.max(probabilities)),
        "mean": safe_float(np.mean(probabilities)),
        "std": safe_float(np.std(probabilities)),
    }

    for name, percentile in percentiles.items():
        result[name] = safe_float(
            np.percentile(
                probabilities,
                percentile,
            )
        )

    return result


def probability_band_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Analyze actual attrition rates across probability bands.
    """

    bands = [
        (0.00, 0.10),
        (0.10, 0.15),
        (0.15, 0.20),
        (0.20, 0.25),
        (0.25, 0.30),
        (0.30, 0.40),
        (0.40, 0.50),
        (0.50, 1.01),
    ]

    results = []

    for lower, upper in bands:

        mask = (
            (probabilities >= lower)
            & (probabilities < upper)
        )

        count = int(mask.sum())

        if count == 0:
            results.append(
                {
                    "lower_probability": lower,
                    "upper_probability": upper,
                    "rows": 0,
                    "actual_attrition_rate": None,
                    "mean_probability": None,
                }
            )

            continue

        actual_rate = (
            y_true[mask].mean()
            * 100.0
        )

        mean_probability = (
            probabilities[mask].mean()
        )

        results.append(
            {
                "lower_probability": lower,
                "upper_probability": upper,
                "rows": count,
                "actual_attrition_rate": safe_float(
                    actual_rate
                ),
                "mean_probability": safe_float(
                    mean_probability
                ),
            }
        )

    return results


# ============================================================
# THRESHOLD DIAGNOSTICS
# ============================================================

def threshold_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, Any]]:
    """Evaluate holdout performance across thresholds."""

    results = []

    for threshold in THRESHOLDS:
        results.append(
            calculate_metrics(
                y_true,
                probabilities,
                threshold,
            )
        )

    return results


# ============================================================
# TRAIN / HOLDOUT DISTRIBUTION DRIFT
# ============================================================

def numerical_drift(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Calculate Kolmogorov-Smirnov drift for numerical features.

    KS statistic interpretation:
        small value -> distributions are similar
        larger value -> stronger distribution difference
    """

    results = []

    numeric_columns = train_df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for feature in numeric_columns:

        if feature not in holdout_df.columns:
            continue

        train_values = (
            pd.to_numeric(
                train_df[feature],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )

        holdout_values = (
            pd.to_numeric(
                holdout_df[feature],
                errors="coerce",
            )
            .dropna()
            .to_numpy()
        )

        if len(train_values) == 0 or len(holdout_values) == 0:
            continue

        statistic, p_value = ks_2samp(
            train_values,
            holdout_values,
        )

        train_mean = np.mean(train_values)
        holdout_mean = np.mean(holdout_values)

        train_std = np.std(train_values)
        holdout_std = np.std(holdout_values)

        pooled_std = math.sqrt(
            (
                train_std ** 2
                + holdout_std ** 2
            )
            / 2.0
        )

        if pooled_std > 0:
            standardized_difference = (
                holdout_mean - train_mean
            ) / pooled_std
        else:
            standardized_difference = 0.0

        results.append(
            {
                "feature": feature,
                "ks_statistic": safe_float(statistic),
                "ks_p_value": safe_float(p_value),
                "train_mean": safe_float(train_mean),
                "holdout_mean": safe_float(holdout_mean),
                "standardized_mean_difference": safe_float(
                    standardized_difference
                ),
            }
        )

    results.sort(
        key=lambda row: row["ks_statistic"] or 0,
        reverse=True,
    )

    return results


# ============================================================
# CATEGORICAL DRIFT
# ============================================================

def categorical_drift(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Compare categorical feature distributions between train
    and holdout using total variation distance and chi-square.
    """

    results = []

    categorical_columns = train_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    for feature in categorical_columns:

        if feature not in holdout_df.columns:
            continue

        train_counts = (
            train_df[feature]
            .astype(str)
            .value_counts()
        )

        holdout_counts = (
            holdout_df[feature]
            .astype(str)
            .value_counts()
        )

        categories = sorted(
            set(train_counts.index)
            | set(holdout_counts.index)
        )

        train_distribution = (
            train_counts
            .reindex(categories, fill_value=0)
            / len(train_df)
        )

        holdout_distribution = (
            holdout_counts
            .reindex(categories, fill_value=0)
            / len(holdout_df)
        )

        total_variation = (
            0.5
            * np.abs(
                train_distribution
                - holdout_distribution
            ).sum()
        )

        contingency = np.array(
            [
                [
                    train_counts.get(
                        category,
                        0,
                    )
                    for category in categories
                ],
                [
                    holdout_counts.get(
                        category,
                        0,
                    )
                    for category in categories
                ],
            ],
            dtype=float,
        )

        try:
            chi2, p_value, _, _ = chi2_contingency(
                contingency
            )
        except ValueError:
            chi2 = np.nan
            p_value = np.nan

        results.append(
            {
                "feature": feature,
                "total_variation_distance": safe_float(
                    total_variation
                ),
                "chi_square": safe_float(chi2),
                "chi_square_p_value": safe_float(p_value),
                "categories": len(categories),
            }
        )

    results.sort(
        key=lambda row: (
            row["total_variation_distance"]
            if row["total_variation_distance"] is not None
            else 0
        ),
        reverse=True,
    )

    return results


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def target_distribution(
    y_train: np.ndarray,
    y_holdout: np.ndarray,
) -> dict[str, Any]:
    """Compare target prevalence."""

    train_rate = (
        y_train.mean()
        * 100.0
    )

    holdout_rate = (
        y_holdout.mean()
        * 100.0
    )

    difference = (
        holdout_rate
        - train_rate
    )

    return {
        "train_rows": int(len(y_train)),
        "holdout_rows": int(len(y_holdout)),
        "train_attrition_count": int(y_train.sum()),
        "holdout_attrition_count": int(y_holdout.sum()),
        "train_attrition_rate": safe_float(
            train_rate
        ),
        "holdout_attrition_rate": safe_float(
            holdout_rate
        ),
        "difference_percentage_points": safe_float(
            difference
        ),
    }


# ============================================================
# CROSS-VALIDATED TRAIN PERFORMANCE
# ============================================================

def cross_validated_training_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> dict[str, Any]:
    """
    Generate out-of-fold probabilities on the training partition.

    This is used to compare:
        CV / OOF performance
        versus
        untouched holdout performance.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print(
        "Generating out-of-fold probabilities on training data..."
    )

    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    metrics = calculate_metrics(
        y_train,
        oof_probabilities,
        OPERATING_THRESHOLD,
    )

    metrics["probability_statistics"] = (
        probability_statistics(
            oof_probabilities
        )
    )

    return {
        "metrics": metrics,
        "probabilities": oof_probabilities,
    }


# ============================================================
# CALIBRATION
# ============================================================

def calibration_analysis(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate probability calibration.

    Brier score:
        lower is better.

    Calibration curve:
        compares predicted probability with observed event rate.
    """

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    fraction_positive, mean_predicted = calibration_curve(
        y_true,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    curve = []

    for predicted, observed in zip(
        mean_predicted,
        fraction_positive,
    ):
        curve.append(
            {
                "mean_predicted_probability": safe_float(
                    predicted
                ),
                "observed_positive_rate": safe_float(
                    observed
                ),
            }
        )

    return {
        "brier_score": safe_float(brier),
        "calibration_curve": curve,
    }


# ============================================================
# MODEL STRUCTURE / FEATURE INFORMATION
# ============================================================

def extract_model_information(
    model: Any,
) -> dict[str, Any]:
    """Extract useful information from a Pipeline if available."""

    information: dict[str, Any] = {
        "model_type": type(model).__name__,
    }

    steps = getattr(
        model,
        "named_steps",
        None,
    )

    if steps is None:
        information["pipeline_steps"] = []
        return information

    information["pipeline_steps"] = list(
        steps.keys()
    )

    estimator = None

    for name in reversed(list(steps.keys())):

        candidate = steps[name]

        if hasattr(candidate, "coef_") or hasattr(
            candidate,
            "feature_importances_",
        ):
            estimator = candidate
            break

    if estimator is None:
        return information

    information["estimator_type"] = (
        type(estimator).__name__
    )

    try:
        params = estimator.get_params()

        selected_params = {}

        for key in [
            "C",
            "class_weight",
            "max_iter",
            "solver",
            "penalty",
            "n_estimators",
            "learning_rate",
            "max_depth",
            "min_samples_leaf",
        ]:
            if key in params:
                selected_params[key] = params[key]

        information["estimator_parameters"] = (
            json_safe(selected_params)
        )

    except Exception:
        pass

    return information


def extract_feature_importance(
    model: Any,
    X: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Attempt to recover feature importance information from the
    final estimator.

    Handles common Pipeline + preprocessing structures.
    """

    steps = getattr(
        model,
        "named_steps",
        None,
    )

    if steps is None:
        return []

    estimator = None

    for name in reversed(list(steps.keys())):

        candidate = steps[name]

        if hasattr(candidate, "coef_") or hasattr(
            candidate,
            "feature_importances_",
        ):
            estimator = candidate
            break

    if estimator is None:
        return []

    preprocessor = None

    for name, step in steps.items():

        if (
            hasattr(step, "get_feature_names_out")
            and step is not estimator
        ):
            preprocessor = step
            break

    feature_names = None

    if preprocessor is not None:

        try:
            feature_names = (
                preprocessor
                .get_feature_names_out()
                .tolist()
            )
        except Exception:
            feature_names = None

    if feature_names is None:
        feature_names = X.columns.tolist()

    importances = None
    importance_type = None

    if hasattr(estimator, "coef_"):

        coefficients = np.asarray(
            estimator.coef_
        )

        if coefficients.ndim == 2:
            coefficients = coefficients[0]

        importances = np.abs(
            coefficients
        )

        importance_type = "absolute_coefficient"

    elif hasattr(
        estimator,
        "feature_importances_",
    ):

        importances = np.asarray(
            estimator.feature_importances_
        )

        importance_type = "feature_importance"

    if importances is None:
        return []

    if len(feature_names) != len(importances):

        feature_names = [
            f"feature_{i}"
            for i in range(len(importances))
        ]

    result = []

    for feature, importance in zip(
        feature_names,
        importances,
    ):
        result.append(
            {
                "feature": str(feature),
                "importance": safe_float(
                    importance
                ),
                "importance_type": importance_type,
            }
        )

    result.sort(
        key=lambda row: (
            row["importance"]
            if row["importance"] is not None
            else 0
        ),
        reverse=True,
    )

    return result


# ============================================================
# GENERALIZATION GAP
# ============================================================

def calculate_generalization_gap(
    cv_metrics: dict[str, Any],
    holdout_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Calculate CV-to-holdout performance gaps."""

    result = {}

    for metric in [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]:

        cv_value = cv_metrics.get(metric)
        holdout_value = holdout_metrics.get(metric)

        if (
            cv_value is None
            or holdout_value is None
        ):
            result[metric] = None
            continue

        result[metric] = safe_float(
            holdout_value - cv_value
        )

    return result


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

def create_diagnosis(
    target_info: dict[str, Any],
    numerical_drift_info: list[dict[str, Any]],
    categorical_drift_info: list[dict[str, Any]],
    cv_metrics: dict[str, Any],
    holdout_metrics: dict[str, Any],
    holdout_probabilities: np.ndarray,
    threshold_metrics: list[dict[str, Any]],
    calibration_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Produce evidence-based diagnostic flags.

    These are indicators, not final modeling decisions.
    """

    flags = []

    # --------------------------------------------------------
    # Target prevalence drift
    # --------------------------------------------------------

    target_difference = abs(
        target_info[
            "difference_percentage_points"
        ]
    )

    if target_difference >= 5.0:

        flags.append(
            "Meaningful train/holdout target prevalence difference detected."
        )

    elif target_difference >= 2.0:

        flags.append(
            "Moderate train/holdout target prevalence difference detected."
        )

    # --------------------------------------------------------
    # Numerical drift
    # --------------------------------------------------------

    strong_numerical_drift = [
        row
        for row in numerical_drift_info
        if (
            row["ks_statistic"] is not None
            and row["ks_statistic"]
            >= DRIFT_STRONG_THRESHOLD
        )
    ]

    warning_numerical_drift = [
        row
        for row in numerical_drift_info
        if (
            row["ks_statistic"] is not None
            and row["ks_statistic"]
            >= DRIFT_WARNING_THRESHOLD
        )
    ]

    if strong_numerical_drift:

        flags.append(
            f"Strong numerical distribution drift detected in "
            f"{len(strong_numerical_drift)} feature(s)."
        )

    elif warning_numerical_drift:

        flags.append(
            f"Moderate numerical distribution drift detected in "
            f"{len(warning_numerical_drift)} feature(s)."
        )

    # --------------------------------------------------------
    # Categorical drift
    # --------------------------------------------------------

    strong_categorical_drift = [
        row
        for row in categorical_drift_info
        if (
            row["total_variation_distance"] is not None
            and row["total_variation_distance"]
            >= DRIFT_STRONG_THRESHOLD
        )
    ]

    warning_categorical_drift = [
        row
        for row in categorical_drift_info
        if (
            row["total_variation_distance"] is not None
            and row["total_variation_distance"]
            >= DRIFT_WARNING_THRESHOLD
        )
    ]

    if strong_categorical_drift:

        flags.append(
            f"Strong categorical distribution drift detected in "
            f"{len(strong_categorical_drift)} feature(s)."
        )

    elif warning_categorical_drift:

        flags.append(
            f"Moderate categorical distribution drift detected in "
            f"{len(warning_categorical_drift)} feature(s)."
        )

    # --------------------------------------------------------
    # Holdout positive prediction rate
    # --------------------------------------------------------

    holdout_positive_rate = (
        holdout_metrics[
            "predicted_positive_rate"
        ]
    )

    actual_rate = (
        target_info[
            "holdout_attrition_rate"
        ]
    )

    if holdout_positive_rate > 2 * actual_rate:

        flags.append(
            "The operating threshold predicts substantially more "
            "positive cases than the observed holdout attrition rate."
        )

    # --------------------------------------------------------
    # Probability compression / low separation
    # --------------------------------------------------------

    probability_range = (
        float(np.max(holdout_probabilities))
        - float(np.min(holdout_probabilities))
    )

    if probability_range < 0.40:

        flags.append(
            "Holdout probability range is relatively compressed, "
            "which may limit threshold separation."
        )

    # --------------------------------------------------------
    # ROC-AUC degradation
    # --------------------------------------------------------

    cv_auc = cv_metrics.get("roc_auc")
    holdout_auc = holdout_metrics.get("roc_auc")

    if (
        cv_auc is not None
        and holdout_auc is not None
        and holdout_auc < cv_auc - 0.05
    ):

        flags.append(
            "Holdout ROC-AUC is more than 0.05 below the training "
            "out-of-fold ROC-AUC."
        )

    # --------------------------------------------------------
    # PR-AUC degradation
    # --------------------------------------------------------

    cv_pr = cv_metrics.get("pr_auc")
    holdout_pr = holdout_metrics.get("pr_auc")

    if (
        cv_pr is not None
        and holdout_pr is not None
        and holdout_pr < cv_pr - 0.05
    ):

        flags.append(
            "Holdout PR-AUC is more than 0.05 below the training "
            "out-of-fold PR-AUC."
        )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    brier = calibration_info.get(
        "brier_score"
    )

    if brier is not None and brier > 0.20:

        flags.append(
            "Brier score indicates potentially poor probability calibration."
        )

    # --------------------------------------------------------
    # Threshold behavior
    # --------------------------------------------------------

    best_f1 = max(
        threshold_metrics,
        key=lambda row: (
            row["f1"]
            if row["f1"] is not None
            else -1
        ),
    )

    if (
        best_f1["threshold"]
        != OPERATING_THRESHOLD
    ):

        flags.append(
            "The previously selected operating threshold is not the "
            "best F1 threshold on this untouched holdout; this is "
            "diagnostic evidence only and must not be used to retune "
            "the final model on the holdout."
        )

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    if strong_numerical_drift or strong_categorical_drift:

        overall = (
            "Potential distribution shift is a major candidate "
            "explanation for the CV-to-holdout degradation."
        )

    elif (
        cv_auc is not None
        and holdout_auc is not None
        and holdout_auc < cv_auc - 0.05
    ):

        overall = (
            "The model shows a meaningful generalization gap. "
            "Further investigation of feature stability, data "
            "construction, and validation methodology is recommended."
        )

    elif holdout_positive_rate > 2 * actual_rate:

        overall = (
            "The current operating threshold is producing excessive "
            "positive predictions on the holdout. Threshold behavior "
            "needs to be separated from underlying ranking performance."
        )

    else:

        overall = (
            "No single dominant generalization failure was detected. "
            "Further model and feature analysis is recommended."
        )

    return {
        "overall_diagnosis": overall,
        "diagnostic_flags": flags,
        "strong_numerical_drift_features": [
            row["feature"]
            for row in strong_numerical_drift
        ],
        "warning_numerical_drift_features": [
            row["feature"]
            for row in warning_numerical_drift
        ],
        "strong_categorical_drift_features": [
            row["feature"]
            for row in strong_categorical_drift
        ],
        "warning_categorical_drift_features": [
            row["feature"]
            for row in warning_categorical_drift
        ],
        "best_holdout_f1_threshold_diagnostic": best_f1[
            "threshold"
        ],
    }


# ============================================================
# TEXT REPORT
# ============================================================

def write_summary_report(
    report: dict[str, Any],
) -> None:
    """Write a human-readable diagnostic summary."""

    dataset = report["dataset"]
    target = report["target_distribution"]
    cv = report["training_oof"]["metrics"]
    holdout = report["holdout"]["metrics"]
    gap = report["generalization_gap"]
    probability = report["holdout"]["probability_statistics"]
    diagnosis = report["diagnosis"]

    lines = []

    lines.append(
        "EMPLOYEE ATTRITION — GENERALIZATION DIAGNOSIS"
    )
    lines.append("=" * 60)
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 {dataset['rows']}"
    )
    lines.append(
        f"Features:             {dataset['features']}"
    )
    lines.append(
        f"Training rows:        {dataset['training_rows']}"
    )
    lines.append(
        f"Holdout rows:         {dataset['holdout_rows']}"
    )
    lines.append("")

    lines.append("[TARGET DISTRIBUTION]")
    lines.append(
        f"Training attrition:   "
        f"{target['train_attrition_rate']:.2f}%"
    )
    lines.append(
        f"Holdout attrition:    "
        f"{target['holdout_attrition_rate']:.2f}%"
    )
    lines.append(
        f"Difference:            "
        f"{target['difference_percentage_points']:+.2f} pp"
    )
    lines.append("")

    lines.append("[TRAINING OUT-OF-FOLD PERFORMANCE]")
    lines.append(
        f"ROC-AUC:              "
        f"{cv['roc_auc']:.4f}"
    )
    lines.append(
        f"PR-AUC:               "
        f"{cv['pr_auc']:.4f}"
    )
    lines.append(
        f"Precision @ 0.15:     "
        f"{cv['precision']:.4f}"
    )
    lines.append(
        f"Recall @ 0.15:        "
        f"{cv['recall']:.4f}"
    )
    lines.append(
        f"F1 @ 0.15:            "
        f"{cv['f1']:.4f}"
    )
    lines.append("")

    lines.append("[HOLDOUT PERFORMANCE]")
    lines.append(
        f"ROC-AUC:              "
        f"{holdout['roc_auc']:.4f}"
    )
    lines.append(
        f"PR-AUC:               "
        f"{holdout['pr_auc']:.4f}"
    )
    lines.append(
        f"Precision @ 0.15:     "
        f"{holdout['precision']:.4f}"
    )
    lines.append(
        f"Recall @ 0.15:        "
        f"{holdout['recall']:.4f}"
    )
    lines.append(
        f"F1 @ 0.15:            "
        f"{holdout['f1']:.4f}"
    )
    lines.append(
        f"Accuracy @ 0.15:      "
        f"{holdout['accuracy']:.4f}"
    )
    lines.append(
        f"Predicted positive:   "
        f"{holdout['predicted_positive_rate']:.2f}%"
    )
    lines.append("")

    lines.append("[GENERALIZATION GAP]")
    lines.append(
        f"ROC-AUC delta:        "
        f"{gap['roc_auc']:+.4f}"
    )
    lines.append(
        f"PR-AUC delta:         "
        f"{gap['pr_auc']:+.4f}"
    )
    lines.append(
        f"F1 delta:             "
        f"{gap['f1']:+.4f}"
    )
    lines.append(
        f"Accuracy delta:       "
        f"{gap['accuracy']:+.4f}"
    )
    lines.append("")

    lines.append("[HOLDOUT PROBABILITY DISTRIBUTION]")
    lines.append(
        f"Minimum:              "
        f"{probability['minimum']:.4f}"
    )
    lines.append(
        f"P05:                  "
        f"{probability['p05']:.4f}"
    )
    lines.append(
        f"P25:                  "
        f"{probability['p25']:.4f}"
    )
    lines.append(
        f"Median:               "
        f"{probability['p50']:.4f}"
    )
    lines.append(
        f"P75:                  "
        f"{probability['p75']:.4f}"
    )
    lines.append(
        f"P95:                  "
        f"{probability['p95']:.4f}"
    )
    lines.append(
        f"Maximum:              "
        f"{probability['maximum']:.4f}"
    )
    lines.append("")

    lines.append("[TOP NUMERICAL DRIFT]")
    for row in report["numerical_drift"][:10]:

        lines.append(
            f"{row['feature']:<35}"
            f"KS={row['ks_statistic']:.4f}  "
            f"p={row['ks_p_value']:.4f}"
        )

    lines.append("")

    lines.append("[TOP CATEGORICAL DRIFT]")
    for row in report["categorical_drift"][:10]:

        lines.append(
            f"{row['feature']:<35}"
            f"TVD={row['total_variation_distance']:.4f}  "
            f"p={row['chi_square_p_value']:.4f}"
        )

    lines.append("")

    lines.append("[DIAGNOSTIC FLAGS]")

    if diagnosis["diagnostic_flags"]:

        for flag in diagnosis["diagnostic_flags"]:
            lines.append(
                f"- {flag}"
            )

    else:
        lines.append(
            "- No major diagnostic flags detected."
        )

    lines.append("")

    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(
        diagnosis["overall_diagnosis"]
    )

    lines.append("")
    lines.append(
        "IMPORTANT: Holdout threshold results are diagnostic only. "
        "Do not retune the final operating threshold using the untouched "
        "holdout partition."
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete generalization diagnosis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Running generalization diagnosis..."
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_dataset()

    y = prepare_target(df)
    X = prepare_features(df)

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Features:             {X.shape[1]}"
    )

    print(
        f"Target prevalence:    "
        f"{y.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Recreate untouched holdout partition
    # --------------------------------------------------------

    from sklearn.model_selection import train_test_split

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training rows:        {len(X_train)}"
    )

    print(
        f"Holdout rows:         {len(X_holdout)}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print(
        "Loading final model artifact..."
    )

    model = load_model()

    print(
        f"Model:                "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # Generate holdout probabilities
    # --------------------------------------------------------

    print()
    print(
        "Generating holdout probabilities..."
    )

    holdout_probabilities = generate_probabilities(
        model,
        X_holdout,
    )

    # --------------------------------------------------------
    # Generate training OOF probabilities
    # --------------------------------------------------------

    training_oof = cross_validated_training_metrics(
        model,
        X_train,
        y_train,
    )

    training_oof_probabilities = (
        training_oof["probabilities"]
    )

    # --------------------------------------------------------
    # Holdout metrics
    # --------------------------------------------------------

    print()
    print(
        "Calculating holdout metrics..."
    )

    holdout_metrics = calculate_metrics(
        y_holdout,
        holdout_probabilities,
        OPERATING_THRESHOLD,
    )

    # --------------------------------------------------------
    # Probability diagnostics
    # --------------------------------------------------------

    print(
        "Analyzing probability distribution..."
    )

    holdout_probability_stats = (
        probability_statistics(
            holdout_probabilities
        )
    )

    oof_probability_stats = (
        probability_statistics(
            training_oof_probabilities
        )
    )

    probability_bands = (
        probability_band_analysis(
            y_holdout,
            holdout_probabilities,
        )
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print(
        "Analyzing threshold behavior..."
    )

    holdout_threshold_metrics = (
        threshold_analysis(
            y_holdout,
            holdout_probabilities,
        )
    )

    # --------------------------------------------------------
    # Train / holdout drift
    # --------------------------------------------------------

    print(
        "Running numerical distribution drift analysis..."
    )

    numerical_drift_results = numerical_drift(
        X_train,
        X_holdout,
    )

    print(
        "Running categorical distribution drift analysis..."
    )

    categorical_drift_results = categorical_drift(
        X_train,
        X_holdout,
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    target_info = target_distribution(
        y_train,
        y_holdout,
    )

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print(
        "Running probability calibration analysis..."
    )

    calibration_info = calibration_analysis(
        y_holdout,
        holdout_probabilities,
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_information = (
        extract_model_information(
            model
        )
    )

    feature_importance = (
        extract_feature_importance(
            model,
            X,
        )
    )

    # --------------------------------------------------------
    # Generalization gap
    # --------------------------------------------------------

    generalization_gap = (
        calculate_generalization_gap(
            training_oof["metrics"],
            holdout_metrics,
        )
    )

    # --------------------------------------------------------
    # Diagnosis
    # --------------------------------------------------------

    diagnosis = create_diagnosis(
        target_info=target_info,
        numerical_drift_info=numerical_drift_results,
        categorical_drift_info=categorical_drift_results,
        cv_metrics=training_oof["metrics"],
        holdout_metrics=holdout_metrics,
        holdout_probabilities=holdout_probabilities,
        threshold_metrics=holdout_threshold_metrics,
        calibration_info=calibration_info,
    )

    # --------------------------------------------------------
    # Assemble report
    # --------------------------------------------------------

    report = {
        "metadata": {
            "analysis": (
                "Employee Attrition Generalization Diagnosis"
            ),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "operating_threshold": OPERATING_THRESHOLD,
        },

        "dataset": {
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "training_rows": int(len(X_train)),
            "holdout_rows": int(len(X_holdout)),
            "data_path": str(DATA_PATH),
            "model_path": str(MODEL_PATH),
        },

        "target_distribution": target_info,

        "model": model_information,

        "training_oof": {
            "metrics": training_oof["metrics"],
            "probability_statistics": (
                oof_probability_stats
            ),
        },

        "holdout": {
            "metrics": holdout_metrics,
            "probability_statistics": (
                holdout_probability_stats
            ),
            "probability_bands": probability_bands,
            "threshold_analysis": (
                holdout_threshold_metrics
            ),
            "calibration": calibration_info,
        },

        "generalization_gap": generalization_gap,

        "numerical_drift": (
            numerical_drift_results
        ),

        "categorical_drift": (
            categorical_drift_results
        ),

        "feature_importance": feature_importance,

        "diagnosis": diagnosis,
    }

    report = json_safe(report)

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save text summary
    # --------------------------------------------------------

    write_summary_report(
        report
    )

    # --------------------------------------------------------
    # Terminal output
    # --------------------------------------------------------

    print_section(
        "EMPLOYEE ATTRITION — GENERALIZATION DIAGNOSIS"
    )

    print(
        "[DATASET]"
    )

    print(
        f"Rows:                 "
        f"{report['dataset']['rows']}"
    )

    print(
        f"Features:             "
        f"{report['dataset']['features']}"
    )

    print(
        f"Training rows:        "
        f"{report['dataset']['training_rows']}"
    )

    print(
        f"Holdout rows:         "
        f"{report['dataset']['holdout_rows']}"
    )

    print()

    print(
        "[TARGET DISTRIBUTION]"
    )

    print(
        f"Training attrition:   "
        f"{target_info['train_attrition_rate']:.2f}%"
    )

    print(
        f"Holdout attrition:    "
        f"{target_info['holdout_attrition_rate']:.2f}%"
    )

    print(
        f"Difference:            "
        f"{target_info['difference_percentage_points']:+.2f} pp"
    )

    print()

    print(
        "[TRAINING OUT-OF-FOLD PERFORMANCE]"
    )

    print(
        f"ROC-AUC:              "
        f"{training_oof['metrics']['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{training_oof['metrics']['pr_auc']:.4f}"
    )

    print(
        f"F1 @ 0.15:            "
        f"{training_oof['metrics']['f1']:.4f}"
    )

    print()

    print(
        "[HOLDOUT PERFORMANCE]"
    )

    print(
        f"ROC-AUC:              "
        f"{holdout_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{holdout_metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision @ 0.15:     "
        f"{holdout_metrics['precision']:.4f}"
    )

    print(
        f"Recall @ 0.15:        "
        f"{holdout_metrics['recall']:.4f}"
    )

    print(
        f"F1 @ 0.15:            "
        f"{holdout_metrics['f1']:.4f}"
    )

    print(
        f"Accuracy @ 0.15:      "
        f"{holdout_metrics['accuracy']:.4f}"
    )

    print(
        f"Predicted positive:   "
        f"{holdout_metrics['predicted_positive_rate']:.2f}%"
    )

    print()

    print(
        "[GENERALIZATION GAP]"
    )

    print(
        f"ROC-AUC delta:        "
        f"{generalization_gap['roc_auc']:+.4f}"
    )

    print(
        f"PR-AUC delta:         "
        f"{generalization_gap['pr_auc']:+.4f}"
    )

    print(
        f"F1 delta:             "
        f"{generalization_gap['f1']:+.4f}"
    )

    print(
        f"Accuracy delta:       "
        f"{generalization_gap['accuracy']:+.4f}"
    )

    print()

    print(
        "[TOP NUMERICAL DRIFT]"
    )

    for row in numerical_drift_results[:10]:

        print(
            f"{row['feature']:<35}"
            f"KS={row['ks_statistic']:.4f}  "
            f"p={row['ks_p_value']:.4f}"
        )

    print()

    print(
        "[TOP CATEGORICAL DRIFT]"
    )

    for row in categorical_drift_results[:10]:

        print(
            f"{row['feature']:<35}"
            f"TVD={row['total_variation_distance']:.4f}  "
            f"p={row['chi_square_p_value']:.4f}"
        )

    print()

    print(
        "[PROBABILITY DISTRIBUTION]"
    )

    print(
        f"Minimum probability: "
        f"{holdout_probability_stats['minimum']:.4f}"
    )

    print(
        f"Maximum probability: "
        f"{holdout_probability_stats['maximum']:.4f}"
    )

    print(
        f"Mean probability:    "
        f"{holdout_probability_stats['mean']:.4f}"
    )

    print(
        f"Median probability:  "
        f"{holdout_probability_stats['p50']:.4f}"
    )

    print()

    print(
        "[CALIBRATION]"
    )

    print(
        f"Brier score:          "
        f"{calibration_info['brier_score']:.4f}"
    )

    print()

    print(
        "[DIAGNOSTIC FLAGS]"
    )

    if diagnosis["diagnostic_flags"]:

        for flag in diagnosis["diagnostic_flags"]:
            print(
                f"- {flag}"
            )

    else:

        print(
            "- No major diagnostic flags detected."
        )

    print()

    print(
        "[OVERALL DIAGNOSIS]"
    )

    print(
        diagnosis["overall_diagnosis"]
    )

    print()

    print(
        "[OUTPUT]"
    )

    print(
        f"Reports:              {OUTPUT_DIR}"
    )

    print(
        f"JSON report:          {JSON_PATH}"
    )

    print(
        f"Summary report:       {SUMMARY_PATH}"
    )

    print()
    print("=" * 60)
    print(
        "GENERALIZATION DIAGNOSIS COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()