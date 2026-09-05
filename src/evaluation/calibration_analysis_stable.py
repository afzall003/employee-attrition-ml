"""
Stable-feature calibration analysis for the employee attrition model.

Purpose
-------
Evaluate whether the selected Random Forest produces well-calibrated
probabilities when trained on the validated stable feature set.

Canonical dataset
-----------------
data/raw/employee_attrition_dataset_v2.csv

Target
------
Attrition:
    No  -> 0
    Yes -> 1

Stable feature set
------------------
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

Selected model
--------------
Random Forest

Configuration
-------------
n_estimators=400
max_features="sqrt"
min_samples_leaf=10
class_weight="balanced"

Validation
----------
5-fold stratified CV x 5 repeats = 25 validation splits

Important
---------
Calibration analysis uses out-of-fold predictions. The same observation
is therefore evaluated using a model that did not train on that observation.
This avoids optimistic calibration estimates caused by evaluating the
training predictions.

Outputs
-------
reports/signal_analysis/calibration_analysis_stable/
    calibration_metrics.csv
    calibration_bins.csv
    calibration_comparison.csv
    calibration_analysis_stable_report.json
    calibration_analysis_stable_summary.txt
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
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
    / "calibration_analysis_stable"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CANONICAL DATASET CONFIGURATION
# ============================================================

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

IDENTIFIER_COLUMN = "Employee_ID"
TARGET_COLUMN = "Attrition"

TARGET_MAPPING = {
    "No": 0,
    "Yes": 1,
}


# ============================================================
# STABLE FEATURE SET
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


# Established feature partition for the stable model.
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


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_features": "sqrt",
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# ============================================================
# VALIDATION CONFIGURATION
# ============================================================

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

DEFAULT_THRESHOLD = 0.50
SELECTED_THRESHOLD = 0.44

CALIBRATION_BINS = 10


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str, width: int = 64) -> None:
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def json_safe(value):
    """
    Convert numpy/pandas objects into JSON-compatible values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if pd.isna(value):
        return None

    return value


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate the canonical dataset.

    IMPORTANT:
    The canonical Attrition target is represented as strings:
        No / Yes

    It is converted to:
        0 / 1

    only after validation.
    """

    checks: Dict[str, bool] = {}

    checks["file_exists"] = DATA_PATH.exists()

    checks["expected_rows"] = len(df) == EXPECTED_ROWS

    checks["expected_columns"] = len(df.columns) == EXPECTED_COLUMNS

    checks["target_exists"] = TARGET_COLUMN in df.columns

    checks["identifier_exists"] = IDENTIFIER_COLUMN in df.columns

    checks["stable_features_exist"] = all(
        feature in df.columns for feature in STABLE_FEATURES
    )

    # --------------------------------------------------------
    # Correct target validation for canonical No/Yes values
    # --------------------------------------------------------

    if TARGET_COLUMN in df.columns:
        raw_target = (
            df[TARGET_COLUMN]
            .astype("string")
            .str.strip()
        )

        normalized_target = raw_target.str.lower()

        target_values = set(
            normalized_target.dropna().unique()
        )

        checks["target_values_valid"] = (
            target_values == {"no", "yes"}
        )
    else:
        checks["target_values_valid"] = False

    checks["no_missing_cells"] = not df.isnull().any().any()

    if IDENTIFIER_COLUMN in df.columns:
        checks["identifier_unique"] = (
            df[IDENTIFIER_COLUMN].nunique(dropna=False)
            == len(df)
        )
    else:
        checks["identifier_unique"] = False

    return checks


# ============================================================
# TARGET CONVERSION
# ============================================================

def convert_target(df: pd.DataFrame) -> pd.Series:
    """
    Convert canonical No/Yes Attrition values to binary 0/1.
    """

    normalized_target = (
        df[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    y = normalized_target.map(
        {
            "no": 0,
            "yes": 1,
        }
    )

    if y.isna().any():
        invalid_values = sorted(
            normalized_target[y.isna()]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Target contains invalid values after normalization: "
            f"{invalid_values}"
        )

    return y.astype(int)


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        median imputation

    Categorical:
        most-frequent imputation
        one-hot encoding
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
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# MODEL
# ============================================================

def build_model(random_state: int = RANDOM_STATE) -> Pipeline:
    """
    Build the final stable-feature Random Forest pipeline.
    """

    model = RandomForestClassifier(
        n_estimators=MODEL_PARAMS["n_estimators"],
        max_features=MODEL_PARAMS["max_features"],
        min_samples_leaf=MODEL_PARAMS["min_samples_leaf"],
        class_weight=MODEL_PARAMS["class_weight"],
        random_state=random_state,
        n_jobs=MODEL_PARAMS["n_jobs"],
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# METRICS
# ============================================================

def calculate_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = np.sum(
        (y_true == 1) & (predictions == 1)
    )

    tn = np.sum(
        (y_true == 0) & (predictions == 0)
    )

    fp = np.sum(
        (y_true == 0) & (predictions == 1)
    )

    fn = np.sum(
        (y_true == 1) & (predictions == 0)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2

    accuracy = (
        (tp + tn) / len(y_true)
    )

    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(accuracy),
        "predicted_positive_percent": float(
            predictions.mean() * 100
        ),
    }


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calculate_calibration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:

    clipped_probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    roc_auc = roc_auc_score(
        y_true,
        clipped_probabilities,
    )

    brier = brier_score_loss(
        y_true,
        clipped_probabilities,
    )

    ll = log_loss(
        y_true,
        clipped_probabilities,
    )

    observed_prevalence = float(
        np.mean(y_true)
    )

    mean_probability = float(
        np.mean(clipped_probabilities)
    )

    calibration_bias = (
        mean_probability
        - observed_prevalence
    )

    return {
        "roc_auc": float(roc_auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "observed_prevalence": observed_prevalence,
        "mean_predicted_probability": mean_probability,
        "calibration_bias": float(calibration_bias),
    }


# ============================================================
# CALIBRATION BIN ANALYSIS
# ============================================================

def calculate_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:

    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "lower_probability": lower,
                    "upper_probability": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_attrition_rate": np.nan,
                    "absolute_calibration_error": np.nan,
                }
            )
            continue

        mean_probability = float(
            probabilities[mask].mean()
        )

        observed_rate = float(
            y_true[mask].mean()
        )

        absolute_error = abs(
            mean_probability
            - observed_rate
        )

        rows.append(
            {
                "bin": i + 1,
                "lower_probability": lower,
                "upper_probability": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_attrition_rate": observed_rate,
                "absolute_calibration_error": absolute_error,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def calculate_ece(
    calibration_bins: pd.DataFrame,
) -> float:

    valid_bins = calibration_bins[
        calibration_bins["count"] > 0
    ].copy()

    if valid_bins.empty:
        return float("nan")

    total = valid_bins["count"].sum()

    weighted_error = (
        valid_bins["count"]
        * valid_bins["absolute_calibration_error"]
    ).sum()

    return float(
        weighted_error / total
    )


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================

def generate_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[np.ndarray, pd.DataFrame]:

    print_header(
        "REPEATED OUT-OF-FOLD CALIBRATION VALIDATION"
    )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    n_observations = len(X)

    probability_sum = np.zeros(
        n_observations,
        dtype=float,
    )

    prediction_count = np.zeros(
        n_observations,
        dtype=int,
    )

    split_records = []

    total_splits = N_SPLITS * N_REPEATS

    for split_number, (train_idx, valid_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"Calibration split "
            f"{split_number}/{total_splits}"
        )

        model = build_model(
            random_state=RANDOM_STATE + split_number
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

        probability_sum[valid_idx] += probabilities

        prediction_count[valid_idx] += 1

        split_auc = roc_auc_score(
            y_valid,
            probabilities,
        )

        split_records.append(
            {
                "split": split_number,
                "roc_auc": float(split_auc),
                "validation_rows": len(valid_idx),
            }
        )

    if np.any(prediction_count == 0):
        raise RuntimeError(
            "Some observations did not receive "
            "an out-of-fold prediction."
        )

    oof_probabilities = (
        probability_sum
        / prediction_count
    )

    split_df = pd.DataFrame(
        split_records
    )

    return (
        oof_probabilities,
        split_df,
    )


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_diagnostic_flags(
    calibration_metrics: Dict[str, float],
    calibration_bins: pd.DataFrame,
    threshold_metrics: Dict[str, float],
) -> List[str]:

    flags: List[str] = []

    brier = calibration_metrics["brier_score"]

    calibration_bias = abs(
        calibration_metrics["calibration_bias"]
    )

    ece = calculate_ece(
        calibration_bins
    )

    mean_probability = (
        calibration_metrics[
            "mean_predicted_probability"
        ]
    )

    prevalence = (
        calibration_metrics[
            "observed_prevalence"
        ]
    )

    # --------------------------------------------------------
    # Calibration diagnostics
    # --------------------------------------------------------

    if calibration_bias >= 0.10:
        flags.append(
            "Mean predicted probability differs "
            "from observed prevalence by at least 0.10."
        )

    if ece >= 0.10:
        flags.append(
            "Expected calibration error is at least 0.10, "
            "indicating material probability miscalibration."
        )

    if brier >= 0.25:
        flags.append(
            "Brier score is relatively high, indicating "
            "limited probability accuracy."
        )

    if mean_probability > prevalence:
        flags.append(
            "The model's average predicted probability "
            "exceeds observed attrition prevalence."
        )

    if mean_probability < prevalence:
        flags.append(
            "The model's average predicted probability "
            "is below observed attrition prevalence."
        )

    # --------------------------------------------------------
    # Operating point diagnostics
    # --------------------------------------------------------

    if threshold_metrics[
        "predicted_positive_percent"
    ] > 50:

        flags.append(
            "The selected threshold flags more than half "
            "of observations; intervention capacity "
            "should be reviewed."
        )

    if (
        threshold_metrics["precision"] < 0.40
    ):
        flags.append(
            "Precision remains below 0.40 at the selected "
            "threshold, indicating a substantial "
            "false-positive burden."
        )

    if (
        threshold_metrics["recall"] >= 0.70
    ):
        flags.append(
            "The selected threshold prioritizes detection "
            "with recall of at least 0.70."
        )

    return flags


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    df: pd.DataFrame,
    calibration_metrics: Dict[str, float],
    threshold_metrics: Dict[str, float],
    split_df: pd.DataFrame,
    flags: List[str],
    ece: float,
) -> str:

    lines = []

    lines.append(
        "EMPLOYEE ATTRITION — "
        "STABLE CALIBRATION ANALYSIS"
    )

    lines.append("")
    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 {len(df)}"
    )
    lines.append(
        f"Columns:              {len(df.columns)}"
    )
    lines.append(
        f"Stable features:      {len(STABLE_FEATURES)}"
    )
    lines.append(
        "Target prevalence:    "
        f"{calibration_metrics['observed_prevalence'] * 100:.2f}%"
    )

    lines.append("")
    lines.append("[MODEL]")
    lines.append(
        "Model:                Random Forest"
    )
    lines.append(
        "Features:             Stable 10-feature subset"
    )
    lines.append(
        "n_estimators:         400"
    )
    lines.append(
        "max_features:         sqrt"
    )
    lines.append(
        "min_samples_leaf:     10"
    )
    lines.append(
        "class_weight:         balanced"
    )

    lines.append("")
    lines.append("[VALIDATION]")
    lines.append(
        f"Folds per repeat:     {N_SPLITS}"
    )
    lines.append(
        f"Repeats:              {N_REPEATS}"
    )
    lines.append(
        f"Total validation:     {N_SPLITS * N_REPEATS}"
    )

    lines.append("")
    lines.append("[OUT-OF-FOLD RANKING]")
    lines.append(
        f"ROC-AUC:              "
        f"{calibration_metrics['roc_auc']:.4f}"
    )

    lines.append("")
    lines.append("[CALIBRATION]")
    lines.append(
        f"Brier Score:          "
        f"{calibration_metrics['brier_score']:.4f}"
    )
    lines.append(
        f"Log Loss:             "
        f"{calibration_metrics['log_loss']:.4f}"
    )
    lines.append(
        f"Mean Probability:     "
        f"{calibration_metrics['mean_predicted_probability']:.4f}"
    )
    lines.append(
        f"Observed Prevalence:  "
        f"{calibration_metrics['observed_prevalence']:.4f}"
    )
    lines.append(
        f"Calibration Bias:     "
        f"{calibration_metrics['calibration_bias']:+.4f}"
    )
    lines.append(
        f"Expected Cal. Error:  "
        f"{ece:.4f}"
    )

    lines.append("")
    lines.append("[OPERATING POINT]")
    lines.append(
        f"Threshold:            "
        f"{threshold_metrics['threshold']:.2f}"
    )
    lines.append(
        f"F1:                   "
        f"{threshold_metrics['f1']:.4f}"
    )
    lines.append(
        f"Precision:            "
        f"{threshold_metrics['precision']:.4f}"
    )
    lines.append(
        f"Recall:               "
        f"{threshold_metrics['recall']:.4f}"
    )
    lines.append(
        f"Specificity:          "
        f"{threshold_metrics['specificity']:.4f}"
    )
    lines.append(
        f"Predicted Positive:   "
        f"{threshold_metrics['predicted_positive_percent']:.2f}%"
    )

    if not split_df.empty:
        lines.append("")
        lines.append("[SPLIT STABILITY]")
        lines.append(
            f"ROC-AUC mean:         "
            f"{split_df['roc_auc'].mean():.4f}"
        )
        lines.append(
            f"ROC-AUC std:          "
            f"{split_df['roc_auc'].std(ddof=0):.4f}"
        )
        lines.append(
            f"ROC-AUC min:          "
            f"{split_df['roc_auc'].min():.4f}"
        )
        lines.append(
            f"ROC-AUC max:          "
            f"{split_df['roc_auc'].max():.4f}"
        )

    lines.append("")
    lines.append("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            lines.append(
                f"- {flag}"
            )
    else:
        lines.append(
            "- No major calibration diagnostic flags."
        )

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    if ece < 0.05 and abs(
        calibration_metrics["calibration_bias"]
    ) < 0.05:

        diagnosis = (
            "The stable-feature Random Forest shows "
            "reasonably good aggregate probability "
            "calibration under repeated out-of-fold "
            "validation."
        )

    elif ece < 0.10 and abs(
        calibration_metrics["calibration_bias"]
    ) < 0.10:

        diagnosis = (
            "The stable-feature Random Forest shows "
            "moderate probability calibration under "
            "repeated out-of-fold validation. "
            "Calibration should be monitored before "
            "and after deployment."
        )

    else:

        diagnosis = (
            "The stable-feature Random Forest shows "
            "material probability calibration error. "
            "Probability outputs should not be treated "
            "as directly calibrated risk estimates "
            "without additional calibration and "
            "subsequent validation."
        )

    lines.append("")
    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(diagnosis)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Running stable-feature calibration analysis..."
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Rows:                 {len(df)}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    print()
    print("Validating canonical dataset...")

    checks = validate_dataset(df)

    for name, passed in checks.items():

        status = "PASS" if passed else "FAIL"

        print(
            f"{status} {name}"
        )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise ValueError(
            "Canonical dataset validation failed: "
            + ", ".join(failed_checks)
        )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    y = convert_target(df)

    X = df[
        STABLE_FEATURES
    ].copy()

    target_prevalence = float(
        y.mean()
    )

    print()
    print(
        f"Stable features:       "
        f"{len(STABLE_FEATURES)}"
    )

    print(
        f"Numerical features:    "
        f"{len(NUMERICAL_FEATURES)}"
    )

    print(
        f"Categorical features:  "
        f"{len(CATEGORICAL_FEATURES)}"
    )

    print(
        f"Target prevalence:     "
        f"{target_prevalence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Generate OOF predictions
    # --------------------------------------------------------

    (
        oof_probabilities,
        split_df,
    ) = generate_oof_predictions(
        X,
        y,
    )

    # --------------------------------------------------------
    # Ranking metrics
    # --------------------------------------------------------

    print()
    print(
        "Calculating out-of-fold calibration metrics..."
    )

    calibration_metrics = (
        calculate_calibration_metrics(
            y.to_numpy(),
            oof_probabilities,
        )
    )

    print(
        f"ROC-AUC: {calibration_metrics['roc_auc']:.4f}"
    )

    print(
        f"Brier Score: "
        f"{calibration_metrics['brier_score']:.4f}"
    )

    print(
        f"Log Loss: "
        f"{calibration_metrics['log_loss']:.4f}"
    )

    # --------------------------------------------------------
    # Calibration bins
    # --------------------------------------------------------

    print(
        "Calculating calibration bins..."
    )

    calibration_bins = (
        calculate_calibration_bins(
            y.to_numpy(),
            oof_probabilities,
            CALIBRATION_BINS,
        )
    )

    ece = calculate_ece(
        calibration_bins
    )

    calibration_metrics[
        "expected_calibration_error"
    ] = ece

    # --------------------------------------------------------
    # Operating threshold
    # --------------------------------------------------------

    threshold_metrics = (
        calculate_classification_metrics(
            y.to_numpy(),
            oof_probabilities,
            SELECTED_THRESHOLD,
        )
    )

    # --------------------------------------------------------
    # Default threshold comparison
    # --------------------------------------------------------

    default_threshold_metrics = (
        calculate_classification_metrics(
            y.to_numpy(),
            oof_probabilities,
            DEFAULT_THRESHOLD,
        )
    )

    threshold_comparison = pd.DataFrame(
        [
            default_threshold_metrics,
            threshold_metrics,
        ]
    )

    threshold_comparison[
        "threshold_label"
    ] = [
        "default_0.50",
        "selected_0.44",
    ]

    # --------------------------------------------------------
    # Calibration curve
    # --------------------------------------------------------

    prob_true, prob_pred = calibration_curve(
        y.to_numpy(),
        oof_probabilities,
        n_bins=CALIBRATION_BINS,
        strategy="uniform",
    )

    calibration_curve_df = pd.DataFrame(
        {
            "mean_predicted_probability": prob_pred,
            "observed_attrition_rate": prob_true,
        }
    )

    # --------------------------------------------------------
    # Diagnostic flags
    # --------------------------------------------------------

    print(
        "Generating diagnostic flags..."
    )

    flags = generate_diagnostic_flags(
        calibration_metrics,
        calibration_bins,
        threshold_metrics,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_header(
        "EMPLOYEE ATTRITION — "
        "STABLE CALIBRATION ANALYSIS"
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
        f"Features:             {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    "
        f"{target_prevalence * 100:.2f}%"
    )

    print()
    print("[MODEL]")
    print(
        "Model:                Random Forest"
    )
    print(
        "Feature set:          Stable 10-feature subset"
    )

    print()
    print("[OUT-OF-FOLD PERFORMANCE]")
    print(
        f"ROC-AUC:              "
        f"{calibration_metrics['roc_auc']:.4f}"
    )
    print(
        f"Brier Score:          "
        f"{calibration_metrics['brier_score']:.4f}"
    )
    print(
        f"Log Loss:             "
        f"{calibration_metrics['log_loss']:.4f}"
    )

    print()
    print("[CALIBRATION]")
    print(
        f"Mean Probability:     "
        f"{calibration_metrics['mean_predicted_probability']:.4f}"
    )
    print(
        f"Observed Prevalence:  "
        f"{calibration_metrics['observed_prevalence']:.4f}"
    )
    print(
        f"Calibration Bias:     "
        f"{calibration_metrics['calibration_bias']:+.4f}"
    )
    print(
        f"Expected Cal. Error:  "
        f"{ece:.4f}"
    )

    print()
    print("[SELECTED OPERATING POINT]")
    print(
        f"Threshold:            "
        f"{SELECTED_THRESHOLD:.2f}"
    )
    print(
        f"F1:                   "
        f"{threshold_metrics['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{threshold_metrics['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{threshold_metrics['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{threshold_metrics['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{threshold_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{threshold_metrics['predicted_positive_percent']:.2f}%"
    )

    print()
    print("[DEFAULT THRESHOLD — 0.50]")
    print(
        f"F1:                   "
        f"{default_threshold_metrics['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{default_threshold_metrics['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{default_threshold_metrics['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{default_threshold_metrics['specificity']:.4f}"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            print(
                f"- {flag}"
            )
    else:
        print(
            "- No major calibration diagnostic flags."
        )

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    if ece < 0.05 and abs(
        calibration_metrics["calibration_bias"]
    ) < 0.05:

        overall_status = (
            "CALIBRATION ACCEPTABLE"
        )

        overall_diagnosis = (
            "The stable-feature Random Forest shows "
            "reasonably good probability calibration "
            "under repeated out-of-fold validation."
        )

    elif ece < 0.10 and abs(
        calibration_metrics["calibration_bias"]
    ) < 0.10:

        overall_status = (
            "CALIBRATION CONDITIONALLY ACCEPTABLE"
        )

        overall_diagnosis = (
            "The stable-feature Random Forest shows "
            "moderate probability calibration. "
            "Probability outputs should be monitored "
            "and recalibration should be considered "
            "if post-deployment evidence indicates "
            "material drift."
        )

    else:

        overall_status = (
            "CALIBRATION REQUIRES REVIEW"
        )

        overall_diagnosis = (
            "The stable-feature Random Forest shows "
            "material probability calibration error. "
            "The model may still provide useful ranking "
            "information, but predicted probabilities "
            "should not be interpreted as calibrated "
            "individual attrition probabilities without "
            "additional calibration."
        )

    print()
    print("[OVERALL STATUS]")
    print(
        f"CALIBRATION STATUS: {overall_status}"
    )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(
        overall_diagnosis
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    calibration_metrics_df = pd.DataFrame(
        [
            calibration_metrics
        ]
    )

    calibration_metrics_path = (
        REPORT_DIR
        / "calibration_metrics.csv"
    )

    calibration_bins_path = (
        REPORT_DIR
        / "calibration_bins.csv"
    )

    threshold_comparison_path = (
        REPORT_DIR
        / "calibration_comparison.csv"
    )

    calibration_curve_path = (
        REPORT_DIR
        / "calibration_curve.csv"
    )

    split_performance_path = (
        REPORT_DIR
        / "calibration_split_performance.csv"
    )

    summary_path = (
        REPORT_DIR
        / "calibration_analysis_stable_summary.txt"
    )

    json_path = (
        REPORT_DIR
        / "calibration_analysis_stable_report.json"
    )

    calibration_metrics_df.to_csv(
        calibration_metrics_path,
        index=False,
    )

    calibration_bins.to_csv(
        calibration_bins_path,
        index=False,
    )

    threshold_comparison.to_csv(
        threshold_comparison_path,
        index=False,
    )

    calibration_curve_df.to_csv(
        calibration_curve_path,
        index=False,
    )

    split_df.to_csv(
        split_performance_path,
        index=False,
    )

    summary = build_summary(
        df=df,
        calibration_metrics=calibration_metrics,
        threshold_metrics=threshold_metrics,
        split_df=split_df,
        flags=flags,
        ece=ece,
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    report = {
        "analysis": "calibration_analysis_stable",
        "dataset": {
            "path": str(DATA_PATH),
            "rows": len(df),
            "columns": len(df.columns),
            "sha256": calculate_sha256(
                DATA_PATH
            ),
            "target_column": TARGET_COLUMN,
            "target_mapping": TARGET_MAPPING,
            "target_prevalence": target_prevalence,
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
            "random_state": RANDOM_STATE,
        },
        "stable_features": STABLE_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model": {
            "name": "Random Forest",
            "parameters": MODEL_PARAMS,
        },
        "thresholds": {
            "default": DEFAULT_THRESHOLD,
            "selected": SELECTED_THRESHOLD,
        },
        "ranking_metrics": calibration_metrics,
        "selected_threshold_metrics": threshold_metrics,
        "default_threshold_metrics": (
            default_threshold_metrics
        ),
        "calibration_bins": calibration_bins.to_dict(
            orient="records"
        ),
        "calibration_curve": calibration_curve_df.to_dict(
            orient="records"
        ),
        "split_performance": split_df.to_dict(
            orient="records"
        ),
        "diagnostic_flags": flags,
        "overall_status": overall_status,
        "overall_diagnosis": overall_diagnosis,
        "validation_checks": checks,
        "outputs": {
            "calibration_metrics": str(
                calibration_metrics_path
            ),
            "calibration_bins": str(
                calibration_bins_path
            ),
            "calibration_comparison": str(
                threshold_comparison_path
            ),
            "calibration_curve": str(
                calibration_curve_path
            ),
            "split_performance": str(
                split_performance_path
            ),
            "json_report": str(
                json_path
            ),
            "summary": str(
                summary_path
            ),
        },
    }

    json_path.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {REPORT_DIR}"
    )
    print(
        f"Calibration metrics:  {calibration_metrics_path}"
    )
    print(
        f"Calibration bins:     {calibration_bins_path}"
    )
    print(
        f"Comparison CSV:       {threshold_comparison_path}"
    )
    print(
        f"Calibration curve:    {calibration_curve_path}"
    )
    print(
        f"Split performance:    {split_performance_path}"
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
        "STABLE CALIBRATION ANALYSIS COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()