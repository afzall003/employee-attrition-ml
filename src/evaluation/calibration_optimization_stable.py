"""
Calibration Optimization for Stable Employee Attrition Model

Purpose
-------
Evaluate whether probability calibration improves the stable-feature
Random Forest model.

The analysis:
1. Loads and validates the canonical dataset.
2. Uses the established stable 10-feature subset.
3. Generates repeated out-of-fold predictions.
4. Compares:
      - Uncalibrated Random Forest
      - Sigmoid / Platt calibration
      - Isotonic calibration
5. Evaluates:
      - ROC-AUC
      - PR-AUC
      - Brier Score
      - Log Loss
6. Compares calibration behavior.
7. Produces CSV, JSON, and TXT reports.

Important
---------
Calibration is evaluated using out-of-fold predictions to avoid evaluating
calibration on the same observations used to fit the calibration model.

This script does NOT automatically replace the production model.
It provides evidence for whether calibration should be considered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "calibration_optimization_stable"
)

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

EXPECTED_CANONICAL_SHA256 = (
    "9bde9a4f6a9c4d3c4b0e9e7d4e7d6d7f"
)

# The canonical hash is intentionally treated as optional evidence.
# If the actual canonical hash is known from deployment_readiness_audit,
# replace EXPECTED_CANONICAL_SHA256 with that value.

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

THRESHOLD = 0.44

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

# Number of bins for calibration diagnostics.
N_CALIBRATION_BINS = 10


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def safe_float(value) -> float:
    if value is None:
        return float("nan")

    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# ================================================================
# DATA VALIDATION
# ================================================================

def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    checks = {}

    checks["file_exists"] = DATA_PATH.exists()
    checks["expected_rows"] = len(df) == EXPECTED_ROWS
    checks["expected_columns"] = len(df.columns) == EXPECTED_COLUMNS
    checks["target_exists"] = TARGET_COLUMN in df.columns
    checks["identifier_exists"] = IDENTIFIER_COLUMN in df.columns

    checks["stable_features_exist"] = all(
        feature in df.columns for feature in STABLE_FEATURES
    )

    if TARGET_COLUMN in df.columns:
        checks["target_values_valid"] = set(
            df[TARGET_COLUMN].dropna().astype(str).str.strip().unique()
        ).issubset({"Yes", "No"})
    else:
        checks["target_values_valid"] = False

    checks["no_missing_cells"] = not df.isnull().any().any()

    if IDENTIFIER_COLUMN in df.columns:
        checks["identifier_unique"] = df[IDENTIFIER_COLUMN].is_unique
    else:
        checks["identifier_unique"] = False

    return checks


# ================================================================
# PREPROCESSING
# ================================================================

def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
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

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ================================================================
# MODEL
# ================================================================

def build_random_forest_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ================================================================
# CALIBRATION METHODS
# ================================================================

def fit_sigmoid_calibrator(
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    """
    Platt-style sigmoid calibration.

    A logistic regression is fitted against the logit of the base model
    probability.
    """

    eps = 1e-6

    clipped = np.clip(predictions, eps, 1.0 - eps)

    logits = np.log(clipped / (1.0 - clipped))

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
    )

    calibrator.fit(
        logits.reshape(-1, 1),
        y_true,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator: LogisticRegression,
    predictions: np.ndarray,
) -> np.ndarray:

    eps = 1e-6

    clipped = np.clip(predictions, eps, 1.0 - eps)

    logits = np.log(clipped / (1.0 - clipped))

    calibrated = calibrator.predict_proba(
        logits.reshape(-1, 1)
    )[:, 1]

    return np.clip(calibrated, 0.0, 1.0)


def fit_isotonic_calibrator(
    predictions: np.ndarray,
    y_true: np.ndarray,
) -> IsotonicRegression:

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        predictions,
        y_true,
    )

    return calibrator


def apply_isotonic_calibrator(
    calibrator: IsotonicRegression,
    predictions: np.ndarray,
) -> np.ndarray:

    calibrated = calibrator.predict(predictions)

    return np.clip(
        np.asarray(calibrated),
        0.0,
        1.0,
    )


# ================================================================
# METRICS
# ================================================================

def calculate_probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:

    probabilities = np.clip(
        probabilities,
        1e-7,
        1.0 - 1e-7,
    )

    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "mean_probability": float(
            np.mean(probabilities)
        ),
    }


# ================================================================
# CLASSIFICATION METRICS
# ================================================================

def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 1)
        )
    )

    tn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 0)
        )
    )

    fp = int(
        np.sum(
            (predictions == 1)
            & (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (predictions == 0)
            & (y_true == 1)
        )
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

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "predicted_positive_percent": (
            np.mean(predictions) * 100.0
        ),
        "flagged_per_1000": (
            np.mean(predictions) * 1000.0
        ),
    }


# ================================================================
# CALIBRATION BINS
# ================================================================

def calculate_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    n_bins: int = 10,
) -> pd.DataFrame:

    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):

        lower = edges[i]
        upper = edges[i + 1]

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

        count = int(np.sum(mask))

        if count == 0:
            rows.append(
                {
                    "method": method,
                    "bin": i + 1,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "count": 0,
                    "mean_probability": np.nan,
                    "observed_rate": np.nan,
                    "absolute_error": np.nan,
                }
            )

            continue

        mean_probability = float(
            np.mean(probabilities[mask])
        )

        observed_rate = float(
            np.mean(y_true[mask])
        )

        absolute_error = abs(
            mean_probability - observed_rate
        )

        rows.append(
            {
                "method": method,
                "bin": i + 1,
                "lower_bound": lower,
                "upper_bound": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_error": absolute_error,
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# REPEATED OOF CALIBRATION
# ================================================================

def generate_repeated_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:

    print("Generating repeated out-of-fold calibration predictions...")
    print(f"Folds per repeat: {N_SPLITS}")
    print(f"Repeats:           {N_REPEATS}")
    print(
        f"Total validation: {N_SPLITS * N_REPEATS}"
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    prediction_records = []

    all_base_predictions = []
    all_sigmoid_predictions = []
    all_isotonic_predictions = []
    all_targets = []

    for split_number, (
        train_idx,
        test_idx,
    ) in enumerate(
        splitter.split(X, y),
        start=1,
    ):

        print(
            f"Calibration split "
            f"{split_number}/"
            f"{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        base_model = build_random_forest_pipeline()

        base_model.fit(
            X_train,
            y_train,
        )

        base_train_probability = (
            base_model.predict_proba(
                X_train
            )[:, 1]
        )

        base_test_probability = (
            base_model.predict_proba(
                X_test
            )[:, 1]
        )

        # --------------------------------------------------------
        # Calibration is fitted ONLY on training predictions.
        # --------------------------------------------------------

        sigmoid_calibrator = (
            fit_sigmoid_calibrator(
                base_train_probability,
                y_train,
            )
        )

        isotonic_calibrator = (
            fit_isotonic_calibrator(
                base_train_probability,
                y_train,
            )
        )

        sigmoid_test_probability = (
            apply_sigmoid_calibrator(
                sigmoid_calibrator,
                base_test_probability,
            )
        )

        isotonic_test_probability = (
            apply_isotonic_calibrator(
                isotonic_calibrator,
                base_test_probability,
            )
        )

        for local_idx, original_idx in enumerate(
            test_idx
        ):

            prediction_records.append(
                {
                    "split": split_number,
                    "row_index": int(original_idx),
                    "y_true": int(
                        y_test[local_idx]
                    ),
                    "base_probability": float(
                        base_test_probability[
                            local_idx
                        ]
                    ),
                    "sigmoid_probability": float(
                        sigmoid_test_probability[
                            local_idx
                        ]
                    ),
                    "isotonic_probability": float(
                        isotonic_test_probability[
                            local_idx
                        ]
                    ),
                }
            )

        all_base_predictions.extend(
            base_test_probability.tolist()
        )

        all_sigmoid_predictions.extend(
            sigmoid_test_probability.tolist()
        )

        all_isotonic_predictions.extend(
            isotonic_test_probability.tolist()
        )

        all_targets.extend(
            y_test.tolist()
        )

    predictions_df = pd.DataFrame(
        prediction_records
    )

    arrays = {
        "y_true": np.asarray(
            all_targets,
            dtype=int,
        ),
        "base": np.asarray(
            all_base_predictions,
            dtype=float,
        ),
        "sigmoid": np.asarray(
            all_sigmoid_predictions,
            dtype=float,
        ),
        "isotonic": np.asarray(
            all_isotonic_predictions,
            dtype=float,
        ),
    }

    return predictions_df, arrays


# ================================================================
# SPLIT-WISE PERFORMANCE
# ================================================================

def calculate_split_performance(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    methods = {
        "Random Forest": "base_probability",
        "Sigmoid Calibration": "sigmoid_probability",
        "Isotonic Calibration": "isotonic_probability",
    }

    for split, split_df in predictions_df.groupby(
        "split"
    ):

        y_true = split_df["y_true"].to_numpy()

        for method_name, probability_column in methods.items():

            probabilities = (
                split_df[
                    probability_column
                ].to_numpy()
            )

            metrics = calculate_probability_metrics(
                y_true,
                probabilities,
            )

            rows.append(
                {
                    "split": split,
                    "method": method_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    print("Running stable-feature calibration optimization...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Rows:                 {len(df)}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    print("\nValidating canonical dataset...")

    checks = validate_dataset(df)

    for name, result in checks.items():

        print(
            f"{'PASS' if result else 'FAIL'} "
            f"{name}"
        )

    if not all(checks.values()):
        failed = [
            name
            for name, result in checks.items()
            if not result
        ]

        raise ValueError(
            "Canonical dataset validation failed: "
            + ", ".join(failed)
        )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
        .to_numpy()
    )

    if np.isnan(y).any():
        raise ValueError(
            "Target contains values that could not be mapped "
            "to 0/1."
        )

    y = y.astype(int)

    X = df[STABLE_FEATURES].copy()

    prevalence = float(
        np.mean(y)
    )

    print(
        f"\nStable features:      "
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
        f"{prevalence:.2%}"
    )

    # ------------------------------------------------------------
    # Generate repeated OOF predictions
    # ------------------------------------------------------------

    print_header(
        "REPEATED OUT-OF-FOLD CALIBRATION OPTIMIZATION"
    )

    predictions_df, arrays = (
        generate_repeated_oof_predictions(
            X,
            y,
        )
    )

    # ------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------

    print(
        "\nCalculating aggregate calibration metrics..."
    )

    method_map = {
        "Random Forest": arrays["base"],
        "Sigmoid Calibration": arrays["sigmoid"],
        "Isotonic Calibration": arrays["isotonic"],
    }

    performance_rows = []

    for method, probabilities in method_map.items():

        metrics = calculate_probability_metrics(
            arrays["y_true"],
            probabilities,
        )

        threshold_metrics = (
            calculate_threshold_metrics(
                arrays["y_true"],
                probabilities,
                THRESHOLD,
            )
        )

        performance_rows.append(
            {
                "method": method,
                **metrics,
                **threshold_metrics,
            }
        )

    performance_df = pd.DataFrame(
        performance_rows
    )

    # ------------------------------------------------------------
    # Split stability
    # ------------------------------------------------------------

    split_performance_df = (
        calculate_split_performance(
            predictions_df
        )
    )

    split_summary_df = (
        split_performance_df
        .groupby("method")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            brier_mean=("brier_score", "mean"),
            brier_std=("brier_score", "std"),
            log_loss_mean=("log_loss", "mean"),
            log_loss_std=("log_loss", "std"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------
    # Calibration bins
    # ------------------------------------------------------------

    print(
        "Calculating calibration bins..."
    )

    calibration_bins = []

    for method, probabilities in method_map.items():

        bins = calculate_calibration_bins(
            arrays["y_true"],
            probabilities,
            method,
            N_CALIBRATION_BINS,
        )

        calibration_bins.append(
            bins
        )

    calibration_bins_df = pd.concat(
        calibration_bins,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Expected Calibration Error
    # ------------------------------------------------------------

    ece_rows = []

    for method, probabilities in method_map.items():

        bins = calculate_calibration_bins(
            arrays["y_true"],
            probabilities,
            method,
            N_CALIBRATION_BINS,
        )

        valid_bins = bins[
            bins["count"] > 0
        ].copy()

        if len(valid_bins) == 0:
            ece = np.nan
        else:
            total = valid_bins["count"].sum()

            ece = float(
                np.sum(
                    (
                        valid_bins["count"]
                        / total
                    )
                    * valid_bins[
                        "absolute_error"
                    ]
                )
            )

        ece_rows.append(
            {
                "method": method,
                "expected_calibration_error": ece,
            }
        )

    ece_df = pd.DataFrame(
        ece_rows
    )

    performance_df = performance_df.merge(
        ece_df,
        on="method",
        how="left",
    )

    # ------------------------------------------------------------
    # Select calibration method
    # ------------------------------------------------------------

    base_row = performance_df[
        performance_df["method"]
        == "Random Forest"
    ].iloc[0]

    sigmoid_row = performance_df[
        performance_df["method"]
        == "Sigmoid Calibration"
    ].iloc[0]

    isotonic_row = performance_df[
        performance_df["method"]
        == "Isotonic Calibration"
    ].iloc[0]

    candidates = [
        sigmoid_row,
        isotonic_row,
    ]

    # Calibration is primarily judged by Brier score and log loss.
    best_calibrated = min(
        candidates,
        key=lambda row: (
            row["brier_score"]
            + row["log_loss"]
        ),
    )

    selected_method = best_calibrated[
        "method"
    ]

    brier_improvement = (
        base_row["brier_score"]
        - best_calibrated["brier_score"]
    )

    log_loss_improvement = (
        base_row["log_loss"]
        - best_calibrated["log_loss"]
    )

    ece_improvement = (
        base_row[
            "expected_calibration_error"
        ]
        - best_calibrated[
            "expected_calibration_error"
        ]
    )

    # ------------------------------------------------------------
    # Diagnostic flags
    # ------------------------------------------------------------

    print(
        "\nGenerating diagnostic flags..."
    )

    diagnostic_flags = []

    if (
        brier_improvement > 0
        and log_loss_improvement > 0
    ):
        diagnostic_flags.append(
            "The selected calibration method improves both "
            "Brier Score and Log Loss relative to the "
            "uncalibrated Random Forest."
        )
    else:
        diagnostic_flags.append(
            "Calibration does not improve both Brier Score "
            "and Log Loss relative to the uncalibrated model."
        )

    if ece_improvement >= 0.05:
        diagnostic_flags.append(
            "Calibration materially reduces expected "
            "calibration error."
        )
    elif ece_improvement > 0:
        diagnostic_flags.append(
            "Calibration reduces expected calibration error "
            "but the improvement is modest."
        )
    else:
        diagnostic_flags.append(
            "Calibration does not reduce expected "
            "calibration error."
        )

    if (
        best_calibrated["roc_auc"]
        < base_row["roc_auc"] - 0.02
    ):
        diagnostic_flags.append(
            "The selected calibration method materially "
            "reduces ROC-AUC; this should be investigated."
        )

    if (
        best_calibrated[
            "predicted_positive_percent"
        ]
        > 50
    ):
        diagnostic_flags.append(
            "The selected threshold still flags more than "
            "half of observations."
        )

    if (
        best_calibrated["precision"]
        < 0.40
    ):
        diagnostic_flags.append(
            "Precision remains below 0.40 at the selected "
            "operating threshold."
        )

    # ------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------

    if (
        brier_improvement > 0
        and log_loss_improvement > 0
        and ece_improvement > 0
    ):
        overall_status = (
            "CALIBRATION IMPROVEMENT SUPPORTED"
        )

        overall_diagnosis = (
            f"{selected_method} calibration improves "
            "probability quality relative to the "
            "uncalibrated Random Forest under repeated "
            "out-of-fold validation. Calibration should be "
            "considered for probability-based decision "
            "support, while threshold selection remains a "
            "separate business decision."
        )

    elif (
        brier_improvement > 0
        or log_loss_improvement > 0
    ):
        overall_status = (
            "CALIBRATION IMPROVEMENT PARTIAL"
        )

        overall_diagnosis = (
            f"{selected_method} provides some improvement "
            "in probability quality, but the evidence is "
            "not uniformly favorable across calibration "
            "metrics. Calibration should therefore be "
            "validated further before being adopted."
        )

    else:
        overall_status = (
            "CALIBRATION IMPROVEMENT NOT SUPPORTED"
        )

        overall_diagnosis = (
            "Neither calibration method provides sufficient "
            "evidence of improved probability quality over "
            "the uncalibrated Random Forest. The existing "
            "model may remain preferable for ranking-based "
            "decision support, subject to further validation."
        )

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------

    print_header(
        "EMPLOYEE ATTRITION — STABLE CALIBRATION OPTIMIZATION"
    )

    print("\n[DATASET]")
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
        f"Target prevalence:    {prevalence:.2%}"
    )

    print("\n[MODEL]")
    print(
        "Model:                Random Forest"
    )
    print(
        "Feature set:          Stable 10-feature subset"
    )

    print("\n[CALIBRATION METHODS]")
    print(
        "1. Random Forest — uncalibrated"
    )
    print(
        "2. Sigmoid / Platt calibration"
    )
    print(
        "3. Isotonic calibration"
    )

    print("\n[PERFORMANCE COMPARISON]")

    display_columns = [
        "method",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    ]

    print(
        performance_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\n[SELECTED CALIBRATION]")
    print(
        f"Method:               {selected_method}"
    )
    print(
        f"Brier Improvement:    "
        f"{brier_improvement:+.4f}"
    )
    print(
        f"Log Loss Improvement:  "
        f"{log_loss_improvement:+.4f}"
    )
    print(
        f"ECE Improvement:      "
        f"{ece_improvement:+.4f}"
    )

    print("\n[OPERATING POINT]")
    print(
        f"Threshold:             {THRESHOLD:.2f}"
    )
    print(
        f"F1:                   "
        f"{best_calibrated['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{best_calibrated['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{best_calibrated['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{best_calibrated['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{best_calibrated['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{best_calibrated['predicted_positive_percent']:.2f}%"
    )

    print("\n[DIAGNOSTIC FLAGS]")

    for flag in diagnostic_flags:
        print(f"- {flag}")

    print("\n[OVERALL STATUS]")
    print(
        f"CALIBRATION OPTIMIZATION STATUS: "
        f"{overall_status}"
    )

    print("\n[OVERALL DIAGNOSIS]")
    print(overall_diagnosis)

    # ------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------

    performance_path = (
        OUTPUT_DIR
        / "calibration_model_performance.csv"
    )

    bins_path = (
        OUTPUT_DIR
        / "calibration_bins.csv"
    )

    split_path = (
        OUTPUT_DIR
        / "calibration_split_performance.csv"
    )

    prediction_path = (
        OUTPUT_DIR
        / "calibration_oof_predictions.csv"
    )

    comparison_path = (
        OUTPUT_DIR
        / "calibration_comparison.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "calibration_optimization_stable_report.json"
    )

    summary_path = (
        OUTPUT_DIR
        / "calibration_optimization_stable_summary.txt"
    )

    performance_df.to_csv(
        performance_path,
        index=False,
    )

    calibration_bins_df.to_csv(
        bins_path,
        index=False,
    )

    split_performance_df.to_csv(
        split_path,
        index=False,
    )

    predictions_df.to_csv(
        prediction_path,
        index=False,
    )

    comparison_df = performance_df[
        [
            "method",
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "expected_calibration_error",
            "mean_probability",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "predicted_positive_percent",
            "flagged_per_1000",
        ]
    ].copy()

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------

    report = {
        "analysis": {
            "name": (
                "stable_calibration_optimization"
            ),
            "status": overall_status,
        },
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "target": TARGET_COLUMN,
            "target_prevalence": prevalence,
        },
        "model": {
            "model": "Random Forest",
            "feature_count": len(
                STABLE_FEATURES
            ),
            "features": STABLE_FEATURES,
            "threshold": THRESHOLD,
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
        },
        "performance": (
            performance_df
            .replace({np.nan: None})
            .to_dict(orient="records")
        ),
        "selected_calibration": {
            "method": selected_method,
            "brier_improvement": safe_float(
                brier_improvement
            ),
            "log_loss_improvement": safe_float(
                log_loss_improvement
            ),
            "ece_improvement": safe_float(
                ece_improvement
            ),
        },
        "diagnostic_flags": diagnostic_flags,
        "overall_diagnosis": overall_diagnosis,
        "outputs": {
            "performance": str(
                performance_path
            ),
            "calibration_bins": str(
                bins_path
            ),
            "split_performance": str(
                split_path
            ),
            "oof_predictions": str(
                prediction_path
            ),
            "comparison": str(
                comparison_path
            ),
            "json_report": str(
                json_path
            ),
            "summary": str(
                summary_path
            ),
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    # ------------------------------------------------------------
    # TXT summary
    # ------------------------------------------------------------

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "STABLE CALIBRATION OPTIMIZATION\n"
        )

        file.write(
            "=" * 64 + "\n\n"
        )

        file.write(
            "[DATASET]\n"
        )

        file.write(
            f"Rows: {len(df)}\n"
        )

        file.write(
            f"Columns: {len(df.columns)}\n"
        )

        file.write(
            f"Features: {len(STABLE_FEATURES)}\n"
        )

        file.write(
            f"Target prevalence: "
            f"{prevalence:.4f}\n\n"
        )

        file.write(
            "[VALIDATION]\n"
        )

        file.write(
            f"Folds per repeat: {N_SPLITS}\n"
        )

        file.write(
            f"Repeats: {N_REPEATS}\n"
        )

        file.write(
            f"Total validation splits: "
            f"{N_SPLITS * N_REPEATS}\n\n"
        )

        file.write(
            "[MODEL]\n"
        )

        file.write(
            "Random Forest\n"
        )

        file.write(
            "Stable 10-feature subset\n\n"
        )

        file.write(
            "[CALIBRATION COMPARISON]\n"
        )

        for _, row in performance_df.iterrows():

            file.write(
                f"{row['method']}\n"
            )

            file.write(
                f"  ROC-AUC: "
                f"{row['roc_auc']:.4f}\n"
            )

            file.write(
                f"  PR-AUC: "
                f"{row['pr_auc']:.4f}\n"
            )

            file.write(
                f"  Brier Score: "
                f"{row['brier_score']:.4f}\n"
            )

            file.write(
                f"  Log Loss: "
                f"{row['log_loss']:.4f}\n"
            )

            file.write(
                f"  ECE: "
                f"{row['expected_calibration_error']:.4f}\n\n"
            )

        file.write(
            "[SELECTED CALIBRATION]\n"
        )

        file.write(
            f"Method: {selected_method}\n"
        )

        file.write(
            f"Brier improvement: "
            f"{brier_improvement:+.4f}\n"
        )

        file.write(
            f"Log Loss improvement: "
            f"{log_loss_improvement:+.4f}\n"
        )

        file.write(
            f"ECE improvement: "
            f"{ece_improvement:+.4f}\n\n"
        )

        file.write(
            "[DIAGNOSTIC FLAGS]\n"
        )

        for flag in diagnostic_flags:

            file.write(
                f"- {flag}\n"
            )

        file.write(
            "\n[OVERALL STATUS]\n"
        )

        file.write(
            f"CALIBRATION OPTIMIZATION STATUS: "
            f"{overall_status}\n\n"
        )

        file.write(
            "[OVERALL DIAGNOSIS]\n"
        )

        file.write(
            overall_diagnosis
        )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print("\n[OUTPUT]")

    print(
        f"Reports:              {OUTPUT_DIR}"
    )

    print(
        f"Performance CSV:      "
        f"{performance_path}"
    )

    print(
        f"Calibration bins:     "
        f"{bins_path}"
    )

    print(
        f"Split performance:    "
        f"{split_path}"
    )

    print(
        f"OOF predictions:      "
        f"{prediction_path}"
    )

    print(
        f"Comparison CSV:       "
        f"{comparison_path}"
    )

    print(
        f"JSON report:          "
        f"{json_path}"
    )

    print(
        f"Summary report:       "
        f"{summary_path}"
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        "STABLE CALIBRATION OPTIMIZATION COMPLETE"
    )


if __name__ == "__main__":
    main()