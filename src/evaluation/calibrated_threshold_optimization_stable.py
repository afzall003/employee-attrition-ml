"""
Calibrated threshold optimization for the stable employee-attrition model.

Purpose
-------
The uncalibrated Random Forest previously used threshold=0.44.

After calibration, the probability scale changes. Therefore the old
0.44 threshold must NOT automatically be reused.

This script:

1. Loads and validates the canonical dataset.
2. Uses the validated stable 10-feature subset.
3. Builds the optimized Random Forest selected previously.
4. Generates repeated out-of-fold predictions.
5. Compares:
       - Uncalibrated Random Forest
       - Sigmoid / Platt calibrated Random Forest
6. Searches thresholds on the calibrated probability scale.
7. Selects the F1-optimal calibrated threshold.
8. Reports business-oriented operating metrics.
9. Compares calibrated and uncalibrated operating points.
10. Writes CSV, JSON and TXT reports.

Validation
----------
5 folds x 5 repeats = 25 validation splits.

Canonical dataset
-----------------
data/raw/employee_attrition_dataset_v2.csv

Target
------
Attrition

Target encoding
---------------
Yes -> 1
No  -> 0

Stable features
---------------
Work_Life_Balance
Job_Satisfaction
Distance_From_Home
Average_Hours_Worked_Per_Week
Years_Since_Last_Promotion
Work_Environment_Satisfaction
Job_Role
Age
Overtime
Absenteeism
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
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
    / "calibrated_threshold_optimization_stable"
)

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

N_SPLITS = 5
N_REPEATS = 5

RANDOM_SEEDS = [
    42,
    52,
    62,
    72,
    82,
]

# Previous uncalibrated operating point.
UNCALIBRATED_THRESHOLD = 0.44

# Candidate thresholds for calibrated probability scale.
THRESHOLDS = [
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
    0.85,
    0.90,
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
# DATASET UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_target(series: pd.Series) -> np.ndarray:
    """
    Convert the canonical Attrition target to binary.

    Supports the current canonical representation:

        Yes -> 1
        No  -> 0

    Also supports numeric/bool-like representations defensively.
    """

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        "true": 1,
        "false": 0,
    }

    unknown = sorted(
        set(values.unique()) - set(mapping.keys())
    )

    if unknown:
        raise ValueError(
            "Unsupported target values: "
            f"{unknown}"
        )

    return (
        values
        .map(mapping)
        .astype(int)
        .to_numpy()
    )


def validate_dataset(
    df: pd.DataFrame,
) -> Dict[str, bool]:
    """Validate canonical dataset structure."""

    checks = {}

    checks["file_exists"] = DATA_PATH.exists()

    checks["expected_rows"] = (
        len(df) == EXPECTED_ROWS
    )

    checks["expected_columns"] = (
        len(df.columns) == EXPECTED_COLUMNS
    )

    checks["target_exists"] = (
        TARGET_COLUMN in df.columns
    )

    checks["identifier_exists"] = (
        IDENTIFIER_COLUMN in df.columns
    )

    checks["stable_features_exist"] = all(
        feature in df.columns
        for feature in STABLE_FEATURES
    )

    if TARGET_COLUMN in df.columns:
        values = set(
            df[TARGET_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )

        checks["target_values_valid"] = values.issubset(
            {
                "yes",
                "no",
                "0",
                "1",
                "true",
                "false",
            }
        )
    else:
        checks["target_values_valid"] = False

    checks["no_missing_cells"] = (
        not df.isna().any().any()
    )

    if IDENTIFIER_COLUMN in df.columns:
        checks["identifier_unique"] = (
            df[IDENTIFIER_COLUMN].is_unique
        )
    else:
        checks["identifier_unique"] = False

    return checks


def print_validation(
    checks: Dict[str, bool],
) -> None:
    """Print validation results."""

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status} {name}")


# ============================================================
# PREPROCESSING
# ============================================================

def make_one_hot_encoder():
    """Create OneHotEncoder compatible with sklearn versions."""

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def identify_feature_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Identify numerical and categorical stable features."""

    numerical_features = []
    categorical_features = []

    for feature in STABLE_FEATURES:
        if pd.api.types.is_numeric_dtype(
            df[feature]
        ):
            numerical_features.append(feature)
        else:
            categorical_features.append(feature)

    return (
        numerical_features,
        categorical_features,
    )


def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Build preprocessing transformer."""

    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "onehot",
                    make_one_hot_encoder(),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODEL
# ============================================================

def build_random_forest_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build the optimized Random Forest used by the previous
    stable-feature model-selection stage.
    """

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
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
# METRICS
# ============================================================

def calculate_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate expected calibration error."""

    y_true = np.asarray(y_true)
    probabilities = np.clip(
        np.asarray(probabilities),
        0.0,
        1.0,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    total = len(y_true)

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

        if not np.any(mask):
            continue

        confidence = np.mean(
            probabilities[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        weight = np.sum(mask) / total

        ece += weight * abs(
            confidence - accuracy
        )

    return float(ece)


def calculate_ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate ranking and probability metrics."""

    probabilities = np.clip(
        np.asarray(probabilities),
        1e-8,
        1.0 - 1e-8,
    )

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
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            )
        ),
        "expected_calibration_error": (
            calculate_ece(
                y_true,
                probabilities,
            )
        ),
    }


def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate classification metrics at a threshold."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (predictions == 0)
        )
    )

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

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    balanced_accuracy = (
        recall + specificity
    ) / 2.0

    predicted_positive_percent = (
        np.mean(predictions) * 100.0
    )

    return {
        "threshold": float(threshold),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "predicted_positive_percent": float(
            predicted_positive_percent
        ),
        "flagged_per_1000": float(
            predicted_positive_percent * 10.0
        ),
        "false_positives_per_1000": float(
            fp / len(y_true) * 1000.0
        ),
        "missed_attrition_per_1000": float(
            fn / len(y_true) * 1000.0
        ),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# REPEATED OOF PREDICTIONS
# ============================================================

def generate_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    numerical_features: List[str],
    categorical_features: List[str],
):
    """
    Generate repeated out-of-fold predictions.

    Two models are evaluated independently:

    1. Uncalibrated Random Forest
    2. Sigmoid-calibrated Random Forest

    Calibration is performed inside the training portion only.
    The validation fold remains unseen by the fitted model/calibrator.
    """

    uncalibrated_records = []
    calibrated_records = []

    split_records = []

    total_splits = (
        N_SPLITS * N_REPEATS
    )

    split_counter = 0

    for repeat_index, seed in enumerate(
        RANDOM_SEEDS,
        start=1,
    ):

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for fold_index, (
            train_index,
            validation_index,
        ) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            split_counter += 1

            print(
                f"Threshold split "
                f"{split_counter}/{total_splits}"
            )

            X_train = X.iloc[train_index]
            X_validation = X.iloc[
                validation_index
            ]

            y_train = y[train_index]
            y_validation = y[
                validation_index
            ]

            # ------------------------------------------------
            # UNCALIBRATED RANDOM FOREST
            # ------------------------------------------------

            rf = build_random_forest_pipeline(
                numerical_features,
                categorical_features,
            )

            rf.fit(
                X_train,
                y_train,
            )

            uncalibrated_probabilities = (
                rf.predict_proba(
                    X_validation
                )[:, 1]
            )

            # ------------------------------------------------
            # SIGMOID-CALIBRATED RANDOM FOREST
            # ------------------------------------------------

            calibration_base = (
                build_random_forest_pipeline(
                    numerical_features,
                    categorical_features,
                )
            )

            calibrated_rf = (
                CalibratedClassifierCV(
                    estimator=calibration_base,
                    method="sigmoid",
                    cv=3,
                )
            )

            calibrated_rf.fit(
                X_train,
                y_train,
            )

            calibrated_probabilities = (
                calibrated_rf.predict_proba(
                    X_validation
                )[:, 1]
            )

            # ------------------------------------------------
            # STORE OOF PREDICTIONS
            # ------------------------------------------------

            uncalibrated_records.append(
                pd.DataFrame(
                    {
                        "row_index": (
                            validation_index
                        ),
                        "y_true": y_validation,
                        "probability": (
                            uncalibrated_probabilities
                        ),
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                    }
                )
            )

            calibrated_records.append(
                pd.DataFrame(
                    {
                        "row_index": (
                            validation_index
                        ),
                        "y_true": y_validation,
                        "probability": (
                            calibrated_probabilities
                        ),
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                    }
                )
            )

            # ------------------------------------------------
            # SPLIT-WISE METRICS
            # ------------------------------------------------

            uncalibrated_ranking = (
                calculate_ranking_metrics(
                    y_validation,
                    uncalibrated_probabilities,
                )
            )

            calibrated_ranking = (
                calculate_ranking_metrics(
                    y_validation,
                    calibrated_probabilities,
                )
            )

            split_records.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "split": split_counter,
                    "uncalibrated_roc_auc": (
                        uncalibrated_ranking[
                            "roc_auc"
                        ]
                    ),
                    "uncalibrated_pr_auc": (
                        uncalibrated_ranking[
                            "pr_auc"
                        ]
                    ),
                    "uncalibrated_brier_score": (
                        uncalibrated_ranking[
                            "brier_score"
                        ]
                    ),
                    "uncalibrated_log_loss": (
                        uncalibrated_ranking[
                            "log_loss"
                        ]
                    ),
                    "uncalibrated_ece": (
                        uncalibrated_ranking[
                            "expected_calibration_error"
                        ]
                    ),
                    "calibrated_roc_auc": (
                        calibrated_ranking[
                            "roc_auc"
                        ]
                    ),
                    "calibrated_pr_auc": (
                        calibrated_ranking[
                            "pr_auc"
                        ]
                    ),
                    "calibrated_brier_score": (
                        calibrated_ranking[
                            "brier_score"
                        ]
                    ),
                    "calibrated_log_loss": (
                        calibrated_ranking[
                            "log_loss"
                        ]
                    ),
                    "calibrated_ece": (
                        calibrated_ranking[
                            "expected_calibration_error"
                        ]
                    ),
                }
            )

    uncalibrated = pd.concat(
        uncalibrated_records,
        ignore_index=True,
    )

    calibrated = pd.concat(
        calibrated_records,
        ignore_index=True,
    )

    split_df = pd.DataFrame(
        split_records
    )

    return (
        uncalibrated,
        calibrated,
        split_df,
    )


# ============================================================
# AGGREGATE REPEATED PREDICTIONS
# ============================================================

def aggregate_repeated_predictions(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average repeated predictions for each observation.

    Each employee is evaluated across the repeated validation
    process. Averaging produces one aggregate OOF probability
    per employee.
    """

    return (
        prediction_df
        .groupby("row_index")
        .agg(
            y_true=("y_true", "first"),
            probability=("probability", "mean"),
        )
        .reset_index()
    )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def evaluate_threshold_grid(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all candidate thresholds."""

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


def select_f1_optimal_threshold(
    threshold_df: pd.DataFrame,
) -> pd.Series:
    """
    Select the F1-optimal threshold.

    Tie-breaking:
        1. Highest F1
        2. Higher recall
        3. Higher specificity
        4. Higher threshold

    The last condition favors a more conservative operating point
    if all earlier metrics are tied.
    """

    ranked = threshold_df.sort_values(
        by=[
            "f1",
            "recall",
            "specificity",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    )

    return ranked.iloc[0]


# ============================================================
# COMPARISON
# ============================================================

def build_operating_comparison(
    y_true: np.ndarray,
    uncalibrated_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    calibrated_threshold: float,
) -> pd.DataFrame:
    """Compare the old and new operating points."""

    old_metrics = calculate_threshold_metrics(
        y_true,
        uncalibrated_probabilities,
        UNCALIBRATED_THRESHOLD,
    )

    new_metrics = calculate_threshold_metrics(
        y_true,
        calibrated_probabilities,
        calibrated_threshold,
    )

    rows = []

    rows.append(
        {
            "model": "Uncalibrated Random Forest",
            "probability_type": "uncalibrated",
            "threshold": (
                UNCALIBRATED_THRESHOLD
            ),
            **old_metrics,
        }
    )

    rows.append(
        {
            "model": "Sigmoid-Calibrated Random Forest",
            "probability_type": "sigmoid_calibrated",
            "threshold": calibrated_threshold,
            **new_metrics,
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# DIAGNOSTICS
# ============================================================

def generate_diagnostic_flags(
    comparison_df: pd.DataFrame,
    calibrated_threshold_df: pd.DataFrame,
    calibrated_metrics: Dict[str, float],
    observed_prevalence: float,
) -> List[str]:
    """Generate diagnostic messages."""

    flags = []

    uncalibrated_row = (
        comparison_df[
            comparison_df["probability_type"]
            == "uncalibrated"
        ]
        .iloc[0]
    )

    calibrated_row = (
        comparison_df[
            comparison_df["probability_type"]
            == "sigmoid_calibrated"
        ]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Threshold movement
    # --------------------------------------------------------

    if (
        calibrated_row["threshold"]
        != UNCALIBRATED_THRESHOLD
    ):
        flags.append(
            "The calibrated model requires a "
            "different operating threshold from "
            "the previous uncalibrated 0.44 threshold."
        )

    # --------------------------------------------------------
    # F1 comparison
    # --------------------------------------------------------

    f1_change = (
        calibrated_row["f1"]
        - uncalibrated_row["f1"]
    )

    if f1_change > 0:
        flags.append(
            "The re-optimized calibrated operating "
            "point improves F1 relative to the previous "
            "uncalibrated operating point."
        )
    else:
        flags.append(
            "The re-optimized calibrated operating "
            "point does not improve F1 relative to the "
            "previous uncalibrated operating point."
        )

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    if calibrated_row["precision"] < 0.40:
        flags.append(
            "Precision remains below 0.40 at the "
            "selected calibrated operating threshold, "
            "indicating a substantial false-positive burden."
        )

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    if calibrated_row["recall"] >= 0.70:
        flags.append(
            "The calibrated operating point prioritizes "
            "detection with recall of at least 0.70."
        )

    # --------------------------------------------------------
    # Predicted-positive volume
    # --------------------------------------------------------

    if (
        calibrated_row[
            "predicted_positive_percent"
        ]
        > 50.0
    ):
        flags.append(
            "The calibrated threshold flags more than "
            "half of observations; intervention capacity "
            "should be explicitly reviewed."
        )

    if (
        calibrated_row[
            "predicted_positive_percent"
        ]
        > observed_prevalence * 100.0 * 2.0
    ):
        flags.append(
            "The calibrated predicted-positive rate "
            "is more than twice observed attrition "
            "prevalence."
        )

    # --------------------------------------------------------
    # Calibration quality
    # --------------------------------------------------------

    if (
        calibrated_metrics[
            "expected_calibration_error"
        ]
        >= 0.10
    ):
        flags.append(
            "Material calibration error remains after "
            "sigmoid calibration; probability estimates "
            "should be interpreted cautiously."
        )
    else:
        flags.append(
            "Expected calibration error is below 0.10 "
            "after sigmoid calibration."
        )

    # --------------------------------------------------------
    # Probability range
    # --------------------------------------------------------

    if calibrated_metrics["brier_score"] < 0.20:
        flags.append(
            "The calibrated model achieves a Brier "
            "Score below 0.20."
        )

    # --------------------------------------------------------
    # Business interpretation
    # --------------------------------------------------------

    flags.append(
        "Threshold selection should ultimately be "
        "validated against intervention capacity and "
        "false-positive versus false-negative costs."
    )

    return flags


# ============================================================
# REPORT GENERATION
# ============================================================

def write_json_report(
    path: Path,
    report: Dict,
) -> None:
    """Write JSON report."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            default=float,
        )


def write_summary_report(
    path: Path,
    report: Dict,
    flags: List[str],
) -> None:
    """Write human-readable TXT summary."""

    model_comparison = report[
        "model_comparison"
    ]

    selected = report[
        "selected_threshold"
    ]

    comparison = pd.DataFrame(
        report["operating_comparison"]
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "CALIBRATED THRESHOLD OPTIMIZATION\n"
        )

        file.write("=" * 72)
        file.write("\n\n")

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
            f"Features:             "
            f"{report['dataset']['feature_count']}\n"
        )
        file.write(
            f"Target prevalence:    "
            f"{report['dataset']['target_prevalence'] * 100:.2f}%\n"
        )

        file.write("\n[MODEL]\n")
        file.write(
            "Model:                "
            "Random Forest\n"
        )
        file.write(
            "Calibration:          "
            "Sigmoid / Platt\n"
        )

        file.write(
            "\n[VALIDATION]\n"
        )
        file.write(
            f"Folds per repeat:     "
            f"{N_SPLITS}\n"
        )
        file.write(
            f"Repeats:              "
            f"{N_REPEATS}\n"
        )
        file.write(
            f"Total validation:     "
            f"{N_SPLITS * N_REPEATS}\n"
        )

        file.write(
            "\n[PROBABILITY QUALITY]\n"
        )

        for row in model_comparison:
            file.write(
                f"{row['method']:<35}"
                f"ROC-AUC={row['roc_auc']:.4f} "
                f"PR-AUC={row['pr_auc']:.4f} "
                f"Brier={row['brier_score']:.4f} "
                f"LogLoss={row['log_loss']:.4f} "
                f"ECE={row['expected_calibration_error']:.4f}\n"
            )

        file.write(
            "\n[CALIBRATED THRESHOLD SEARCH]\n"
        )

        file.write(
            f"Selected threshold:   "
            f"{selected['threshold']:.2f}\n"
        )

        file.write(
            f"F1:                   "
            f"{selected['f1']:.4f}\n"
        )

        file.write(
            f"Precision:            "
            f"{selected['precision']:.4f}\n"
        )

        file.write(
            f"Recall:               "
            f"{selected['recall']:.4f}\n"
        )

        file.write(
            f"Specificity:          "
            f"{selected['specificity']:.4f}\n"
        )

        file.write(
            f"Balanced Accuracy:    "
            f"{selected['balanced_accuracy']:.4f}\n"
        )

        file.write(
            f"Predicted Positive:   "
            f"{selected['predicted_positive_percent']:.2f}%\n"
        )

        file.write(
            f"Flagged per 1000:     "
            f"{selected['flagged_per_1000']:.1f}\n"
        )

        file.write(
            "\n[OPERATING COMPARISON]\n"
        )

        for _, row in comparison.iterrows():
            file.write(
                f"{row['model']:<35}"
                f"Threshold={row['threshold']:.2f} "
                f"F1={row['f1']:.4f} "
                f"Precision={row['precision']:.4f} "
                f"Recall={row['recall']:.4f} "
                f"Specificity={row['specificity']:.4f} "
                f"Flagged/1000={row['flagged_per_1000']:.1f}\n"
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
            report["overall_diagnosis"]
        )

        file.write("\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run calibrated threshold optimization."""

    print(
        "Running calibrated stable-feature "
        "threshold optimization..."
    )

    print(
        "Loading canonical dataset..."
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: "
            f"{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    # --------------------------------------------------------
    # DATASET VALIDATION
    # --------------------------------------------------------

    print(
        "\nValidating canonical dataset..."
    )

    checks = validate_dataset(df)

    print_validation(checks)

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

    y = normalize_target(
        df[TARGET_COLUMN]
    )

    X = df[
        STABLE_FEATURES
    ].copy()

    numerical_features, categorical_features = (
        identify_feature_types(df)
    )

    prevalence = float(
        np.mean(y)
    )

    print(
        f"\nStable features:      "
        f"{len(STABLE_FEATURES)}"
    )

    print(
        f"Numerical features:    "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features:  "
        f"{len(categorical_features)}"
    )

    print(
        f"Target prevalence:     "
        f"{prevalence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\n================================================================"
    )
    print(
        "REPEATED OOF CALIBRATED THRESHOLD OPTIMIZATION"
    )
    print(
        "================================================================"
    )

    print(
        "Generating repeated out-of-fold predictions..."
    )

    print(
        f"Folds per repeat: {N_SPLITS}"
    )

    print(
        f"Repeats:           {N_REPEATS}"
    )

    print(
        f"Total validation:  "
        f"{N_SPLITS * N_REPEATS}"
    )

    (
        uncalibrated_predictions,
        calibrated_predictions,
        split_performance,
    ) = generate_oof_predictions(
        X,
        y,
        numerical_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # AGGREGATE
    # --------------------------------------------------------

    uncalibrated_aggregate = (
        aggregate_repeated_predictions(
            uncalibrated_predictions
        )
    )

    calibrated_aggregate = (
        aggregate_repeated_predictions(
            calibrated_predictions
        )
    )

    y_aggregate = (
        calibrated_aggregate[
            "y_true"
        ].to_numpy()
    )

    uncalibrated_probabilities = (
        uncalibrated_aggregate[
            "probability"
        ].to_numpy()
    )

    calibrated_probabilities = (
        calibrated_aggregate[
            "probability"
        ].to_numpy()
    )

    # --------------------------------------------------------
    # RANKING / CALIBRATION METRICS
    # --------------------------------------------------------

    print(
        "\nCalculating aggregate probability metrics..."
    )

    uncalibrated_metrics = (
        calculate_ranking_metrics(
            y_aggregate,
            uncalibrated_probabilities,
        )
    )

    calibrated_metrics = (
        calculate_ranking_metrics(
            y_aggregate,
            calibrated_probabilities,
        )
    )

    model_comparison = pd.DataFrame(
        [
            {
                "method": (
                    "Random Forest"
                ),
                **uncalibrated_metrics,
            },
            {
                "method": (
                    "Sigmoid Calibration"
                ),
                **calibrated_metrics,
            },
        ]
    )

    # --------------------------------------------------------
    # THRESHOLD SEARCH
    # --------------------------------------------------------

    print(
        "\nEvaluating calibrated candidate thresholds..."
    )

    threshold_results = (
        evaluate_threshold_grid(
            y_aggregate,
            calibrated_probabilities,
        )
    )

    selected_threshold = (
        select_f1_optimal_threshold(
            threshold_results
        )
    )

    selected_threshold_value = float(
        selected_threshold[
            "threshold"
        ]
    )

    # --------------------------------------------------------
    # OLD OPERATING POINT
    # --------------------------------------------------------

    old_operating_metrics = (
        calculate_threshold_metrics(
            y_aggregate,
            uncalibrated_probabilities,
            UNCALIBRATED_THRESHOLD,
        )
    )

    # --------------------------------------------------------
    # NEW OPERATING POINT
    # --------------------------------------------------------

    new_operating_metrics = (
        calculate_threshold_metrics(
            y_aggregate,
            calibrated_probabilities,
            selected_threshold_value,
        )
    )

    # --------------------------------------------------------
    # OPERATING COMPARISON
    # --------------------------------------------------------

    operating_comparison = (
        build_operating_comparison(
            y_aggregate,
            uncalibrated_probabilities,
            calibrated_probabilities,
            selected_threshold_value,
        )
    )

    # --------------------------------------------------------
    # METRIC CHANGES
    # --------------------------------------------------------

    f1_change = (
        new_operating_metrics["f1"]
        - old_operating_metrics["f1"]
    )

    precision_change = (
        new_operating_metrics["precision"]
        - old_operating_metrics["precision"]
    )

    recall_change = (
        new_operating_metrics["recall"]
        - old_operating_metrics["recall"]
    )

    specificity_change = (
        new_operating_metrics["specificity"]
        - old_operating_metrics["specificity"]
    )

    # --------------------------------------------------------
    # DIAGNOSTICS
    # --------------------------------------------------------

    flags = generate_diagnostic_flags(
        operating_comparison,
        threshold_results,
        calibrated_metrics,
        prevalence,
    )

    # --------------------------------------------------------
    # OVERALL DIAGNOSIS
    # --------------------------------------------------------

    if (
        calibrated_metrics[
            "expected_calibration_error"
        ]
        < uncalibrated_metrics[
            "expected_calibration_error"
        ]
        and f1_change >= 0
    ):
        overall_diagnosis = (
            "Sigmoid calibration improves probability "
            "quality while the re-optimized calibrated "
            "threshold preserves or improves the previous "
            "classification operating point. The calibrated "
            "threshold is a stronger candidate for further "
            "business validation, subject to intervention "
            "capacity and error-cost review."
        )
    elif (
        calibrated_metrics[
            "expected_calibration_error"
        ]
        < uncalibrated_metrics[
            "expected_calibration_error"
        ]
    ):
        overall_diagnosis = (
            "Sigmoid calibration improves probability "
            "quality, but the calibrated threshold does "
            "not improve the previous classification "
            "operating point. Calibration should therefore "
            "be treated primarily as a probability-quality "
            "improvement rather than an automatic deployment "
            "upgrade."
        )
    else:
        overall_diagnosis = (
            "Sigmoid calibration does not provide sufficient "
            "evidence of improvement to justify replacing "
            "the current uncalibrated operating configuration. "
            "Further validation is recommended before adoption."
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n================================================================"
    )
    print(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED THRESHOLD OPTIMIZATION"
    )
    print(
        "================================================================"
    )

    print(
        "\n[DATASET]"
    )

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
        f"{prevalence * 100:.2f}%"
    )

    print(
        "\n[MODEL]"
    )

    print(
        "Model:                Random Forest"
    )

    print(
        "Feature set:          Stable 10-feature subset"
    )

    print(
        "Calibration:          Sigmoid / Platt"
    )

    print(
        "\n[PROBABILITY QUALITY]"
    )

    print(
        model_comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\n[CALIBRATED THRESHOLD SEARCH]"
    )

    print(
        threshold_results[
            [
                "threshold",
                "f1",
                "precision",
                "recall",
                "specificity",
                "balanced_accuracy",
                "predicted_positive_percent",
                "flagged_per_1000",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print(
        "\n[F1-OPTIMAL CALIBRATED OPERATING POINT]"
    )

    print(
        f"Threshold:            "
        f"{selected_threshold_value:.2f}"
    )

    print(
        f"F1:                   "
        f"{new_operating_metrics['f1']:.4f}"
    )

    print(
        f"Precision:            "
        f"{new_operating_metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{new_operating_metrics['recall']:.4f}"
    )

    print(
        f"Specificity:          "
        f"{new_operating_metrics['specificity']:.4f}"
    )

    print(
        f"Balanced Accuracy:    "
        f"{new_operating_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Predicted Positive:   "
        f"{new_operating_metrics['predicted_positive_percent']:.2f}%"
    )

    print(
        f"Flagged per 1000:     "
        f"{new_operating_metrics['flagged_per_1000']:.1f}"
    )

    print(
        "\n[PREVIOUS UNCALIBRATED OPERATING POINT]"
    )

    print(
        f"Threshold:            "
        f"{UNCALIBRATED_THRESHOLD:.2f}"
    )

    print(
        f"F1:                   "
        f"{old_operating_metrics['f1']:.4f}"
    )

    print(
        f"Precision:            "
        f"{old_operating_metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{old_operating_metrics['recall']:.4f}"
    )

    print(
        f"Specificity:          "
        f"{old_operating_metrics['specificity']:.4f}"
    )

    print(
        "\n[OPERATING POINT CHANGE]"
    )

    print(
        f"F1 change:             "
        f"{f1_change:+.4f}"
    )

    print(
        f"Precision change:      "
        f"{precision_change:+.4f}"
    )

    print(
        f"Recall change:         "
        f"{recall_change:+.4f}"
    )

    print(
        f"Specificity change:    "
        f"{specificity_change:+.4f}"
    )

    print(
        "\n[DIAGNOSTIC FLAGS]"
    )

    for flag in flags:
        print(
            f"- {flag}"
        )

    print(
        "\n[OVERALL DIAGNOSIS]"
    )

    print(
        overall_diagnosis
    )

    # --------------------------------------------------------
    # REPORT DIRECTORY
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV 1 — THRESHOLD RESULTS
    # --------------------------------------------------------

    threshold_csv = (
        REPORT_DIR
        / "calibrated_threshold_results.csv"
    )

    threshold_results.to_csv(
        threshold_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CSV 2 — OPERATING COMPARISON
    # --------------------------------------------------------

    comparison_csv = (
        REPORT_DIR
        / "calibrated_operating_comparison.csv"
    )

    operating_comparison.to_csv(
        comparison_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CSV 3 — MODEL COMPARISON
    # --------------------------------------------------------

    model_comparison_csv = (
        REPORT_DIR
        / "calibrated_model_comparison.csv"
    )

    model_comparison.to_csv(
        model_comparison_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CSV 4 — SPLIT PERFORMANCE
    # --------------------------------------------------------

    split_csv = (
        REPORT_DIR
        / "calibrated_threshold_split_performance.csv"
    )

    split_performance.to_csv(
        split_csv,
        index=False,
    )

    # --------------------------------------------------------
    # CSV 5 — OOF PREDICTIONS
    # --------------------------------------------------------

    oof_csv = (
        REPORT_DIR
        / "calibrated_threshold_oof_predictions.csv"
    )

    oof_output = calibrated_aggregate.copy()

    oof_output[
        "uncalibrated_probability"
    ] = uncalibrated_aggregate[
        "probability"
    ]

    oof_output[
        "calibrated_probability"
    ] = calibrated_aggregate[
        "probability"
    ]

    oof_output[
        "uncalibrated_prediction_044"
    ] = (
        oof_output[
            "uncalibrated_probability"
        ]
        >= UNCALIBRATED_THRESHOLD
    ).astype(int)

    oof_output[
        "calibrated_prediction_selected"
    ] = (
        oof_output[
            "calibrated_probability"
        ]
        >= selected_threshold_value
    ).astype(int)

    oof_output.to_csv(
        oof_csv,
        index=False,
    )

    # --------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "sha256": sha256_file(DATA_PATH),
            "rows": len(df),
            "columns": len(df.columns),
            "feature_count": len(
                STABLE_FEATURES
            ),
            "target_prevalence": prevalence,
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
            "seeds": RANDOM_SEEDS,
        },
        "model": {
            "name": "Random Forest",
            "feature_set": STABLE_FEATURES,
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "calibration": {
            "method": "sigmoid",
            "name": "Sigmoid Calibration",
        },
        "uncalibrated_metrics": (
            uncalibrated_metrics
        ),
        "calibrated_metrics": (
            calibrated_metrics
        ),
        "model_comparison": (
            model_comparison.to_dict(
                orient="records"
            )
        ),
        "selected_threshold": (
            selected_threshold.to_dict()
        ),
        "old_operating_point": (
            old_operating_metrics
        ),
        "new_operating_point": (
            new_operating_metrics
        ),
        "operating_changes": {
            "f1_change": f1_change,
            "precision_change": precision_change,
            "recall_change": recall_change,
            "specificity_change": (
                specificity_change
            ),
        },
        "operating_comparison": (
            operating_comparison.to_dict(
                orient="records"
            )
        ),
        "diagnostic_flags": flags,
        "overall_diagnosis": (
            overall_diagnosis
        ),
        "dataset_validation": checks,
    }

    json_path = (
        REPORT_DIR
        / "calibrated_threshold_optimization_stable_report.json"
    )

    write_json_report(
        json_path,
        report,
    )

    # --------------------------------------------------------
    # TXT SUMMARY
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "calibrated_threshold_optimization_stable_summary.txt"
    )

    write_summary_report(
        summary_path,
        report,
        flags,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        "\n[OUTPUT]"
    )

    print(
        f"Reports:              "
        f"{REPORT_DIR}"
    )

    print(
        f"Threshold CSV:        "
        f"{threshold_csv}"
    )

    print(
        f"Operating comparison: "
        f"{comparison_csv}"
    )

    print(
        f"Model comparison:     "
        f"{model_comparison_csv}"
    )

    print(
        f"Split performance:    "
        f"{split_csv}"
    )

    print(
        f"OOF predictions:      "
        f"{oof_csv}"
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
        "\n================================================================"
    )

    print(
        "CALIBRATED THRESHOLD OPTIMIZATION COMPLETE"
    )

    print(
        "================================================================"
    )


if __name__ == "__main__":
    main()