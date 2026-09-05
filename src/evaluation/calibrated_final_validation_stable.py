"""
Final validation of the calibrated stable-feature employee attrition model.

This stage performs an apples-to-apples repeated out-of-fold comparison of:

1. Uncalibrated optimized Random Forest
   - stable 10-feature subset
   - threshold 0.44

2. Sigmoid / Platt calibrated optimized Random Forest
   - stable 10-feature subset
   - threshold 0.25

Validation design:
    5 folds x 5 repeats = 25 validation splits

Important:
    Calibration is fitted inside each training fold only. The validation
    fold is never used to fit the calibrator.

Outputs:
    reports/signal_analysis/calibrated_final_validation_stable/
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List

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
from sklearn.preprocessing import OneHotEncoder

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
    / "calibrated_final_validation_stable"
)

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26

N_SPLITS = 5
N_REPEATS = 5

RANDOM_SEEDS = [42, 52, 62, 72, 82]

UNCALIBRATED_THRESHOLD = 0.44
CALIBRATED_THRESHOLD = 0.25

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
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def normalize_target(series: pd.Series) -> np.ndarray:
    """
    Canonical target:
        Yes -> 1
        No  -> 0
    """
    normalized = series.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
    }

    values = normalized.map(mapping)

    if values.isna().any():
        invalid = sorted(normalized[values.isna()].unique().tolist())
        raise ValueError(
            f"Unsupported target values in {TARGET_COLUMN}: {invalid}"
        )

    return values.astype(int).to_numpy()


def validate_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}

    checks["file_exists"] = DATA_PATH.exists()
    checks["expected_rows"] = len(df) == EXPECTED_ROWS
    checks["expected_columns"] = len(df.columns) == EXPECTED_COLUMNS
    checks["target_exists"] = TARGET_COLUMN in df.columns
    checks["identifier_exists"] = IDENTIFIER_COLUMN in df.columns

    checks["stable_features_exist"] = all(
        feature in df.columns for feature in STABLE_FEATURES
    )

    if TARGET_COLUMN in df.columns:
        normalized = df[TARGET_COLUMN].astype(str).str.strip().str.lower()
        checks["target_values_valid"] = normalized.isin(
            {"yes", "no", "1", "0"}
        ).all()
    else:
        checks["target_values_valid"] = False

    checks["no_missing_cells"] = not df.isna().any().any()

    if IDENTIFIER_COLUMN in df.columns:
        checks["identifier_unique"] = df[IDENTIFIER_COLUMN].is_unique
    else:
        checks["identifier_unique"] = False

    checks["stable_feature_count"] = len(STABLE_FEATURES) == 10

    return checks


def print_validation(checks: Dict[str, bool]) -> None:
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")


def identify_feature_types(
    df: pd.DataFrame,
) -> tuple[List[str], List[str]]:
    model_features = [
        column
        for column in df.columns
        if column not in {IDENTIFIER_COLUMN, TARGET_COLUMN}
    ]

    numerical_features = [
        feature
        for feature in STABLE_FEATURES
        if pd.api.types.is_numeric_dtype(df[feature])
    ]

    categorical_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in numerical_features
    ]

    if set(numerical_features + categorical_features) != set(STABLE_FEATURES):
        raise ValueError("Stable feature type partition is incomplete.")

    if len(model_features) != 24:
        raise ValueError(
            f"Expected 24 model features after identifier exclusion; "
            f"found {len(model_features)}."
        )

    return numerical_features, categorical_features


# ============================================================
# MODEL
# ============================================================

def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    transformers = []

    if numerical_features:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
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


def build_random_forest_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
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
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def safe_log_loss(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return float(log_loss(y_true, probabilities))


def calculate_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    probabilities = np.asarray(probabilities, dtype=float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)

        if not np.any(mask):
            continue

        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(probabilities[mask]))
        weight = float(np.mean(mask))

        ece += weight * abs(observed - predicted)

    return float(ece)


def ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "brier_score": float(
            brier_score_loss(y_true, probabilities)
        ),
        "log_loss": safe_log_loss(y_true, probabilities),
        "expected_calibration_error": calculate_ece(
            y_true,
            probabilities,
        ),
    }


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = (
        __import__("sklearn.metrics", fromlist=["confusion_matrix"])
        .confusion_matrix(y_true, predictions)
        .ravel()
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

    predicted_positive = float(np.mean(predictions))

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
        "balanced_accuracy": float(balanced_accuracy),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "predicted_positive_percent": (
            predicted_positive * 100.0
        ),
        "flagged_per_1000": (
            predicted_positive * 1000.0
        ),
        "false_positives_per_1000": (
            fp / len(y_true) * 1000.0
        ),
        "missed_attrition_per_1000": (
            fn / len(y_true) * 1000.0
        ),
    }


# ============================================================
# REPEATED OOF VALIDATION
# ============================================================

def generate_repeated_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    numerical_features: List[str],
    categorical_features: List[str],
):
    uncalibrated_records = []
    calibrated_records = []
    split_records = []

    total_splits = N_SPLITS * N_REPEATS
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
                f"Validation split "
                f"{split_counter}/{total_splits}"
            )

            X_train = X.iloc[train_index]
            X_validation = X.iloc[validation_index]

            y_train = y[train_index]
            y_validation = y[validation_index]

            # ------------------------------------------------
            # UNCALIBRATED MODEL
            # ------------------------------------------------

            uncalibrated_model = build_random_forest_pipeline(
                numerical_features,
                categorical_features,
            )

            uncalibrated_model.fit(
                X_train,
                y_train,
            )

            uncalibrated_probability = (
                uncalibrated_model.predict_proba(
                    X_validation
                )[:, 1]
            )

            # ------------------------------------------------
            # SIGMOID-CALIBRATED MODEL
            # ------------------------------------------------

            calibration_base = build_random_forest_pipeline(
                numerical_features,
                categorical_features,
            )

            calibrated_model = CalibratedClassifierCV(
                estimator=calibration_base,
                method="sigmoid",
                cv=3,
            )

            calibrated_model.fit(
                X_train,
                y_train,
            )

            calibrated_probability = (
                calibrated_model.predict_proba(
                    X_validation
                )[:, 1]
            )

            uncalibrated_records.append(
                pd.DataFrame(
                    {
                        "row_index": validation_index,
                        "y_true": y_validation,
                        "probability": uncalibrated_probability,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                    }
                )
            )

            calibrated_records.append(
                pd.DataFrame(
                    {
                        "row_index": validation_index,
                        "y_true": y_validation,
                        "probability": calibrated_probability,
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                    }
                )
            )

            unc_rank = ranking_metrics(
                y_validation,
                uncalibrated_probability,
            )

            cal_rank = ranking_metrics(
                y_validation,
                calibrated_probability,
            )

            unc_cls = classification_metrics(
                y_validation,
                uncalibrated_probability,
                UNCALIBRATED_THRESHOLD,
            )

            cal_cls = classification_metrics(
                y_validation,
                calibrated_probability,
                CALIBRATED_THRESHOLD,
            )

            split_records.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "split": split_counter,
                    "uncalibrated_roc_auc": unc_rank["roc_auc"],
                    "uncalibrated_pr_auc": unc_rank["pr_auc"],
                    "uncalibrated_brier_score": unc_rank["brier_score"],
                    "uncalibrated_log_loss": unc_rank["log_loss"],
                    "uncalibrated_ece": unc_rank[
                        "expected_calibration_error"
                    ],
                    "uncalibrated_f1": unc_cls["f1"],
                    "uncalibrated_precision": unc_cls["precision"],
                    "uncalibrated_recall": unc_cls["recall"],
                    "uncalibrated_specificity": unc_cls[
                        "specificity"
                    ],
                    "uncalibrated_predicted_positive_percent": (
                        unc_cls["predicted_positive_percent"]
                    ),
                    "calibrated_roc_auc": cal_rank["roc_auc"],
                    "calibrated_pr_auc": cal_rank["pr_auc"],
                    "calibrated_brier_score": cal_rank["brier_score"],
                    "calibrated_log_loss": cal_rank["log_loss"],
                    "calibrated_ece": cal_rank[
                        "expected_calibration_error"
                    ],
                    "calibrated_f1": cal_cls["f1"],
                    "calibrated_precision": cal_cls["precision"],
                    "calibrated_recall": cal_cls["recall"],
                    "calibrated_specificity": cal_cls[
                        "specificity"
                    ],
                    "calibrated_predicted_positive_percent": (
                        cal_cls["predicted_positive_percent"]
                    ),
                }
            )

    return (
        pd.concat(
            uncalibrated_records,
            ignore_index=True,
        ),
        pd.concat(
            calibrated_records,
            ignore_index=True,
        ),
        pd.DataFrame(split_records),
    )


# ============================================================
# REPORTING
# ============================================================

def build_model_comparison(
    uncalibrated_oof: pd.DataFrame,
    calibrated_oof: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for name, frame in [
        ("Random Forest", uncalibrated_oof),
        ("Sigmoid Calibration", calibrated_oof),
    ]:
        y_true = frame["y_true"].to_numpy()
        probabilities = frame["probability"].to_numpy()

        metrics = ranking_metrics(
            y_true,
            probabilities,
        )

        threshold = (
            UNCALIBRATED_THRESHOLD
            if name == "Random Forest"
            else CALIBRATED_THRESHOLD
        )

        cls = classification_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(
            {
                "method": name,
                **metrics,
                "threshold": threshold,
                "f1": cls["f1"],
                "precision": cls["precision"],
                "recall": cls["recall"],
                "specificity": cls["specificity"],
                "balanced_accuracy": cls[
                    "balanced_accuracy"
                ],
                "predicted_positive_percent": cls[
                    "predicted_positive_percent"
                ],
                "flagged_per_1000": cls[
                    "flagged_per_1000"
                ],
            }
        )

    return pd.DataFrame(rows)


def build_threshold_comparison(
    uncalibrated_oof: pd.DataFrame,
    calibrated_oof: pd.DataFrame,
) -> pd.DataFrame:
    y_unc = uncalibrated_oof["y_true"].to_numpy()
    p_unc = uncalibrated_oof["probability"].to_numpy()

    y_cal = calibrated_oof["y_true"].to_numpy()
    p_cal = calibrated_oof["probability"].to_numpy()

    unc = classification_metrics(
        y_unc,
        p_unc,
        UNCALIBRATED_THRESHOLD,
    )

    cal = classification_metrics(
        y_cal,
        p_cal,
        CALIBRATED_THRESHOLD,
    )

    return pd.DataFrame(
        [
            {
                "operating_point": "uncalibrated_0.44",
                **unc,
            },
            {
                "operating_point": "sigmoid_calibrated_0.25",
                **cal,
            },
        ]
    )


def build_calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    rows = []

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    for index, (lower, upper) in enumerate(
        zip(bins[:-1], bins[1:]),
        start=1,
    ):
        if upper == 1.0:
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
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_rate": np.nan,
                    "absolute_gap": np.nan,
                }
            )
            continue

        mean_probability = float(
            np.mean(probabilities[mask])
        )
        observed_rate = float(
            np.mean(y_true[mask])
        )

        rows.append(
            {
                "method": method,
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": abs(
                    mean_probability - observed_rate
                ),
            }
        )

    return pd.DataFrame(rows)


def build_diagnostic_flags(
    model_comparison: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
) -> List[str]:
    flags: List[str] = []

    unc = model_comparison.iloc[0]
    cal = model_comparison.iloc[1]

    unc_op = threshold_comparison.iloc[0]
    cal_op = threshold_comparison.iloc[1]

    if cal["brier_score"] < unc["brier_score"]:
        flags.append(
            "Sigmoid calibration improves Brier Score."
        )

    if cal["log_loss"] < unc["log_loss"]:
        flags.append(
            "Sigmoid calibration improves Log Loss."
        )

    if cal["expected_calibration_error"] < unc[
        "expected_calibration_error"
    ]:
        flags.append(
            "Sigmoid calibration materially improves expected "
            "calibration error."
        )

    if cal["roc_auc"] < unc["roc_auc"]:
        flags.append(
            "Sigmoid calibration slightly reduces ROC-AUC; "
            "calibration should be viewed primarily as a "
            "probability-quality intervention."
        )

    if cal_op["f1"] <= unc_op["f1"]:
        flags.append(
            "The calibrated operating point does not improve F1 "
            "relative to the previous uncalibrated operating point."
        )

    if cal_op["precision"] < 0.40:
        flags.append(
            "Precision remains below 0.40 at the calibrated "
            "operating point."
        )

    if cal_op["predicted_positive_percent"] > 50:
        flags.append(
            "The calibrated operating point still flags more than "
            "half of observations."
        )

    if cal_op["predicted_positive_percent"] > 2 * 23.60:
        flags.append(
            "The calibrated operating point flags more than twice "
            "the observed attrition prevalence."
        )

    if cal_op["recall"] >= 0.70:
        flags.append(
            "The calibrated operating point prioritizes detection "
            "with recall of at least 0.70."
        )

    if cal_op["specificity"] < 0.60:
        flags.append(
            "Specificity remains below 0.60 at the calibrated "
            "operating point."
        )

    if not flags:
        flags.append(
            "No material diagnostic concerns were identified."
        )

    return flags


def write_summary_report(
    path: Path,
    df: pd.DataFrame,
    prevalence: float,
    model_comparison: pd.DataFrame,
    threshold_comparison: pd.DataFrame,
    split_performance: pd.DataFrame,
    flags: List[str],
    status: str,
    diagnosis: str,
) -> None:
    unc = model_comparison.iloc[0]
    cal = model_comparison.iloc[1]

    unc_op = threshold_comparison.iloc[0]
    cal_op = threshold_comparison.iloc[1]

    with path.open("w", encoding="utf-8") as file:
        file.write(
            "EMPLOYEE ATTRITION — "
            "CALIBRATED FINAL VALIDATION\n"
        )
        file.write("=" * 64 + "\n\n")

        file.write("[DATASET]\n")
        file.write(f"Rows:                 {len(df)}\n")
        file.write(f"Columns:              {len(df.columns)}\n")
        file.write(
            f"Stable features:      {len(STABLE_FEATURES)}\n"
        )
        file.write(
            f"Target prevalence:    {prevalence * 100:.2f}%\n\n"
        )

        file.write("[VALIDATION]\n")
        file.write(
            f"Folds per repeat:      {N_SPLITS}\n"
        )
        file.write(
            f"Repeats:               {N_REPEATS}\n"
        )
        file.write(
            f"Total validation:      {N_SPLITS * N_REPEATS}\n\n"
        )

        file.write("[MODEL COMPARISON]\n")
        file.write(
            model_comparison.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        file.write("\n\n")

        file.write("[UNCALIBRATED OPERATING POINT]\n")
        file.write(
            f"Threshold:             {UNCALIBRATED_THRESHOLD:.2f}\n"
        )
        file.write(
            f"F1:                    {unc_op['f1']:.4f}\n"
        )
        file.write(
            f"Precision:             {unc_op['precision']:.4f}\n"
        )
        file.write(
            f"Recall:                {unc_op['recall']:.4f}\n"
        )
        file.write(
            f"Specificity:           {unc_op['specificity']:.4f}\n"
        )
        file.write(
            f"Predicted Positive:    "
            f"{unc_op['predicted_positive_percent']:.2f}%\n"
        )
        file.write(
            f"Flagged per 1000:      "
            f"{unc_op['flagged_per_1000']:.1f}\n\n"
        )

        file.write("[CALIBRATED OPERATING POINT]\n")
        file.write(
            f"Threshold:             {CALIBRATED_THRESHOLD:.2f}\n"
        )
        file.write(
            f"F1:                    {cal_op['f1']:.4f}\n"
        )
        file.write(
            f"Precision:             {cal_op['precision']:.4f}\n"
        )
        file.write(
            f"Recall:                {cal_op['recall']:.4f}\n"
        )
        file.write(
            f"Specificity:           {cal_op['specificity']:.4f}\n"
        )
        file.write(
            f"Predicted Positive:    "
            f"{cal_op['predicted_positive_percent']:.2f}%\n"
        )
        file.write(
            f"Flagged per 1000:      "
            f"{cal_op['flagged_per_1000']:.1f}\n\n"
        )

        file.write("[CALIBRATION CHANGE]\n")
        file.write(
            f"Brier change:          "
            f"{cal['brier_score'] - unc['brier_score']:+.4f}\n"
        )
        file.write(
            f"Log Loss change:       "
            f"{cal['log_loss'] - unc['log_loss']:+.4f}\n"
        )
        file.write(
            f"ECE change:            "
            f"{cal['expected_calibration_error'] - unc['expected_calibration_error']:+.4f}\n"
        )
        file.write(
            f"ROC-AUC change:        "
            f"{cal['roc_auc'] - unc['roc_auc']:+.4f}\n"
        )
        file.write(
            f"PR-AUC change:         "
            f"{cal['pr_auc'] - unc['pr_auc']:+.4f}\n\n"
        )

        file.write("[OPERATING POINT CHANGE]\n")
        file.write(
            f"F1 change:             "
            f"{cal_op['f1'] - unc_op['f1']:+.4f}\n"
        )
        file.write(
            f"Precision change:      "
            f"{cal_op['precision'] - unc_op['precision']:+.4f}\n"
        )
        file.write(
            f"Recall change:         "
            f"{cal_op['recall'] - unc_op['recall']:+.4f}\n"
        )
        file.write(
            f"Specificity change:    "
            f"{cal_op['specificity'] - unc_op['specificity']:+.4f}\n"
        )
        file.write(
            f"Flagged/1000 change:   "
            f"{cal_op['flagged_per_1000'] - unc_op['flagged_per_1000']:+.1f}\n\n"
        )

        file.write("[STABILITY]\n")

        for column in [
            "uncalibrated_roc_auc",
            "calibrated_roc_auc",
        ]:
            values = split_performance[column]

            label = (
                "Uncalibrated"
                if column.startswith("uncalibrated")
                else "Sigmoid calibrated"
            )

            file.write(
                f"{label} ROC-AUC mean:  "
                f"{values.mean():.4f}\n"
            )
            file.write(
                f"{label} ROC-AUC std:   "
                f"{values.std(ddof=1):.4f}\n"
            )
            file.write(
                f"{label} ROC-AUC min:   "
                f"{values.min():.4f}\n"
            )
            file.write(
                f"{label} ROC-AUC max:   "
                f"{values.max():.4f}\n"
            )

        file.write("\n[DIAGNOSTIC FLAGS]\n")

        for flag in flags:
            file.write(f"- {flag}\n")

        file.write("\n[OVERALL STATUS]\n")
        file.write(
            f"CALIBRATED FINAL VALIDATION STATUS: {status}\n"
        )

        file.write("\n[OVERALL DIAGNOSIS]\n")
        file.write(diagnosis + "\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "Running calibrated stable-feature final validation..."
    )

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    print("\nValidating canonical dataset...")

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

    y = normalize_target(df[TARGET_COLUMN])

    X = df[STABLE_FEATURES].copy()

    numerical_features, categorical_features = (
        identify_feature_types(df)
    )

    prevalence = float(np.mean(y))

    print(
        f"\nStable features:      {len(STABLE_FEATURES)}"
    )
    print(
        f"Numerical features:    {len(numerical_features)}"
    )
    print(
        f"Categorical features:  {len(categorical_features)}"
    )
    print(
        f"Target prevalence:     {prevalence * 100:.2f}%"
    )

    print("\n" + "=" * 64)
    print("FINAL REPEATED OOF CALIBRATED VALIDATION")
    print("=" * 64)

    print("Generating repeated out-of-fold predictions...")
    print(f"Folds per repeat: {N_SPLITS}")
    print(f"Repeats:          {N_REPEATS}")
    print(
        f"Total validation: {N_SPLITS * N_REPEATS}"
    )

    (
        uncalibrated_oof,
        calibrated_oof,
        split_performance,
    ) = generate_repeated_oof_predictions(
        X,
        y,
        numerical_features,
        categorical_features,
    )

    print("\nCalculating aggregate validation metrics...")

    model_comparison = build_model_comparison(
        uncalibrated_oof,
        calibrated_oof,
    )

    threshold_comparison = build_threshold_comparison(
        uncalibrated_oof,
        calibrated_oof,
    )

    unc_bins = build_calibration_bins(
        uncalibrated_oof["y_true"].to_numpy(),
        uncalibrated_oof["probability"].to_numpy(),
        "Random Forest",
    )

    cal_bins = build_calibration_bins(
        calibrated_oof["y_true"].to_numpy(),
        calibrated_oof["probability"].to_numpy(),
        "Sigmoid Calibration",
    )

    calibration_bins = pd.concat(
        [unc_bins, cal_bins],
        ignore_index=True,
    )

    flags = build_diagnostic_flags(
        model_comparison,
        threshold_comparison,
    )

    unc = model_comparison.iloc[0]
    cal = model_comparison.iloc[1]
    unc_op = threshold_comparison.iloc[0]
    cal_op = threshold_comparison.iloc[1]

    # --------------------------------------------------------
    # DECISION LOGIC
    # --------------------------------------------------------

    probability_quality_improved = (
        cal["brier_score"] < unc["brier_score"]
        and cal["log_loss"] < unc["log_loss"]
        and cal["expected_calibration_error"]
        < unc["expected_calibration_error"]
    )

    classification_improved = (
        cal_op["f1"] > unc_op["f1"]
        and cal_op["recall"] >= unc_op["recall"]
    )

    if probability_quality_improved and classification_improved:
        status = "PASS"

        diagnosis = (
            "Sigmoid calibration improves probability quality and "
            "also improves the selected classification operating "
            "point under repeated out-of-fold validation. The "
            "calibrated configuration is supported for the next "
            "deployment-readiness stage, subject to business review."
        )

    elif probability_quality_improved:
        status = "CONDITIONAL PASS"

        diagnosis = (
            "Sigmoid calibration materially improves probability "
            "quality, but the calibrated operating point does not "
            "improve the previous classification operating point "
            "on all decision metrics. Calibration is supported for "
            "probability-quality use, but the deployment threshold "
            "must remain a business decision."
        )

    else:
        status = "REVIEW REQUIRED"

        diagnosis = (
            "Sigmoid calibration does not provide sufficiently "
            "consistent improvement in probability quality under "
            "repeated out-of-fold validation. The uncalibrated "
            "configuration should remain the reference until "
            "further calibration work is completed."
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 64)
    print(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED FINAL VALIDATION"
    )
    print("=" * 64)

    print("\n[DATASET]")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print(
        f"Stable features:      {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    {prevalence * 100:.2f}%"
    )

    print("\n[MODEL COMPARISON]")
    print(
        model_comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\n[UNCALIBRATED OPERATING POINT]")
    print(
        f"Threshold:            {UNCALIBRATED_THRESHOLD:.2f}"
    )
    print(f"F1:                   {unc_op['f1']:.4f}")
    print(
        f"Precision:            {unc_op['precision']:.4f}"
    )
    print(f"Recall:               {unc_op['recall']:.4f}")
    print(
        f"Specificity:          {unc_op['specificity']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{unc_op['predicted_positive_percent']:.2f}%"
    )
    print(
        f"Flagged per 1000:     "
        f"{unc_op['flagged_per_1000']:.1f}"
    )

    print("\n[SIGMOID CALIBRATED OPERATING POINT]")
    print(
        f"Threshold:            {CALIBRATED_THRESHOLD:.2f}"
    )
    print(f"F1:                   {cal_op['f1']:.4f}")
    print(
        f"Precision:            {cal_op['precision']:.4f}"
    )
    print(f"Recall:               {cal_op['recall']:.4f}")
    print(
        f"Specificity:          {cal_op['specificity']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{cal_op['predicted_positive_percent']:.2f}%"
    )
    print(
        f"Flagged per 1000:     "
        f"{cal_op['flagged_per_1000']:.1f}"
    )

    print("\n[CALIBRATION CHANGE]")
    print(
        f"Brier change:         "
        f"{cal['brier_score'] - unc['brier_score']:+.4f}"
    )
    print(
        f"Log Loss change:      "
        f"{cal['log_loss'] - unc['log_loss']:+.4f}"
    )
    print(
        f"ECE change:           "
        f"{cal['expected_calibration_error'] - unc['expected_calibration_error']:+.4f}"
    )
    print(
        f"ROC-AUC change:       "
        f"{cal['roc_auc'] - unc['roc_auc']:+.4f}"
    )
    print(
        f"PR-AUC change:        "
        f"{cal['pr_auc'] - unc['pr_auc']:+.4f}"
    )

    print("\n[OPERATING POINT CHANGE]")
    print(
        f"F1 change:            "
        f"{cal_op['f1'] - unc_op['f1']:+.4f}"
    )
    print(
        f"Precision change:     "
        f"{cal_op['precision'] - unc_op['precision']:+.4f}"
    )
    print(
        f"Recall change:        "
        f"{cal_op['recall'] - unc_op['recall']:+.4f}"
    )
    print(
        f"Specificity change:   "
        f"{cal_op['specificity'] - unc_op['specificity']:+.4f}"
    )
    print(
        f"Flagged/1000 change:  "
        f"{cal_op['flagged_per_1000'] - unc_op['flagged_per_1000']:+.1f}"
    )

    print("\n[SPLIT STABILITY]")

    for prefix, label in [
        ("uncalibrated", "Uncalibrated"),
        ("calibrated", "Sigmoid Calibrated"),
    ]:
        values = split_performance[
            f"{prefix}_roc_auc"
        ]

        print(
            f"{label} ROC-AUC mean: "
            f"{values.mean():.4f}"
        )
        print(
            f"{label} ROC-AUC std:  "
            f"{values.std(ddof=1):.4f}"
        )
        print(
            f"{label} ROC-AUC min:  "
            f"{values.min():.4f}"
        )
        print(
            f"{label} ROC-AUC max:  "
            f"{values.max():.4f}"
        )

    print("\n[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(f"- {flag}")

    print("\n[OVERALL STATUS]")
    print(
        "CALIBRATED FINAL VALIDATION STATUS: "
        f"{status}"
    )

    print("\n[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV OUTPUTS
    # --------------------------------------------------------

    model_comparison_csv = (
        REPORT_DIR
        / "calibrated_final_model_comparison.csv"
    )

    threshold_comparison_csv = (
        REPORT_DIR
        / "calibrated_final_threshold_comparison.csv"
    )

    split_csv = (
        REPORT_DIR
        / "calibrated_final_split_performance.csv"
    )

    bins_csv = (
        REPORT_DIR
        / "calibrated_final_calibration_bins.csv"
    )

    oof_csv = (
        REPORT_DIR
        / "calibrated_final_oof_predictions.csv"
    )

    model_comparison.to_csv(
        model_comparison_csv,
        index=False,
    )

    threshold_comparison.to_csv(
        threshold_comparison_csv,
        index=False,
    )

    split_performance.to_csv(
        split_csv,
        index=False,
    )

    calibration_bins.to_csv(
        bins_csv,
        index=False,
    )

    oof_output = pd.DataFrame(
        {
            "row_index": uncalibrated_oof[
                "row_index"
            ],
            "y_true": uncalibrated_oof["y_true"],
            "uncalibrated_probability": (
                uncalibrated_oof["probability"]
            ),
            "calibrated_probability": (
                calibrated_oof["probability"]
            ),
        }
    )

    oof_output[
        "uncalibrated_prediction_044"
    ] = (
        oof_output["uncalibrated_probability"]
        >= UNCALIBRATED_THRESHOLD
    ).astype(int)

    oof_output[
        "calibrated_prediction_025"
    ] = (
        oof_output["calibrated_probability"]
        >= CALIBRATED_THRESHOLD
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
            "stable_features": STABLE_FEATURES,
            "feature_count": len(STABLE_FEATURES),
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
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "operating_points": {
            "uncalibrated_threshold": (
                UNCALIBRATED_THRESHOLD
            ),
            "calibrated_threshold": (
                CALIBRATED_THRESHOLD
            ),
        },
        "model_comparison": (
            model_comparison.to_dict(
                orient="records"
            )
        ),
        "threshold_comparison": (
            threshold_comparison.to_dict(
                orient="records"
            )
        ),
        "stability": {
            "uncalibrated_roc_auc_mean": float(
                split_performance[
                    "uncalibrated_roc_auc"
                ].mean()
            ),
            "uncalibrated_roc_auc_std": float(
                split_performance[
                    "uncalibrated_roc_auc"
                ].std(ddof=1)
            ),
            "uncalibrated_roc_auc_min": float(
                split_performance[
                    "uncalibrated_roc_auc"
                ].min()
            ),
            "uncalibrated_roc_auc_max": float(
                split_performance[
                    "uncalibrated_roc_auc"
                ].max()
            ),
            "calibrated_roc_auc_mean": float(
                split_performance[
                    "calibrated_roc_auc"
                ].mean()
            ),
            "calibrated_roc_auc_std": float(
                split_performance[
                    "calibrated_roc_auc"
                ].std(ddof=1)
            ),
            "calibrated_roc_auc_min": float(
                split_performance[
                    "calibrated_roc_auc"
                ].min()
            ),
            "calibrated_roc_auc_max": float(
                split_performance[
                    "calibrated_roc_auc"
                ].max()
            ),
        },
        "diagnostic_flags": flags,
        "status": status,
        "diagnosis": diagnosis,
    }

    json_path = (
        REPORT_DIR
        / "calibrated_final_validation_stable_report.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "calibrated_final_validation_stable_summary.txt"
    )

    write_summary_report(
        summary_path,
        df,
        prevalence,
        model_comparison,
        threshold_comparison,
        split_performance,
        flags,
        status,
        diagnosis,
    )

    print("\n[OUTPUT]")
    print(f"Reports:              {REPORT_DIR}")
    print(
        f"Model comparison:     {model_comparison_csv}"
    )
    print(
        f"Threshold comparison: {threshold_comparison_csv}"
    )
    print(
        f"Split performance:    {split_csv}"
    )
    print(
        f"Calibration bins:     {bins_csv}"
    )
    print(
        f"OOF predictions:      {oof_csv}"
    )
    print(
        f"JSON report:          {json_path}"
    )
    print(
        f"Summary report:       {summary_path}"
    )

    print("\n" + "=" * 64)
    print("CALIBRATED FINAL VALIDATION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()
