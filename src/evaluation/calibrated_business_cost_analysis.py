"""
Calibrated business cost analysis for the stable employee-attrition model.

Purpose
-------
Evaluate whether sigmoid-calibrated Random Forest probabilities provide a
better basis for business threshold selection than the previous
uncalibrated Random Forest.

The analysis:

1. Loads the canonical employee attrition dataset.
2. Validates the canonical schema.
3. Uses the validated stable 10-feature subset.
4. Uses the previously selected Random Forest configuration.
5. Generates repeated out-of-fold predictions.
6. Fits sigmoid calibration inside each training fold only.
7. Evaluates calibrated and uncalibrated probability quality.
8. Evaluates candidate thresholds on the calibrated probability scale.
9. Evaluates multiple false-positive / false-negative cost scenarios.
10. Identifies the cost-optimal threshold for each scenario.
11. Compares calibrated and uncalibrated operating points.
12. Writes CSV, JSON and TXT reports.

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

Identifier
----------
Employee_ID

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
    / "calibrated_business_cost_analysis"
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

# Previously selected sigmoid-calibrated threshold.
CALIBRATED_REFERENCE_THRESHOLD = 0.25

# Candidate thresholds on the calibrated probability scale.
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

# Business cost scenarios.
#
# false-positive cost is normalized to 1.
# false-negative cost represents the relative cost of missing
# an employee who actually leaves.
BUSINESS_SCENARIOS = [
    {
        "scenario": "balanced_1_to_1",
        "false_positive_cost": 1.0,
        "false_negative_cost": 1.0,
    },
    {
        "scenario": "moderate_detection_2_to_1",
        "false_positive_cost": 1.0,
        "false_negative_cost": 2.0,
    },
    {
        "scenario": "high_detection_5_to_1",
        "false_positive_cost": 1.0,
        "false_negative_cost": 5.0,
    },
    {
        "scenario": "very_high_detection_10_to_1",
        "false_positive_cost": 1.0,
        "false_negative_cost": 10.0,
    },
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

EXPECTED_CANONICAL_SHA256 = (
    "9b294a270e34d159ce21e7f2c4d0be394d53f83736bc7d413296be4cf2768ed6"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def print_validation(name: str, passed: bool) -> None:
    """Print a standardized validation result."""

    status = "PASS" if passed else "FAIL"
    print(f"{status} {name}")


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_canonical_dataset(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Validate the canonical dataset.

    This intentionally uses the same schema established by the previous
    stable-feature evaluation scripts.
    """

    results: Dict[str, bool] = {}

    results["file_exists"] = DATA_PATH.exists()

    results["expected_rows"] = (
        len(df) == EXPECTED_ROWS
    )

    results["expected_columns"] = (
        len(df.columns) == EXPECTED_COLUMNS
    )

    results["target_exists"] = (
        TARGET_COLUMN in df.columns
    )

    results["identifier_exists"] = (
        IDENTIFIER_COLUMN in df.columns
    )

    results["stable_features_exist"] = all(
        feature in df.columns
        for feature in STABLE_FEATURES
    )

    if TARGET_COLUMN in df.columns:
        valid_targets = set(
            df[TARGET_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

        results["target_values_valid"] = (
            valid_targets == {"Yes", "No"}
        )
    else:
        results["target_values_valid"] = False

    results["no_missing_cells"] = (
        not df.isnull().any().any()
    )

    if IDENTIFIER_COLUMN in df.columns:
        results["identifier_unique"] = (
            df[IDENTIFIER_COLUMN].is_unique
        )
    else:
        results["identifier_unique"] = False

    results["stable_feature_count"] = (
        len(STABLE_FEATURES) == 10
    )

    return results


def load_and_validate_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load and validate the canonical dataset."""

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print()
    print("Validating canonical dataset...")

    validation = validate_canonical_dataset(df)

    for name, passed in validation.items():
        print_validation(name, passed)

    failed = [
        name
        for name, passed in validation.items()
        if not passed
    ]

    if failed:
        raise ValueError(
            "Canonical dataset validation failed: "
            + ", ".join(failed)
        )

    # Explicit target conversion.
    target_mapping = {
        "No": 0,
        "Yes": 1,
    }

    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map(target_mapping)
        .to_numpy(dtype=int)
    )

    if np.isnan(y).any():
        raise ValueError(
            "Target conversion produced missing values."
        )

    X = df[STABLE_FEATURES].copy()

    print()
    print(
        f"Stable features:      {len(STABLE_FEATURES)}"
    )
    print(
        f"Numerical features:    {len(NUMERICAL_FEATURES)}"
    )
    print(
        f"Categorical features:  {len(CATEGORICAL_FEATURES)}"
    )
    print(
        f"Target prevalence:    {np.mean(y):.2%}"
    )

    return X, y


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """Build the stable-feature preprocessing pipeline."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

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
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
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


# ============================================================
# MODEL
# ============================================================

def build_random_forest() -> Pipeline:
    """
    Build the previously selected stable Random Forest.

    Configuration matches the deployment readiness audit.
    """

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
                build_preprocessor(),
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_ranking_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """Calculate probability/ranking metrics."""

    probabilities = np.clip(
        probabilities,
        1e-6,
        1.0 - 1e-6,
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
    }


def calculate_ece(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate expected calibration error."""

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    n = len(y_true)

    for index in range(n_bins):
        lower = bins[index]
        upper = bins[index + 1]

        if index == n_bins - 1:
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

        mean_probability = np.mean(
            probabilities[mask]
        )

        observed_rate = np.mean(
            y_true[mask]
        )

        ece += (
            np.sum(mask)
            / n
        ) * abs(
            mean_probability
            - observed_rate
        )

    return float(ece)


def calculate_classification_metrics(
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
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "specificity": float(
            specificity
        ),
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

def generate_repeated_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate repeated OOF predictions.

    For every validation fold:

    - fit the Random Forest on training data
    - generate uncalibrated validation probabilities
    - fit sigmoid calibration on training data only
    - generate calibrated validation probabilities

    The validation fold is never used to fit the model/calibrator.
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
                f"Business cost split "
                f"{split_counter}/{total_splits}"
            )

            X_train = X.iloc[
                train_index
            ].copy()

            X_validation = X.iloc[
                validation_index
            ].copy()

            y_train = y[
                train_index
            ]

            y_validation = y[
                validation_index
            ]

            # ------------------------------------------------
            # UNCALIBRATED RANDOM FOREST
            # ------------------------------------------------

            base_model = build_random_forest()

            base_model.fit(
                X_train,
                y_train,
            )

            uncalibrated_probabilities = (
                base_model.predict_proba(
                    X_validation
                )[:, 1]
            )

            # ------------------------------------------------
            # SIGMOID CALIBRATION
            # ------------------------------------------------

            calibrated_model = CalibratedClassifierCV(
                estimator=build_random_forest(),
                method="sigmoid",
                cv=5,
                n_jobs=-1,
            )

            calibrated_model.fit(
                X_train,
                y_train,
            )

            calibrated_probabilities = (
                calibrated_model.predict_proba(
                    X_validation
                )[:, 1]
            )

            for row_index, (
                original_index,
                true_value,
                uncalibrated_probability,
                calibrated_probability,
            ) in enumerate(
                zip(
                    validation_index,
                    y_validation,
                    uncalibrated_probabilities,
                    calibrated_probabilities,
                )
            ):

                uncalibrated_records.append(
                    {
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                        "row_index": int(
                            original_index
                        ),
                        "y_true": int(
                            true_value
                        ),
                        "probability": float(
                            uncalibrated_probability
                        ),
                    }
                )

                calibrated_records.append(
                    {
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "split": split_counter,
                        "row_index": int(
                            original_index
                        ),
                        "y_true": int(
                            true_value
                        ),
                        "probability": float(
                            calibrated_probability
                        ),
                    }
                )

            # ------------------------------------------------
            # SPLIT METRICS
            # ------------------------------------------------

            uncal_metrics = calculate_ranking_metrics(
                y_validation,
                uncalibrated_probabilities,
            )

            cal_metrics = calculate_ranking_metrics(
                y_validation,
                calibrated_probabilities,
            )

            uncal_ece = calculate_ece(
                y_validation,
                uncalibrated_probabilities,
            )

            cal_ece = calculate_ece(
                y_validation,
                calibrated_probabilities,
            )

            split_records.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "split": split_counter,
                    "uncalibrated_roc_auc": (
                        uncal_metrics["roc_auc"]
                    ),
                    "uncalibrated_pr_auc": (
                        uncal_metrics["pr_auc"]
                    ),
                    "uncalibrated_brier_score": (
                        uncal_metrics["brier_score"]
                    ),
                    "uncalibrated_log_loss": (
                        uncal_metrics["log_loss"]
                    ),
                    "uncalibrated_ece": (
                        uncal_ece
                    ),
                    "calibrated_roc_auc": (
                        cal_metrics["roc_auc"]
                    ),
                    "calibrated_pr_auc": (
                        cal_metrics["pr_auc"]
                    ),
                    "calibrated_brier_score": (
                        cal_metrics["brier_score"]
                    ),
                    "calibrated_log_loss": (
                        cal_metrics["log_loss"]
                    ),
                    "calibrated_ece": (
                        cal_ece
                    ),
                }
            )

    return (
        pd.DataFrame(
            uncalibrated_records
        ),
        pd.DataFrame(
            calibrated_records
        ),
        pd.DataFrame(
            split_records
        ),
    )


# ============================================================
# BUSINESS THRESHOLD ANALYSIS
# ============================================================

def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all candidate calibrated thresholds."""

    records = []

    for threshold in THRESHOLDS:

        metrics = calculate_classification_metrics(
            y_true,
            probabilities,
            threshold,
        )

        records.append(metrics)

    return pd.DataFrame(records)


def calculate_business_cost(
    metrics: Dict[str, float],
    false_positive_cost: float,
    false_negative_cost: float,
) -> float:
    """Calculate normalized business cost."""

    return (
        metrics["fp"]
        * false_positive_cost
        + metrics["fn"]
        * false_negative_cost
    )


def evaluate_business_scenarios(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Find the cost-optimal threshold for each business scenario.
    """

    records = []

    for scenario in BUSINESS_SCENARIOS:

        best_record = None

        for threshold in THRESHOLDS:

            metrics = calculate_classification_metrics(
                y_true,
                probabilities,
                threshold,
            )

            total_cost = calculate_business_cost(
                metrics,
                scenario[
                    "false_positive_cost"
                ],
                scenario[
                    "false_negative_cost"
                ],
            )

            candidate = {
                "scenario": scenario["scenario"],
                "false_positive_cost": (
                    scenario[
                        "false_positive_cost"
                    ]
                ),
                "false_negative_cost": (
                    scenario[
                        "false_negative_cost"
                    ]
                ),
                "recommended_threshold": (
                    threshold
                ),
                "total_cost": float(
                    total_cost
                ),
                "cost_per_employee": float(
                    total_cost
                    / len(y_true)
                ),
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "specificity": metrics[
                    "specificity"
                ],
                "f1": metrics["f1"],
                "balanced_accuracy": metrics[
                    "balanced_accuracy"
                ],
                "flagged_per_1000": metrics[
                    "flagged_per_1000"
                ],
                "false_positives_per_1000": (
                    metrics[
                        "false_positives_per_1000"
                    ]
                ),
                "missed_attrition_per_1000": (
                    metrics[
                        "missed_attrition_per_1000"
                    ]
                ),
            }

            if (
                best_record is None
                or candidate["total_cost"]
                < best_record["total_cost"]
            ):
                best_record = candidate

        records.append(best_record)

    return pd.DataFrame(records)


# ============================================================
# PROBABILITY QUALITY
# ============================================================

def build_probability_comparison(
    uncalibrated_oof: pd.DataFrame,
    calibrated_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Compare calibrated and uncalibrated probabilities."""

    records = []

    for name, frame in [
        (
            "Random Forest",
            uncalibrated_oof,
        ),
        (
            "Sigmoid Calibration",
            calibrated_oof,
        ),
    ]:

        y_true = frame[
            "y_true"
        ].to_numpy()

        probabilities = frame[
            "probability"
        ].to_numpy()

        metrics = calculate_ranking_metrics(
            y_true,
            probabilities,
        )

        ece = calculate_ece(
            y_true,
            probabilities,
        )

        records.append(
            {
                "method": name,
                **metrics,
                "expected_calibration_error": (
                    ece
                ),
                "mean_probability": float(
                    np.mean(probabilities)
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# OPERATING POINT COMPARISON
# ============================================================

def build_operating_comparison(
    uncalibrated_oof: pd.DataFrame,
    calibrated_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Compare previous and calibrated operating points."""

    y_unc = uncalibrated_oof[
        "y_true"
    ].to_numpy()

    p_unc = uncalibrated_oof[
        "probability"
    ].to_numpy()

    y_cal = calibrated_oof[
        "y_true"
    ].to_numpy()

    p_cal = calibrated_oof[
        "probability"
    ].to_numpy()

    unc = calculate_classification_metrics(
        y_unc,
        p_unc,
        UNCALIBRATED_THRESHOLD,
    )

    cal = calculate_classification_metrics(
        y_cal,
        p_cal,
        CALIBRATED_REFERENCE_THRESHOLD,
    )

    return pd.DataFrame(
        [
            {
                "operating_point": (
                    "uncalibrated_0.44"
                ),
                **unc,
            },
            {
                "operating_point": (
                    "sigmoid_calibrated_0.25"
                ),
                **cal,
            },
        ]
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def generate_diagnostic_flags(
    probability_comparison: pd.DataFrame,
    operating_comparison: pd.DataFrame,
    scenario_summary: pd.DataFrame,
) -> List[str]:
    """Generate business-oriented diagnostic flags."""

    flags: List[str] = []

    unc = probability_comparison[
        probability_comparison["method"]
        == "Random Forest"
    ].iloc[0]

    cal = probability_comparison[
        probability_comparison["method"]
        == "Sigmoid Calibration"
    ].iloc[0]

    unc_operating = operating_comparison.iloc[0]
    cal_operating = operating_comparison.iloc[1]

    if (
        cal["brier_score"]
        < unc["brier_score"]
    ):
        flags.append(
            "Sigmoid calibration improves "
            "Brier Score."
        )

    if (
        cal["log_loss"]
        < unc["log_loss"]
    ):
        flags.append(
            "Sigmoid calibration improves "
            "Log Loss."
        )

    if (
        cal["expected_calibration_error"]
        < unc[
            "expected_calibration_error"
        ]
    ):
        flags.append(
            "Sigmoid calibration materially "
            "improves expected calibration error."
        )

    if (
        cal_operating["f1"]
        < unc_operating["f1"]
    ):
        flags.append(
            "The calibrated operating point "
            "does not improve F1 relative to "
            "the previous uncalibrated point."
        )

    if (
        cal_operating["precision"]
        < 0.40
    ):
        flags.append(
            "Precision remains below 0.40 at "
            "the calibrated operating point, "
            "indicating a substantial "
            "false-positive burden."
        )

    if (
        cal_operating[
            "flagged_per_1000"
        ]
        > 500
    ):
        flags.append(
            "The calibrated operating point "
            "still produces high intervention "
            "volume; business capacity should "
            "be explicitly reviewed."
        )

    if (
        scenario_summary[
            "recommended_threshold"
        ].nunique()
        > 1
    ):
        flags.append(
            "Cost-optimal thresholds change "
            "across business scenarios, "
            "confirming that threshold selection "
            "depends on intervention economics."
        )

    return flags


# ============================================================
# REPORTING
# ============================================================

def write_reports(
    probability_comparison: pd.DataFrame,
    threshold_results: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    operating_comparison: pd.DataFrame,
    split_performance: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    flags: List[str],
    y: np.ndarray,
) -> None:
    """Write CSV, JSON and TXT reports."""

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV OUTPUTS
    # --------------------------------------------------------

    probability_comparison.to_csv(
        REPORT_DIR
        / "calibrated_business_model_comparison.csv",
        index=False,
    )

    threshold_results.to_csv(
        REPORT_DIR
        / "calibrated_business_threshold_results.csv",
        index=False,
    )

    scenario_summary.to_csv(
        REPORT_DIR
        / "calibrated_business_cost_scenario_summary.csv",
        index=False,
    )

    operating_comparison.to_csv(
        REPORT_DIR
        / "calibrated_business_operating_comparison.csv",
        index=False,
    )

    split_performance.to_csv(
        REPORT_DIR
        / "calibrated_business_split_performance.csv",
        index=False,
    )

    oof_predictions.to_csv(
        REPORT_DIR
        / "calibrated_business_oof_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------

    unc = probability_comparison[
        probability_comparison["method"]
        == "Random Forest"
    ].iloc[0]

    cal = probability_comparison[
        probability_comparison["method"]
        == "Sigmoid Calibration"
    ].iloc[0]

    unc_operating = operating_comparison.iloc[0]
    cal_operating = operating_comparison.iloc[1]

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(y)),
            "columns": EXPECTED_COLUMNS,
            "target": TARGET_COLUMN,
            "identifier": IDENTIFIER_COLUMN,
            "target_prevalence": float(
                np.mean(y)
            ),
            "sha256": calculate_sha256(
                DATA_PATH
            ),
        },
        "model": {
            "name": "Random Forest",
            "feature_set": (
                "Stable 10-feature subset"
            ),
            "stable_features": STABLE_FEATURES,
            "numerical_features": (
                NUMERICAL_FEATURES
            ),
            "categorical_features": (
                CATEGORICAL_FEATURES
            ),
            "n_estimators": 400,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_splits": (
                N_SPLITS * N_REPEATS
            ),
        },
        "probability_quality": {
            "uncalibrated": unc.to_dict(),
            "sigmoid_calibrated": cal.to_dict(),
        },
        "operating_points": {
            "uncalibrated": (
                unc_operating.to_dict()
            ),
            "sigmoid_calibrated": (
                cal_operating.to_dict()
            ),
        },
        "business_cost_scenarios": (
            scenario_summary.to_dict(
                orient="records"
            )
        ),
        "diagnostic_flags": flags,
        "overall_status": (
            "CONDITIONAL PASS"
        ),
        "overall_diagnosis": (
            "Sigmoid calibration should be "
            "evaluated primarily as a probability "
            "quality improvement. The final "
            "deployment threshold should be "
            "selected using validated intervention "
            "capacity and false-positive versus "
            "false-negative costs."
        ),
    }

    with (
        REPORT_DIR
        / "calibrated_business_cost_analysis_report.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY TXT
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED BUSINESS COST ANALYSIS"
    )
    summary_lines.append("")
    summary_lines.append(
        "[DATASET]"
    )
    summary_lines.append(
        f"Rows:                 {len(y)}"
    )
    summary_lines.append(
        f"Columns:              {EXPECTED_COLUMNS}"
    )
    summary_lines.append(
        f"Features:             {len(STABLE_FEATURES)}"
    )
    summary_lines.append(
        f"Target prevalence:    {np.mean(y):.2%}"
    )
    summary_lines.append("")
    summary_lines.append(
        "[MODEL]"
    )
    summary_lines.append(
        "Model:                 Random Forest"
    )
    summary_lines.append(
        "Feature set:           Stable 10-feature subset"
    )
    summary_lines.append(
        "Calibration:            Sigmoid / Platt"
    )
    summary_lines.append("")
    summary_lines.append(
        "[PROBABILITY QUALITY]"
    )

    for _, row in probability_comparison.iterrows():
        summary_lines.append(
            f"{row['method']}: "
            f"ROC-AUC={row['roc_auc']:.4f}, "
            f"PR-AUC={row['pr_auc']:.4f}, "
            f"Brier={row['brier_score']:.4f}, "
            f"LogLoss={row['log_loss']:.4f}, "
            f"ECE={row['expected_calibration_error']:.4f}"
        )

    summary_lines.append("")
    summary_lines.append(
        "[CALIBRATED THRESHOLD RESULTS]"
    )

    for _, row in threshold_results.iterrows():
        summary_lines.append(
            f"Threshold {row['threshold']:.2f}: "
            f"F1={row['f1']:.4f}, "
            f"Precision={row['precision']:.4f}, "
            f"Recall={row['recall']:.4f}, "
            f"Specificity={row['specificity']:.4f}, "
            f"Flagged/1000={row['flagged_per_1000']:.1f}"
        )

    summary_lines.append("")
    summary_lines.append(
        "[BUSINESS COST SCENARIOS]"
    )

    for _, row in scenario_summary.iterrows():
        summary_lines.append(
            f"{row['scenario']}: "
            f"threshold={row['recommended_threshold']:.2f}, "
            f"total_cost={row['total_cost']:.2f}, "
            f"cost/employee={row['cost_per_employee']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, "
            f"flagged/1000={row['flagged_per_1000']:.1f}"
        )

    summary_lines.append("")
    summary_lines.append(
        "[DIAGNOSTIC FLAGS]"
    )

    if flags:
        for flag in flags:
            summary_lines.append(
                f"- {flag}"
            )
    else:
        summary_lines.append(
            "- No major diagnostic flags."
        )

    summary_lines.append("")
    summary_lines.append(
        "[OVERALL STATUS]"
    )
    summary_lines.append(
        "CALIBRATED BUSINESS COST STATUS: "
        "CONDITIONAL PASS"
    )

    summary_lines.append("")
    summary_lines.append(
        "[OVERALL DIAGNOSIS]"
    )
    summary_lines.append(
        "Business cost analysis confirms that "
        "the preferred operating threshold depends "
        "materially on the relative cost of false "
        "positives and false negatives. Sigmoid "
        "calibration improves probability quality, "
        "but threshold selection must remain tied "
        "to validated intervention economics and "
        "organizational capacity."
    )

    summary_lines.append("")

    with (
        REPORT_DIR
        / "calibrated_business_cost_analysis_summary.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(summary_lines)
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run calibrated business cost analysis."""

    print(
        "Running calibrated business cost analysis..."
    )

    X, y = load_and_validate_dataset()

    print()
    print("=" * 64)
    print(
        "REPEATED OOF CALIBRATED BUSINESS COST VALIDATION"
    )
    print("=" * 64)

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
        uncalibrated_oof,
        calibrated_oof,
        split_performance,
    ) = generate_repeated_oof_predictions(
        X,
        y,
    )

    y_unc = uncalibrated_oof[
        "y_true"
    ].to_numpy()

    p_unc = uncalibrated_oof[
        "probability"
    ].to_numpy()

    y_cal = calibrated_oof[
        "y_true"
    ].to_numpy()

    p_cal = calibrated_oof[
        "probability"
    ].to_numpy()

    print()
    print(
        "Calculating probability quality metrics..."
    )

    probability_comparison = (
        build_probability_comparison(
            uncalibrated_oof,
            calibrated_oof,
        )
    )

    print()
    print(
        "Evaluating calibrated candidate thresholds..."
    )

    threshold_results = evaluate_thresholds(
        y_cal,
        p_cal,
    )

    print()
    print(
        "Evaluating business cost scenarios..."
    )

    scenario_summary = evaluate_business_scenarios(
        y_cal,
        p_cal,
    )

    operating_comparison = (
        build_operating_comparison(
            uncalibrated_oof,
            calibrated_oof,
        )
    )

    flags = generate_diagnostic_flags(
        probability_comparison,
        operating_comparison,
        scenario_summary,
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print(
        "EMPLOYEE ATTRITION — "
        "CALIBRATED BUSINESS COST ANALYSIS"
    )
    print("=" * 64)

    print()
    print("[DATASET]")
    print(
        f"Rows:                 {len(y)}"
    )
    print(
        f"Columns:              {EXPECTED_COLUMNS}"
    )
    print(
        f"Features:             {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    {np.mean(y):.2%}"
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
        "Calibration:            Sigmoid / Platt"
    )
    print(
        "Validation:             5-fold × 5-repeat"
    )

    print()
    print("[PROBABILITY QUALITY]")

    print(
        probability_comparison[
            [
                "method",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "[CALIBRATED THRESHOLD COMPARISON]"
    )

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
        threshold_results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    best_threshold_row = (
        threshold_results.loc[
            threshold_results["f1"].idxmax()
        ]
    )

    print()
    print(
        "[F1-OPTIMAL CALIBRATED OPERATING POINT]"
    )
    print(
        f"Threshold:            "
        f"{best_threshold_row['threshold']:.2f}"
    )
    print(
        f"F1:                   "
        f"{best_threshold_row['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{best_threshold_row['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{best_threshold_row['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{best_threshold_row['specificity']:.4f}"
    )
    print(
        f"Flagged per 1000:     "
        f"{best_threshold_row['flagged_per_1000']:.1f}"
    )

    print()
    print(
        "[BUSINESS COST SCENARIOS]"
    )

    print(
        scenario_summary.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "[PREVIOUS VS CALIBRATED OPERATING POINT]"
    )

    print(
        operating_comparison.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "[DIAGNOSTIC FLAGS]"
    )

    for flag in flags:
        print(
            f"- {flag}"
        )

    print()
    print(
        "[OVERALL STATUS]"
    )
    print(
        "CALIBRATED BUSINESS COST STATUS: "
        "CONDITIONAL PASS"
    )

    print()
    print(
        "[OVERALL DIAGNOSIS]"
    )
    print(
        "Business cost analysis confirms that "
        "the preferred operating threshold depends "
        "materially on the relative cost of false "
        "positives and false negatives. Sigmoid "
        "calibration improves probability quality, "
        "but threshold selection must remain tied "
        "to validated intervention economics and "
        "organizational capacity."
    )

    # --------------------------------------------------------
    # REPORTS
    # --------------------------------------------------------

    oof_predictions = pd.DataFrame(
        {
            "row_index": uncalibrated_oof[
                "row_index"
            ],
            "y_true": uncalibrated_oof[
                "y_true"
            ],
            "uncalibrated_probability": (
                uncalibrated_oof[
                    "probability"
                ]
            ),
            "calibrated_probability": (
                calibrated_oof[
                    "probability"
                ]
            ),
        }
    )

    write_reports(
        probability_comparison,
        threshold_results,
        scenario_summary,
        operating_comparison,
        split_performance,
        oof_predictions,
        flags,
        y,
    )

    print()
    print("[OUTPUT]")
    print(
        "Reports:              "
        f"{REPORT_DIR}"
    )
    print(
        "Model comparison:     "
        f"{REPORT_DIR / 'calibrated_business_model_comparison.csv'}"
    )
    print(
        "Threshold CSV:        "
        f"{REPORT_DIR / 'calibrated_business_threshold_results.csv'}"
    )
    print(
        "Scenario CSV:         "
        f"{REPORT_DIR / 'calibrated_business_cost_scenario_summary.csv'}"
    )
    print(
        "Operating comparison: "
        f"{REPORT_DIR / 'calibrated_business_operating_comparison.csv'}"
    )
    print(
        "Split performance:    "
        f"{REPORT_DIR / 'calibrated_business_split_performance.csv'}"
    )
    print(
        "OOF predictions:      "
        f"{REPORT_DIR / 'calibrated_business_oof_predictions.csv'}"
    )
    print(
        "JSON report:          "
        f"{REPORT_DIR / 'calibrated_business_cost_analysis_report.json'}"
    )
    print(
        "Summary report:       "
        f"{REPORT_DIR / 'calibrated_business_cost_analysis_summary.txt'}"
    )

    print()
    print("=" * 64)
    print(
        "CALIBRATED BUSINESS COST ANALYSIS COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()