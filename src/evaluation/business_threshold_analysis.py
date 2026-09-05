"""
Business Threshold Analysis
===========================

Evaluates practical operating thresholds for the final stable-feature
employee attrition model.

Pipeline:
    canonical dataset
        -> stable 10-feature subset
        -> optimized Random Forest
        -> repeated stratified 5-fold CV x 5 repeats
        -> out-of-fold probabilities
        -> threshold comparison
        -> business-oriented operating-point analysis

Run:
    python -m src.evaluation.business_threshold_analysis
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "business_threshold_analysis"
)


# ============================================================================
# DATASET SPECIFICATION
# ============================================================================

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

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


# ============================================================================
# VALIDATION SPECIFICATION
# ============================================================================

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42


# ============================================================================
# THRESHOLDS
# ============================================================================

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


# ============================================================================
# MODEL
# ============================================================================

def build_model(
    numerical_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build the optimized Random Forest pipeline established during
    stable-feature model optimization.
    """

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

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
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


# ============================================================================
# DATA LOADING / VALIDATION
# ============================================================================

def load_dataset() -> pd.DataFrame:
    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    print("\nValidating canonical dataset...")

    required_columns = (
        [IDENTIFIER_COLUMN, TARGET_COLUMN]
        + STABLE_FEATURES
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns missing from canonical dataset: "
            + ", ".join(missing_columns)
        )

    if len(df) != 1000:
        raise ValueError(
            f"Expected 1000 rows, found {len(df)}."
        )

    if len(df.columns) != 26:
        raise ValueError(
            f"Expected 26 columns, found {len(df.columns)}."
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .astype(str)
        .unique()
    )

    allowed_values = {"Yes", "No"}

    if not target_values.issubset(allowed_values):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    if df[IDENTIFIER_COLUMN].duplicated().any():
        raise ValueError(
            "Employee_ID contains duplicate identifiers."
        )

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "Canonical dataset contains missing cells."
        )

    print("PASS file_exists")
    print("PASS expected_rows")
    print("PASS expected_columns")
    print("PASS target_exists")
    print("PASS stable_features_exist")
    print("PASS target_values_valid")
    print("PASS no_missing_cells")
    print("PASS identifier_unique")


# ============================================================================
# TARGET PREPARATION
# ============================================================================

def prepare_target(df: pd.DataFrame) -> np.ndarray:
    return (
        df[TARGET_COLUMN]
        .astype(str)
        .map({"No": 0, "Yes": 1})
        .astype(int)
        .to_numpy()
    )


# ============================================================================
# FEATURE TYPE DETECTION
# ============================================================================

def get_feature_types(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    numerical_features = [
        column
        for column in STABLE_FEATURES
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_features = [
        column
        for column in STABLE_FEATURES
        if column not in numerical_features
    ]

    return numerical_features, categorical_features


# ============================================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================================

def generate_oof_predictions(
    X: pd.DataFrame,
    y: np.ndarray,
    numerical_features: List[str],
    categorical_features: List[str],
) -> Tuple[np.ndarray, np.ndarray]:

    print("\n" + "=" * 64)
    print("REPEATED OUT-OF-FOLD BUSINESS VALIDATION")
    print("=" * 64)

    print("Generating repeated out-of-fold predictions...")
    print(f"Folds per repeat:      {N_SPLITS}")
    print(f"Repeats:               {N_REPEATS}")
    print(f"Total validation:      {N_SPLITS * N_REPEATS}")

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    predictions: List[np.ndarray] = []
    actuals: List[np.ndarray] = []

    split_number = 0

    for train_idx, validation_idx in splitter.split(X, y):

        split_number += 1

        print(
            f"Validation split "
            f"{split_number}/{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y[train_idx]
        y_validation = y[validation_idx]

        model = build_model(
            numerical_features,
            categorical_features,
        )

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        predictions.extend(probabilities.tolist())
        actuals.extend(y_validation.tolist())

    return (
        np.asarray(actuals),
        np.asarray(predictions),
    )


# ============================================================================
# METRICS
# ============================================================================

def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, float]:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(y_true)

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

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    predicted_positive_rate = (
        predictions.mean()
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    false_negative_rate = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "predicted_positive_rate": predicted_positive_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "predicted_positive_count": int(
            predictions.sum()
        ),
        "observed_positive_count": int(
            y_true.sum()
        ),
        "total_observations": int(total),
    }


# ============================================================================
# THRESHOLD EVALUATION
# ============================================================================

def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:

    rows = []

    for threshold in CANDIDATE_THRESHOLDS:

        metrics = calculate_threshold_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(metrics)

    return pd.DataFrame(rows)


# ============================================================================
# BUSINESS CAPACITY METRICS
# ============================================================================

def add_business_metrics(
    results: pd.DataFrame,
    prevalence: float,
) -> pd.DataFrame:

    results = results.copy()

    results[
        "predicted_positive_vs_prevalence_ratio"
    ] = (
        results["predicted_positive_rate"]
        / prevalence
    )

    results["flagged_per_1000"] = (
        results["predicted_positive_rate"]
        * 1000
    )

    results["false_positives_per_1000"] = (
        results["false_positive"]
        / results["total_observations"]
        * 1000
    )

    results["false_negatives_per_1000"] = (
        results["false_negative"]
        / results["total_observations"]
        * 1000
    )

    results["precision_percent"] = (
        results["precision"] * 100
    )

    results["recall_percent"] = (
        results["recall"] * 100
    )

    results["specificity_percent"] = (
        results["specificity"] * 100
    )

    results["predicted_positive_percent"] = (
        results["predicted_positive_rate"] * 100
    )

    return results


# ============================================================================
# THRESHOLD RECOMMENDATIONS
# ============================================================================

def determine_operating_points(
    results: pd.DataFrame,
) -> Dict[str, Dict]:

    recommendations: Dict[str, Dict] = {}

    # Maximum F1
    f1_row = results.loc[
        results["f1"].idxmax()
    ]

    recommendations["best_f1"] = (
        f1_row.to_dict()
    )

    # Highest recall with at least 30% precision
    precision_30 = results[
        results["precision"] >= 0.30
    ]

    if not precision_30.empty:
        recall_row = precision_30.loc[
            precision_30["recall"].idxmax()
        ]

        recommendations[
            "high_recall_precision_30"
        ] = recall_row.to_dict()

    # Highest precision while maintaining recall >= 0.60
    recall_60 = results[
        results["recall"] >= 0.60
    ]

    if not recall_60.empty:
        precision_row = recall_60.loc[
            recall_60["precision"].idxmax()
        ]

        recommendations[
            "precision_priority_recall_60"
        ] = precision_row.to_dict()

    # Highest specificity while maintaining recall >= 0.50
    recall_50 = results[
        results["recall"] >= 0.50
    ]

    if not recall_50.empty:
        specificity_row = recall_50.loc[
            recall_50["specificity"].idxmax()
        ]

        recommendations[
            "specificity_priority_recall_50"
        ] = specificity_row.to_dict()

    return recommendations


# ============================================================================
# DIAGNOSTIC FLAGS
# ============================================================================

def generate_diagnostic_flags(
    results: pd.DataFrame,
    prevalence: float,
    oof_roc_auc: float,
) -> List[str]:

    flags = []

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    if abs(
        best_f1["threshold"] - 0.50
    ) <= 0.05:
        flags.append(
            "The F1-optimal threshold remains reasonably "
            "close to the default 0.50 operating point."
        )
    else:
        flags.append(
            "The F1-optimal threshold differs materially "
            "from the default 0.50 operating point."
        )

    default_row = results[
        np.isclose(
            results["threshold"],
            0.50,
        )
    ]

    if not default_row.empty:

        default_f1 = float(
            default_row.iloc[0]["f1"]
        )

        f1_gain = (
            best_f1["f1"]
            - default_f1
        )

        if f1_gain >= 0.02:
            flags.append(
                f"Threshold optimization improves F1 "
                f"by {f1_gain:.4f} relative to 0.50."
            )

    if best_f1["predicted_positive_rate"] > (
        prevalence * 2
    ):
        flags.append(
            "The F1-optimal threshold flags more than "
            "twice the observed attrition prevalence; "
            "business capacity should be reviewed."
        )

    if best_f1["precision"] < 0.40:
        flags.append(
            "Precision remains below 0.40 at the "
            "F1-optimal threshold, indicating a substantial "
            "false-positive burden."
        )

    if best_f1["recall"] >= 0.70:
        flags.append(
            "The F1-optimal threshold prioritizes detection "
            "with recall of at least 0.70."
        )

    if best_f1["specificity"] < 0.60:
        flags.append(
            "Specificity is below 0.60 at the selected "
            "F1-optimal threshold."
        )

    if oof_roc_auc >= 0.60:
        flags.append(
            "Out-of-fold ROC-AUC indicates useful ranking "
            "information before thresholding."
        )
    else:
        flags.append(
            "Out-of-fold ROC-AUC remains below 0.60, "
            "limiting confidence in operational ranking."
        )

    return flags


# ============================================================================
# OVERALL DIAGNOSIS
# ============================================================================

def generate_diagnosis(
    results: pd.DataFrame,
    recommendations: Dict[str, Dict],
    oof_roc_auc: float,
) -> str:

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    if oof_roc_auc < 0.60:
        return (
            "The model provides limited ranking separation, "
            "so threshold selection should be treated cautiously. "
            "Additional model improvement or feature investigation "
            "is recommended before operational deployment."
        )

    return (
        f"The threshold analysis identifies {best_f1['threshold']:.2f} "
        f"as the F1-optimal operating point, producing F1="
        f"{best_f1['f1']:.4f}, precision="
        f"{best_f1['precision']:.4f}, and recall="
        f"{best_f1['recall']:.4f}. "
        "However, threshold selection should be based on the "
        "business cost of false positives versus false negatives "
        "rather than F1 alone. The model therefore remains "
        "appropriate for controlled decision-support use, subject "
        "to business review of intervention capacity and error costs."
    )


# ============================================================================
# REPORT WRITING
# ============================================================================

def write_summary(
    df: pd.DataFrame,
    results: pd.DataFrame,
    recommendations: Dict[str, Dict],
    flags: List[str],
    diagnosis: str,
    oof_roc_auc: float,
    oof_pr_auc: float,
    prevalence: float,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        REPORT_DIR
        / "business_threshold_analysis_summary.txt"
    )

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 64
            + "\n"
        )
        file.write(
            "EMPLOYEE ATTRITION — BUSINESS THRESHOLD ANALYSIS\n"
        )
        file.write(
            "=" * 64
            + "\n\n"
        )

        file.write("[DATASET]\n")
        file.write(
            f"Rows:                 {len(df)}\n"
        )
        file.write(
            f"Columns:              {len(df.columns)}\n"
        )
        file.write(
            f"Features:             {len(STABLE_FEATURES)}\n"
        )
        file.write(
            f"Target prevalence:    "
            f"{prevalence * 100:.2f}%\n\n"
        )

        file.write("[MODEL]\n")
        file.write(
            "Model:                 Random Forest\n"
        )
        file.write(
            "Feature set:           Stable 10-feature subset\n"
        )
        file.write(
            "Validation:            5-fold × 5-repeat\n\n"
        )

        file.write("[OUT-OF-FOLD RANKING]\n")
        file.write(
            f"ROC-AUC:              "
            f"{oof_roc_auc:.4f}\n"
        )
        file.write(
            f"PR-AUC:               "
            f"{oof_pr_auc:.4f}\n\n"
        )

        file.write("[THRESHOLD COMPARISON]\n")

        display_columns = [
            "threshold",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "predicted_positive_rate",
            "flagged_per_1000",
            "false_positives_per_1000",
            "false_negatives_per_1000",
        ]

        file.write(
            results[display_columns]
            .to_string(
                index=False,
                float_format=lambda value:
                f"{value:.4f}",
            )
        )

        file.write("\n\n")

        file.write("[F1-OPTIMAL OPERATING POINT]\n")
        file.write(
            f"Threshold:            "
            f"{best_f1['threshold']:.2f}\n"
        )
        file.write(
            f"F1:                   "
            f"{best_f1['f1']:.4f}\n"
        )
        file.write(
            f"Precision:            "
            f"{best_f1['precision']:.4f}\n"
        )
        file.write(
            f"Recall:               "
            f"{best_f1['recall']:.4f}\n"
        )
        file.write(
            f"Specificity:          "
            f"{best_f1['specificity']:.4f}\n"
        )
        file.write(
            f"Balanced Accuracy:    "
            f"{best_f1['balanced_accuracy']:.4f}\n"
        )
        file.write(
            f"Predicted Positive:   "
            f"{best_f1['predicted_positive_rate'] * 100:.2f}%\n"
        )
        file.write(
            f"Flagged per 1000:     "
            f"{best_f1['flagged_per_1000']:.1f}\n\n"
        )

        file.write("[DIAGNOSTIC FLAGS]\n")

        for flag in flags:
            file.write(
                f"- {flag}\n"
            )

        file.write("\n")

        file.write("[OVERALL DIAGNOSIS]\n")
        file.write(
            diagnosis
            + "\n"
        )

        file.write("\n")


def write_json_report(
    df: pd.DataFrame,
    results: pd.DataFrame,
    recommendations: Dict[str, Dict],
    flags: List[str],
    diagnosis: str,
    oof_roc_auc: float,
    oof_pr_auc: float,
    prevalence: float,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "features": STABLE_FEATURES,
            "target": TARGET_COLUMN,
            "target_prevalence": prevalence,
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
            "random_state": RANDOM_STATE,
        },
        "model": {
            "name": "Random Forest",
            "n_estimators": 400,
            "max_depth": None,
            "max_features": "sqrt",
            "min_samples_leaf": 10,
            "class_weight": "balanced",
        },
        "ranking_metrics": {
            "roc_auc": float(oof_roc_auc),
            "pr_auc": float(oof_pr_auc),
        },
        "threshold_results": (
            results.to_dict(orient="records")
        ),
        "operating_points": recommendations,
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    report_path = (
        REPORT_DIR
        / "business_threshold_analysis_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "Running business threshold analysis..."
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------------------
    # Load and validate
    # ----------------------------------------------------------------------

    df = load_dataset()

    validate_dataset(df)

    numerical_features, categorical_features = (
        get_feature_types(df)
    )

    print(
        f"\nStable features:       "
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

    y = prepare_target(df)

    prevalence = float(
        y.mean()
    )

    X = df[STABLE_FEATURES].copy()

    # ----------------------------------------------------------------------
    # Generate OOF predictions
    # ----------------------------------------------------------------------

    y_oof, probabilities_oof = (
        generate_oof_predictions(
            X,
            y,
            numerical_features,
            categorical_features,
        )
    )

    # ----------------------------------------------------------------------
    # Ranking metrics
    # ----------------------------------------------------------------------

    print(
        "\nCalculating ranking metrics..."
    )

    oof_roc_auc = roc_auc_score(
        y_oof,
        probabilities_oof,
    )

    oof_pr_auc = average_precision_score(
        y_oof,
        probabilities_oof,
    )

    print(
        f"ROC-AUC: {oof_roc_auc:.4f}"
    )

    print(
        f"PR-AUC:  {oof_pr_auc:.4f}"
    )

    # ----------------------------------------------------------------------
    # Threshold analysis
    # ----------------------------------------------------------------------

    print(
        "\nEvaluating candidate thresholds..."
    )

    results = evaluate_thresholds(
        y_oof,
        probabilities_oof,
    )

    results = add_business_metrics(
        results,
        prevalence,
    )

    # ----------------------------------------------------------------------
    # Recommendations
    # ----------------------------------------------------------------------

    recommendations = (
        determine_operating_points(
            results
        )
    )

    flags = generate_diagnostic_flags(
        results,
        prevalence,
        oof_roc_auc,
    )

    diagnosis = generate_diagnosis(
        results,
        recommendations,
        oof_roc_auc,
    )

    # ----------------------------------------------------------------------
    # Console report
    # ----------------------------------------------------------------------

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    print("\n" + "=" * 64)
    print(
        "EMPLOYEE ATTRITION — BUSINESS THRESHOLD ANALYSIS"
    )
    print("=" * 64)

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
        f"Target prevalence:    "
        f"{prevalence * 100:.2f}%"
    )

    print("\n[MODEL]")
    print(
        "Model:                 Random Forest"
    )
    print(
        "Feature set:           Stable 10-feature subset"
    )
    print(
        "Validation:            5-fold × 5-repeat"
    )

    print("\n[OUT-OF-FOLD RANKING]")
    print(
        f"ROC-AUC:              "
        f"{oof_roc_auc:.4f}"
    )
    print(
        f"PR-AUC:               "
        f"{oof_pr_auc:.4f}"
    )

    print("\n[THRESHOLD COMPARISON]")

    display_columns = [
        "threshold",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "predicted_positive_percent",
        "flagged_per_1000",
    ]

    display = results[
        display_columns
    ].copy()

    print(
        display.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "f1": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "specificity": "{:.4f}".format,
                "balanced_accuracy": "{:.4f}".format,
                "predicted_positive_percent":
                    "{:.2f}".format,
                "flagged_per_1000":
                    "{:.1f}".format,
            },
        )
    )

    print("\n[F1-OPTIMAL OPERATING POINT]")
    print(
        f"Threshold:            "
        f"{best_f1['threshold']:.2f}"
    )
    print(
        f"F1:                   "
        f"{best_f1['f1']:.4f}"
    )
    print(
        f"Precision:            "
        f"{best_f1['precision']:.4f}"
    )
    print(
        f"Recall:               "
        f"{best_f1['recall']:.4f}"
    )
    print(
        f"Specificity:          "
        f"{best_f1['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:    "
        f"{best_f1['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:   "
        f"{best_f1['predicted_positive_rate'] * 100:.2f}%"
    )
    print(
        f"Flagged per 1000:     "
        f"{best_f1['flagged_per_1000']:.1f}"
    )

    print("\n[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(
            f"- {flag}"
        )

    print("\n[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # ----------------------------------------------------------------------
    # Save reports
    # ----------------------------------------------------------------------

    threshold_csv = (
        REPORT_DIR
        / "business_threshold_results.csv"
    )

    comparison_csv = (
        REPORT_DIR
        / "business_threshold_comparison.csv"
    )

    results.to_csv(
        threshold_csv,
        index=False,
    )

    # Compact comparison containing the most useful columns.
    comparison_columns = [
        "threshold",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "accuracy",
        "predicted_positive_rate",
        "predicted_positive_percent",
        "flagged_per_1000",
        "false_positives_per_1000",
        "false_negatives_per_1000",
        "predicted_positive_vs_prevalence_ratio",
    ]

    results[
        comparison_columns
    ].to_csv(
        comparison_csv,
        index=False,
    )

    write_json_report(
        df=df,
        results=results,
        recommendations=recommendations,
        flags=flags,
        diagnosis=diagnosis,
        oof_roc_auc=oof_roc_auc,
        oof_pr_auc=oof_pr_auc,
        prevalence=prevalence,
    )

    write_summary(
        df=df,
        results=results,
        recommendations=recommendations,
        flags=flags,
        diagnosis=diagnosis,
        oof_roc_auc=oof_roc_auc,
        oof_pr_auc=oof_pr_auc,
        prevalence=prevalence,
    )

    print("\n[OUTPUT]")
    print(
        f"Reports:              "
        f"{REPORT_DIR}"
    )
    print(
        f"Threshold CSV:        "
        f"{threshold_csv}"
    )
    print(
        f"Comparison CSV:       "
        f"{comparison_csv}"
    )
    print(
        f"JSON report:          "
        f"{REPORT_DIR / 'business_threshold_analysis_report.json'}"
    )
    print(
        f"Summary report:       "
        f"{REPORT_DIR / 'business_threshold_analysis_summary.txt'}"
    )

    print("\n" + "=" * 64)
    print(
        "BUSINESS THRESHOLD ANALYSIS COMPLETE"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()