from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
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
    / "final_model_selection"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attrition"
ID_COLUMNS = ["Employee_ID"]

RANDOM_STATE = 42

CV_FOLDS = 5

HOLDOUT_SIZE = 0.20

OPTIMIZED_C = 0.01

CLASS_WEIGHT = None

PRODUCTION_THRESHOLD = 0.15


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Dataset V2 and separate features from target.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    y_raw = df[TARGET_COLUMN].copy()

    X = df.drop(columns=[TARGET_COLUMN]).copy()

    for column in ID_COLUMNS:
        if column in X.columns:
            X = X.drop(columns=[column])

    y = (
        y_raw
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if y.isna().any():
        invalid_values = sorted(
            y_raw[y.isna()].astype(str).unique().tolist()
        )

        raise ValueError(
            "Target contains unsupported values: "
            f"{invalid_values}"
        )

    y = y.astype(int)

    return X, y


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        median imputation
        standard scaling

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

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

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No numerical or categorical features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    X: pd.DataFrame,
) -> Pipeline:
    """
    Build the optimized Logistic Regression model.
    """

    preprocessor = build_preprocessor(X)

    model = LogisticRegression(
        C=OPTIMIZED_C,
        class_weight=CLASS_WEIGHT,
        max_iter=5000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
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

    return pipeline


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Calculate threshold-independent and
    threshold-dependent metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
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
        predictions.mean() * 100
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "predicted_positive_rate": float(
            predicted_positive_rate
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# CROSS-VALIDATED PERFORMANCE
# ============================================================

def cross_validated_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict, np.ndarray]:
    """
    Generate out-of-fold probabilities using
    stratified K-fold cross-validation.

    These probabilities are used only for evaluating
    the selected model before the final holdout check.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_model(X)

    print()
    print(
        "Generating out-of-fold probabilities..."
    )

    probabilities = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=None,
    )[:, 1]

    metrics = calculate_metrics(
        y,
        probabilities,
        PRODUCTION_THRESHOLD,
    )

    return metrics, probabilities


# ============================================================
# HOLDOUT EVALUATION
# ============================================================

def holdout_evaluation(
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
) -> tuple[Pipeline, dict, np.ndarray]:
    """
    Train the final candidate on the training partition
    and evaluate it once on the untouched holdout partition.
    """

    print()
    print(
        "Training final candidate on training partition..."
    )

    pipeline = build_model(X_train)

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_holdout
    )[:, 1]

    metrics = calculate_metrics(
        y_holdout,
        probabilities,
        PRODUCTION_THRESHOLD,
    )

    return (
        pipeline,
        metrics,
        probabilities,
    )


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

def threshold_comparison(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate several operating thresholds.

    Threshold selection remains separate from
    model fitting.
    """

    thresholds = [
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

    rows = []

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "predicted_positive_rate": (
                    metrics[
                        "predicted_positive_rate"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_safe(value):
    """
    Convert NumPy/Pandas values into JSON-safe values.
    """

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.int64,
            np.int32,
        ),
    ):
        return int(value)

    if isinstance(
        value,
        (
            np.floating,
            np.float64,
            np.float32,
        ),
    ):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


# ============================================================
# REPORT GENERATION
# ============================================================

def save_reports(
    X: pd.DataFrame,
    y: pd.Series,
    cv_metrics: dict,
    holdout_metrics: dict,
    threshold_df: pd.DataFrame,
) -> None:
    """
    Save CSV and JSON reports.
    """

    threshold_path = (
        REPORT_DIR
        / "threshold_comparison.csv"
    )

    threshold_df.to_csv(
        threshold_path,
        index=False,
    )

    summary = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target_column": TARGET_COLUMN,
            "positive_class": "Yes",
            "negative_class": "No",
            "target_prevalence": float(
                y.mean()
            ),
        },
        "model": {
            "family": "Logistic Regression",
            "C": OPTIMIZED_C,
            "class_weight": CLASS_WEIGHT,
            "solver": "lbfgs",
            "max_iter": 5000,
        },
        "validation": {
            "cv_folds": CV_FOLDS,
            "holdout_size": HOLDOUT_SIZE,
            "random_state": RANDOM_STATE,
        },
        "production_threshold": (
            PRODUCTION_THRESHOLD
        ),
        "cross_validated_metrics": cv_metrics,
        "holdout_metrics": holdout_metrics,
    }

    summary = make_json_safe(summary)

    summary_path = (
        REPORT_DIR
        / "final_model_selection.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    dataset_summary = pd.DataFrame(
        [
            {
                "rows": len(X),
                "features": X.shape[1],
                "target_prevalence": y.mean(),
            }
        ]
    )

    dataset_summary.to_csv(
        REPORT_DIR
        / "dataset_summary.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print(
        "Loading employee attrition Dataset V2..."
    )

    X, y = load_dataset()

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 {len(X)}"
    )

    print(
        f"Features:             {X.shape[1]}"
    )

    print(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "EMPLOYEE ATTRITION — FINAL MODEL SELECTION"
    )
    print(
        "============================================================"
    )

    print()
    print("[CONFIGURATION]")

    print(
        f"Model:                Logistic Regression"
    )

    print(
        f"C:                    {OPTIMIZED_C}"
    )

    print(
        f"Class weight:         {CLASS_WEIGHT}"
    )

    print(
        f"CV folds:             {CV_FOLDS}"
    )

    print(
        f"Holdout size:         {HOLDOUT_SIZE * 100:.0f}%"
    )

    print(
        f"Production threshold: {PRODUCTION_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # OUT-OF-FOLD EVALUATION
    # --------------------------------------------------------

    cv_metrics, cv_probabilities = (
        cross_validated_evaluation(
            X,
            y,
        )
    )

    # --------------------------------------------------------
    # HOLDOUT SPLIT
    # --------------------------------------------------------

    print()
    print(
        "Creating untouched stratified holdout..."
    )

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

    print(
        f"Training rows:       {len(X_train)}"
    )

    print(
        f"Holdout rows:        {len(X_holdout)}"
    )

    print(
        f"Training prevalence: {y_train.mean() * 100:.2f}%"
    )

    print(
        f"Holdout prevalence:  {y_holdout.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # FINAL HOLDOUT MODEL
    # --------------------------------------------------------

    (
        final_pipeline,
        holdout_metrics,
        holdout_probabilities,
    ) = holdout_evaluation(
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    )

    # --------------------------------------------------------
    # THRESHOLD ANALYSIS ON OOF PROBABILITIES
    # --------------------------------------------------------

    print()
    print(
        "Evaluating operating thresholds..."
    )

    threshold_df = threshold_comparison(
        y,
        cv_probabilities,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    model_path = (
        REPORT_DIR
        / "final_model.joblib"
    )

    joblib.dump(
        final_pipeline,
        model_path,
    )

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    save_reports(
        X=X,
        y=y,
        cv_metrics=cv_metrics,
        holdout_metrics=holdout_metrics,
        threshold_df=threshold_df,
    )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "EMPLOYEE ATTRITION — FINAL MODEL SELECTION"
    )
    print(
        "============================================================"
    )

    print()
    print("[DATASET]")

    print(
        f"Rows:                 {len(X)}"
    )

    print(
        f"Features:             {X.shape[1]}"
    )

    print(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )

    print()
    print("[SELECTED MODEL]")

    print(
        "Model:                Logistic Regression"
    )

    print(
        f"C:                    {OPTIMIZED_C}"
    )

    print(
        f"Class weight:         {CLASS_WEIGHT}"
    )

    print()
    print(
        "[CROSS-VALIDATED PERFORMANCE]"
    )

    print(
        f"ROC-AUC:              "
        f"{cv_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{cv_metrics['pr_auc']:.4f}"
    )

    print(
        f"Precision @ {PRODUCTION_THRESHOLD:.2f}: "
        f"{cv_metrics['precision']:.4f}"
    )

    print(
        f"Recall @ {PRODUCTION_THRESHOLD:.2f}:    "
        f"{cv_metrics['recall']:.4f}"
    )

    print(
        f"F1 @ {PRODUCTION_THRESHOLD:.2f}:        "
        f"{cv_metrics['f1']:.4f}"
    )

    print(
        f"Accuracy @ {PRODUCTION_THRESHOLD:.2f}: "
        f"{cv_metrics['accuracy']:.4f}"
    )

    print(
        f"Predicted positive:   "
        f"{cv_metrics['predicted_positive_rate']:.2f}%"
    )

    print()
    print(
        "[FINAL HOLDOUT PERFORMANCE]"
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
        f"Precision @ {PRODUCTION_THRESHOLD:.2f}: "
        f"{holdout_metrics['precision']:.4f}"
    )

    print(
        f"Recall @ {PRODUCTION_THRESHOLD:.2f}:    "
        f"{holdout_metrics['recall']:.4f}"
    )

    print(
        f"F1 @ {PRODUCTION_THRESHOLD:.2f}:        "
        f"{holdout_metrics['f1']:.4f}"
    )

    print(
        f"Accuracy @ {PRODUCTION_THRESHOLD:.2f}: "
        f"{holdout_metrics['accuracy']:.4f}"
    )

    print(
        f"Predicted positive:   "
        f"{holdout_metrics['predicted_positive_rate']:.2f}%"
    )

    print()
    print("[CONFUSION MATRIX — HOLDOUT]")

    print(
        f"True Negative:        "
        f"{holdout_metrics['true_negative']}"
    )

    print(
        f"False Positive:       "
        f"{holdout_metrics['false_positive']}"
    )

    print(
        f"False Negative:       "
        f"{holdout_metrics['false_negative']}"
    )

    print(
        f"True Positive:        "
        f"{holdout_metrics['true_positive']}"
    )

    print()
    print(
        "[THRESHOLD CANDIDATES — OOF]"
    )

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "predicted_positive_rate",
    ]

    print(
        threshold_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda value:
                f"{value:.4f}",
        )
    )

    best_f1_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    print()
    print(
        "[BEST OOF F1 THRESHOLD]"
    )

    print(
        f"Threshold:            "
        f"{best_f1_row['threshold']:.2f}"
    )

    print(
        f"F1:                   "
        f"{best_f1_row['f1']:.4f}"
    )

    print(
        f"Precision:            "
        f"{best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{best_f1_row['recall']:.4f}"
    )

    print()
    print(
        "[FINAL DECISION]"
    )

    print(
        "Selected model:       Logistic Regression"
    )

    print(
        f"Selected C:           {OPTIMIZED_C}"
    )

    print(
        f"Operating threshold:  {PRODUCTION_THRESHOLD:.2f}"
    )

    print(
        "Threshold source:     "
        "previous controlled threshold analysis"
    )

    print()
    print(
        "[INTERPRETATION]"
    )

    print(
        "The optimized Logistic Regression model is "
        "retained as the final candidate because it "
        "provided the strongest controlled ROC-AUC "
        "during model optimization."
    )

    print(
        "The classification threshold is handled "
        "separately from model fitting."
    )

    print(
        f"The current operating threshold is "
        f"{PRODUCTION_THRESHOLD:.2f}, based on the "
        "previous threshold analysis."
    )

    print(
        "The untouched holdout partition is used only "
        "for final performance verification."
    )

    print()
    print("[OUTPUT]")

    print(
        f"Reports:              {REPORT_DIR}"
    )

    print(
        f"Model artifact:       {model_path}"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "FINAL MODEL SELECTION COMPLETE"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()