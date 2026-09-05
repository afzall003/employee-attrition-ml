from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.engineering import (
    build_preprocessor,
    prepare_model_data,
)


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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "model_benchmark"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CV_SPLITS = 5

CLASSIFICATION_THRESHOLD = 0.50

TARGET_COLUMN = "Attrition"

ID_COLUMN = "Employee_ID"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required by the benchmark."""

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the V2 employee attrition dataset.

    The project's engineering helper already knows how to
    separate predictors and target, so we intentionally call:

        prepare_model_data(df)

    and do not pass target_column or any other argument.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    X, y = prepare_model_data(
        df
    )

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X = X.reset_index(
        drop=True
    )

    y = y.reset_index(
        drop=True
    )

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if y.nunique() != 2:
        raise ValueError(
            "The target must contain exactly two classes."
        )

    return X, y


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_pipeline(
    estimator,
) -> Pipeline:
    """
    Build preprocessing + classifier pipeline.

    IMPORTANT:
    build_preprocessor() takes ZERO arguments in the current
    project architecture.

    Therefore this must remain:

        build_preprocessor()

    and NOT:

        build_preprocessor(X)
    """

    preprocessor = build_preprocessor()

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                estimator,
            ),
        ]
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models() -> dict[str, object]:
    """
    Return the models used in the benchmark.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=RANDOM_STATE,
        ),
    }

    return models


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, float]:
    """
    Evaluate a model using stratified cross-validation.

    Preprocessing is fitted independently inside every fold.

    ROC-AUC and PR-AUC are threshold-independent.

    F1, precision, recall, accuracy, and predicted-positive
    rate use CLASSIFICATION_THRESHOLD.
    """

    roc_auc_scores = []

    pr_auc_scores = []

    f1_scores = []

    precision_scores = []

    recall_scores = []

    accuracy_scores = []

    predicted_positive_rates = []

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        f"Evaluating: {model_name}"
    )

    print(
        "------------------------------------------------------------"
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"  {model_name}: "
            f"fold {fold_number}/{CV_SPLITS}"
        )

        X_train = X.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        # ----------------------------------------------------
        # Build a fresh pipeline for this fold.
        # ----------------------------------------------------

        pipeline = build_pipeline(
            estimator
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Probability predictions
        # ----------------------------------------------------

        probabilities = (
            pipeline
            .predict_proba(
                X_validation
            )[:, 1]
        )

        # ----------------------------------------------------
        # Threshold predictions
        # ----------------------------------------------------

        predictions = (
            probabilities
            >= CLASSIFICATION_THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        roc_auc_scores.append(
            float(roc_auc)
        )

        # ----------------------------------------------------
        # PR-AUC
        # ----------------------------------------------------

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        pr_auc_scores.append(
            float(pr_auc)
        )

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1_scores.append(
            float(f1)
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        precision_scores.append(
            float(precision)
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall_scores.append(
            float(recall)
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        accuracy_scores.append(
            float(accuracy)
        )

        # ----------------------------------------------------
        # Predicted positive rate
        # ----------------------------------------------------

        predicted_positive_rate = (
            predictions.mean()
        )

        predicted_positive_rates.append(
            float(
                predicted_positive_rate
            )
        )

    # ========================================================
    # Aggregate fold results
    # ========================================================

    return {
        "model": model_name,

        "roc_auc_mean": float(
            np.mean(
                roc_auc_scores
            )
        ),

        "roc_auc_std": float(
            np.std(
                roc_auc_scores,
                ddof=1,
            )
        ),

        "pr_auc_mean": float(
            np.mean(
                pr_auc_scores
            )
        ),

        "pr_auc_std": float(
            np.std(
                pr_auc_scores,
                ddof=1,
            )
        ),

        "f1_mean": float(
            np.mean(
                f1_scores
            )
        ),

        "precision_mean": float(
            np.mean(
                precision_scores
            )
        ),

        "recall_mean": float(
            np.mean(
                recall_scores
            )
        ),

        "accuracy_mean": float(
            np.mean(
                accuracy_scores
            )
        ),

        "predicted_positive_rate_mean": float(
            np.mean(
                predicted_positive_rates
            )
            * 100.0
        ),

        "folds": int(
            len(roc_auc_scores)
        ),
    }


# ============================================================
# MODEL RANKING
# ============================================================

def rank_models(
    results: list[dict[str, float]],
) -> pd.DataFrame:
    """
    Rank models by:

    1. ROC-AUC descending
    2. PR-AUC descending
    """

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by=[
            "roc_auc_mean",
            "pr_auc_mean",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    result_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result_df) + 1,
        ),
    )

    return result_df


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    result_df: pd.DataFrame,
    metadata: dict[str, object],
) -> None:
    """
    Save benchmark results in CSV and JSON formats.
    """

    comparison_path = (
        REPORTS_DIR
        / "model_comparison.csv"
    )

    summary_path = (
        REPORTS_DIR
        / "model_benchmark_summary.json"
    )

    metrics_path = (
        METRICS_DIR
        / "model_benchmark.csv"
    )

    # --------------------------------------------------------
    # Detailed comparison
    # --------------------------------------------------------

    result_df.to_csv(
        comparison_path,
        index=False,
    )

    # --------------------------------------------------------
    # General metrics copy
    # --------------------------------------------------------

    result_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best_candidate = (
        result_df
        .iloc[0]
        .to_dict()
    )

    summary = {
        "metadata": metadata,
        "best_candidate": best_candidate,
        "models": result_df.to_dict(
            orient="records"
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_report(
    X: pd.DataFrame,
    y: pd.Series,
    result_df: pd.DataFrame,
) -> None:
    """
    Print final benchmark results.
    """

    target_prevalence = float(
        y.mean()
    )

    print()
    print(
        "============================================================"
    )

    print(
        "EMPLOYEE ATTRITION — MODEL BENCHMARK"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print(
        "[MODEL COMPARISON]"
    )

    display_df = result_df[
        [
            "rank",
            "model",
            "roc_auc_mean",
            "roc_auc_std",
            "pr_auc_mean",
            "pr_auc_std",
            "f1_mean",
            "precision_mean",
            "recall_mean",
            "accuracy_mean",
            "predicted_positive_rate_mean",
        ]
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    best = result_df.iloc[0]

    print()
    print(
        "[BEST CANDIDATE]"
    )

    print(
        f"Model:                "
        f"{best['model']}"
    )

    print(
        f"ROC-AUC:              "
        f"{best['roc_auc_mean']:.4f} "
        f"+/- {best['roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{best['pr_auc_mean']:.4f} "
        f"+/- {best['pr_auc_std']:.4f}"
    )

    print(
        f"F1:                   "
        f"{best['f1_mean']:.4f}"
    )

    print(
        f"Precision:            "
        f"{best['precision_mean']:.4f}"
    )

    print(
        f"Recall:               "
        f"{best['recall_mean']:.4f}"
    )

    print(
        f"Accuracy:             "
        f"{best['accuracy_mean']:.4f}"
    )

    print(
        f"Predicted positive:   "
        f"{best['predicted_positive_rate_mean']:.1f}%"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print(
        "[INTERPRETATION]"
    )

    print(
        "Models are ranked primarily by "
        "cross-validated ROC-AUC and "
        "secondarily by PR-AUC."
    )

    print(
        f"The {CLASSIFICATION_THRESHOLD:.2f} "
        "classification threshold is used "
        "only for threshold-dependent metrics."
    )

    print(
        "It is NOT being selected as the "
        "final production threshold."
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print(
        "[DATASET]"
    )

    print(
        f"Rows:                 "
        f"{len(X)}"
    )

    print(
        f"Features:             "
        f"{X.shape[1]}"
    )

    print(
        f"Target prevalence:    "
        f"{target_prevalence * 100:.2f}%"
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "[OUTPUT]"
    )

    print(
        f"Reports:              "
        f"{REPORTS_DIR}"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "MODEL BENCHMARK COMPLETE"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete model benchmark.
    """

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y = load_dataset()

    target_prevalence = float(
        y.mean()
    )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 "
        f"{len(X)}"
    )

    print(
        f"Target prevalence:    "
        f"{target_prevalence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Benchmark configuration
    # --------------------------------------------------------

    print()
    print(
        "Starting model benchmark..."
    )

    print(
        f"Cross-validation folds: "
        f"{CV_SPLITS}"
    )

    print(
        f"Classification threshold: "
        f"{CLASSIFICATION_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = get_models()

    # --------------------------------------------------------
    # Evaluate models
    # --------------------------------------------------------

    results = []

    for model_name, estimator in models.items():

        result = evaluate_model(
            model_name=model_name,
            estimator=estimator,
            X=X,
            y=y,
            cv=cv,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Rank models
    # --------------------------------------------------------

    result_df = rank_models(
        results
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "dataset": str(
            DATA_PATH
        ),

        "rows": int(
            len(X)
        ),

        "features": int(
            X.shape[1]
        ),

        "target_column": (
            TARGET_COLUMN
        ),

        "target_prevalence": (
            target_prevalence
        ),

        "cv_type": (
            "StratifiedKFold"
        ),

        "cv_folds": (
            CV_SPLITS
        ),

        "shuffle": True,

        "random_state": (
            RANDOM_STATE
        ),

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_reports(
        result_df=result_df,
        metadata=metadata,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_report(
        X=X,
        y=y,
        result_df=result_df,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()