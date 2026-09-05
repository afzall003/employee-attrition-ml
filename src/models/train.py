from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
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

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    ),
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required for model artifacts."""

    MODELS_DIR.mkdir(
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

def load_training_data() -> pd.DataFrame:
    """Load the raw dataset used for model development."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The training dataset is empty."
        )

    return df


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_train_test_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create a stratified train/test split.

    The test set is held out and is not used for model
    selection or cross-validation.
    """

    X, y = prepare_model_data(df)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# PIPELINE CONSTRUCTION
# ============================================================

def build_pipeline(
    estimator,
) -> Pipeline:
    """
    Build a complete preprocessing + estimator pipeline.

    Keeping preprocessing inside the pipeline prevents
    information leakage during cross-validation.
    """

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
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

    return pipeline


# ============================================================
# CROSS-VALIDATION
# ============================================================

def perform_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a model using stratified k-fold cross-validation.

    The held-out test set is never used here.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise",
    )

    summary = {}

    for metric_name in scoring:

        scores = results[
            f"test_{metric_name}"
        ]

        summary[
            f"cv_{metric_name}_mean"
        ] = float(
            np.mean(scores)
        )

        summary[
            f"cv_{metric_name}_std"
        ] = float(
            np.std(scores)
        )

    return summary


# ============================================================
# MODEL TRAINING
# ============================================================

def train_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit the complete pipeline on the training data."""

    pipeline.fit(
        X_train,
        y_train,
    )

    return pipeline


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_test_set(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a trained model on the untouched test set.
    """

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    metrics = {
        "test_roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "test_pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "test_accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "test_precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "test_f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    return metrics


# ============================================================
# MODEL COMPARISON
# ============================================================

def run_model_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[
    dict[str, Pipeline],
    pd.DataFrame,
]:
    """
    Run cross-validation for all candidate models.

    Returns:
        trained_pipelines
        comparison dataframe
    """

    trained_pipelines = []

    comparison_records = []

    for model_name, estimator in MODELS.items():

        print(
            f"\nTraining and validating: "
            f"{model_name}"
        )

        pipeline = build_pipeline(
            estimator
        )

        cv_metrics = perform_cross_validation(
            pipeline,
            X_train,
            y_train,
        )

        trained_pipeline = train_model(
            pipeline,
            X_train,
            y_train,
        )

        trained_pipelines.append(
            (
                model_name,
                trained_pipeline,
            )
        )

        comparison_records.append(
            {
                "model": model_name,
                "cv_roc_auc_mean": cv_metrics[
                    "cv_roc_auc_mean"
                ],
                "cv_roc_auc_std": cv_metrics[
                    "cv_roc_auc_std"
                ],
                "cv_pr_auc_mean": cv_metrics[
                    "cv_average_precision_mean"
                ],
                "cv_pr_auc_std": cv_metrics[
                    "cv_average_precision_std"
                ],
                "cv_accuracy_mean": cv_metrics[
                    "cv_accuracy_mean"
                ],
                "cv_precision_mean": cv_metrics[
                    "cv_precision_mean"
                ],
                "cv_recall_mean": cv_metrics[
                    "cv_recall_mean"
                ],
                "cv_f1_mean": cv_metrics[
                    "cv_f1_mean"
                ],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_records
    )

    comparison_df = comparison_df.sort_values(
        by="cv_roc_auc_mean",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return (
        dict(trained_pipelines),
        comparison_df,
    )


# ============================================================
# BEST MODEL SELECTION
# ============================================================

def select_best_model(
    comparison_df: pd.DataFrame,
    trained_pipelines: dict[str, Pipeline],
) -> tuple[
    str,
    Pipeline,
]:
    """
    Select the model with the highest mean CV ROC-AUC.

    The test set is not involved in this decision.
    """

    best_model_name = comparison_df.iloc[
        0
    ]["model"]

    best_pipeline = trained_pipelines[
        best_model_name
    ]

    return (
        best_model_name,
        best_pipeline,
    )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model_name: str,
    pipeline: Pipeline,
) -> Path:
    """Save a trained model pipeline."""

    model_path = (
        MODELS_DIR
        / f"{model_name}.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    return model_path


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

def save_comparison(
    comparison_df: pd.DataFrame,
) -> Path:
    """Save cross-validation model comparison."""

    output_path = (
        METRICS_DIR
        / "model_comparison.csv"
    )

    comparison_df.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# SAVE FINAL METRICS
# ============================================================

def save_final_metrics(
    best_model_name: str,
    test_metrics: dict[str, float],
    comparison_df: pd.DataFrame,
    model_path: Path,
) -> Path:
    """Save final model evaluation metadata."""

    best_row = comparison_df[
        comparison_df["model"]
        == best_model_name
    ].iloc[0]

    metadata = {
        "selected_model": best_model_name,
        "selection_metric": "cv_roc_auc_mean",
        "selection_reason": (
            "Highest mean ROC-AUC across "
            "stratified cross-validation"
        ),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),
        "model_path": str(model_path),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "cross_validation": {
            "roc_auc_mean": float(
                best_row[
                    "cv_roc_auc_mean"
                ]
            ),
            "roc_auc_std": float(
                best_row[
                    "cv_roc_auc_std"
                ]
            ),
            "pr_auc_mean": float(
                best_row[
                    "cv_pr_auc_mean"
                ]
            ),
            "pr_auc_std": float(
                best_row[
                    "cv_pr_auc_std"
                ]
            ),
            "accuracy_mean": float(
                best_row[
                    "cv_accuracy_mean"
                ]
            ),
            "precision_mean": float(
                best_row[
                    "cv_precision_mean"
                ]
            ),
            "recall_mean": float(
                best_row[
                    "cv_recall_mean"
                ]
            ),
            "f1_mean": float(
                best_row[
                    "cv_f1_mean"
                ]
            ),
        },
        "held_out_test": test_metrics,
    }

    output_path = (
        METRICS_DIR
        / "selected_model_metrics.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# TERMINAL REPORT
# ============================================================

def print_model_comparison(
    comparison_df: pd.DataFrame,
) -> None:
    """Print cross-validation comparison."""

    print("\n" + "=" * 60)
    print(
        "CROSS-VALIDATION MODEL COMPARISON"
    )
    print("=" * 60)

    display_df = comparison_df.copy()

    numeric_columns = [
        column
        for column in display_df.columns
        if column != "model"
    ]

    display_df[numeric_columns] = (
        display_df[numeric_columns]
        .round(4)
    )

    print(
        display_df.to_string(
            index=False
        )
    )


def print_final_evaluation(
    best_model_name: str,
    test_metrics: dict[str, float],
    model_path: Path,
    comparison_path: Path,
    metadata_path: Path,
) -> None:
    """Print final test-set evaluation."""

    print("\n" + "=" * 60)
    print(
        "SELECTED MODEL — HELD-OUT TEST EVALUATION"
    )
    print("=" * 60)

    print(
        f"\nSelected model:       "
        f"{best_model_name}"
    )

    print(
        "\n[Test Set Metrics]"
    )

    print(
        f"ROC-AUC:              "
        f"{test_metrics['test_roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{test_metrics['test_pr_auc']:.4f}"
    )

    print(
        f"Accuracy:             "
        f"{test_metrics['test_accuracy']:.4f}"
    )

    print(
        f"Precision:            "
        f"{test_metrics['test_precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{test_metrics['test_recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{test_metrics['test_f1']:.4f}"
    )

    print("\n[ARTIFACTS]")

    print(
        f"Selected model:       "
        f"{model_path}"
    )

    print(
        f"Comparison:           "
        f"{comparison_path}"
    )

    print(
        f"Metadata:             "
        f"{metadata_path}"
    )

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Execute the complete model comparison workflow.
    """

    create_output_directories()

    df = load_training_data()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_train_test_split(df)

    print("\n" + "=" * 60)
    print(
        "EMPLOYEE ATTRITION — MODEL COMPARISON"
    )
    print("=" * 60)

    print(
        f"\nTraining rows:       {len(X_train)}"
    )

    print(
        f"Test rows:           {len(X_test)}"
    )

    print(
        f"Training attrition:  {int(y_train.sum())}"
    )

    print(
        f"Test attrition:      {int(y_test.sum())}"
    )

    print(
        "\nModels:"
    )

    for model_name in MODELS:
        print(
            f"  - {model_name}"
        )

    (
        trained_pipelines,
        comparison_df,
    ) = run_model_comparison(
        X_train,
        y_train,
    )

    print_model_comparison(
        comparison_df
    )

    (
        best_model_name,
        best_pipeline,
    ) = select_best_model(
        comparison_df,
        trained_pipelines,
    )

    test_metrics = evaluate_test_set(
        best_pipeline,
        X_test,
        y_test,
    )

    model_path = save_model(
        best_model_name,
        best_pipeline,
    )

    comparison_path = save_comparison(
        comparison_df
    )

    metadata_path = save_final_metrics(
        best_model_name,
        test_metrics,
        comparison_df,
        model_path,
    )

    print_final_evaluation(
        best_model_name,
        test_metrics,
        model_path,
        comparison_path,
        metadata_path,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "MODEL COMPARISON COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()