from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.features.engineering import prepare_model_data


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

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression_baseline.joblib"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

METRICS_DIR = (
    REPORTS_DIR
    / "metrics"
)

FIGURES_DIR = (
    REPORTS_DIR
    / "figures"
)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

DEFAULT_THRESHOLD = 0.50


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required for evaluation outputs."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the raw employee attrition dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    return df


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """Load the trained model pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run `python -m src.models.train` first."
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# TEST DATA RECREATION
# ============================================================

def create_test_data(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Recreate the same stratified train/test split used during
    baseline training.

    The split configuration is intentionally identical to
    train.py so that evaluation is performed against the same
    held-out test population.
    """

    X, y = prepare_model_data(df)

    (
        _,
        X_test,
        _,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    return X_test, y_test


# ============================================================
# PROBABILITY GENERATION
# ============================================================

def generate_probabilities(
    model,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate attrition probabilities for test employees."""

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return probabilities


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_thresholds(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate classification performance across multiple
    probability thresholds.

    This is diagnostic only. We are not selecting a production
    threshold from the test set.
    """

    thresholds = np.arange(
        0.10,
        0.91,
        0.05,
    )

    records = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        records.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "predicted_attrition_count": int(
                    predictions.sum()
                ),
                "predicted_attrition_rate": round(
                    float(predictions.mean() * 100),
                    2,
                ),
                "precision": round(
                    float(
                        precision_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        recall_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "f1": round(
                    float(
                        f1_score(
                            y_test,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "accuracy": round(
                    float(
                        accuracy_score(
                            y_test,
                            predictions,
                        )
                    ),
                    4,
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# BASELINE COMPARISON
# ============================================================

def evaluate_dummy_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a majority-class dummy classifier.

    This establishes the minimum useful baseline for accuracy.
    """

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        np.zeros(
            (len(y_train), 1)
        ),
        y_train,
    )

    predictions = dummy.predict(
        np.zeros(
            (len(y_test), 1)
        )
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Calculate classification and ranking metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Stayed",
            "Left",
        ],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "threshold": threshold,
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                probabilities,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


# ============================================================
# ROC CURVE
# ============================================================

def plot_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> Path:
    """Generate and save the ROC curve."""

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities,
    )

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=f"Model ROC-AUC = {auc:.3f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Employee Attrition — ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output_path = (
        FIGURES_DIR
        / "baseline_roc_curve.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

def plot_precision_recall_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> Path:
    """Generate and save the precision-recall curve."""

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            probabilities,
        )
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    positive_rate = float(
        y_test.mean()
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        recall,
        precision,
        label=f"Model PR-AUC = {pr_auc:.3f}",
    )

    plt.axhline(
        positive_rate,
        linestyle="--",
        label=(
            f"Positive-class prevalence = "
            f"{positive_rate:.3f}"
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Employee Attrition — Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output_path = (
        FIGURES_DIR
        / "baseline_precision_recall_curve.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def plot_probability_distribution(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> Path:
    """
    Plot predicted attrition probabilities separated by
    actual outcome.
    """

    stayed_probabilities = probabilities[
        y_test.to_numpy() == 0
    ]

    left_probabilities = probabilities[
        y_test.to_numpy() == 1
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        stayed_probabilities,
        bins=20,
        alpha=0.6,
        label="Actually stayed",
    )

    plt.hist(
        left_probabilities,
        bins=20,
        alpha=0.6,
        label="Actually left",
    )

    plt.axvline(
        DEFAULT_THRESHOLD,
        linestyle="--",
        label="Threshold = 0.50",
    )

    plt.xlabel(
        "Predicted probability of attrition"
    )

    plt.ylabel(
        "Number of employees"
    )

    plt.title(
        "Employee Attrition — Prediction Probability Distribution"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    output_path = (
        FIGURES_DIR
        / "baseline_probability_distribution.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Generate and save the confusion matrix."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title(
        "Employee Attrition — Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(
        tick_marks,
        [
            "Stayed",
            "Left",
        ],
    )

    plt.yticks(
        tick_marks,
        [
            "Stayed",
            "Left",
        ],
    )

    threshold_value = (
        cm.max() / 2
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold_value
                    else "black"
                ),
                fontsize=14,
            )

    plt.ylabel(
        "Actual"
    )

    plt.xlabel(
        "Predicted"
    )

    output_path = (
        FIGURES_DIR
        / "baseline_confusion_matrix.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    return output_path


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data: dict,
    filename: str,
) -> Path:
    """Save a dictionary as formatted JSON."""

    output_path = (
        METRICS_DIR
        / filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    return output_path


# ============================================================
# SAVE CSV
# ============================================================

def save_dataframe(
    dataframe: pd.DataFrame,
    filename: str,
) -> Path:
    """Save a DataFrame as CSV."""

    output_path = (
        METRICS_DIR
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# TERMINAL REPORT
# ============================================================

def print_report(
    metrics: dict,
    dummy_metrics: dict,
    threshold_table: pd.DataFrame,
    roc_path: Path,
    pr_path: Path,
    probability_path: Path,
    confusion_path: Path,
) -> None:
    """Print the evaluation results."""

    print("\n" + "=" * 60)
    print(
        "EMPLOYEE ATTRITION — MODEL DIAGNOSTICS"
    )
    print("=" * 60)

    print("\n[BASELINE MODEL]")

    print(
        f"ROC-AUC:              "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:            "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{metrics['f1']:.4f}"
    )

    print("\n[DUMMY MAJORITY-CLASS BASELINE]")

    print(
        f"Accuracy:             "
        f"{dummy_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:            "
        f"{dummy_metrics['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{dummy_metrics['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{dummy_metrics['f1']:.4f}"
    )

    print("\n[THRESHOLD DIAGNOSTICS]")

    print(
        threshold_table.to_string(
            index=False
        )
    )

    print("\n[FIGURES]")

    print(
        f"ROC curve:            {roc_path}"
    )

    print(
        f"Precision-Recall:     {pr_path}"
    )

    print(
        f"Probability:          {probability_path}"
    )

    print(
        f"Confusion matrix:     {confusion_path}"
    )

    print("\n" + "=" * 60)
    print(
        "MODEL DIAGNOSTICS COMPLETE"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete baseline evaluation workflow."""

    create_output_directories()

    df = load_data()

    model = load_model()

    (
        X_test,
        y_test,
    ) = create_test_data(df)

    probabilities = generate_probabilities(
        model,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    (
        X_train,
        _,
        y_train,
        _,
    ) = train_test_split(
        prepare_model_data(df)[0],
        prepare_model_data(df)[1],
        test_size=TEST_SIZE,
        stratify=prepare_model_data(df)[1],
        random_state=RANDOM_STATE,
    )

    dummy_metrics = evaluate_dummy_baseline(
        y_train,
        y_test,
    )

    threshold_table = evaluate_thresholds(
        y_test,
        probabilities,
    )

    roc_path = plot_roc_curve(
        y_test,
        probabilities,
    )

    pr_path = plot_precision_recall_curve(
        y_test,
        probabilities,
    )

    probability_path = plot_probability_distribution(
        y_test,
        probabilities,
    )

    confusion_path = plot_confusion_matrix(
        y_test,
        probabilities,
        DEFAULT_THRESHOLD,
    )

    save_json(
        metrics,
        "baseline_diagnostics.json",
    )

    save_json(
        dummy_metrics,
        "dummy_baseline.json",
    )

    save_dataframe(
        threshold_table,
        "threshold_diagnostics.csv",
    )

    print_report(
        metrics,
        dummy_metrics,
        threshold_table,
        roc_path,
        pr_path,
        probability_path,
        confusion_path,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()