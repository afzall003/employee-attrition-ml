from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation.signal_analysis import (
    build_signal_model,
    load_data,
    prepare_target,
)
from src.features.engineering import prepare_model_data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

SIGNAL_REPORT_DIR = (
    REPORTS_DIR
    / "signal_analysis"
)

THRESHOLD_REPORT_DIR = (
    SIGNAL_REPORT_DIR
    / "threshold_analysis"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CV_SPLITS = 5

TARGET_COLUMN = "Attrition"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required for threshold analysis."""

    THRESHOLD_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CROSS-VALIDATED PREDICTIONS
# ============================================================

def generate_cross_validated_probabilities(
    df: pd.DataFrame,
    target: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate out-of-fold probabilities using ordinary
    StratifiedKFold.

    Unlike RepeatedStratifiedKFold, ordinary StratifiedKFold
    creates a true partition of the dataset, allowing every
    employee to receive exactly one out-of-fold prediction.
    """

    X, y = prepare_model_data(df)

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = np.zeros(
        len(y),
        dtype=float,
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"Running threshold CV fold "
            f"{fold_number}/{CV_SPLITS}..."
        )

        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]

        model = build_signal_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities[
            validation_indices
        ] = model.predict_proba(
            X_validation
        )[:, 1]

    return (
        y.to_numpy(),
        probabilities,
    )


# ============================================================
# THRESHOLD METRICS
# ============================================================

def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Calculate classification metrics across a range
    of probability thresholds.
    """

    thresholds = np.arange(
        0.05,
        0.96,
        0.05,
    )

    records = []

    actual_positive_rate = float(
        y_true.mean()
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

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

        predicted_positive_rate = float(
            predictions.mean()
        )

        records.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "predicted_positive_rate": (
                    predicted_positive_rate
                ),
                "actual_positive_rate": (
                    actual_positive_rate
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# BEST THRESHOLDS
# ============================================================

def identify_best_thresholds(
    threshold_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Identify thresholds that optimize common metrics."""

    best_f1 = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_recall = threshold_df.loc[
        threshold_df["recall"].idxmax()
    ]

    best_precision = threshold_df.loc[
        threshold_df["precision"].idxmax()
    ]

    best_accuracy = threshold_df.loc[
        threshold_df["accuracy"].idxmax()
    ]

    return {
        "best_f1": {
            "threshold": float(
                best_f1["threshold"]
            ),
            "f1": float(
                best_f1["f1"]
            ),
            "precision": float(
                best_f1["precision"]
            ),
            "recall": float(
                best_f1["recall"]
            ),
        },
        "best_recall": {
            "threshold": float(
                best_recall["threshold"]
            ),
            "recall": float(
                best_recall["recall"]
            ),
            "precision": float(
                best_recall["precision"]
            ),
            "f1": float(
                best_recall["f1"]
            ),
        },
        "best_precision": {
            "threshold": float(
                best_precision["threshold"]
            ),
            "precision": float(
                best_precision["precision"]
            ),
            "recall": float(
                best_precision["recall"]
            ),
            "f1": float(
                best_precision["f1"]
            ),
        },
        "best_accuracy": {
            "threshold": float(
                best_accuracy["threshold"]
            ),
            "accuracy": float(
                best_accuracy["accuracy"]
            ),
            "precision": float(
                best_accuracy["precision"]
            ),
            "recall": float(
                best_accuracy["recall"]
            ),
            "f1": float(
                best_accuracy["f1"]
            ),
        },
    }


# ============================================================
# MODEL-LEVEL METRICS
# ============================================================

def calculate_model_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Calculate threshold-independent ranking metrics."""

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "actual_positive_rate": float(
            y_true.mean()
        ),
        "minimum_probability": float(
            probabilities.min()
        ),
        "maximum_probability": float(
            probabilities.max()
        ),
        "mean_probability": float(
            probabilities.mean()
        ),
    }


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    threshold_df: pd.DataFrame,
    best_thresholds: dict[str, dict[str, float]],
    model_metrics: dict[str, float],
) -> None:
    """Save threshold analysis reports."""

    threshold_df.to_csv(
        THRESHOLD_REPORT_DIR
        / "threshold_metrics.csv",
        index=False,
    )

    with open(
        THRESHOLD_REPORT_DIR
        / "best_thresholds.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_thresholds,
            file,
            indent=4,
        )

    with open(
        THRESHOLD_REPORT_DIR
        / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            model_metrics,
            file,
            indent=4,
        )


# ============================================================
# TERMINAL REPORT
# ============================================================

def print_report(
    threshold_df: pd.DataFrame,
    best_thresholds: dict[str, dict[str, float]],
    model_metrics: dict[str, float],
) -> None:
    """Print a concise threshold analysis report."""

    print("\n" + "=" * 60)

    print(
        "EMPLOYEE ATTRITION — THRESHOLD ANALYSIS"
    )

    print("=" * 60)

    print("\n[MODEL-LEVEL METRICS]")

    print(
        f"ROC-AUC:              "
        f"{model_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{model_metrics['pr_auc']:.4f}"
    )

    print(
        f"Actual attrition:     "
        f"{model_metrics['actual_positive_rate'] * 100:.2f}%"
    )

    print(
        f"Minimum probability:  "
        f"{model_metrics['minimum_probability']:.4f}"
    )

    print(
        f"Maximum probability:  "
        f"{model_metrics['maximum_probability']:.4f}"
    )

    print(
        f"Mean probability:     "
        f"{model_metrics['mean_probability']:.4f}"
    )

    print("\n[THRESHOLD PERFORMANCE]")

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "predicted_positive_rate",
    ]

    display_df = threshold_df[
        display_columns
    ].copy()

    display_df[
        "predicted_positive_rate"
    ] *= 100

    print(
        display_df
        .round(4)
        .to_string(index=False)
    )

    print("\n[BEST F1 THRESHOLD]")

    best_f1 = best_thresholds[
        "best_f1"
    ]

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

    print("\n[BEST RECALL THRESHOLD]")

    best_recall = best_thresholds[
        "best_recall"
    ]

    print(
        f"Threshold:            "
        f"{best_recall['threshold']:.2f}"
    )

    print(
        f"Recall:               "
        f"{best_recall['recall']:.4f}"
    )

    print(
        f"Precision:            "
        f"{best_recall['precision']:.4f}"
    )

    print(
        f"F1:                   "
        f"{best_recall['f1']:.4f}"
    )

    print("\n[BEST PRECISION THRESHOLD]")

    best_precision = best_thresholds[
        "best_precision"
    ]

    print(
        f"Threshold:            "
        f"{best_precision['threshold']:.2f}"
    )

    print(
        f"Precision:            "
        f"{best_precision['precision']:.4f}"
    )

    print(
        f"Recall:               "
        f"{best_precision['recall']:.4f}"
    )

    print(
        f"F1:                   "
        f"{best_precision['f1']:.4f}"
    )

    print("\n[OUTPUT]")

    print(
        f"Reports:              "
        f"{THRESHOLD_REPORT_DIR}"
    )

    print("\n" + "=" * 60)

    print(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run complete threshold analysis."""

    create_output_directories()

    df = load_data()

    target = prepare_target(
        df
    )

    print(
        "\nGenerating out-of-fold probabilities..."
    )

    y_true, probabilities = (
        generate_cross_validated_probabilities(
            df,
            target,
        )
    )

    print(
        "\nCalculating threshold metrics..."
    )

    threshold_df = (
        calculate_threshold_metrics(
            y_true,
            probabilities,
        )
    )

    best_thresholds = (
        identify_best_thresholds(
            threshold_df
        )
    )

    model_metrics = (
        calculate_model_metrics(
            y_true,
            probabilities,
        )
    )

    save_reports(
        threshold_df,
        best_thresholds,
        model_metrics,
    )

    print_report(
        threshold_df,
        best_thresholds,
        model_metrics,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()