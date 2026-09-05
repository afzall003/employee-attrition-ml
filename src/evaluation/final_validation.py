"""
Final Holdout Validation
========================

Validates the selected final model exactly once on the untouched holdout
partition.

Important:
- Model selection has already been completed.
- Threshold selection has already been completed.
- This script must NOT optimize the model or threshold.
- The final operating threshold is fixed at 0.15.
- The holdout is used only for final performance verification.

Dataset:
    data/raw/employee_attrition_dataset_v2.csv

Final model artifact:
    reports/signal_analysis/final_model_selection/final_model.joblib

Outputs:
    reports/signal_analysis/final_validation/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


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

MODEL_PATH = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "final_model_selection"
    / "final_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "final_validation"
)

REPORT_PATH = OUTPUT_DIR / "final_validation_report.json"
SUMMARY_PATH = OUTPUT_DIR / "final_validation_summary.txt"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

TARGET_COLUMN = "Attrition"

POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"

# This threshold was selected in the previous threshold analysis.
FINAL_THRESHOLD = 0.15


# ============================================================
# TERMINAL HELPERS
# ============================================================


def print_header(title: str) -> None:
    """Print a consistent terminal section header."""
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_metric(name: str, value: float, decimals: int = 4) -> None:
    """Print a formatted metric."""
    print(f"{name:<28} {value:.{decimals}f}")


# ============================================================
# DATA LOADING
# ============================================================


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the V2 employee attrition dataset.

    Returns:
        X: feature dataframe
        y: binary target series
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found "
            f"in dataset."
        )

    # Preserve the same feature columns used during model training.
    X = df.drop(columns=[TARGET_COLUMN]).copy()

    y_raw = df[TARGET_COLUMN].copy()

    # Convert Yes/No to binary representation.
    y = (
        y_raw
        .map(
            {
                NEGATIVE_LABEL: 0,
                POSITIVE_LABEL: 1,
            }
        )
        .astype(int)
    )

    if y.isna().any():
        raise ValueError(
            "Target contains values other than "
            f"'{NEGATIVE_LABEL}' and '{POSITIVE_LABEL}'."
        )

    return X, y


# ============================================================
# HOLDOUT CREATION
# ============================================================


def create_holdout(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Recreate the same deterministic train/holdout partition.

    Stratification preserves the target prevalence between the
    training and holdout partitions.
    """

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    )


# ============================================================
# MODEL LOADING
# ============================================================


def load_final_model():
    """
    Load the already-selected final model artifact.

    No model fitting or optimization occurs here.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model artifact not found:\n{MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


# ============================================================
# PROBABILITY PREDICTION
# ============================================================


def get_positive_probabilities(
    model,
    X_holdout: pd.DataFrame,
) -> np.ndarray:
    """
    Generate positive-class probabilities from the final model.
    """

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The final model does not expose predict_proba(). "
            "Final validation requires probability estimates."
        )

    probabilities = model.predict_proba(X_holdout)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "Unexpected probability output shape: "
            f"{probabilities.shape}"
        )

    positive_probabilities = probabilities[:, 1]

    return np.asarray(positive_probabilities, dtype=float)


# ============================================================
# THRESHOLD PREDICTION
# ============================================================


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convert probabilities into binary predictions.

    Positive class is predicted when:

        probability >= threshold
    """

    return (
        probabilities >= threshold
    ).astype(int)


# ============================================================
# METRIC CALCULATION
# ============================================================


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """Calculate final holdout metrics."""

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

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    actual_positive_rate = (
        float(np.mean(y_true))
        * 100.0
    )

    predicted_positive_rate = (
        float(np.mean(predictions))
        * 100.0
    )

    mean_probability = float(
        np.mean(probabilities)
    )

    minimum_probability = float(
        np.min(probabilities)
    )

    maximum_probability = float(
        np.max(probabilities)
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "threshold": float(FINAL_THRESHOLD),
        "actual_positive_rate_percent": actual_positive_rate,
        "predicted_positive_rate_percent": predicted_positive_rate,
        "mean_probability": mean_probability,
        "minimum_probability": minimum_probability,
        "maximum_probability": maximum_probability,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================


def extract_model_information(model) -> dict:
    """
    Extract useful information from the final model artifact.

    This function intentionally does not modify the model.
    """

    information = {
        "model_type": type(model).__name__,
    }

    # Pipeline information
    if hasattr(model, "named_steps"):
        information["pipeline_steps"] = list(
            model.named_steps.keys()
        )

        if "model" in model.named_steps:
            estimator = model.named_steps["model"]

            information["estimator_type"] = (
                type(estimator).__name__
            )

            if hasattr(estimator, "get_params"):
                params = estimator.get_params()

                selected_params = {}

                for key in [
                    "C",
                    "class_weight",
                    "solver",
                    "penalty",
                    "max_iter",
                    "random_state",
                ]:
                    if key in params:
                        value = params[key]

                        if isinstance(value, np.generic):
                            value = value.item()

                        selected_params[key] = value

                information["estimator_parameters"] = (
                    selected_params
                )

    elif hasattr(model, "get_params"):
        params = model.get_params()

        selected_params = {}

        for key in [
            "C",
            "class_weight",
            "solver",
            "penalty",
            "max_iter",
            "random_state",
        ]:
            if key in params:
                value = params[key]

                if isinstance(value, np.generic):
                    value = value.item()

                selected_params[key] = value

        information["estimator_parameters"] = (
            selected_params
        )

    return information


# ============================================================
# PRODUCTION READINESS ASSESSMENT
# ============================================================


def determine_conclusion(
    metrics: dict,
) -> str:
    """
    Produce a conservative interpretation of final holdout
    performance.

    This does not silently declare production readiness based
    on an arbitrary single metric.
    """

    roc_auc = metrics["roc_auc"]
    pr_auc = metrics["pr_auc"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    f1 = metrics["f1"]

    # Conservative criteria.
    #
    # These are deliberately stricter than simply beating random
    # performance. The project has already demonstrated only
    # moderate signal, so final validation should not overstate
    # model quality.

    if (
        roc_auc >= 0.70
        and pr_auc >= 0.40
        and precision >= 0.40
        and recall >= 0.40
        and f1 >= 0.40
    ):
        return (
            "The final holdout results meet the configured "
            "minimum performance criteria for a potentially "
            "production-capable model. Further business "
            "validation and monitoring are still required."
        )

    return (
        "The final holdout results do not meet the configured "
        "minimum performance criteria for declaring the model "
        "production-ready. The model may still be useful as "
        "an experimental or decision-support model, but further "
        "improvement and validation are recommended."
    )


# ============================================================
# REPORT WRITING
# ============================================================


def write_reports(
    metrics: dict,
    model_information: dict,
    dataset_information: dict,
    conclusion: str,
) -> None:
    """Write JSON and human-readable final validation reports."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "project": "Employee Attrition ML",
        "stage": "Final Holdout Validation",
        "dataset": dataset_information,
        "model": model_information,
        "validation": {
            "threshold": FINAL_THRESHOLD,
            "holdout_only": True,
            "optimization_performed": False,
            "threshold_optimization_performed": False,
        },
        "metrics": metrics,
        "conclusion": conclusion,
    }

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    lines = []

    lines.append(
        "EMPLOYEE ATTRITION — FINAL HOLDOUT VALIDATION"
    )
    lines.append("=" * 60)
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 "
        f"{dataset_information['rows']}"
    )
    lines.append(
        f"Features:             "
        f"{dataset_information['features']}"
    )
    lines.append(
        f"Training rows:        "
        f"{dataset_information['training_rows']}"
    )
    lines.append(
        f"Holdout rows:         "
        f"{dataset_information['holdout_rows']}"
    )
    lines.append(
        f"Actual attrition:     "
        f"{dataset_information['actual_attrition_rate']:.2f}%"
    )
    lines.append("")

    lines.append("[FINAL MODEL]")
    lines.append(
        f"Model type:           "
        f"{model_information.get('model_type', 'Unknown')}"
    )

    if "estimator_type" in model_information:
        lines.append(
            f"Estimator:            "
            f"{model_information['estimator_type']}"
        )

    if "estimator_parameters" in model_information:
        lines.append(
            "Parameters:           "
            + json.dumps(
                model_information[
                    "estimator_parameters"
                ],
                sort_keys=True,
            )
        )

    lines.append(
        f"Operating threshold:  "
        f"{FINAL_THRESHOLD:.2f}"
    )
    lines.append("")

    lines.append("[FINAL HOLDOUT METRICS]")
    lines.append(
        f"ROC-AUC:              "
        f"{metrics['roc_auc']:.4f}"
    )
    lines.append(
        f"PR-AUC:               "
        f"{metrics['pr_auc']:.4f}"
    )
    lines.append(
        f"Precision:            "
        f"{metrics['precision']:.4f}"
    )
    lines.append(
        f"Recall:               "
        f"{metrics['recall']:.4f}"
    )
    lines.append(
        f"F1:                   "
        f"{metrics['f1']:.4f}"
    )
    lines.append(
        f"Accuracy:             "
        f"{metrics['accuracy']:.4f}"
    )
    lines.append(
        f"Predicted positive:   "
        f"{metrics['predicted_positive_rate_percent']:.2f}%"
    )
    lines.append("")

    lines.append("[CONFUSION MATRIX]")
    lines.append(
        "                    Predicted No    Predicted Yes"
    )
    lines.append(
        f"Actual No           "
        f"{metrics['true_negative']:>12}    "
        f"{metrics['false_positive']:>13}"
    )
    lines.append(
        f"Actual Yes          "
        f"{metrics['false_negative']:>12}    "
        f"{metrics['true_positive']:>13}"
    )
    lines.append("")

    lines.append("[PROBABILITY RANGE]")
    lines.append(
        f"Minimum probability: "
        f"{metrics['minimum_probability']:.4f}"
    )
    lines.append(
        f"Maximum probability: "
        f"{metrics['maximum_probability']:.4f}"
    )
    lines.append(
        f"Mean probability:    "
        f"{metrics['mean_probability']:.4f}"
    )
    lines.append("")

    lines.append("[CONCLUSION]")
    lines.append(conclusion)
    lines.append("")

    lines.append("[IMPORTANT]")
    lines.append(
        "This holdout partition was evaluated only after "
        "model optimization and threshold selection."
    )
    lines.append(
        "No model or threshold optimization was performed "
        "during this validation stage."
    )
    lines.append("")

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Run final holdout validation."""

    print()
    print("Running final holdout validation...")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(X)}")
    print(f"Features:             {X.shape[1]}")
    print(
        f"Target prevalence:    "
        f"{float(y.mean()) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Recreate deterministic holdout
    # --------------------------------------------------------

    (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
    ) = create_holdout(
        X,
        y,
    )

    print(
        f"Training rows:        {len(X_train)}"
    )
    print(
        f"Holdout rows:         {len(X_holdout)}"
    )

    # --------------------------------------------------------
    # Load selected final model
    # --------------------------------------------------------

    print()
    print("Loading final model artifact...")

    model = load_final_model()

    print(
        f"Model:                "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # Generate holdout probabilities
    # --------------------------------------------------------

    print()
    print("Generating holdout probabilities...")

    probabilities = get_positive_probabilities(
        model,
        X_holdout,
    )

    # --------------------------------------------------------
    # Apply fixed threshold
    # --------------------------------------------------------

    print(
        f"Applying operating threshold: "
        f"{FINAL_THRESHOLD:.2f}"
    )

    predictions = apply_threshold(
        probabilities,
        FINAL_THRESHOLD,
    )

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    print()
    print("Calculating final holdout metrics...")

    metrics = calculate_metrics(
        y_holdout,
        probabilities,
        predictions,
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_information = (
        extract_model_information(model)
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    dataset_information = {
        "dataset_path": str(DATA_PATH),
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "training_rows": int(len(X_train)),
        "holdout_rows": int(len(X_holdout)),
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "overall_attrition_rate": float(
            y.mean() * 100.0
        ),
        "actual_attrition_rate": float(
            y_holdout.mean() * 100.0
        ),
    }

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    conclusion = determine_conclusion(
        metrics
    )

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print_header(
        "EMPLOYEE ATTRITION — FINAL HOLDOUT VALIDATION"
    )

    print()
    print("[DATASET]")
    print(
        f"Rows:                 "
        f"{dataset_information['rows']}"
    )
    print(
        f"Features:             "
        f"{dataset_information['features']}"
    )
    print(
        f"Training rows:        "
        f"{dataset_information['training_rows']}"
    )
    print(
        f"Holdout rows:         "
        f"{dataset_information['holdout_rows']}"
    )
    print(
        f"Actual attrition:     "
        f"{dataset_information['actual_attrition_rate']:.2f}%"
    )

    print()
    print("[FINAL MODEL]")
    print(
        f"Model:                "
        f"{model_information.get('model_type', 'Unknown')}"
    )

    if "estimator_type" in model_information:
        print(
            f"Estimator:            "
            f"{model_information['estimator_type']}"
        )

    if "estimator_parameters" in model_information:
        print(
            "Parameters:           "
            + json.dumps(
                model_information[
                    "estimator_parameters"
                ],
                sort_keys=True,
            )
        )

    print(
        f"Operating threshold:  "
        f"{FINAL_THRESHOLD:.2f}"
    )

    print()
    print("[FINAL HOLDOUT METRICS]")

    print_metric(
        "ROC-AUC:",
        metrics["roc_auc"],
    )

    print_metric(
        "PR-AUC:",
        metrics["pr_auc"],
    )

    print_metric(
        "Precision:",
        metrics["precision"],
    )

    print_metric(
        "Recall:",
        metrics["recall"],
    )

    print_metric(
        "F1:",
        metrics["f1"],
    )

    print_metric(
        "Accuracy:",
        metrics["accuracy"],
    )

    print(
        f"{'Predicted positive:':<28}"
        f"{metrics['predicted_positive_rate_percent']:.2f}%"
    )

    print()
    print("[CONFUSION MATRIX]")
    print(
        "                    Predicted No    Predicted Yes"
    )
    print(
        f"Actual No           "
        f"{metrics['true_negative']:>12}    "
        f"{metrics['false_positive']:>13}"
    )
    print(
        f"Actual Yes          "
        f"{metrics['false_negative']:>12}    "
        f"{metrics['true_positive']:>13}"
    )

    print()
    print("[PROBABILITY RANGE]")
    print(
        f"Minimum probability: "
        f"{metrics['minimum_probability']:.4f}"
    )
    print(
        f"Maximum probability: "
        f"{metrics['maximum_probability']:.4f}"
    )
    print(
        f"Mean probability:    "
        f"{metrics['mean_probability']:.4f}"
    )

    print()
    print("[CONCLUSION]")
    print(conclusion)

    # --------------------------------------------------------
    # Write reports
    # --------------------------------------------------------

    write_reports(
        metrics=metrics,
        model_information=model_information,
        dataset_information=dataset_information,
        conclusion=conclusion,
    )

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              "
        f"{OUTPUT_DIR}"
    )
    print(
        f"JSON report:          "
        f"{REPORT_PATH}"
    )
    print(
        f"Summary report:       "
        f"{SUMMARY_PATH}"
    )

    print()
    print("=" * 60)
    print("FINAL HOLDOUT VALIDATION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Validation interrupted by user.")
        sys.exit(1)

    except Exception as exc:
        print()
        print("=" * 60)
        print("FINAL HOLDOUT VALIDATION FAILED")
        print("=" * 60)
        print()
        print(f"Error: {exc}")
        print()

        raise