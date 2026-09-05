"""
Final stable-feature model selection.

Purpose
-------
Consolidates the validated employee-attrition modeling decisions:

1. Canonical dataset
2. Stable 10-feature subset
3. Candidate model families
4. Repeated stratified validation
5. ROC-AUC / PR-AUC comparison
6. Threshold = 0.44
7. Business operating characteristics
8. Final model selection

This script is intended as the final model-selection gate before
artifact generation / deployment validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

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
    / "final_model_selection_stable"
)

TARGET = "Attrition"
IDENTIFIER_COLUMNS = ["Employee_ID"]

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

THRESHOLD = 0.44

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42


# ============================================================
# OUTPUT SETUP
# ============================================================

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=20,
            n_estimators=50,
            subsample=0.7,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            max_features="sqrt",
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def validate_dataset(df: pd.DataFrame) -> dict:
    checks = {}

    checks["file_exists"] = DATASET_PATH.exists()
    checks["expected_rows"] = len(df) == 1000
    checks["expected_columns"] = len(df.columns) == 26
    checks["target_exists"] = TARGET in df.columns

    checks["stable_features_exist"] = all(
        feature in df.columns
        for feature in STABLE_FEATURES
    )

    checks["target_values_valid"] = set(
        df[TARGET].dropna().unique()
    ).issubset({"Yes", "No", 0, 1})

    checks["no_missing_cells"] = int(
        df.isna().sum().sum()
    ) == 0

    checks["identifier_unique"] = (
        IDENTIFIER_COLUMNS[0] in df.columns
        and df[IDENTIFIER_COLUMNS[0]].is_unique
    )

    return checks


def evaluate_model(
    model_name: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Evaluate model across repeated validation splits.

    Returns
    -------
    split_df:
        One row per validation split.

    oof_predictions:
        Averaged out-of-fold probabilities across repeated validation.
    """

    split_records = []

    prediction_sum = np.zeros(len(X), dtype=float)
    prediction_count = np.zeros(len(X), dtype=float)

    for split_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        preprocessor = build_preprocessor(X_train)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]

        prediction_sum[test_idx] += probabilities
        prediction_count[test_idx] += 1

        predictions = (
            probabilities >= THRESHOLD
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        balanced_accuracy = (
            recall_score(y_test, predictions)
            + specificity
        ) / 2

        split_records.append(
            {
                "model": model_name,
                "split": split_number,
                "roc_auc": roc_auc_score(
                    y_test,
                    probabilities,
                ),
                "pr_auc": average_precision_score(
                    y_test,
                    probabilities,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
                "predicted_positive_rate": predictions.mean(),
            }
        )

    oof_predictions = np.divide(
        prediction_sum,
        prediction_count,
        out=np.zeros_like(prediction_sum),
        where=prediction_count > 0,
    )

    return pd.DataFrame(split_records), oof_predictions


def summarize_results(split_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "accuracy",
        "predicted_positive_rate",
    ]

    rows = []

    for model_name, group in split_df.groupby("model"):
        row = {"model": model_name}

        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_min"] = group[metric].min()
            row[f"{metric}_max"] = group[metric].max()

        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_oof_threshold(
    y: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(
            y,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y,
            probabilities,
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,
        ),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
        "specificity": specificity,
        "balanced_accuracy": (
            recall_score(y, predictions)
            + specificity
        ) / 2,
        "accuracy": accuracy_score(
            y,
            predictions,
        ),
        "predicted_positive_rate": predictions.mean(),
        "flagged_per_1000": predictions.mean() * 1000,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Running final stable-feature model selection...")

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("Loading canonical dataset...")

    df = pd.read_csv(DATASET_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("Validating canonical dataset...")

    checks = validate_dataset(df)

    for name, passed in checks.items():
        print(
            f"{'PASS' if passed else 'FAIL'} "
            f"{name}"
        )

    if not all(checks.values()):
        raise ValueError(
            "Canonical dataset validation failed."
        )

    # --------------------------------------------------------
    # PREPARE FEATURES
    # --------------------------------------------------------

    X = df[STABLE_FEATURES].copy()

    y = (
        df[TARGET]
        .map({"No": 0, "Yes": 1})
        .astype(int)
    )

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print()
    print(f"Stable features:       {len(STABLE_FEATURES)}")
    print(f"Numerical features:    {len(numerical_features)}")
    print(f"Categorical features:  {len(categorical_features)}")
    print(
        f"Target prevalence:     "
        f"{y.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # VALIDATION DESIGN
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    print_header(
        "REPEATED VALIDATION MODEL COMPARISON"
    )

    print(f"Folds per repeat:      {N_SPLITS}")
    print(f"Repeats:               {N_REPEATS}")
    print(
        f"Total validation:      "
        f"{N_SPLITS * N_REPEATS}"
    )

    models = build_models()

    all_split_results = []
    oof_results = {}

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    for model_name, model in models.items():

        print(
            f"Evaluating {model_name}..."
        )

        split_df, oof_predictions = evaluate_model(
            model_name,
            model,
            X,
            y,
            cv,
        )

        all_split_results.append(split_df)

        oof_results[model_name] = oof_predictions

    split_results = pd.concat(
        all_split_results,
        ignore_index=True,
    )

    summary = summarize_results(
        split_results
    )

    summary = summary.sort_values(
        "roc_auc_mean",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(1, len(summary) + 1),
    )

    # --------------------------------------------------------
    # OOF COMPARISON
    # --------------------------------------------------------

    oof_rows = []

    for model_name, probabilities in oof_results.items():

        metrics = evaluate_oof_threshold(
            y,
            probabilities,
            THRESHOLD,
        )

        metrics["model"] = model_name

        oof_rows.append(metrics)

    oof_df = pd.DataFrame(oof_rows)

    oof_df = oof_df[
        [
            "model",
            "threshold",
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "accuracy",
            "predicted_positive_rate",
            "flagged_per_1000",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ]
    ]

    # --------------------------------------------------------
    # FINAL MODEL SELECTION
    # --------------------------------------------------------

    selected_model_name = summary.iloc[0]["model"]

    selected_summary = summary[
        summary["model"] == selected_model_name
    ].iloc[0]

    selected_oof = oof_df[
        oof_df["model"] == selected_model_name
    ].iloc[0]

    # --------------------------------------------------------
    # BUSINESS CHECKS
    # --------------------------------------------------------

    prevalence = y.mean()

    flags_ratio = (
        selected_oof["predicted_positive_rate"]
        / prevalence
    )

    diagnostic_flags = []

    if selected_oof["roc_auc"] < 0.60:
        diagnostic_flags.append(
            "Selected model has ROC-AUC below 0.60."
        )

    if selected_summary["roc_auc_std"] >= 0.04:
        diagnostic_flags.append(
            "Selected model shows elevated split sensitivity."
        )

    if selected_oof["precision"] < 0.40:
        diagnostic_flags.append(
            "Precision remains below 0.40, indicating "
            "a substantial false-positive burden."
        )

    if selected_oof["recall"] >= 0.70:
        diagnostic_flags.append(
            "Threshold prioritizes detection with recall "
            "of at least 0.70."
        )

    if selected_oof["predicted_positive_rate"] > (
        2 * prevalence
    ):
        diagnostic_flags.append(
            "The selected threshold flags more than twice "
            "the observed attrition prevalence."
        )

    if selected_oof["specificity"] < 0.60:
        diagnostic_flags.append(
            "Specificity is below 0.60 at the selected "
            "operating threshold."
        )

    if not diagnostic_flags:
        diagnostic_flags.append(
            "No major diagnostic concern detected."
        )

    # --------------------------------------------------------
    # CONSOLE REPORT
    # --------------------------------------------------------

    print_header(
        "EMPLOYEE ATTRITION — FINAL STABLE MODEL SELECTION"
    )

    print()
    print("[DATASET]")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print(
        f"Stable features:      {len(STABLE_FEATURES)}"
    )
    print(
        f"Target prevalence:    {prevalence * 100:.2f}%"
    )

    print()
    print("[MODEL COMPARISON]")

    display_columns = [
        "rank",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "accuracy_mean",
    ]

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("[SELECTED MODEL]")
    print(
        f"Model:                 "
        f"{selected_model_name}"
    )
    print(
        f"Features:              "
        f"{len(STABLE_FEATURES)}"
    )
    print(
        f"Threshold:             "
        f"{THRESHOLD:.2f}"
    )
    print(
        f"Repeated ROC-AUC:      "
        f"{selected_summary['roc_auc_mean']:.4f}"
    )
    print(
        f"ROC-AUC Std:           "
        f"{selected_summary['roc_auc_std']:.4f}"
    )
    print(
        f"Repeated PR-AUC:       "
        f"{selected_summary['pr_auc_mean']:.4f}"
    )

    print()
    print("[OUT-OF-FOLD BUSINESS PERFORMANCE]")
    print(
        f"ROC-AUC:               "
        f"{selected_oof['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:                "
        f"{selected_oof['pr_auc']:.4f}"
    )
    print(
        f"F1:                    "
        f"{selected_oof['f1']:.4f}"
    )
    print(
        f"Precision:             "
        f"{selected_oof['precision']:.4f}"
    )
    print(
        f"Recall:                "
        f"{selected_oof['recall']:.4f}"
    )
    print(
        f"Specificity:           "
        f"{selected_oof['specificity']:.4f}"
    )
    print(
        f"Balanced Accuracy:     "
        f"{selected_oof['balanced_accuracy']:.4f}"
    )
    print(
        f"Predicted Positive:    "
        f"{selected_oof['predicted_positive_rate'] * 100:.2f}%"
    )
    print(
        f"Flagged per 1000:      "
        f"{selected_oof['flagged_per_1000']:.1f}"
    )
    print(
        f"Flag / prevalence ratio: "
        f"{flags_ratio:.2f}x"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in diagnostic_flags:
        print(f"- {flag}")

    # --------------------------------------------------------
    # OVERALL DIAGNOSIS
    # --------------------------------------------------------

    if (
        selected_oof["roc_auc"] >= 0.60
        and selected_summary["roc_auc_std"] < 0.04
    ):
        overall_status = "CONDITIONAL PASS"
        diagnosis = (
            "The stable-feature model demonstrates useful "
            "and reasonably stable predictive separation "
            "under repeated validation. The selected "
            "threshold provides a detection-oriented "
            "operating point, but the high flagged rate and "
            "moderate precision require business-capacity "
            "and error-cost review before deployment."
        )
    else:
        overall_status = "REVIEW REQUIRED"
        diagnosis = (
            "The selected stable-feature model provides "
            "limited or unstable predictive separation. "
            "Further investigation is recommended before "
            "production deployment."
        )

    print()
    print("[OVERALL STATUS]")
    print(
        f"FINAL MODEL SELECTION STATUS: "
        f"{overall_status}"
    )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # --------------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "final_model_comparison.csv"
    )

    split_path = (
        REPORT_DIR
        / "final_model_split_performance.csv"
    )

    oof_path = (
        REPORT_DIR
        / "final_model_oof_performance.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    split_results.to_csv(
        split_path,
        index=False,
    )

    oof_df.to_csv(
        oof_path,
        index=False,
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(df),
            "columns": len(df.columns),
            "target": TARGET,
            "target_prevalence": float(prevalence),
        },
        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": (
                N_SPLITS * N_REPEATS
            ),
        },
        "stable_features": STABLE_FEATURES,
        "selected_model": selected_model_name,
        "threshold": THRESHOLD,
        "model_comparison": summary.to_dict(
            orient="records"
        ),
        "oof_performance": oof_df.to_dict(
            orient="records"
        ),
        "diagnostic_flags": diagnostic_flags,
        "overall_status": overall_status,
        "overall_diagnosis": diagnosis,
    }

    json_path = (
        REPORT_DIR
        / "final_model_selection_stable_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            default=str,
        )

    summary_txt_path = (
        REPORT_DIR
        / "final_model_selection_stable_summary.txt"
    )

    with open(
        summary_txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "EMPLOYEE ATTRITION — "
            "FINAL STABLE MODEL SELECTION\n"
        )
        f.write("=" * 64 + "\n\n")

        f.write(
            f"Dataset: {DATASET_PATH}\n"
        )
        f.write(
            f"Rows: {len(df)}\n"
        )
        f.write(
            f"Columns: {len(df.columns)}\n"
        )
        f.write(
            f"Stable features: {len(STABLE_FEATURES)}\n"
        )
        f.write(
            f"Target prevalence: "
            f"{prevalence * 100:.2f}%\n\n"
        )

        f.write(
            f"Selected model: "
            f"{selected_model_name}\n"
        )
        f.write(
            f"Threshold: {THRESHOLD:.2f}\n\n"
        )

        f.write(
            f"Repeated ROC-AUC: "
            f"{selected_summary['roc_auc_mean']:.4f}\n"
        )
        f.write(
            f"ROC-AUC Std: "
            f"{selected_summary['roc_auc_std']:.4f}\n"
        )
        f.write(
            f"Repeated PR-AUC: "
            f"{selected_summary['pr_auc_mean']:.4f}\n\n"
        )

        f.write(
            "OUT-OF-FOLD BUSINESS PERFORMANCE\n"
        )
        f.write("-" * 40 + "\n")

        for key in [
            "roc_auc",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "specificity",
            "balanced_accuracy",
            "accuracy",
            "predicted_positive_rate",
            "flagged_per_1000",
        ]:
            f.write(
                f"{key}: "
                f"{selected_oof[key]:.4f}\n"
                if key != "flagged_per_1000"
                else
                f"{key}: "
                f"{selected_oof[key]:.1f}\n"
            )

        f.write("\nDIAGNOSTIC FLAGS\n")
        f.write("-" * 40 + "\n")

        for flag in diagnostic_flags:
            f.write(f"- {flag}\n")

        f.write("\nOVERALL STATUS\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"FINAL MODEL SELECTION STATUS: "
            f"{overall_status}\n\n"
        )
        f.write(diagnosis)
        f.write("\n")

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              "
        f"{REPORT_DIR}"
    )
    print(
        f"Model comparison:     "
        f"{summary_path}"
    )
    print(
        f"Split performance:    "
        f"{split_path}"
    )
    print(
        f"OOF performance:      "
        f"{oof_path}"
    )
    print(
        f"JSON report:          "
        f"{json_path}"
    )
    print(
        f"Summary report:       "
        f"{summary_txt_path}"
    )

    print()
    print("=" * 64)
    print("FINAL STABLE MODEL SELECTION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()