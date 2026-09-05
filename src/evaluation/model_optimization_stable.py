"""
Model Optimization — Stable Feature Set

Purpose
-------
Optimize and compare candidate classification models using the
validated stable 10-feature subset.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

Stable features:
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

The script:
    1. Loads and validates the canonical dataset.
    2. Uses only the frozen stable feature set.
    3. Builds preprocessing pipelines.
    4. Performs repeated stratified cross-validation.
    5. Optimizes Logistic Regression, Gradient Boosting,
       and Random Forest hyperparameters.
    6. Compares ROC-AUC and PR-AUC.
    7. Reports F1, precision, recall and accuracy.
    8. Identifies the best candidate model.
    9. Evaluates the selected model across repeated splits.
   10. Saves JSON, CSV and TXT reports.

Important:
    This is a development/optimization stage.
    The final untouched validation set must NOT be used for
    hyperparameter selection.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
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
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_attrition_dataset_v2.csv"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "model_optimization_stable"
)

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 5

N_JOBS = -1

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMN = "Employee_ID"

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26
EXPECTED_FEATURE_COUNT = 24

EXPECTED_POSITIVE_COUNT = 236
EXPECTED_NEGATIVE_COUNT = 764
EXPECTED_PREVALENCE = 0.236


# ----------------------------------------------------------------------------
# FROZEN STABLE FEATURE SET
# ----------------------------------------------------------------------------

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
# UTILITY FUNCTIONS
# ============================================================================


def ensure_report_directory() -> None:
    """Create the report directory."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def convert_numpy_types(value: Any) -> Any:
    """
    Recursively convert NumPy/Pandas objects into JSON-safe Python types.
    """
    if isinstance(value, dict):
        return {
            str(k): convert_numpy_types(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [convert_numpy_types(v) for v in value]

    if isinstance(value, tuple):
        return [convert_numpy_types(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return None

    return value


def save_json_report(report: dict[str, Any]) -> None:
    """Save JSON report."""
    output_path = REPORT_DIR / "model_optimization_stable_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            convert_numpy_types(report),
            file,
            indent=2,
        )


def save_summary(summary_lines: list[str]) -> None:
    """Save human-readable summary."""
    output_path = REPORT_DIR / "model_optimization_stable_summary.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))


def safe_metric_mean(values: np.ndarray) -> float:
    """Return a safe metric mean."""
    return float(np.nanmean(values))


def safe_metric_std(values: np.ndarray) -> float:
    """Return a safe metric standard deviation."""
    return float(np.nanstd(values))


# ============================================================================
# DATASET VALIDATION
# ============================================================================


def load_canonical_dataset() -> pd.DataFrame:
    """
    Load and validate the canonical dataset.

    This intentionally uses an explicit path rather than dataset discovery.
    """

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    # ------------------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------------------

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected row count: {len(df)} "
            f"(expected {EXPECTED_ROWS})"
        )

    if len(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected column count: {len(df.columns)} "
            f"(expected {EXPECTED_COLUMNS})"
        )

    # ------------------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------------------

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    target_values = set(df[TARGET_COLUMN].dropna().unique())

    if not target_values.issubset({"Yes", "No"}):
        raise ValueError(
            f"Unexpected target values: {target_values}"
        )

    positive_count = int((df[TARGET_COLUMN] == "Yes").sum())
    negative_count = int((df[TARGET_COLUMN] == "No").sum())

    prevalence = positive_count / len(df)

    print(f"Target prevalence:    {prevalence:.2%}")

    if positive_count != EXPECTED_POSITIVE_COUNT:
        raise ValueError(
            f"Unexpected positive count: {positive_count} "
            f"(expected {EXPECTED_POSITIVE_COUNT})"
        )

    if negative_count != EXPECTED_NEGATIVE_COUNT:
        raise ValueError(
            f"Unexpected negative count: {negative_count} "
            f"(expected {EXPECTED_NEGATIVE_COUNT})"
        )

    if not np.isclose(
        prevalence,
        EXPECTED_PREVALENCE,
        atol=1e-9,
    ):
        raise ValueError(
            "Canonical target prevalence does not match "
            "the established 23.60% prevalence."
        )

    # ------------------------------------------------------------------------
    # Stable feature validation
    # ------------------------------------------------------------------------

    missing_features = [
        feature
        for feature in STABLE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Stable features missing from dataset: "
            + ", ".join(missing_features)
        )

    # ------------------------------------------------------------------------
    # Identifier must not be used
    # ------------------------------------------------------------------------

    if IDENTIFIER_COLUMN in STABLE_FEATURES:
        raise ValueError(
            f"Identifier column '{IDENTIFIER_COLUMN}' "
            "must not be part of the stable feature set."
        )

    return df


# ============================================================================
# PREPROCESSING
# ============================================================================


def identify_feature_types(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Identify numerical and categorical stable features."""

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

    return numerical_features, categorical_features


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessing transformer."""

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

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================


def build_model_searches(
    preprocessor: ColumnTransformer,
) -> dict[str, GridSearchCV]:
    """
    Build candidate model pipelines and hyperparameter searches.
    """

    # ------------------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------------------

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    logistic_grid = {
        "model__C": [
            0.01,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],
        "model__solver": [
            "liblinear",
            "lbfgs",
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    }

    logistic_search = GridSearchCV(
        estimator=logistic_pipeline,
        param_grid=logistic_grid,
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=2,
            random_state=RANDOM_STATE,
        ),
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=True,
    )

    # ------------------------------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------------------------------

    gradient_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    gradient_grid = {
        "model__n_estimators": [
            50,
            100,
            150,
            200,
        ],
        "model__learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1,
        ],
        "model__max_depth": [
            1,
            2,
            3,
        ],
        "model__min_samples_leaf": [
            5,
            10,
            20,
        ],
        "model__subsample": [
            0.7,
            0.85,
            1.0,
        ],
    }

    gradient_search = GridSearchCV(
        estimator=gradient_pipeline,
        param_grid=gradient_grid,
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=2,
            random_state=RANDOM_STATE,
        ),
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=True,
    )

    # ------------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------------

    random_forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=N_JOBS,
                ),
            ),
        ]
    )

    random_forest_grid = {
        "model__n_estimators": [
            200,
            400,
        ],
        "model__max_depth": [
            None,
            3,
            5,
            8,
        ],
        "model__min_samples_leaf": [
            2,
            5,
            10,
            20,
        ],
        "model__max_features": [
            "sqrt",
            "log2",
            0.7,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
    }

    random_forest_search = GridSearchCV(
        estimator=random_forest_pipeline,
        param_grid=random_forest_grid,
        scoring="roc_auc",
        cv=RepeatedStratifiedKFold(
            n_splits=N_SPLITS,
            n_repeats=2,
            random_state=RANDOM_STATE,
        ),
        n_jobs=N_JOBS,
        refit=True,
        return_train_score=True,
    )

    return {
        "Logistic Regression": logistic_search,
        "Gradient Boosting": gradient_search,
        "Random Forest": random_forest_search,
    }


# ============================================================================
# REPEATED VALIDATION
# ============================================================================


def evaluate_model_repeated_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a fitted model family using repeated stratified CV.

    The model passed here is an unfitted pipeline.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=N_JOBS,
        return_train_score=False,
    )

    return {
        "roc_auc_mean": safe_metric_mean(
            results["test_roc_auc"]
        ),
        "roc_auc_std": safe_metric_std(
            results["test_roc_auc"]
        ),
        "roc_auc_min": float(
            np.nanmin(results["test_roc_auc"])
        ),
        "roc_auc_max": float(
            np.nanmax(results["test_roc_auc"])
        ),
        "pr_auc_mean": safe_metric_mean(
            results["test_pr_auc"]
        ),
        "pr_auc_std": safe_metric_std(
            results["test_pr_auc"]
        ),
        "accuracy_mean": safe_metric_mean(
            results["test_accuracy"]
        ),
        "f1_mean": safe_metric_mean(
            results["test_f1"]
        ),
        "precision_mean": safe_metric_mean(
            results["test_precision"]
        ),
        "recall_mean": safe_metric_mean(
            results["test_recall"]
        ),
    }


# ============================================================================
# SPLIT-WISE VALIDATION
# ============================================================================


def evaluate_selected_model_splitwise(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate the optimized model across 25 repeated validation folds.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []

    for split_number, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        rows.append(
            {
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
                "accuracy": accuracy_score(
                    y_test,
                    predictions,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    print("Running stable-feature model optimization...")
    print()

    ensure_report_directory()

    # ------------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------------

    df = load_canonical_dataset()

    # ------------------------------------------------------------------------
    # Prepare target
    # ------------------------------------------------------------------------

    y = (
        df[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
        .astype(int)
    )

    X = df[STABLE_FEATURES].copy()

    numerical_features, categorical_features = (
        identify_feature_types(df)
    )

    print()
    print("Stable feature set:")
    for index, feature in enumerate(
        STABLE_FEATURES,
        start=1,
    ):
        print(f"  {index:2d}. {feature}")

    print()
    print("Numerical features:", len(numerical_features))
    print("Categorical features:", len(categorical_features))

    # ------------------------------------------------------------------------
    # Build searches
    # ------------------------------------------------------------------------

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    searches = build_model_searches(preprocessor)

    # ------------------------------------------------------------------------
    # Hyperparameter optimization
    # ------------------------------------------------------------------------

    optimization_rows = []
    best_models: dict[str, Pipeline] = {}

    print()
    print("============================================================")
    print("HYPERPARAMETER OPTIMIZATION")
    print("============================================================")

    for model_name, search in searches.items():

        print()
        print(f"Optimizing {model_name}...")

        search.fit(X, y)

        best_models[model_name] = search.best_estimator_

        optimization_rows.append(
            {
                "model": model_name,
                "best_cv_roc_auc": search.best_score_,
                "best_params": json.dumps(
                    convert_numpy_types(
                        search.best_params_
                    ),
                    sort_keys=True,
                ),
            }
        )

        print(
            f"Best ROC-AUC: {search.best_score_:.4f}"
        )

        print("Best parameters:")

        for parameter, value in search.best_params_.items():
            print(
                f"  {parameter}: {value}"
            )

    optimization_df = pd.DataFrame(
        optimization_rows
    )

    optimization_df.to_csv(
        REPORT_DIR / "hyperparameter_optimization.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Repeated validation of optimized models
    # ------------------------------------------------------------------------

    print()
    print("============================================================")
    print("REPEATED VALIDATION OF OPTIMIZED MODELS")
    print("============================================================")

    validation_rows = []

    for model_name, optimized_model in best_models.items():

        print(
            f"Evaluating {model_name}..."
        )

        # Reconstruct an unfitted pipeline using the
        # optimized parameters.
        model_clone = optimized_model

        metrics = evaluate_model_repeated_cv(
            model_clone,
            X,
            y,
        )

        validation_rows.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df = validation_df.sort_values(
        by=[
            "roc_auc_mean",
            "pr_auc_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    validation_df.insert(
        0,
        "rank",
        range(
            1,
            len(validation_df) + 1,
        ),
    )

    validation_df.to_csv(
        REPORT_DIR / "optimized_model_performance.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Select best candidate
    # ------------------------------------------------------------------------

    best_model_name = validation_df.iloc[0]["model"]

    best_model = best_models[
        best_model_name
    ]

    # ------------------------------------------------------------------------
    # Split-wise stability
    # ------------------------------------------------------------------------

    print()
    print("Evaluating selected model split-wise...")

    split_df = evaluate_selected_model_splitwise(
        best_model,
        X,
        y,
    )

    split_df.insert(
        0,
        "model",
        best_model_name,
    )

    split_df.to_csv(
        REPORT_DIR / "selected_model_split_performance.csv",
        index=False,
    )

    # ------------------------------------------------------------------------
    # Split statistics
    # ------------------------------------------------------------------------

    split_summary = {
        "model": best_model_name,
        "roc_auc_mean": float(
            split_df["roc_auc"].mean()
        ),
        "roc_auc_std": float(
            split_df["roc_auc"].std()
        ),
        "roc_auc_min": float(
            split_df["roc_auc"].min()
        ),
        "roc_auc_max": float(
            split_df["roc_auc"].max()
        ),
        "pr_auc_mean": float(
            split_df["pr_auc"].mean()
        ),
        "f1_mean": float(
            split_df["f1"].mean()
        ),
        "precision_mean": float(
            split_df["precision"].mean()
        ),
        "recall_mean": float(
            split_df["recall"].mean()
        ),
        "accuracy_mean": float(
            split_df["accuracy"].mean()
        ),
    }

    # ------------------------------------------------------------------------
    # Diagnostic flags
    # ------------------------------------------------------------------------

    diagnostic_flags: list[str] = []

    if len(validation_df) >= 2:

        roc_gap = (
            validation_df.iloc[0]["roc_auc_mean"]
            - validation_df.iloc[1]["roc_auc_mean"]
        )

        if abs(roc_gap) < 0.01:
            diagnostic_flags.append(
                "Top candidate models are separated by less than "
                "0.01 ROC-AUC, indicating a practically small "
                "performance difference."
            )

    if split_summary["roc_auc_std"] >= 0.04:
        diagnostic_flags.append(
            f"{best_model_name} shows substantial "
            "split-to-split ROC-AUC variability."
        )

    if split_summary["roc_auc_mean"] < 0.60:
        diagnostic_flags.append(
            "The selected optimized model remains below "
            "0.60 mean ROC-AUC, indicating weak predictive "
            "separation."
        )
    elif split_summary["roc_auc_mean"] < 0.65:
        diagnostic_flags.append(
            "The selected optimized model provides modest "
            "predictive separation."
        )
    else:
        diagnostic_flags.append(
            "The selected optimized model provides useful "
            "predictive separation in repeated validation."
        )

    if (
        validation_df.iloc[0]["pr_auc_mean"]
        <= EXPECTED_PREVALENCE
    ):
        diagnostic_flags.append(
            "Mean PR-AUC is close to or below the target "
            "prevalence baseline."
        )

    # ------------------------------------------------------------------------
    # Overall diagnosis
    # ------------------------------------------------------------------------

    if (
        split_summary["roc_auc_mean"] >= 0.65
        and split_summary["roc_auc_std"] < 0.04
    ):
        overall_diagnosis = (
            "The optimized stable-feature model demonstrates "
            "useful and reasonably stable predictive performance "
            "under repeated validation. It is a strong candidate "
            "for threshold optimization and final validation."
        )

    elif split_summary["roc_auc_mean"] >= 0.60:
        overall_diagnosis = (
            "The optimized stable-feature model demonstrates "
            "modest predictive performance. The stable feature "
            "set is suitable for continued threshold, calibration, "
            "and final validation analysis, but performance should "
            "not yet be considered production-ready."
        )

    else:
        overall_diagnosis = (
            "The optimized stable-feature model remains weak "
            "under repeated validation. Further model optimization "
            "alone may have limited value; feature quality and "
            "target construction should be reconsidered."
        )

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    report = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "rows": len(df),
            "columns": len(df.columns),
            "model_features": len(STABLE_FEATURES),
            "target": TARGET_COLUMN,
            "positive_count": int(y.sum()),
            "negative_count": int((y == 0).sum()),
            "target_prevalence": float(y.mean()),
        },
        "stable_features": STABLE_FEATURES,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "validation_design": {
            "folds": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_splits": N_SPLITS * N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "optimization": optimization_rows,
        "optimized_model_performance": validation_df.to_dict(
            orient="records"
        ),
        "selected_model": {
            "name": best_model_name,
            "split_summary": split_summary,
        },
        "diagnostic_flags": diagnostic_flags,
        "overall_diagnosis": overall_diagnosis,
    }

    save_json_report(report)

    # ------------------------------------------------------------------------
    # Summary text
    # ------------------------------------------------------------------------

    summary_lines = [
        "EMPLOYEE ATTRITION — STABLE FEATURE MODEL OPTIMIZATION",
        "=" * 60,
        "",
        "[DATASET]",
        f"Rows:                 {len(df)}",
        f"Columns:              {len(df.columns)}",
        f"Model features:       {len(STABLE_FEATURES)}",
        f"Target prevalence:    {y.mean():.2%}",
        "",
        "[STABLE FEATURE SET]",
    ]

    for index, feature in enumerate(
        STABLE_FEATURES,
        start=1,
    ):
        summary_lines.append(
            f"{index:2d}. {feature}"
        )

    summary_lines.extend(
        [
            "",
            "[OPTIMIZED MODEL PERFORMANCE]",
        ]
    )

    for _, row in validation_df.iterrows():

        summary_lines.append(
            f"{row['rank']}. "
            f"{row['model']:<24} "
            f"ROC-AUC={row['roc_auc_mean']:.4f} "
            f"Std={row['roc_auc_std']:.4f} "
            f"PR-AUC={row['pr_auc_mean']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "[SELECTED MODEL]",
            f"Model:                {best_model_name}",
            f"ROC-AUC mean:         {split_summary['roc_auc_mean']:.4f}",
            f"ROC-AUC std:          {split_summary['roc_auc_std']:.4f}",
            f"ROC-AUC min:          {split_summary['roc_auc_min']:.4f}",
            f"ROC-AUC max:          {split_summary['roc_auc_max']:.4f}",
            f"PR-AUC mean:          {split_summary['pr_auc_mean']:.4f}",
            f"F1 mean:              {split_summary['f1_mean']:.4f}",
            f"Precision mean:       {split_summary['precision_mean']:.4f}",
            f"Recall mean:          {split_summary['recall_mean']:.4f}",
            f"Accuracy mean:        {split_summary['accuracy_mean']:.4f}",
            "",
            "[DIAGNOSTIC FLAGS]",
        ]
    )

    if diagnostic_flags:
        for flag in diagnostic_flags:
            summary_lines.append(
                f"- {flag}"
            )
    else:
        summary_lines.append(
            "- No major diagnostic flags."
        )

    summary_lines.extend(
        [
            "",
            "[OVERALL DIAGNOSIS]",
            overall_diagnosis,
            "",
            "[OUTPUT]",
            f"Reports:              {REPORT_DIR}",
            f"Optimization CSV:     {REPORT_DIR / 'hyperparameter_optimization.csv'}",
            f"Performance CSV:      {REPORT_DIR / 'optimized_model_performance.csv'}",
            f"Split CSV:            {REPORT_DIR / 'selected_model_split_performance.csv'}",
            f"JSON report:          {REPORT_DIR / 'model_optimization_stable_report.json'}",
            f"Summary report:       {REPORT_DIR / 'model_optimization_stable_summary.txt'}",
            "",
            "============================================================",
            "STABLE MODEL OPTIMIZATION COMPLETE",
            "============================================================",
        ]
    )

    save_summary(summary_lines)

    # ------------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------------

    print()
    print("=" * 60)
    print("EMPLOYEE ATTRITION — STABLE MODEL OPTIMIZATION")
    print("=" * 60)

    print()
    print("[DATASET]")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print(f"Model features:       {len(STABLE_FEATURES)}")
    print(f"Target prevalence:    {y.mean():.2%}")

    print()
    print("[OPTIMIZED MODEL PERFORMANCE]")

    print(
        validation_df[
            [
                "rank",
                "model",
                "roc_auc_mean",
                "roc_auc_std",
                "roc_auc_min",
                "roc_auc_max",
                "pr_auc_mean",
                "f1_mean",
                "precision_mean",
                "recall_mean",
                "accuracy_mean",
            ]
        ].to_string(index=False)
    )

    print()
    print("[SELECTED MODEL]")
    print(
        f"{best_model_name}"
    )

    print(
        f"ROC-AUC:              "
        f"{split_summary['roc_auc_mean']:.4f}"
    )

    print(
        f"ROC-AUC Std:          "
        f"{split_summary['roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{split_summary['pr_auc_mean']:.4f}"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if diagnostic_flags:
        for flag in diagnostic_flags:
            print(f"- {flag}")
    else:
        print("- No major diagnostic flags.")

    print()
    print("[OVERALL DIAGNOSIS]")
    print(overall_diagnosis)

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {REPORT_DIR}"
    )
    print(
        f"Optimization CSV:     "
        f"{REPORT_DIR / 'hyperparameter_optimization.csv'}"
    )
    print(
        f"Performance CSV:      "
        f"{REPORT_DIR / 'optimized_model_performance.csv'}"
    )
    print(
        f"Split CSV:            "
        f"{REPORT_DIR / 'selected_model_split_performance.csv'}"
    )
    print(
        f"JSON report:          "
        f"{REPORT_DIR / 'model_optimization_stable_report.json'}"
    )
    print(
        f"Summary report:       "
        f"{REPORT_DIR / 'model_optimization_stable_summary.txt'}"
    )

    print()
    print("=" * 60)
    print("STABLE MODEL OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()