"""
Employee Attrition — Feature Stability Analysis

Purpose
-------
Diagnose whether feature relationships with Attrition are stable across
different train/holdout partitions.

This analysis does NOT modify the final model or untouched holdout.
It evaluates:
    1. Repeated stratified splits
    2. Feature-level correlation stability
    3. Logistic-regression coefficient stability
    4. Feature importance stability
    5. AUC stability across splits
    6. Direction consistency of feature effects

Outputs
-------
reports/signal_analysis/feature_stability/
    feature_stability_report.json
    feature_stability_summary.txt
    feature_stability.csv
    coefficient_stability.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pointbiserialr

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "feature_stability"
)

SUMMARY_PATH = OUTPUT_DIR / "feature_stability_summary.txt"
JSON_PATH = OUTPUT_DIR / "feature_stability_report.json"
FEATURE_CSV_PATH = OUTPUT_DIR / "feature_stability.csv"
COEFFICIENT_CSV_PATH = OUTPUT_DIR / "coefficient_stability.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_SPLITS = 5
N_REPEATS = 5

LOGISTIC_C = 0.01

TARGET_COLUMN = "Attrition"

ID_COLUMNS = [
    "Employee_ID",
]

DROP_COLUMNS = [
    "Employee_ID",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and prepare target."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map({"No": 0, "Yes": 1})
    )

    if y.isna().any():
        invalid = df.loc[y.isna(), TARGET_COLUMN].unique()
        raise ValueError(
            f"Unexpected target values found: {invalid}"
        )

    X = df.drop(columns=[TARGET_COLUMN])

    for column in DROP_COLUMNS:
        if column in X.columns:
            X = X.drop(columns=[column])

    return X, y.astype(int)


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """Build preprocessing pipeline."""

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
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
                    drop=None,
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


# ============================================================
# MODEL
# ============================================================

def build_model(
    X: pd.DataFrame,
) -> Pipeline:
    """Build the reference logistic regression model."""

    preprocessor = build_preprocessor(X)

    model = LogisticRegression(
        C=LOGISTIC_C,
        class_weight=None,
        max_iter=5000,
        solver="lbfgs",
        random_state=RANDOM_SEED,
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
# NUMERICAL FEATURE SIGNAL
# ============================================================

def numerical_feature_signal(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """
    Calculate point-biserial correlation for numerical features.
    """

    rows = []

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for feature in numerical_features:

        values = X[feature]

        if values.nunique(dropna=True) <= 1:
            correlation = 0.0
            p_value = 1.0
        else:
            valid = values.notna() & y.notna()

            correlation, p_value = pointbiserialr(
                y.loc[valid],
                values.loc[valid],
            )

            if not np.isfinite(correlation):
                correlation = 0.0

            if not np.isfinite(p_value):
                p_value = 1.0

        rows.append(
            {
                "feature": feature,
                "correlation": float(correlation),
                "p_value": float(p_value),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# COEFFICIENT EXTRACTION
# ============================================================

def extract_coefficients(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extract standardized logistic regression coefficients.

    Coefficients are aggregated back to their original feature names.
    For one-hot categorical variables, mean absolute coefficient is used
    as the feature-level magnitude and mean coefficient is used for
    direction.
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [
            f"feature_{i}"
            for i in range(len(model.coef_[0]))
        ]

    coefficients = model.coef_[0]

    rows = []

    for feature_name, coefficient in zip(
        feature_names,
        coefficients,
    ):

        original_name = feature_name

        if feature_name.startswith("numerical__"):
            original_name = feature_name.replace(
                "numerical__",
                "",
                1,
            )

        elif feature_name.startswith("categorical__"):
            remaining = feature_name.replace(
                "categorical__",
                "",
                1,
            )

            # Match against original categorical column names.
            matched = None

            for column in preprocessor.transformers_[1][2]:
                if remaining == column:
                    matched = column
                    break

                if remaining.startswith(f"{column}_"):
                    matched = column
                    break

            if matched is not None:
                original_name = matched
            else:
                original_name = remaining

        rows.append(
            {
                "encoded_feature": feature_name,
                "original_feature": original_name,
                "coefficient": float(coefficient),
                "absolute_coefficient": float(
                    abs(coefficient)
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FEATURE-LEVEL COEFFICIENT AGGREGATION
# ============================================================

def aggregate_coefficients(
    coefficient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate encoded coefficients to original feature level.
    """

    grouped = (
        coefficient_df
        .groupby("original_feature")
        .agg(
            coefficient_mean=(
                "coefficient",
                "mean",
            ),
            coefficient_abs_mean=(
                "absolute_coefficient",
                "mean",
            ),
            coefficient_max_abs=(
                "absolute_coefficient",
                "max",
            ),
            encoded_feature_count=(
                "encoded_feature",
                "count",
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================
# SINGLE SPLIT ANALYSIS
# ============================================================

def evaluate_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    split_number: int,
) -> tuple[dict, pd.DataFrame]:

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[validation_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[validation_idx]

    pipeline = build_model(X_train)

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_valid
    )[:, 1]

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    coefficients = extract_coefficients(
        pipeline
    )

    aggregated = aggregate_coefficients(
        coefficients
    )

    aggregated["split"] = split_number

    result = {
        "split": split_number,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "train_prevalence": float(y_train.mean()),
        "validation_prevalence": float(y_valid.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    return result, aggregated


# ============================================================
# STABILITY METRICS
# ============================================================

def calculate_stability(
    coefficient_results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate coefficient stability across repeated splits."""

    grouped = (
        coefficient_results
        .groupby("original_feature")
    )

    rows = []

    for feature, group in grouped:

        coefficients = group[
            "coefficient_mean"
        ].to_numpy()

        absolute_coefficients = group[
            "coefficient_abs_mean"
        ].to_numpy()

        positive_count = int(
            np.sum(coefficients > 0)
        )

        negative_count = int(
            np.sum(coefficients < 0)
        )

        zero_count = int(
            np.sum(np.isclose(coefficients, 0.0))
        )

        total = len(coefficients)

        if total > 0:
            direction_consistency = (
                max(
                    positive_count,
                    negative_count,
                    zero_count,
                )
                / total
            )
        else:
            direction_consistency = 0.0

        rows.append(
            {
                "feature": feature,
                "mean_coefficient": float(
                    np.mean(coefficients)
                ),
                "std_coefficient": float(
                    np.std(coefficients, ddof=1)
                )
                if len(coefficients) > 1
                else 0.0,
                "mean_absolute_coefficient": float(
                    np.mean(absolute_coefficients)
                ),
                "std_absolute_coefficient": float(
                    np.std(
                        absolute_coefficients,
                        ddof=1,
                    )
                )
                if len(absolute_coefficients) > 1
                else 0.0,
                "positive_splits": positive_count,
                "negative_splits": negative_count,
                "zero_splits": zero_count,
                "direction_consistency": float(
                    direction_consistency
                ),
                "split_count": int(total),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by="mean_absolute_coefficient",
            ascending=False,
        )

    return result


# ============================================================
# NUMERICAL CORRELATION STABILITY
# ============================================================

def calculate_correlation_stability(
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Calculate numerical signal stability across splits."""

    rows = []

    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    for feature in numerical_features:

        correlations = []

        for train_idx, _ in splits:

            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            values = X_train[feature]

            valid = values.notna() & y_train.notna()

            if (
                valid.sum() < 3
                or values.loc[valid].nunique() <= 1
            ):
                correlation = 0.0
            else:
                correlation, _ = pointbiserialr(
                    y_train.loc[valid],
                    values.loc[valid],
                )

                if not np.isfinite(correlation):
                    correlation = 0.0

            correlations.append(
                float(correlation)
            )

        correlations_array = np.array(
            correlations
        )

        positive = int(
            np.sum(correlations_array > 0)
        )

        negative = int(
            np.sum(correlations_array < 0)
        )

        same_direction = max(
            positive,
            negative,
        )

        direction_consistency = (
            same_direction
            / len(correlations_array)
        )

        rows.append(
            {
                "feature": feature,
                "mean_correlation": float(
                    np.mean(correlations_array)
                ),
                "std_correlation": float(
                    np.std(
                        correlations_array,
                        ddof=1,
                    )
                )
                if len(correlations_array) > 1
                else 0.0,
                "minimum_correlation": float(
                    np.min(correlations_array)
                ),
                "maximum_correlation": float(
                    np.max(correlations_array)
                ),
                "positive_splits": positive,
                "negative_splits": negative,
                "direction_consistency": float(
                    direction_consistency
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by="mean_correlation",
            key=lambda x: x.abs(),
            ascending=False,
        )

    return result


# ============================================================
# SPLIT GENERATION
# ============================================================

def generate_repeated_splits(
    X: pd.DataFrame,
    y: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:

    all_splits = []

    for repeat in range(N_REPEATS):

        random_state = (
            RANDOM_SEED + repeat
        )

        cv = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=random_state,
        )

        for train_idx, validation_idx in cv.split(
            X,
            y,
        ):
            all_splits.append(
                (
                    train_idx,
                    validation_idx,
                )
            )

    return all_splits


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main() -> None:

    print()
    print(
        "Running feature stability analysis..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X, y = load_dataset()

    print("Dataset loaded successfully.")
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
        "Generating repeated stratified splits..."
    )

    splits = generate_repeated_splits(
        X,
        y,
    )

    print(
        f"Splits:                {len(splits)}"
    )

    print()
    print(
        "Evaluating feature stability..."
    )

    split_results = []
    coefficient_results = []

    for index, (
        train_idx,
        validation_idx,
    ) in enumerate(splits, start=1):

        print(
            f"Split {index}/{len(splits)}"
        )

        result, coefficients = evaluate_split(
            X,
            y,
            train_idx,
            validation_idx,
            index,
        )

        split_results.append(result)

        coefficient_results.append(
            coefficients
        )

    split_df = pd.DataFrame(
        split_results
    )

    all_coefficients = pd.concat(
        coefficient_results,
        ignore_index=True,
    )

    stability_df = calculate_stability(
        all_coefficients
    )

    correlation_df = (
        calculate_correlation_stability(
            X,
            y,
            splits,
        )
    )

    # --------------------------------------------------------
    # MODEL PERFORMANCE STABILITY
    # --------------------------------------------------------

    roc_auc_values = split_df[
        "roc_auc"
    ].to_numpy()

    pr_auc_values = split_df[
        "pr_auc"
    ].to_numpy()

    roc_auc_mean = float(
        np.mean(roc_auc_values)
    )

    roc_auc_std = float(
        np.std(
            roc_auc_values,
            ddof=1,
        )
    )

    roc_auc_min = float(
        np.min(roc_auc_values)
    )

    roc_auc_max = float(
        np.max(roc_auc_values)
    )

    pr_auc_mean = float(
        np.mean(pr_auc_values)
    )

    pr_auc_std = float(
        np.std(
            pr_auc_values,
            ddof=1,
        )
    )

    pr_auc_min = float(
        np.min(pr_auc_values)
    )

    pr_auc_max = float(
        np.max(pr_auc_values)
    )

    # --------------------------------------------------------
    # DIRECTION INSTABILITY
    # --------------------------------------------------------

    unstable_coefficient_features = (
        stability_df[
            stability_df[
                "direction_consistency"
            ] < 0.70
        ]
        if not stability_df.empty
        else pd.DataFrame()
    )

    unstable_numerical_features = (
        correlation_df[
            correlation_df[
                "direction_consistency"
            ] < 0.70
        ]
        if not correlation_df.empty
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # MOST STABLE FEATURES
    # --------------------------------------------------------

    stable_features = (
        stability_df[
            stability_df[
                "direction_consistency"
            ] >= 0.90
        ]
        .sort_values(
            by="mean_absolute_coefficient",
            ascending=False,
        )
        .head(10)
        if not stability_df.empty
        else pd.DataFrame()
    )

    unstable_features = (
        stability_df[
            stability_df[
                "direction_consistency"
            ] < 0.90
        ]
        .sort_values(
            by="direction_consistency",
            ascending=True,
        )
        .head(10)
        if not stability_df.empty
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------------

    stability_df.to_csv(
        FEATURE_CSV_PATH,
        index=False,
    )

    all_coefficients.to_csv(
        COEFFICIENT_CSV_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # OVERALL DIAGNOSIS
    # --------------------------------------------------------

    flags = []

    if roc_auc_std >= 0.05:
        flags.append(
            "ROC-AUC varies substantially across repeated splits."
        )

    if (
        roc_auc_min < roc_auc_mean - 0.05
    ):
        flags.append(
            "Some validation splits produce materially weaker ROC-AUC."
        )

    if len(
        unstable_coefficient_features
    ) > 0:
        flags.append(
            "Several features show unstable coefficient direction across splits."
        )

    if len(
        unstable_numerical_features
    ) > 0:
        flags.append(
            "Several numerical features show unstable correlation direction across splits."
        )

    if not flags:
        flags.append(
            "No major feature-direction instability was detected across repeated splits."
        )

    if (
        roc_auc_std >= 0.05
        or len(unstable_coefficient_features) >= 5
        or len(unstable_numerical_features) >= 3
    ):
        overall_diagnosis = (
            "Feature and/or validation instability is present and may contribute "
            "to the observed generalization gap. Further investigation of data "
            "construction and validation sensitivity is recommended."
        )
    else:
        overall_diagnosis = (
            "Feature relationships are reasonably stable across repeated splits. "
            "The observed holdout generalization gap is therefore less likely to "
            "be caused by broad feature-direction instability alone."
        )

    # --------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------

    report = {
        "dataset": {
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target_column": TARGET_COLUMN,
            "target_prevalence": float(y.mean()),
        },
        "configuration": {
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "total_validation_splits": len(splits),
            "logistic_regression_C": LOGISTIC_C,
            "random_seed": RANDOM_SEED,
        },
        "performance_stability": {
            "roc_auc_mean": roc_auc_mean,
            "roc_auc_std": roc_auc_std,
            "roc_auc_min": roc_auc_min,
            "roc_auc_max": roc_auc_max,
            "pr_auc_mean": pr_auc_mean,
            "pr_auc_std": pr_auc_std,
            "pr_auc_min": pr_auc_min,
            "pr_auc_max": pr_auc_max,
        },
        "feature_stability": (
            stability_df.to_dict(
                orient="records"
            )
            if not stability_df.empty
            else []
        ),
        "numerical_signal_stability": (
            correlation_df.to_dict(
                orient="records"
            )
            if not correlation_df.empty
            else []
        ),
        "diagnostic_flags": flags,
        "overall_diagnosis": overall_diagnosis,
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append(
        "============================================================"
    )
    summary_lines.append(
        "EMPLOYEE ATTRITION — FEATURE STABILITY ANALYSIS"
    )
    summary_lines.append(
        "============================================================"
    )
    summary_lines.append("")

    summary_lines.append(
        "[DATASET]"
    )
    summary_lines.append(
        f"Rows:                 {len(X)}"
    )
    summary_lines.append(
        f"Features:             {X.shape[1]}"
    )
    summary_lines.append(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )
    summary_lines.append("")

    summary_lines.append(
        "[VALIDATION DESIGN]"
    )
    summary_lines.append(
        f"Folds per repeat:      {N_SPLITS}"
    )
    summary_lines.append(
        f"Repeats:               {N_REPEATS}"
    )
    summary_lines.append(
        f"Total validation:      {len(splits)}"
    )
    summary_lines.append("")

    summary_lines.append(
        "[PERFORMANCE STABILITY]"
    )
    summary_lines.append(
        f"ROC-AUC mean:          {roc_auc_mean:.4f}"
    )
    summary_lines.append(
        f"ROC-AUC std:           {roc_auc_std:.4f}"
    )
    summary_lines.append(
        f"ROC-AUC minimum:       {roc_auc_min:.4f}"
    )
    summary_lines.append(
        f"ROC-AUC maximum:       {roc_auc_max:.4f}"
    )
    summary_lines.append(
        f"PR-AUC mean:           {pr_auc_mean:.4f}"
    )
    summary_lines.append(
        f"PR-AUC std:            {pr_auc_std:.4f}"
    )
    summary_lines.append(
        f"PR-AUC minimum:        {pr_auc_min:.4f}"
    )
    summary_lines.append(
        f"PR-AUC maximum:        {pr_auc_max:.4f}"
    )
    summary_lines.append("")

    summary_lines.append(
        "[MOST STABLE FEATURES]"
    )

    if stable_features.empty:
        summary_lines.append(
            "No features met the >= 90% direction consistency criterion."
        )
    else:
        for _, row in stable_features.iterrows():
            summary_lines.append(
                f"{row['feature']:<35}"
                f"mean_coef={row['mean_coefficient']:+.4f}  "
                f"direction_consistency="
                f"{row['direction_consistency']:.2f}"
            )

    summary_lines.append("")

    summary_lines.append(
        "[POTENTIALLY UNSTABLE FEATURES]"
    )

    if unstable_features.empty:
        summary_lines.append(
            "No major coefficient-direction instability detected."
        )
    else:
        for _, row in unstable_features.iterrows():
            summary_lines.append(
                f"{row['feature']:<35}"
                f"mean_coef={row['mean_coefficient']:+.4f}  "
                f"direction_consistency="
                f"{row['direction_consistency']:.2f}"
            )

    summary_lines.append("")

    summary_lines.append(
        "[TOP NUMERICAL SIGNAL STABILITY]"
    )

    if correlation_df.empty:
        summary_lines.append(
            "No numerical features available."
        )
    else:
        for _, row in correlation_df.head(10).iterrows():
            summary_lines.append(
                f"{row['feature']:<35}"
                f"mean_r={row['mean_correlation']:+.4f}  "
                f"std={row['std_correlation']:.4f}  "
                f"direction_consistency="
                f"{row['direction_consistency']:.2f}"
            )

    summary_lines.append("")

    summary_lines.append(
        "[DIAGNOSTIC FLAGS]"
    )

    for flag in flags:
        summary_lines.append(
            f"- {flag}"
        )

    summary_lines.append("")

    summary_lines.append(
        "[OVERALL DIAGNOSIS]"
    )
    summary_lines.append(
        overall_diagnosis
    )

    summary_lines.append("")
    summary_lines.append(
        "[OUTPUT]"
    )
    summary_lines.append(
        f"Reports:              {OUTPUT_DIR}"
    )
    summary_lines.append(
        f"JSON report:          {JSON_PATH}"
    )
    summary_lines.append(
        f"Feature stability:    {FEATURE_CSV_PATH}"
    )
    summary_lines.append(
        f"Coefficient detail:   {COEFFICIENT_CSV_PATH}"
    )
    summary_lines.append(
        f"Summary report:       {SUMMARY_PATH}"
    )

    summary_lines.append("")
    summary_lines.append(
        "============================================================"
    )
    summary_lines.append(
        "FEATURE STABILITY ANALYSIS COMPLETE"
    )
    summary_lines.append(
        "============================================================"
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(summary_lines)
        )

    # --------------------------------------------------------
    # CONSOLE OUTPUT
    # --------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "EMPLOYEE ATTRITION — FEATURE STABILITY ANALYSIS"
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
    print("[VALIDATION DESIGN]")
    print(
        f"Folds per repeat:      {N_SPLITS}"
    )
    print(
        f"Repeats:               {N_REPEATS}"
    )
    print(
        f"Total validation:      {len(splits)}"
    )

    print()
    print("[PERFORMANCE STABILITY]")
    print(
        f"ROC-AUC mean:          {roc_auc_mean:.4f}"
    )
    print(
        f"ROC-AUC std:           {roc_auc_std:.4f}"
    )
    print(
        f"ROC-AUC min:           {roc_auc_min:.4f}"
    )
    print(
        f"ROC-AUC max:           {roc_auc_max:.4f}"
    )
    print(
        f"PR-AUC mean:           {pr_auc_mean:.4f}"
    )
    print(
        f"PR-AUC std:            {pr_auc_std:.4f}"
    )
    print(
        f"PR-AUC min:            {pr_auc_min:.4f}"
    )
    print(
        f"PR-AUC max:            {pr_auc_max:.4f}"
    )

    print()
    print("[TOP FEATURE STABILITY]")

    if not stability_df.empty:

        display_columns = [
            "feature",
            "mean_coefficient",
            "std_coefficient",
            "direction_consistency",
        ]

        print(
            stability_df[
                display_columns
            ]
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "mean_coefficient": "{:.4f}".format,
                    "std_coefficient": "{:.4f}".format,
                    "direction_consistency": "{:.2f}".format,
                },
            )
        )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(
            f"- {flag}"
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(
        overall_diagnosis
    )

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {OUTPUT_DIR}"
    )
    print(
        f"JSON report:          {JSON_PATH}"
    )
    print(
        f"Feature stability:    {FEATURE_CSV_PATH}"
    )
    print(
        f"Coefficient detail:   {COEFFICIENT_CSV_PATH}"
    )
    print(
        f"Summary report:       {SUMMARY_PATH}"
    )

    print()
    print(
        "============================================================"
    )
    print(
        "FEATURE STABILITY ANALYSIS COMPLETE"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()