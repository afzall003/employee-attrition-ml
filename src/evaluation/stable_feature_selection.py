"""
Stable Feature Selection
========================

Employee Attrition ML Project

Purpose
-------
Identify features that remain consistently useful across repeated
stratified validation splits.

This analysis is intentionally diagnostic. It does NOT permanently
remove features from the project dataset.

Outputs
-------
reports/signal_analysis/stable_feature_selection/
    stable_feature_selection_report.json
    feature_stability.csv
    feature_importance.csv
    subset_performance.csv
    stable_feature_selection_summary.txt

Canonical dataset
-----------------
data/raw/employee_attrition_dataset_v2.csv
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "employee_attrition_dataset_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "stable_feature_selection"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attrition"
POSITIVE_LABEL = "Yes"

N_SPLITS = 5
N_REPEATS = 5
RANDOM_STATE = 42

PERMUTATION_REPEATS = 10

# Feature subset sizes to investigate.
SUBSET_SIZES = [3, 5, 7, 10, 15, 20, 24]

# Selection threshold.
#
# A feature is considered "selected" on a split when its
# permutation importance is greater than or equal to the
# 75th percentile of the positive permutation importances
# for that split.
#
# This prevents tiny/noisy importances from automatically
# being treated as useful.
SELECTION_QUANTILE = 0.75

# Minimum frequency to classify a feature as stable.
STABLE_FREQUENCY_THRESHOLD = 0.70

# Importance threshold below which a feature is considered weak.
WEAK_IMPORTANCE_THRESHOLD = 0.001


# ============================================================
# GENERAL HELPERS
# ============================================================

def make_json_serializable(value: Any) -> Any:
    """
    Recursively convert NumPy/Pandas objects into native Python
    objects suitable for json.dump().
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serializable(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(v)
            for v in value.tolist()
        ]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Interval):
        return str(value)

    if isinstance(value, pd.Categorical):
        return value.astype(str).tolist()

    if isinstance(value, Path):
        return str(value)

    if pd.isna(value):
        return None

    return value


def save_json_report(report: dict, path: Path) -> None:
    """
    Save a JSON report after converting all objects to native
    Python representations.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    clean_report = make_json_serializable(report)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            clean_report,
            file,
            indent=2,
            ensure_ascii=False,
        )


def safe_float(value: Any) -> float | None:
    """
    Convert a numeric value to a JSON-safe Python float.
    """

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


# ============================================================
# DATASET
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load and validate the canonical dataset.
    """

    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    target_values = df[TARGET_COLUMN].astype(str).str.strip()

    unexpected = set(target_values.unique()) - {"Yes", "No"}

    if unexpected:
        raise ValueError(
            "Unexpected target values detected: "
            f"{sorted(unexpected)}"
        )

    y = (target_values == POSITIVE_LABEL).astype(int)

    X = df.drop(columns=[TARGET_COLUMN]).copy()

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Features:             {X.shape[1]}")
    print(
        f"Target prevalence:    "
        f"{y.mean() * 100:.2f}%"
    )

    return X, y


# ============================================================
# FEATURE TYPES
# ============================================================

def detect_feature_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Identify numerical and categorical columns.
    """

    numerical_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.

    Numerical:
        median imputation
        standardization

    Categorical:
        most-frequent imputation
        one-hot encoding
    """

    from sklearn.impute import SimpleImputer

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
                    drop="first",
                ),
            ),
        ]
    )

    transformers = []

    if numerical_columns:
        transformers.append(
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODELS
# ============================================================

def build_logistic_model(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """
    Logistic Regression diagnostic model.
    """

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    model = LogisticRegression(
        max_iter=3000,
        C=1.0,
        class_weight=None,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_gradient_boosting_model(
    numerical_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    """
    Gradient Boosting diagnostic model.

    Gradient Boosting requires numerical encoded input,
    so the categorical preprocessing is handled through
    OneHotEncoder.
    """

    preprocessor = build_preprocessor(
        numerical_columns,
        categorical_columns,
    )

    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def calculate_feature_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Calculate permutation importance at the ORIGINAL FEATURE level.

    This is important because one categorical feature may generate
    multiple one-hot columns. We aggregate those transformed
    columns back to the original feature where possible.
    """

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_mean = result.importances_mean
    importance_std = result.importances_std

    feature_names = X_test.columns.tolist()

    if len(feature_names) != len(importance_mean):
        raise RuntimeError(
            "Permutation importance did not return one importance "
            "value per original feature."
        )

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
        }
    )


# ============================================================
# FEATURE SELECTION RULE
# ============================================================

def select_features_from_importance(
    importance_df: pd.DataFrame,
    total_features: int,
) -> set[str]:
    """
    Select features using positive permutation importance.

    Selection logic:

    1. Keep features with positive importance.
    2. If there are enough positive features, use the upper
       quantile of positive importance as the selection threshold.
    3. Always retain at least 3 features where possible.
    """

    work = importance_df.copy()

    positive = work[
        work["importance_mean"] > 0
    ].copy()

    if positive.empty:
        ranked = (
            work.sort_values(
                "importance_mean",
                ascending=False,
            )
            .head(min(3, total_features))
        )

        return set(ranked["feature"].tolist())

    threshold = positive[
        "importance_mean"
    ].quantile(SELECTION_QUANTILE)

    selected = positive[
        positive["importance_mean"] >= threshold
    ]

    # Prevent an unexpectedly tiny subset.
    if len(selected) < min(3, total_features):
        selected = positive.nlargest(
            min(3, total_features),
            "importance_mean",
        )

    return set(selected["feature"].tolist())


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[float, float]:

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    return roc_auc, pr_auc


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_stability_analysis(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    numerical_columns, categorical_columns = detect_feature_types(X)

    feature_names = X.columns.tolist()

    print()
    print("Generating repeated validation splits...")
    print(
        f"Splits:               "
        f"{N_SPLITS * N_REPEATS}"
    )

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    total_splits = N_SPLITS * N_REPEATS

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    selection_counts = {
        feature: 0
        for feature in feature_names
    }

    importance_records = []

    coefficient_records = []

    split_metrics = []

    subset_records = []

    # --------------------------------------------------------
    # Repeated validation
    # --------------------------------------------------------

    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y),
        start=1,
    ):

        print(
            f"Split {split_number}/{total_splits}"
        )

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        logistic_model = build_logistic_model(
            numerical_columns,
            categorical_columns,
        )

        logistic_model.fit(
            X_train,
            y_train,
        )

        logistic_prob = logistic_model.predict_proba(
            X_test
        )[:, 1]

        logistic_roc = roc_auc_score(
            y_test,
            logistic_prob,
        )

        logistic_pr = average_precision_score(
            y_test,
            logistic_prob,
        )

        split_metrics.append(
            {
                "split": split_number,
                "model": "Logistic Regression",
                "roc_auc": logistic_roc,
                "pr_auc": logistic_pr,
            }
        )

        # ----------------------------------------------------
        # Logistic permutation importance
        # ----------------------------------------------------

        logistic_importance = (
            calculate_feature_permutation_importance(
                logistic_model,
                X_test,
                y_test,
            )
        )

        logistic_importance[
            "split"
        ] = split_number

        logistic_importance[
            "model"
        ] = "Logistic Regression"

        importance_records.append(
            logistic_importance
        )

        selected_features = (
            select_features_from_importance(
                logistic_importance[
                    [
                        "feature",
                        "importance_mean",
                    ]
                ],
                len(feature_names),
            )
        )

        for feature in selected_features:
            selection_counts[feature] += 1

        # ----------------------------------------------------
        # Logistic coefficient direction
        # ----------------------------------------------------

        try:
            transformed_names = (
                logistic_model
                .named_steps["preprocessor"]
                .get_feature_names_out()
            )

            coefficients = (
                logistic_model
                .named_steps["model"]
                .coef_[0]
            )

            coefficient_records.extend(
                extract_original_feature_coefficients(
                    transformed_names,
                    coefficients,
                    split_number,
                )
            )

        except Exception as exc:
            warnings.warn(
                "Could not extract Logistic Regression "
                f"coefficients on split {split_number}: {exc}"
            )

        # ====================================================
        # GRADIENT BOOSTING
        # ====================================================

        gb_model = build_gradient_boosting_model(
            numerical_columns,
            categorical_columns,
        )

        gb_model.fit(
            X_train,
            y_train,
        )

        gb_prob = gb_model.predict_proba(
            X_test
        )[:, 1]

        gb_roc = roc_auc_score(
            y_test,
            gb_prob,
        )

        gb_pr = average_precision_score(
            y_test,
            gb_prob,
        )

        split_metrics.append(
            {
                "split": split_number,
                "model": "Gradient Boosting",
                "roc_auc": gb_roc,
                "pr_auc": gb_pr,
            }
        )

        # ----------------------------------------------------
        # Gradient Boosting permutation importance
        # ----------------------------------------------------

        gb_importance = (
            calculate_feature_permutation_importance(
                gb_model,
                X_test,
                y_test,
            )
        )

        gb_importance[
            "split"
        ] = split_number

        gb_importance[
            "model"
        ] = "Gradient Boosting"

        importance_records.append(
            gb_importance
        )

        # ====================================================
        # FEATURE SUBSET PERFORMANCE
        # ====================================================

        # Rank features according to Logistic Regression
        # permutation importance on this split.
        ranking = (
            logistic_importance
            .sort_values(
                "importance_mean",
                ascending=False,
            )["feature"]
            .tolist()
        )

        for subset_size in SUBSET_SIZES:

            subset_size = min(
                subset_size,
                len(feature_names),
            )

            subset_features = ranking[
                :subset_size
            ]

            X_train_subset = X_train[
                subset_features
            ]

            X_test_subset = X_test[
                subset_features
            ]

            subset_num, subset_cat = detect_feature_types(
                X_train_subset
            )

            # Logistic Regression
            subset_lr = build_logistic_model(
                subset_num,
                subset_cat,
            )

            lr_roc, lr_pr = evaluate_model(
                subset_lr,
                X_train_subset,
                X_test_subset,
                y_train,
                y_test,
            )

            subset_records.append(
                {
                    "split": split_number,
                    "model": "Logistic Regression",
                    "feature_count": subset_size,
                    "features": "|".join(
                        subset_features
                    ),
                    "roc_auc": lr_roc,
                    "pr_auc": lr_pr,
                }
            )

            # Gradient Boosting
            subset_gb = build_gradient_boosting_model(
                subset_num,
                subset_cat,
            )

            gb_subset_roc, gb_subset_pr = (
                evaluate_model(
                    subset_gb,
                    X_train_subset,
                    X_test_subset,
                    y_train,
                    y_test,
                )
            )

            subset_records.append(
                {
                    "split": split_number,
                    "model": "Gradient Boosting",
                    "feature_count": subset_size,
                    "features": "|".join(
                        subset_features
                    ),
                    "roc_auc": gb_subset_roc,
                    "pr_auc": gb_subset_pr,
                }
            )

    # ========================================================
    # DATAFRAMES
    # ========================================================

    importance_df = pd.concat(
        importance_records,
        ignore_index=True,
    )

    coefficient_df = pd.DataFrame(
        coefficient_records
    )

    metrics_df = pd.DataFrame(
        split_metrics
    )

    subset_df = pd.DataFrame(
        subset_records
    )

    # ========================================================
    # AGGREGATE FEATURE STABILITY
    # ========================================================

    feature_summary_records = []

    for feature in feature_names:

        feature_importance = importance_df[
            importance_df["feature"] == feature
        ]

        lr_importance = feature_importance[
            feature_importance["model"]
            == "Logistic Regression"
        ]

        gb_importance = feature_importance[
            feature_importance["model"]
            == "Gradient Boosting"
        ]

        selection_frequency = (
            selection_counts[feature]
            / total_splits
        )

        mean_importance = (
            feature_importance[
                "importance_mean"
            ].mean()
        )

        std_importance = (
            feature_importance[
                "importance_mean"
            ].std(ddof=1)
        )

        positive_importance_frequency = (
            (
                feature_importance[
                    "importance_mean"
                ] > 0
            ).mean()
        )

        # Logistic coefficient stability.
        feature_coefficients = coefficient_df[
            coefficient_df["feature"] == feature
        ]

        if not feature_coefficients.empty:

            coefficient_mean = (
                feature_coefficients[
                    "coefficient"
                ].mean()
            )

            coefficient_std = (
                feature_coefficients[
                    "coefficient"
                ].std(ddof=1)
            )

            positive_direction = (
                feature_coefficients[
                    "coefficient"
                ] > 0
            ).mean()

            negative_direction = (
                feature_coefficients[
                    "coefficient"
                ] < 0
            ).mean()

            direction_consistency = max(
                positive_direction,
                negative_direction,
            )

            if coefficient_mean > 0:
                dominant_direction = "positive"

            elif coefficient_mean < 0:
                dominant_direction = "negative"

            else:
                dominant_direction = "neutral"

        else:

            coefficient_mean = np.nan
            coefficient_std = np.nan
            direction_consistency = np.nan
            dominant_direction = "unavailable"

        # Model-specific importance.
        lr_mean_importance = (
            lr_importance[
                "importance_mean"
            ].mean()
            if not lr_importance.empty
            else np.nan
        )

        gb_mean_importance = (
            gb_importance[
                "importance_mean"
            ].mean()
            if not gb_importance.empty
            else np.nan
        )

        feature_summary_records.append(
            {
                "feature": feature,
                "selection_frequency": (
                    selection_frequency
                ),
                "mean_permutation_importance": (
                    mean_importance
                ),
                "std_permutation_importance": (
                    std_importance
                ),
                "positive_importance_frequency": (
                    positive_importance_frequency
                ),
                "logistic_mean_importance": (
                    lr_mean_importance
                ),
                "gradient_boosting_mean_importance": (
                    gb_mean_importance
                ),
                "coefficient_mean": (
                    coefficient_mean
                ),
                "coefficient_std": (
                    coefficient_std
                ),
                "direction_consistency": (
                    direction_consistency
                ),
                "dominant_direction": (
                    dominant_direction
                ),
            }
        )

    feature_stability_df = pd.DataFrame(
        feature_summary_records
    )

    # ========================================================
    # STABILITY SCORE
    # ========================================================

    # Rank normalized importance.
    max_importance = (
        feature_stability_df[
            "mean_permutation_importance"
        ].abs().max()
    )

    if (
        pd.isna(max_importance)
        or max_importance == 0
    ):
        normalized_importance = 0.0

        feature_stability_df[
            "normalized_importance"
        ] = 0.0

    else:

        feature_stability_df[
            "normalized_importance"
        ] = (
            feature_stability_df[
                "mean_permutation_importance"
            ].clip(lower=0)
            / max_importance
        )

    feature_stability_df[
        "stability_score"
    ] = (
        0.50
        * feature_stability_df[
            "selection_frequency"
        ].fillna(0)
        + 0.30
        * feature_stability_df[
            "positive_importance_frequency"
        ].fillna(0)
        + 0.20
        * feature_stability_df[
            "normalized_importance"
        ].fillna(0)
    )

    feature_stability_df[
        "stability_class"
    ] = "weak"

    feature_stability_df.loc[
        feature_stability_df[
            "selection_frequency"
        ] >= STABLE_FREQUENCY_THRESHOLD,
        "stability_class",
    ] = "stable"

    feature_stability_df.loc[
        (
            feature_stability_df[
                "selection_frequency"
            ] >= 0.40
        )
        & (
            feature_stability_df[
                "selection_frequency"
            ] < STABLE_FREQUENCY_THRESHOLD
        ),
        "stability_class",
    ] = "moderately_stable"

    feature_stability_df = (
        feature_stability_df
        .sort_values(
            [
                "stability_score",
                "mean_permutation_importance",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SUBSET SUMMARY
    # ========================================================

    subset_summary_df = (
        subset_df
        .groupby(
            [
                "model",
                "feature_count",
            ]
        )
        .agg(
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            roc_auc_min=(
                "roc_auc",
                "min",
            ),
            roc_auc_max=(
                "roc_auc",
                "max",
            ),
            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),
            pr_auc_std=(
                "pr_auc",
                "std",
            ),
        )
        .reset_index()
    )

    # ========================================================
    # BASELINE SUMMARY
    # ========================================================

    baseline_summary = (
        metrics_df
        .groupby("model")
        .agg(
            roc_auc_mean=(
                "roc_auc",
                "mean",
            ),
            roc_auc_std=(
                "roc_auc",
                "std",
            ),
            roc_auc_min=(
                "roc_auc",
                "min",
            ),
            roc_auc_max=(
                "roc_auc",
                "max",
            ),
            pr_auc_mean=(
                "pr_auc",
                "mean",
            ),
            pr_auc_std=(
                "pr_auc",
                "std",
            ),
        )
        .reset_index()
    )

    return (
        feature_stability_df,
        importance_df,
        subset_summary_df,
        {
            "metrics_df": metrics_df,
            "subset_df": subset_df,
            "baseline_summary": baseline_summary,
        },
    )


# ============================================================
# COEFFICIENT EXTRACTION
# ============================================================

def extract_original_feature_coefficients(
    transformed_names: np.ndarray,
    coefficients: np.ndarray,
    split_number: int,
) -> list[dict]:

    records = []

    for transformed_name, coefficient in zip(
        transformed_names,
        coefficients,
    ):

        transformed_name = str(
            transformed_name
        )

        # ColumnTransformer names normally look like:
        #
        # num__Age
        # cat__Job_Role_Manager
        #
        # Recover original feature.

        if "__" in transformed_name:
            raw_name = transformed_name.split(
                "__",
                1,
            )[1]
        else:
            raw_name = transformed_name

        # For categorical one-hot variables, match against
        # the original feature prefix where possible.
        #
        # This function keeps the transformed coefficient
        # as a separate observation. The feature-level
        # aggregation later groups by the recovered name.

        original_feature = raw_name

        records.append(
            {
                "split": split_number,
                "transformed_feature": (
                    transformed_name
                ),
                "feature": original_feature,
                "coefficient": float(
                    coefficient
                ),
            }
        )

    return records


# ============================================================
# REPORT GENERATION
# ============================================================

def build_diagnostic_flags(
    feature_stability_df: pd.DataFrame,
    subset_summary_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> list[str]:

    flags = []

    stable_count = int(
        (
            feature_stability_df[
                "selection_frequency"
            ]
            >= STABLE_FREQUENCY_THRESHOLD
        ).sum()
    )

    weak_count = int(
        (
            feature_stability_df[
                "mean_permutation_importance"
            ]
            <= WEAK_IMPORTANCE_THRESHOLD
        ).sum()
    )

    if stable_count == 0:

        flags.append(
            "No feature reaches the 70% repeated-selection "
            "frequency threshold."
        )

    else:

        flags.append(
            f"{stable_count} feature(s) are selected on at "
            f"least {STABLE_FREQUENCY_THRESHOLD:.0%} of "
            "validation splits."
        )

    if weak_count > 0:

        flags.append(
            f"{weak_count} feature(s) have near-zero or "
            "negative mean permutation importance."
        )

    # --------------------------------------------------------
    # Compare subset performance
    # --------------------------------------------------------

    for model_name in [
        "Logistic Regression",
        "Gradient Boosting",
    ]:

        model_subsets = subset_summary_df[
            subset_summary_df["model"]
            == model_name
        ]

        if model_subsets.empty:
            continue

        best_row = model_subsets.loc[
            model_subsets[
                "roc_auc_mean"
            ].idxmax()
        ]

        full_rows = model_subsets[
            model_subsets[
                "feature_count"
            ]
            == model_subsets[
                "feature_count"
            ].max()
        ]

        if full_rows.empty:
            continue

        full_auc = float(
            full_rows.iloc[0][
                "roc_auc_mean"
            ]
        )

        best_auc = float(
            best_row["roc_auc_mean"]
        )

        best_count = int(
            best_row["feature_count"]
        )

        if best_count < len(
            feature_stability_df
        ) and best_auc > full_auc + 0.005:

            flags.append(
                f"{model_name}: the best reduced feature "
                f"subset ({best_count} features) outperforms "
                "the full feature set in repeated validation."
            )

        elif abs(best_auc - full_auc) <= 0.005:

            flags.append(
                f"{model_name}: reduced feature subsets perform "
                "within 0.005 ROC-AUC of the full feature set."
            )

    return flags


def generate_diagnosis(
    feature_stability_df: pd.DataFrame,
    subset_summary_df: pd.DataFrame,
) -> str:

    stable_features = feature_stability_df[
        feature_stability_df[
            "selection_frequency"
        ] >= STABLE_FREQUENCY_THRESHOLD
    ]

    if stable_features.empty:

        return (
            "No feature demonstrates sufficiently strong "
            "selection stability across repeated validation "
            "splits. The predictive signal appears distributed "
            "and/or noisy, so aggressive feature elimination "
            "is not currently justified."
        )

    stable_count = len(
        stable_features
    )

    total_features = len(
        feature_stability_df
    )

    # Determine whether a reduced subset is competitive.
    reduced_competitive = False

    for model_name in [
        "Logistic Regression",
        "Gradient Boosting",
    ]:

        rows = subset_summary_df[
            subset_summary_df["model"]
            == model_name
        ]

        if rows.empty:
            continue

        full_row = rows.loc[
            rows["feature_count"].idxmax()
        ]

        best_row = rows.loc[
            rows["roc_auc_mean"].idxmax()
        ]

        if (
            best_row["feature_count"]
            < full_row["feature_count"]
            and best_row["roc_auc_mean"]
            >= full_row["roc_auc_mean"] - 0.005
        ):
            reduced_competitive = True

    if reduced_competitive:

        return (
            f"{stable_count} of {total_features} features show "
            "strong repeated-selection stability. At least one "
            "reduced feature subset performs competitively with "
            "the full feature set, suggesting that a smaller "
            "stable feature set may improve interpretability "
            "without materially sacrificing predictive "
            "performance."
        )

    return (
        f"{stable_count} of {total_features} features show "
        "strong repeated-selection stability, but the evidence "
        "does not justify aggressive feature reduction. "
        "Predictive information appears distributed across "
        "multiple features, and feature selection should "
        "therefore remain conservative."
    )


def write_summary_report(
    path: Path,
    X: pd.DataFrame,
    y: pd.Series,
    feature_stability_df: pd.DataFrame,
    subset_summary_df: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    flags: list[str],
    diagnosis: str,
) -> None:

    stable_features = feature_stability_df[
        feature_stability_df[
            "selection_frequency"
        ] >= STABLE_FREQUENCY_THRESHOLD
    ]

    weak_features = feature_stability_df[
        feature_stability_df[
            "mean_permutation_importance"
        ] <= WEAK_IMPORTANCE_THRESHOLD
    ]

    lines = []

    lines.append(
        "=" * 60
    )
    lines.append(
        "EMPLOYEE ATTRITION — STABLE FEATURE SELECTION"
    )
    lines.append(
        "=" * 60
    )
    lines.append("")

    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 {len(X)}"
    )
    lines.append(
        f"Features:             {X.shape[1]}"
    )
    lines.append(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )
    lines.append("")

    lines.append("[VALIDATION DESIGN]")
    lines.append(
        f"Folds per repeat:      {N_SPLITS}"
    )
    lines.append(
        f"Repeats:               {N_REPEATS}"
    )
    lines.append(
        f"Total validation:      "
        f"{N_SPLITS * N_REPEATS}"
    )
    lines.append(
        f"Permutation repeats:   "
        f"{PERMUTATION_REPEATS}"
    )
    lines.append("")

    lines.append("[STABLE FEATURES]")
    lines.append(
        f"Threshold:             "
        f"{STABLE_FREQUENCY_THRESHOLD:.0%}"
    )
    lines.append(
        f"Stable feature count:  "
        f"{len(stable_features)}"
    )
    lines.append("")

    if stable_features.empty:

        lines.append(
            "No features reached the stability threshold."
        )

    else:

        display_columns = [
            "feature",
            "selection_frequency",
            "mean_permutation_importance",
            "std_permutation_importance",
            "direction_consistency",
            "stability_score",
        ]

        table = stable_features[
            display_columns
        ].copy()

        for _, row in table.iterrows():

            lines.append(
                f"{row['feature']:<38}"
                f"selection={row['selection_frequency']:.2f}  "
                f"importance="
                f"{row['mean_permutation_importance']:.4f}  "
                f"stability="
                f"{row['stability_score']:.3f}"
            )

    lines.append("")

    lines.append("[WEAK / UNSTABLE FEATURES]")

    if weak_features.empty:

        lines.append(
            "No features meet the weak-importance threshold."
        )

    else:

        for _, row in weak_features.iterrows():

            lines.append(
                f"{row['feature']:<38}"
                f"selection={row['selection_frequency']:.2f}  "
                f"importance="
                f"{row['mean_permutation_importance']:.4f}"
            )

    lines.append("")

    lines.append("[BASELINE PERFORMANCE]")

    for _, row in baseline_summary.iterrows():

        lines.append(
            f"{row['model']:<25}"
            f"ROC-AUC={row['roc_auc_mean']:.4f} "
            f"Std={row['roc_auc_std']:.4f} "
            f"PR-AUC={row['pr_auc_mean']:.4f}"
        )

    lines.append("")

    lines.append("[SUBSET PERFORMANCE]")

    for model_name in [
        "Logistic Regression",
        "Gradient Boosting",
    ]:

        rows = subset_summary_df[
            subset_summary_df["model"]
            == model_name
        ]

        if rows.empty:
            continue

        lines.append("")
        lines.append(model_name)

        for _, row in rows.iterrows():

            lines.append(
                f"  Features={int(row['feature_count']):2d} "
                f"ROC-AUC={row['roc_auc_mean']:.4f} "
                f"Std={row['roc_auc_std']:.4f} "
                f"PR-AUC={row['pr_auc_mean']:.4f}"
            )

    lines.append("")

    lines.append("[DIAGNOSTIC FLAGS]")

    if flags:

        for flag in flags:
            lines.append(
                f"- {flag}"
            )

    else:

        lines.append(
            "- No major diagnostic flags."
        )

    lines.append("")

    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(diagnosis)
    lines.append("")

    lines.append("[OUTPUT]")
    lines.append(
        f"Reports:              {OUTPUT_DIR}"
    )
    lines.append(
        f"Feature stability:    "
        f"{OUTPUT_DIR / 'feature_stability.csv'}"
    )
    lines.append(
        f"Feature importance:   "
        f"{OUTPUT_DIR / 'feature_importance.csv'}"
    )
    lines.append(
        f"Subset performance:   "
        f"{OUTPUT_DIR / 'subset_performance.csv'}"
    )
    lines.append(
        f"JSON report:          "
        f"{OUTPUT_DIR / 'stable_feature_selection_report.json'}"
    )
    lines.append(
        f"Summary report:       "
        f"{path}"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Running stable feature selection..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y = load_dataset()

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------

    (
        feature_stability_df,
        importance_df,
        subset_summary_df,
        additional_results,
    ) = run_stability_analysis(
        X,
        y,
    )

    metrics_df = additional_results[
        "metrics_df"
    ]

    subset_df = additional_results[
        "subset_df"
    ]

    baseline_summary = additional_results[
        "baseline_summary"
    ]

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print(
        "Generating diagnostic flags..."
    )

    flags = build_diagnostic_flags(
        feature_stability_df,
        subset_summary_df,
        baseline_summary,
    )

    diagnosis = generate_diagnosis(
        feature_stability_df,
        subset_summary_df,
    )

    # --------------------------------------------------------
    # Save CSV reports
    # --------------------------------------------------------

    feature_stability_path = (
        OUTPUT_DIR
        / "feature_stability.csv"
    )

    feature_importance_path = (
        OUTPUT_DIR
        / "feature_importance.csv"
    )

    subset_performance_path = (
        OUTPUT_DIR
        / "subset_performance.csv"
    )

    feature_stability_df.to_csv(
        feature_stability_path,
        index=False,
    )

    importance_df.to_csv(
        feature_importance_path,
        index=False,
    )

    subset_df.to_csv(
        subset_performance_path,
        index=False,
    )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    stable_features = (
        feature_stability_df[
            feature_stability_df[
                "selection_frequency"
            ]
            >= STABLE_FREQUENCY_THRESHOLD
        ]["feature"]
        .tolist()
    )

    top_features = (
        feature_stability_df
        .head(10)["feature"]
        .tolist()
    )

    report = {
        "analysis": {
            "name": (
                "stable_feature_selection"
            ),
            "version": "1.0",
        },

        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(X),
            "features": X.shape[1],
            "target": TARGET_COLUMN,
            "positive_label": POSITIVE_LABEL,
            "positive_count": int(y.sum()),
            "negative_count": int(
                len(y) - y.sum()
            ),
            "target_prevalence": float(
                y.mean()
            ),
        },

        "validation": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_splits": (
                N_SPLITS * N_REPEATS
            ),
            "permutation_repeats": (
                PERMUTATION_REPEATS
            ),
            "random_state": RANDOM_STATE,
        },

        "selection": {
            "frequency_threshold": (
                STABLE_FREQUENCY_THRESHOLD
            ),
            "stable_feature_count": (
                len(stable_features)
            ),
            "stable_features": (
                stable_features
            ),
            "top_features": top_features,
        },

        "baseline_performance": (
            baseline_summary
            .to_dict(orient="records")
        ),

        "subset_performance": (
            subset_summary_df
            .to_dict(orient="records")
        ),

        "diagnostic_flags": flags,

        "overall_diagnosis": diagnosis,
    }

    json_path = (
        OUTPUT_DIR
        / "stable_feature_selection_report.json"
    )

    save_json_report(
        report,
        json_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "stable_feature_selection_summary.txt"
    )

    write_summary_report(
        summary_path,
        X,
        y,
        feature_stability_df,
        subset_summary_df,
        baseline_summary,
        flags,
        diagnosis,
    )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — "
        "STABLE FEATURE SELECTION"
    )
    print("=" * 60)
    print()

    print("[DATASET]")
    print(
        f"Rows:                 {len(X)}"
    )
    print(
        f"Features:             {X.shape[1]}"
    )
    print(
        f"Target prevalence:    "
        f"{y.mean() * 100:.2f}%"
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
        f"Total validation:      "
        f"{N_SPLITS * N_REPEATS}"
    )
    print()

    print("[STABLE FEATURES]")

    if stable_features:

        display = feature_stability_df[
            feature_stability_df[
                "feature"
            ].isin(stable_features)
        ]

        display = display.sort_values(
            "stability_score",
            ascending=False,
        )

        for _, row in display.iterrows():

            print(
                f"{row['feature']:<38}"
                f"Selection="
                f"{row['selection_frequency']:.2f} "
                f"Importance="
                f"{row['mean_permutation_importance']:.4f} "
                f"Stability="
                f"{row['stability_score']:.3f}"
            )

    else:

        print(
            "No features reached the stability threshold."
        )

    print()

    print("[BASELINE PERFORMANCE]")

    for _, row in baseline_summary.iterrows():

        print(
            f"{row['model']:<25}"
            f"ROC-AUC="
            f"{row['roc_auc_mean']:.4f} "
            f"Std="
            f"{row['roc_auc_std']:.4f} "
            f"PR-AUC="
            f"{row['pr_auc_mean']:.4f}"
        )

    print()

    print("[SUBSET PERFORMANCE]")

    for model_name in [
        "Logistic Regression",
        "Gradient Boosting",
    ]:

        rows = subset_summary_df[
            subset_summary_df["model"]
            == model_name
        ]

        if rows.empty:
            continue

        best_row = rows.loc[
            rows["roc_auc_mean"].idxmax()
        ]

        print(
            f"{model_name:<25}"
            f"Best Features="
            f"{int(best_row['feature_count']):2d} "
            f"ROC-AUC="
            f"{best_row['roc_auc_mean']:.4f} "
            f"Std="
            f"{best_row['roc_auc_std']:.4f} "
            f"PR-AUC="
            f"{best_row['pr_auc_mean']:.4f}"
        )

    print()

    print("[DIAGNOSTIC FLAGS]")

    if flags:

        for flag in flags:
            print(f"- {flag}")

    else:

        print(
            "- No major diagnostic flags."
        )

    print()

    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)
    print()

    print("[OUTPUT]")
    print(
        f"Reports:              {OUTPUT_DIR}"
    )
    print(
        f"Feature stability:    "
        f"{feature_stability_path}"
    )
    print(
        f"Feature importance:   "
        f"{feature_importance_path}"
    )
    print(
        f"Subset performance:   "
        f"{subset_performance_path}"
    )
    print(
        f"JSON report:          "
        f"{json_path}"
    )
    print(
        f"Summary report:       "
        f"{summary_path}"
    )

    print()
    print(
        "=" * 60
    )
    print(
        "STABLE FEATURE SELECTION COMPLETE"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()