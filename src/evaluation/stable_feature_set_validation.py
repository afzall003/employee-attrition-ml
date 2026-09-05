"""
Stable Feature Set Validation
=============================

Validates whether the reduced feature subset identified by
stable_feature_selection.py provides genuinely stable predictive
performance compared with the full canonical feature set.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

Outputs:
    reports/signal_analysis/stable_feature_set_validation/
        stable_feature_set_validation_report.json
        feature_selection_summary.csv
        model_performance.csv
        split_performance.csv
        stable_feature_set_validation_summary.txt
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "employee_attrition_dataset_v2.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "stable_feature_set_validation"
)

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 5

SUBSET_SIZE = 10

TARGET_COLUMN = "Attrition"

# Columns that must NEVER be used as predictive features.
IDENTIFIER_COLUMNS = {
    "Employee_ID",
    "EmployeeID",
    "Employee_Id",
    "ID",
    "Id",
}

# Columns that should also be excluded if they appear.
NON_FEATURE_COLUMNS = {
    TARGET_COLUMN,
    *IDENTIFIER_COLUMNS,
}


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load and validate the canonical dataset."""

    print("Loading canonical dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")

    return df


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def identify_features(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Identify model features and split them into numerical/categorical.

    Employee_ID and other identifiers are explicitly excluded.
    """

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    numerical_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_features = [
        column
        for column in feature_columns
        if column not in numerical_features
    ]

    return (
        feature_columns,
        numerical_features,
        categorical_features,
    )


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(series: pd.Series) -> np.ndarray:
    """Convert Attrition values to binary 0/1."""

    if pd.api.types.is_numeric_dtype(series):
        values = series.astype(int).to_numpy()

        unique_values = set(np.unique(values))

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Numeric target must contain only 0/1. "
                f"Found: {unique_values}"
            )

        return values

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
        "0": 0,
        "1": 1,
    }

    unknown = set(normalized.unique()) - set(mapping)

    if unknown:
        raise ValueError(
            f"Unknown target values: {unknown}"
        )

    return normalized.map(mapping).astype(int).to_numpy()


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline."""

    transformers = []

    if numerical_features:
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

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# MODELS
# ============================================================

def build_logistic_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> Pipeline:

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    model = LogisticRegression(
        max_iter=3000,
        class_weight=None,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def build_gradient_boosting_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
) -> Pipeline:

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
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

def calculate_feature_importance(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_columns: List[str],
    numerical_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """
    Estimate feature importance using repeated permutation importance.

    Feature importance is calculated at the original dataframe feature
    level, not one-hot encoded column level.
    """

    print("Calculating feature stability importance...")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    importance_values = {
        feature: []
        for feature in feature_columns
    }

    selection_counts = {
        feature: 0
        for feature in feature_columns
    }

    split_number = 0

    for train_idx, test_idx in cv.split(X, y):

        split_number += 1

        print(
            f"Importance split "
            f"{split_number}/{N_SPLITS * N_REPEATS}"
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_logistic_pipeline(
            numerical_features,
            categorical_features,
        )

        model.fit(X_train, y_train)

        baseline_auc = roc_auc_score(
            y_test,
            model.predict_proba(X_test)[:, 1],
        )

        baseline_score = baseline_auc

        for feature in feature_columns:

            X_permuted = X_test.copy()

            rng = np.random.default_rng(
                RANDOM_STATE + split_number
            )

            X_permuted[feature] = rng.permutation(
                X_permuted[feature].to_numpy()
            )

            permuted_auc = roc_auc_score(
                y_test,
                model.predict_proba(X_permuted)[:, 1],
            )

            importance = baseline_score - permuted_auc

            importance_values[feature].append(
                importance
            )

    rows = []

    for feature in feature_columns:

        values = np.asarray(
            importance_values[feature],
            dtype=float,
        )

        mean_importance = float(np.mean(values))
        std_importance = float(np.std(values))

        positive_frequency = float(
            np.mean(values > 0)
        )

        rows.append(
            {
                "feature": feature,
                "mean_permutation_importance": mean_importance,
                "std_permutation_importance": std_importance,
                "positive_importance_frequency": positive_frequency,
            }
        )

    importance_df = pd.DataFrame(rows)

    importance_df = importance_df.sort_values(
        by="mean_permutation_importance",
        ascending=False,
    ).reset_index(drop=True)

    return importance_df


# ============================================================
# SELECT STABLE FEATURE SUBSET
# ============================================================

def select_stable_features(
    importance_df: pd.DataFrame,
    subset_size: int = SUBSET_SIZE,
) -> List[str]:
    """
    Select the top stable features.

    Primary criterion:
        mean permutation importance

    Secondary criterion:
        positive importance frequency
    """

    ranked = importance_df.copy()

    ranked = ranked.sort_values(
        by=[
            "mean_permutation_importance",
            "positive_importance_frequency",
        ],
        ascending=[
            False,
            False,
        ],
    )

    selected = (
        ranked.head(subset_size)["feature"]
        .tolist()
    )

    return selected


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_subset(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_features: List[str],
    model_name: str,
) -> pd.DataFrame:
    """Evaluate a feature subset using repeated stratified CV."""

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    records = []

    if model_name == "Logistic Regression":
        builder = build_logistic_pipeline

    elif model_name == "Gradient Boosting":
        builder = build_gradient_boosting_pipeline

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    X_subset = X[selected_features]

    numerical_features = [
        feature
        for feature in selected_features
        if pd.api.types.is_numeric_dtype(
            X_subset[feature]
        )
    ]

    categorical_features = [
        feature
        for feature in selected_features
        if feature not in numerical_features
    ]

    split_number = 0

    for train_idx, test_idx in cv.split(
        X_subset,
        y,
    ):

        split_number += 1

        X_train = X_subset.iloc[train_idx]
        X_test = X_subset.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = builder(
            numerical_features,
            categorical_features,
        )

        model.fit(
            X_train,
            y_train,
        )

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

        records.append(
            {
                "model": model_name,
                "feature_set": (
                    "stable_10"
                    if len(selected_features) == SUBSET_SIZE
                    else "full"
                ),
                "feature_count": len(
                    selected_features
                ),
                "split": split_number,
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

def summarize_performance(
    split_results: pd.DataFrame,
) -> pd.DataFrame:

    summaries = []

    for (
        model_name,
        feature_set,
    ), group in split_results.groupby(
        ["model", "feature_set"]
    ):

        summaries.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "feature_count": int(
                    group["feature_count"].iloc[0]
                ),
                "roc_auc_mean": float(
                    group["roc_auc"].mean()
                ),
                "roc_auc_std": float(
                    group["roc_auc"].std()
                ),
                "roc_auc_min": float(
                    group["roc_auc"].min()
                ),
                "roc_auc_max": float(
                    group["roc_auc"].max()
                ),
                "pr_auc_mean": float(
                    group["pr_auc"].mean()
                ),
                "pr_auc_std": float(
                    group["pr_auc"].std()
                ),
                "pr_auc_min": float(
                    group["pr_auc"].min()
                ),
                "pr_auc_max": float(
                    group["pr_auc"].max()
                ),
            }
        )

    return pd.DataFrame(summaries)


# ============================================================
# STABILITY COMPARISON
# ============================================================

def compare_split_stability(
    split_results: pd.DataFrame,
) -> Dict:

    comparison = {}

    for model_name in split_results["model"].unique():

        model_data = split_results[
            split_results["model"] == model_name
        ]

        full = model_data[
            model_data["feature_set"] == "full"
        ].sort_values("split")

        stable = model_data[
            model_data["feature_set"] == "stable_10"
        ].sort_values("split")

        merged = full.merge(
            stable,
            on="split",
            suffixes=("_full", "_stable"),
        )

        roc_deltas = (
            merged["roc_auc_stable"]
            - merged["roc_auc_full"]
        )

        comparison[model_name] = {
            "stable_beats_full_frequency": float(
                np.mean(roc_deltas > 0)
            ),
            "stable_equal_or_better_frequency": float(
                np.mean(roc_deltas >= 0)
            ),
            "mean_roc_auc_delta": float(
                roc_deltas.mean()
            ),
            "median_roc_auc_delta": float(
                roc_deltas.median()
            ),
            "minimum_roc_auc_delta": float(
                roc_deltas.min()
            ),
            "maximum_roc_auc_delta": float(
                roc_deltas.max()
            ),
        }

    return comparison


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_diagnostic_flags(
    importance_df: pd.DataFrame,
    performance_df: pd.DataFrame,
    comparison: Dict,
) -> List[str]:

    flags = []

    positive_importance = importance_df[
        importance_df[
            "mean_permutation_importance"
        ] > 0
    ]

    if len(positive_importance) < len(importance_df):
        flags.append(
            f"{len(importance_df) - len(positive_importance)} "
            "feature(s) have non-positive mean permutation importance."
        )

    for model_name, stats in comparison.items():

        if stats[
            "stable_beats_full_frequency"
        ] >= 0.60:

            flags.append(
                f"{model_name}: stable 10-feature subset "
                f"outperforms the full feature set on "
                f"{stats['stable_beats_full_frequency']:.1%} "
                "of repeated validation splits."
            )

        elif stats[
            "stable_beats_full_frequency"
        ] <= 0.40:

            flags.append(
                f"{model_name}: stable 10-feature subset "
                "rarely outperforms the full feature set "
                "across repeated validation splits."
            )

    for model_name in performance_df["model"].unique():

        stable_row = performance_df[
            (performance_df["model"] == model_name)
            & (
                performance_df["feature_set"]
                == "stable_10"
            )
        ]

        full_row = performance_df[
            (performance_df["model"] == model_name)
            & (
                performance_df["feature_set"]
                == "full"
            )
        ]

        if stable_row.empty or full_row.empty:
            continue

        stable_auc = float(
            stable_row["roc_auc_mean"].iloc[0]
        )

        full_auc = float(
            full_row["roc_auc_mean"].iloc[0]
        )

        if stable_auc > full_auc:
            flags.append(
                f"{model_name}: reduced stable subset "
                f"improves mean ROC-AUC by "
                f"{stable_auc - full_auc:.4f}."
            )

        else:
            flags.append(
                f"{model_name}: full feature set "
                f"retains higher mean ROC-AUC by "
                f"{full_auc - stable_auc:.4f}."
            )

    return flags


# ============================================================
# OVERALL DIAGNOSIS
# ============================================================

def generate_diagnosis(
    performance_df: pd.DataFrame,
    comparison: Dict,
) -> str:

    recommendations = []

    for model_name in comparison:

        stable = performance_df[
            (performance_df["model"] == model_name)
            & (
                performance_df["feature_set"]
                == "stable_10"
            )
        ]

        full = performance_df[
            (performance_df["model"] == model_name)
            & (
                performance_df["feature_set"]
                == "full"
            )
        ]

        if stable.empty or full.empty:
            continue

        stable_auc = float(
            stable["roc_auc_mean"].iloc[0]
        )

        full_auc = float(
            full["roc_auc_mean"].iloc[0]
        )

        beat_frequency = comparison[
            model_name
        ]["stable_beats_full_frequency"]

        if (
            stable_auc >= full_auc
            and beat_frequency >= 0.55
        ):
            recommendations.append(
                f"{model_name} supports the stable "
                "10-feature subset."
            )

        elif (
            stable_auc < full_auc
            and beat_frequency < 0.45
        ):
            recommendations.append(
                f"{model_name} supports retaining "
                "the full feature set."
            )

        else:
            recommendations.append(
                f"{model_name} provides inconclusive "
                "evidence between the stable subset "
                "and full feature set."
            )

    if not recommendations:
        return (
            "Evidence is insufficient to determine "
            "whether the reduced feature set should "
            "replace the full feature set."
        )

    if all(
        "supports the stable" in item
        for item in recommendations
    ):
        return (
            "The stable 10-feature subset consistently "
            "matches or improves the full feature set "
            "across the evaluated model families and "
            "repeated validation splits. The reduced "
            "feature set is recommended for further "
            "model development and interpretability "
            "analysis."
        )

    if all(
        "supports retaining" in item
        for item in recommendations
    ):
        return (
            "The full feature set remains more reliable "
            "across the evaluated model families. "
            "Aggressive feature reduction is not "
            "recommended at this stage."
        )

    return (
        "Model-family results are mixed. The stable "
        "10-feature subset provides evidence of useful "
        "feature reduction, but the improvement is not "
        "consistent enough across all models to make "
        "a universal replacement decision."
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def convert_for_json(value):
    """Convert NumPy/Pandas objects into JSON-safe values."""

    if isinstance(value, dict):
        return {
            str(key): convert_for_json(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            convert_for_json(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_for_json(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return [
            convert_for_json(item)
            for item in value.tolist()
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

    if pd.isna(value):
        return None

    return value


# ============================================================
# SAVE REPORT
# ============================================================

def save_outputs(
    df: pd.DataFrame,
    feature_columns: List[str],
    importance_df: pd.DataFrame,
    selected_features: List[str],
    split_results: pd.DataFrame,
    performance_df: pd.DataFrame,
    comparison: Dict,
    flags: List[str],
    diagnosis: str,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Feature selection summary
    # --------------------------------------------------------

    feature_summary = importance_df.copy()

    feature_summary[
        "selected_in_stable_subset"
    ] = feature_summary[
        "feature"
    ].isin(selected_features)

    feature_summary.to_csv(
        OUTPUT_DIR
        / "feature_selection_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Model performance
    # --------------------------------------------------------

    performance_df.to_csv(
        OUTPUT_DIR
        / "model_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Split performance
    # --------------------------------------------------------

    split_results.to_csv(
        OUTPUT_DIR
        / "split_performance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    target = encode_target(
        df[TARGET_COLUMN]
    )

    summary_lines = [
        "=" * 60,
        "EMPLOYEE ATTRITION — STABLE FEATURE SET VALIDATION",
        "=" * 60,
        "",
        "[DATASET]",
        f"Rows:                 {len(df)}",
        f"Columns:              {len(df.columns)}",
        f"Model features:       {len(feature_columns)}",
        f"Target prevalence:    {np.mean(target):.2%}",
        "",
        "[FEATURE SET]",
        f"Stable subset size:   {len(selected_features)}",
        "",
        "Stable features:",
    ]

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        summary_lines.append(
            f"{index:2d}. {feature}"
        )

    summary_lines.extend(
        [
            "",
            "[MODEL PERFORMANCE]",
        ]
    )

    for _, row in performance_df.iterrows():

        summary_lines.append(
            f"{row['model']:22s} "
            f"{row['feature_set']:10s} "
            f"ROC-AUC={row['roc_auc_mean']:.4f} "
            f"Std={row['roc_auc_std']:.4f} "
            f"Min={row['roc_auc_min']:.4f} "
            f"Max={row['roc_auc_max']:.4f} "
            f"PR-AUC={row['pr_auc_mean']:.4f}"
        )

    summary_lines.extend(
        [
            "",
            "[SPLIT-WISE STABILITY]",
        ]
    )

    for model_name, stats in comparison.items():

        summary_lines.append(
            f"{model_name:22s} "
            f"Stable > Full="
            f"{stats['stable_beats_full_frequency']:.1%} "
            f"Mean Delta="
            f"{stats['mean_roc_auc_delta']:+.4f}"
        )

    summary_lines.extend(
        [
            "",
            "[DIAGNOSTIC FLAGS]",
        ]
    )

    if flags:
        for flag in flags:
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
            diagnosis,
            "",
            "[OUTPUT]",
            f"Reports:              {OUTPUT_DIR}",
            f"Feature selection:    "
            f"{OUTPUT_DIR / 'feature_selection_summary.csv'}",
            f"Model performance:    "
            f"{OUTPUT_DIR / 'model_performance.csv'}",
            f"Split performance:    "
            f"{OUTPUT_DIR / 'split_performance.csv'}",
            "",
            "=" * 60,
            "STABLE FEATURE SET VALIDATION COMPLETE",
            "=" * 60,
        ]
    )

    summary_text = "\n".join(
        summary_lines
    )

    summary_path = (
        OUTPUT_DIR
        / "stable_feature_set_validation_summary.txt"
    )

    summary_path.write_text(
        summary_text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "model_feature_count": int(
                len(feature_columns)
            ),
            "target": TARGET_COLUMN,
            "target_prevalence": float(
                np.mean(target)
            ),
        },
        "validation_design": {
            "folds_per_repeat": N_SPLITS,
            "repeats": N_REPEATS,
            "total_validation_runs": (
                N_SPLITS * N_REPEATS
            ),
        },
        "stable_feature_subset": {
            "size": len(selected_features),
            "features": selected_features,
        },
        "feature_importance": (
            importance_df.to_dict(
                orient="records"
            )
        ),
        "performance": (
            performance_df.to_dict(
                orient="records"
            )
        ),
        "split_comparison": comparison,
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    json_path = (
        OUTPUT_DIR
        / "stable_feature_set_validation_report.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            convert_for_json(report),
            file,
            indent=2,
        )

    return summary_path, json_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("Running stable feature set validation...")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Identify features
    # --------------------------------------------------------

    (
        feature_columns,
        numerical_features,
        categorical_features,
    ) = identify_features(df)

    print(
        f"Model features after identifier exclusion: "
        f"{len(feature_columns)}"
    )

    print(
        f"Numerical features:     "
        f"{len(numerical_features)}"
    )

    print(
        f"Categorical features:   "
        f"{len(categorical_features)}"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = encode_target(
        df[TARGET_COLUMN]
    )

    X = df[feature_columns].copy()

    print(
        f"Target prevalence:      "
        f"{np.mean(y):.2%}"
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance_df = calculate_feature_importance(
        X,
        y,
        feature_columns,
        numerical_features,
        categorical_features,
    )

    # --------------------------------------------------------
    # Stable subset
    # --------------------------------------------------------

    selected_features = select_stable_features(
        importance_df,
        SUBSET_SIZE,
    )

    print()
    print("Selected stable feature subset:")

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(
            f"  {index:2d}. {feature}"
        )

    # --------------------------------------------------------
    # Evaluate full feature set
    # --------------------------------------------------------

    print()
    print("Evaluating full feature set...")

    full_logistic = evaluate_subset(
        X,
        y,
        feature_columns,
        "Logistic Regression",
    )

    full_gb = evaluate_subset(
        X,
        y,
        feature_columns,
        "Gradient Boosting",
    )

    # --------------------------------------------------------
    # Evaluate stable subset
    # --------------------------------------------------------

    print()
    print("Evaluating stable 10-feature subset...")

    stable_logistic = evaluate_subset(
        X,
        y,
        selected_features,
        "Logistic Regression",
    )

    stable_gb = evaluate_subset(
        X,
        y,
        selected_features,
        "Gradient Boosting",
    )

    split_results = pd.concat(
        [
            full_logistic,
            full_gb,
            stable_logistic,
            stable_gb,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    performance_df = summarize_performance(
        split_results
    )

    comparison = compare_split_stability(
        split_results
    )

    flags = generate_diagnostic_flags(
        importance_df,
        performance_df,
        comparison,
    )

    diagnosis = generate_diagnosis(
        performance_df,
        comparison,
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — "
        "STABLE FEATURE SET VALIDATION"
    )
    print("=" * 60)

    print()
    print("[DATASET]")
    print(
        f"Rows:                 {len(df)}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )
    print(
        f"Model features:       {len(feature_columns)}"
    )
    print(
        f"Target prevalence:    {np.mean(y):.2%}"
    )

    print()
    print("[STABLE FEATURE SUBSET]")
    print(
        f"Feature count:         {len(selected_features)}"
    )

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(
            f"{index:2d}. {feature}"
        )

    print()
    print("[MODEL PERFORMANCE]")

    print(
        performance_df[
            [
                "model",
                "feature_set",
                "feature_count",
                "roc_auc_mean",
                "roc_auc_std",
                "roc_auc_min",
                "roc_auc_max",
                "pr_auc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("[SPLIT-WISE STABILITY]")

    for model_name, stats in comparison.items():

        print(
            f"{model_name:22s} "
            f"Stable > Full="
            f"{stats['stable_beats_full_frequency']:.1%} "
            f"Mean ROC-AUC Delta="
            f"{stats['mean_roc_auc_delta']:+.4f}"
        )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            print(
                f"- {flag}"
            )
    else:
        print(
            "- No major diagnostic flags."
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_path, json_path = save_outputs(
        df=df,
        feature_columns=feature_columns,
        importance_df=importance_df,
        selected_features=selected_features,
        split_results=split_results,
        performance_df=performance_df,
        comparison=comparison,
        flags=flags,
        diagnosis=diagnosis,
    )

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              {OUTPUT_DIR}"
    )
    print(
        f"JSON report:          {json_path}"
    )
    print(
        f"Feature selection:    "
        f"{OUTPUT_DIR / 'feature_selection_summary.csv'}"
    )
    print(
        f"Model performance:    "
        f"{OUTPUT_DIR / 'model_performance.csv'}"
    )
    print(
        f"Split performance:    "
        f"{OUTPUT_DIR / 'split_performance.csv'}"
    )
    print(
        f"Summary report:       {summary_path}"
    )

    print()
    print("=" * 60)
    print(
        "STABLE FEATURE SET VALIDATION COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()