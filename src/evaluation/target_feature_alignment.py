from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, pointbiserialr

from sklearn.ensemble import (
    GradientBoostingClassifier,
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

from sklearn.model_selection import train_test_split
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

REPORTS_DIR = PROJECT_ROOT / "reports"

ALIGNMENT_REPORT_DIR = (
    REPORTS_DIR
    / "signal_analysis"
    / "target_feature_alignment"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "Attrition"

ID_COLUMN = "Employee_ID"

TEST_SIZE = 0.20

OPERATING_THRESHOLD = 0.15

TOP_N_FEATURES = 12

N_BINS = 5


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_output_directories() -> None:
    """
    Create directories required for the target-feature
    alignment analysis.
    """

    ALIGNMENT_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(
    value: Any,
) -> Any:
    """
    Recursively convert pandas / NumPy / sklearn-related
    objects into JSON-compatible Python objects.

    This specifically handles pandas Interval objects created
    by pd.qcut(), which caused the previous failure.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        if isinstance(value, float):
            if not np.isfinite(value):
                return None

        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Timedelta):
        return str(value)

    if isinstance(value, pd.Interval):
        return str(value)

    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.Series):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.Index):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, pd.Categorical):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return str(value)


def save_json_report(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a JSON report after recursively converting all
    unsupported objects into JSON-compatible values.
    """

    safe_data = make_json_serializable(data)

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            safe_data,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the raw employee attrition dataset.
    """

    print("Loading dataset...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    return df


# ============================================================
# TARGET PREPARATION
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Convert Attrition Yes/No into binary values.
    """

    target = (
        df[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    if target.isna().any():
        unexpected = (
            df.loc[
                target.isna(),
                TARGET_COLUMN,
            ]
            .dropna()
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected values found in "
            f"{TARGET_COLUMN}: {unexpected}"
        )

    return target.astype(int)


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def get_numerical_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return numerical predictor columns.
    """

    columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    return [
        column
        for column in columns
        if column not in {
            TARGET_COLUMN,
            ID_COLUMN,
        }
    ]


def get_categorical_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return categorical predictor columns.
    """

    columns = df.select_dtypes(
        include=[
            "object",
            "string",
            "category",
        ]
    ).columns.tolist()

    return [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]


# ============================================================
# NUMERICAL TARGET RELATIONSHIP
# ============================================================

def analyze_numerical_relationships(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Calculate point-biserial correlations between numerical
    features and Attrition.
    """

    records: list[dict[str, Any]] = []

    for feature in get_numerical_features(df):

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid_mask = (
            values.notna()
            & target.notna()
        )

        x = values.loc[valid_mask]
        y = target.loc[valid_mask]

        if x.nunique() <= 1:

            correlation = 0.0
            p_value = 1.0

        else:

            result = pointbiserialr(
                y,
                x,
            )

            correlation = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        stayed_values = x[
            y == 0
        ]

        attrition_values = x[
            y == 1
        ]

        stayed_mean = (
            float(stayed_values.mean())
            if not stayed_values.empty
            else np.nan
        )

        attrition_mean = (
            float(attrition_values.mean())
            if not attrition_values.empty
            else np.nan
        )

        records.append(
            {
                "feature": feature,
                "stayed_mean": stayed_mean,
                "attrition_mean": attrition_mean,
                "mean_difference": (
                    attrition_mean
                    - stayed_mean
                ),
                "point_biserial_correlation": (
                    correlation
                ),
                "absolute_correlation": abs(
                    correlation
                ),
                "p_value": p_value,
            }
        )

    result_df = pd.DataFrame(
        records
    )

    if not result_df.empty:

        result_df = (
            result_df
            .sort_values(
                by="absolute_correlation",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

    return result_df


# ============================================================
# CATEGORICAL TARGET RELATIONSHIP
# ============================================================

def calculate_cramers_v(
    table: pd.DataFrame,
) -> tuple[float, float]:
    """
    Calculate bias-corrected Cramer's V and chi-square
    p-value.
    """

    if table.empty:
        return 0.0, 1.0

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return 0.0, 1.0

    chi2, p_value, _, _ = (
        chi2_contingency(table)
    )

    n = table.values.sum()

    if n == 0:
        return 0.0, 1.0

    phi2 = chi2 / n

    rows, columns = table.shape

    phi2_corrected = max(
        0.0,
        phi2
        - (
            (columns - 1)
            * (rows - 1)
            / max(n - 1, 1)
        ),
    )

    rows_corrected = (
        rows
        - (
            (rows - 1) ** 2
            / max(n - 1, 1)
        )
    )

    columns_corrected = (
        columns
        - (
            (columns - 1) ** 2
            / max(n - 1, 1)
        )
    )

    denominator = min(
        rows_corrected - 1,
        columns_corrected - 1,
    )

    if denominator <= 0:
        return 0.0, float(p_value)

    cramers_v = np.sqrt(
        phi2_corrected
        / denominator
    )

    return (
        float(cramers_v),
        float(p_value),
    )


def analyze_categorical_relationships(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Analyze categorical feature relationships with Attrition.
    """

    records: list[dict[str, Any]] = []

    for feature in get_categorical_features(df):

        working = pd.DataFrame(
            {
                "feature": df[feature],
                "target": target,
            }
        ).dropna()

        if working.empty:
            continue

        contingency = pd.crosstab(
            working["feature"],
            working["target"],
        )

        cramers_v, p_value = (
            calculate_cramers_v(
                contingency
            )
        )

        for category in sorted(
            working["feature"]
            .astype(str)
            .unique()
        ):

            category_mask = (
                working["feature"]
                .astype(str)
                == category
            )

            total = int(
                category_mask.sum()
            )

            attrition_count = int(
                (
                    working.loc[
                        category_mask,
                        "target",
                    ]
                    == 1
                ).sum()
            )

            attrition_rate = (
                attrition_count / total
                if total > 0
                else 0.0
            )

            records.append(
                {
                    "feature": feature,
                    "category": category,
                    "total_employees": total,
                    "attrition_count": (
                        attrition_count
                    ),
                    "attrition_rate": (
                        attrition_rate
                    ),
                    "cramers_v": cramers_v,
                    "p_value": p_value,
                }
            )

    result_df = pd.DataFrame(
        records
    )

    if not result_df.empty:

        result_df = (
            result_df
            .sort_values(
                by=[
                    "cramers_v",
                    "attrition_rate",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    return result_df


# ============================================================
# NUMERICAL TARGET-RATE ANALYSIS
# ============================================================

def analyze_numerical_target_rates(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Divide numerical features into quantile bins and calculate
    observed attrition rates.

    pd.qcut() creates Interval objects. These are intentionally
    converted to strings before saving to CSV/JSON.
    """

    records: list[dict[str, Any]] = []

    working_df = df.copy()

    for feature in get_numerical_features(
        working_df
    ):

        values = pd.to_numeric(
            working_df[feature],
            errors="coerce",
        )

        if values.nunique(
            dropna=True
        ) < 3:
            continue

        try:

            bins = pd.qcut(
                values,
                q=N_BINS,
                duplicates="drop",
            )

        except ValueError:
            continue

        temporary = pd.DataFrame(
            {
                "bin": bins,
                "target": target,
            }
        ).dropna()

        if temporary.empty:
            continue

        grouped = (
            temporary
            .groupby(
                "bin",
                observed=True,
            )
            .agg(
                employees=(
                    "target",
                    "size",
                ),
                attrition_count=(
                    "target",
                    "sum",
                ),
                attrition_rate=(
                    "target",
                    "mean",
                ),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():

            bin_value = row["bin"]

            records.append(
                {
                    "feature": feature,
                    "bin": str(bin_value),
                    "employees": int(
                        row["employees"]
                    ),
                    "attrition_count": int(
                        row["attrition_count"]
                    ),
                    "attrition_rate": float(
                        row["attrition_rate"]
                    ),
                }
            )

    result_df = pd.DataFrame(
        records
    )

    return result_df


# ============================================================
# TRAIN / HOLDOUT RELATIONSHIP COMPARISON
# ============================================================

def compare_numerical_relationships(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    train_target: pd.Series,
    holdout_target: pd.Series,
) -> pd.DataFrame:
    """
    Compare numerical feature-target relationships between
    training and holdout partitions.
    """

    train_relationships = (
        analyze_numerical_relationships(
            train_df,
            train_target,
        )
    )

    holdout_relationships = (
        analyze_numerical_relationships(
            holdout_df,
            holdout_target,
        )
    )

    train_relationships = (
        train_relationships[
            [
                "feature",
                "point_biserial_correlation",
            ]
        ]
        .rename(
            columns={
                "point_biserial_correlation":
                    "train_correlation"
            }
        )
    )

    holdout_relationships = (
        holdout_relationships[
            [
                "feature",
                "point_biserial_correlation",
            ]
        ]
        .rename(
            columns={
                "point_biserial_correlation":
                    "holdout_correlation"
            }
        )
    )

    comparison = train_relationships.merge(
        holdout_relationships,
        on="feature",
        how="outer",
    )

    comparison[
        "correlation_delta"
    ] = (
        comparison["holdout_correlation"]
        - comparison["train_correlation"]
    )

    comparison[
        "direction_consistent"
    ] = (
        np.sign(
            comparison["train_correlation"]
        )
        ==
        np.sign(
            comparison["holdout_correlation"]
        )
    )

    comparison[
        "absolute_delta"
    ] = comparison[
        "correlation_delta"
    ].abs()

    return (
        comparison
        .sort_values(
            "absolute_delta",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def compare_categorical_relationships(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    train_target: pd.Series,
    holdout_target: pd.Series,
) -> pd.DataFrame:
    """
    Compare Cramer's V for categorical features between
    training and holdout partitions.
    """

    train_relationships = (
        analyze_categorical_relationships(
            train_df,
            train_target,
        )
    )

    holdout_relationships = (
        analyze_categorical_relationships(
            holdout_df,
            holdout_target,
        )
    )

    if train_relationships.empty:
        return pd.DataFrame()

    train_v = (
        train_relationships[
            [
                "feature",
                "cramers_v",
            ]
        ]
        .drop_duplicates(
            subset=["feature"]
        )
        .rename(
            columns={
                "cramers_v":
                    "train_cramers_v"
            }
        )
    )

    holdout_v = (
        holdout_relationships[
            [
                "feature",
                "cramers_v",
            ]
        ]
        .drop_duplicates(
            subset=["feature"]
        )
        .rename(
            columns={
                "cramers_v":
                    "holdout_cramers_v"
            }
        )
    )

    comparison = train_v.merge(
        holdout_v,
        on="feature",
        how="outer",
    )

    comparison[
        "cramers_v_delta"
    ] = (
        comparison["holdout_cramers_v"]
        - comparison["train_cramers_v"]
    )

    comparison[
        "absolute_delta"
    ] = comparison[
        "cramers_v_delta"
    ].abs()

    return (
        comparison
        .sort_values(
            "absolute_delta",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_logistic_model() -> Pipeline:
    """
    Build the optimized Logistic Regression candidate.

    This is the candidate selected by the previous controlled
    optimization stage:
        C = 0.01
        class_weight = None
    """

    preprocessor = (
        build_preprocessor()
    )

    classifier = LogisticRegression(
        C=0.01,
        class_weight=None,
        max_iter=5000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def build_gradient_boosting_model() -> Pipeline:
    """
    Build the strongest useful Gradient Boosting candidate
    from the previous optimization stage.
    """

    preprocessor = (
        build_preprocessor()
    )

    classifier = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def build_random_forest_model() -> Pipeline:
    """
    Build Random Forest as a nonlinear comparison model.
    """

    preprocessor = (
        build_preprocessor()
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
) -> dict[str, Any]:
    """
    Fit and evaluate one model on the fixed train/holdout split.
    """

    print(
        f"Evaluating {model_name}..."
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = (
        model.predict_proba(
            X_holdout
        )[:, 1]
    )

    predictions = (
        probabilities
        >= OPERATING_THRESHOLD
    ).astype(int)

    result = {
        "model": model_name,
        "roc_auc": float(
            roc_auc_score(
                y_holdout,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_holdout,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_holdout,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_holdout,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_holdout,
                predictions,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_holdout,
                predictions,
            )
        ),
        "predicted_positive_rate": float(
            predictions.mean()
        ),
        "mean_probability": float(
            probabilities.mean()
        ),
        "minimum_probability": float(
            probabilities.min()
        ),
        "maximum_probability": float(
            probabilities.max()
        ),
    }

    return result


def evaluate_models(
    X_train: pd.DataFrame,
    X_holdout: pd.DataFrame,
    y_train: pd.Series,
    y_holdout: pd.Series,
) -> pd.DataFrame:
    """
    Compare linear and nonlinear models.
    """

    models = [
        (
            "Logistic Regression",
            build_logistic_model(),
        ),
        (
            "Gradient Boosting",
            build_gradient_boosting_model(),
        ),
        (
            "Random Forest",
            build_random_forest_model(),
        ),
    ]

    results: list[dict[str, Any]] = []

    for model_name, model in models:

        result = evaluate_model(
            model_name,
            model,
            X_train,
            X_holdout,
            y_train,
            y_holdout,
        )

        results.append(
            result
        )

    result_df = pd.DataFrame(
        results
    )

    if not result_df.empty:

        result_df = (
            result_df
            .sort_values(
                by=[
                    "roc_auc",
                    "pr_auc",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
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
# SIGNAL CONCENTRATION
# ============================================================

def calculate_signal_concentration(
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Summarize how concentrated the target signal is.
    """

    if numerical_df.empty:

        maximum_numerical_correlation = 0.0

        numerical_ge_010 = 0

        numerical_ge_005 = 0

    else:

        maximum_numerical_correlation = (
            float(
                numerical_df[
                    "absolute_correlation"
                ].max()
            )
        )

        numerical_ge_010 = int(
            (
                numerical_df[
                    "absolute_correlation"
                ]
                >= 0.10
            ).sum()
        )

        numerical_ge_005 = int(
            (
                numerical_df[
                    "absolute_correlation"
                ]
                >= 0.05
            ).sum()
        )

    if categorical_df.empty:

        maximum_cramers_v = 0.0

        maximum_category_deviation = 0.0

    else:

        maximum_cramers_v = float(
            categorical_df[
                "cramers_v"
            ].max()
        )

        category_rates = (
            categorical_df[
                "attrition_rate"
            ]
        )

        overall_rate = (
            categorical_df[
                "attrition_count"
            ].sum()
            /
            categorical_df[
                "total_employees"
            ].sum()
        )

        maximum_category_deviation = float(
            (
                category_rates
                - overall_rate
            ).abs().max()
        )

    return {
        "maximum_numerical_absolute_correlation":
            maximum_numerical_correlation,

        "numerical_features_absolute_correlation_ge_0.10":
            numerical_ge_010,

        "numerical_features_absolute_correlation_ge_0.05":
            numerical_ge_005,

        "maximum_categorical_cramers_v":
            maximum_cramers_v,

        "maximum_category_rate_deviation":
            maximum_category_deviation,
    }


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_diagnostic_flags(
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    train_holdout_numerical: pd.DataFrame,
    train_holdout_categorical: pd.DataFrame,
    model_results: pd.DataFrame,
    train_prevalence: float,
    holdout_prevalence: float,
) -> list[str]:
    """
    Generate conservative diagnostic observations.
    """

    flags: list[str] = []

    # --------------------------------------------------------
    # Numerical signal
    # --------------------------------------------------------

    if not numerical_df.empty:

        max_corr = float(
            numerical_df[
                "absolute_correlation"
            ].max()
        )

        if max_corr >= 0.10:

            flags.append(
                "At least one numerical feature "
                "shows a potentially meaningful "
                "marginal relationship with Attrition."
            )

        elif max_corr >= 0.05:

            flags.append(
                "Numerical features show modest "
                "marginal relationships with Attrition."
            )

        else:

            flags.append(
                "Marginal numerical relationships "
                "with Attrition are generally weak."
            )

    # --------------------------------------------------------
    # Categorical signal
    # --------------------------------------------------------

    if not categorical_df.empty:

        max_v = float(
            categorical_df[
                "cramers_v"
            ].max()
        )

        if max_v >= 0.10:

            flags.append(
                "At least one categorical feature "
                "shows a potentially meaningful "
                "association with Attrition."
            )

        else:

            flags.append(
                "Categorical relationships exist "
                "but remain modest."
            )

    # --------------------------------------------------------
    # Train / holdout relationship stability
    # --------------------------------------------------------

    if (
        not train_holdout_numerical.empty
    ):

        inconsistent = (
            ~train_holdout_numerical[
                "direction_consistent"
            ]
        )

        if inconsistent.any():

            flags.append(
                "Some numerical feature-target "
                "relationships change direction "
                "between training and holdout partitions."
            )

        large_changes = (
            train_holdout_numerical[
                "absolute_delta"
            ]
            >= 0.10
        )

        if large_changes.any():

            flags.append(
                "Some numerical feature-target "
                "relationships change materially "
                "between training and holdout partitions."
            )

    if (
        not train_holdout_categorical.empty
    ):

        large_categorical_changes = (
            train_holdout_categorical[
                "absolute_delta"
            ]
            >= 0.10
        )

        if large_categorical_changes.any():

            flags.append(
                "Some categorical feature associations "
                "change materially between training "
                "and holdout partitions."
            )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    if not model_results.empty:

        best_auc = float(
            model_results[
                "roc_auc"
            ].max()
        )

        if best_auc < 0.60:

            flags.append(
                "The best fixed-holdout ROC-AUC "
                "is below 0.60, indicating weak "
                "out-of-sample ranking performance."
            )

        elif best_auc < 0.65:

            flags.append(
                "The best fixed-holdout ROC-AUC "
                "indicates modest predictive separation."
            )

        else:

            flags.append(
                "The best fixed-holdout ROC-AUC "
                "indicates meaningful predictive separation."
            )

    # --------------------------------------------------------
    # Target prevalence
    # --------------------------------------------------------

    prevalence_difference = (
        holdout_prevalence
        - train_prevalence
    )

    if abs(
        prevalence_difference
    ) > 0.05:

        flags.append(
            "Training and holdout target prevalence "
            "differ materially."
        )

    else:

        flags.append(
            "Training and holdout target prevalence "
            "are broadly aligned."
        )

    return flags


# ============================================================
# OVERALL DIAGNOSIS
# ============================================================

def generate_overall_diagnosis(
    model_results: pd.DataFrame,
    train_holdout_numerical: pd.DataFrame,
    train_holdout_categorical: pd.DataFrame,
) -> str:
    """
    Produce the overall diagnostic conclusion.
    """

    best_auc = 0.0

    if not model_results.empty:

        best_auc = float(
            model_results[
                "roc_auc"
            ].max()
        )

    numerical_instability = False

    if not train_holdout_numerical.empty:

        numerical_instability = bool(
            (
                train_holdout_numerical[
                    "absolute_delta"
                ]
                >= 0.10
            ).any()
            or
            (
                ~train_holdout_numerical[
                    "direction_consistent"
                ]
            ).any()
        )

    categorical_instability = False

    if not train_holdout_categorical.empty:

        categorical_instability = bool(
            (
                train_holdout_categorical[
                    "absolute_delta"
                ]
                >= 0.10
            ).any()
        )

    if best_auc < 0.60:

        if (
            numerical_instability
            or categorical_instability
        ):

            return (
                "The target contains identifiable "
                "relationships, but fixed-holdout "
                "performance is weak and some "
                "feature-target relationships "
                "are unstable across partitions. "
                "The evidence points toward a "
                "combination of weak signal strength "
                "and generalization instability. "
                "Further investigation of data "
                "construction, feature quality, and "
                "validation methodology is recommended."
            )

        return (
            "The target contains identifiable "
            "relationships, but the available "
            "features provide weak out-of-sample "
            "predictive separation. The primary "
            "issue appears to be limited predictive "
            "signal rather than a simple feature-"
            "direction instability problem."
        )

    if best_auc < 0.65:

        if (
            numerical_instability
            or categorical_instability
        ):

            return (
                "The target contains genuine "
                "predictive structure, but the "
                "strength of that structure varies "
                "between training and holdout data. "
                "This supports further investigation "
                "of feature stability, data construction, "
                "and validation methodology before "
                "production deployment."
            )

        return (
            "The target contains genuine predictive "
            "structure and the models achieve modest "
            "out-of-sample separation. However, the "
            "signal remains relatively weak, so "
            "additional feature engineering and "
            "independent validation are recommended "
            "before production deployment."
        )

    return (
        "The target contains meaningful predictive "
        "structure and at least one model demonstrates "
        "reasonable out-of-sample separation. "
        "The findings support continued controlled "
        "model development and independent validation."
    )


# ============================================================
# REPORT SAVING
# ============================================================

def save_reports(
    numerical_relationships: pd.DataFrame,
    categorical_relationships: pd.DataFrame,
    numerical_target_rates: pd.DataFrame,
    train_holdout_numerical: pd.DataFrame,
    train_holdout_categorical: pd.DataFrame,
    model_results: pd.DataFrame,
    signal_concentration: dict[str, Any],
    flags: list[str],
    diagnosis: str,
    dataset_summary: dict[str, Any],
) -> None:
    """
    Save CSV, JSON and TXT reports.
    """

    numerical_relationships.to_csv(
        ALIGNMENT_REPORT_DIR
        / "numerical_target_relationships.csv",
        index=False,
    )

    categorical_relationships.to_csv(
        ALIGNMENT_REPORT_DIR
        / "categorical_target_relationships.csv",
        index=False,
    )

    numerical_target_rates.to_csv(
        ALIGNMENT_REPORT_DIR
        / "numerical_binned_target_rates.csv",
        index=False,
    )

    train_holdout_numerical.to_csv(
        ALIGNMENT_REPORT_DIR
        / "train_holdout_numerical_alignment.csv",
        index=False,
    )

    train_holdout_categorical.to_csv(
        ALIGNMENT_REPORT_DIR
        / "train_holdout_categorical_alignment.csv",
        index=False,
    )

    model_results.to_csv(
        ALIGNMENT_REPORT_DIR
        / "model_comparison.csv",
        index=False,
    )

    report = {
        "metadata": {
            "generated_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "analysis": (
                "target_feature_alignment"
            ),
            "random_state": RANDOM_STATE,
            "operating_threshold": (
                OPERATING_THRESHOLD
            ),
        },

        "dataset": dataset_summary,

        "signal_concentration":
            signal_concentration,

        "diagnostic_flags":
            flags,

        "overall_diagnosis":
            diagnosis,

        "top_numerical_relationships":
            numerical_relationships
            .head(TOP_N_FEATURES)
            .to_dict(
                orient="records"
            ),

        "top_categorical_relationships":
            categorical_relationships
            .head(TOP_N_FEATURES)
            .to_dict(
                orient="records"
            ),

        "train_holdout_numerical_alignment":
            train_holdout_numerical
            .head(TOP_N_FEATURES)
            .to_dict(
                orient="records"
            ),

        "train_holdout_categorical_alignment":
            train_holdout_categorical
            .head(TOP_N_FEATURES)
            .to_dict(
                orient="records"
            ),

        "model_comparison":
            model_results.to_dict(
                orient="records"
            ),
    }

    save_json_report(
        report,
        ALIGNMENT_REPORT_DIR
        / "target_feature_alignment_report.json",
    )

    summary_path = (
        ALIGNMENT_REPORT_DIR
        / "target_feature_alignment_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "TARGET-FEATURE ALIGNMENT\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            "[DATASET]\n"
        )

        file.write(
            f"Rows:                 "
            f"{dataset_summary['rows']}\n"
        )

        file.write(
            f"Features:             "
            f"{dataset_summary['features']}\n"
        )

        file.write(
            f"Target prevalence:    "
            f"{dataset_summary['target_prevalence'] * 100:.2f}%\n"
        )

        file.write(
            f"Training rows:        "
            f"{dataset_summary['training_rows']}\n"
        )

        file.write(
            f"Holdout rows:         "
            f"{dataset_summary['holdout_rows']}\n"
        )

        file.write(
            "\n[MODEL COMPARISON]\n"
        )

        if not model_results.empty:

            file.write(
                model_results.to_string(
                    index=False
                )
            )

        file.write(
            "\n\n[SIGNAL CONCENTRATION]\n"
        )

        for key, value in (
            signal_concentration.items()
        ):

            file.write(
                f"{key}: {value}\n"
            )

        file.write(
            "\n[DIAGNOSTIC FLAGS]\n"
        )

        for flag in flags:

            file.write(
                f"- {flag}\n"
            )

        file.write(
            "\n[OVERALL DIAGNOSIS]\n"
        )

        file.write(
            diagnosis
        )

        file.write(
            "\n"
        )


# ============================================================
# TERMINAL REPORT
# ============================================================

def print_report(
    numerical_relationships: pd.DataFrame,
    categorical_relationships: pd.DataFrame,
    train_holdout_numerical: pd.DataFrame,
    train_holdout_categorical: pd.DataFrame,
    model_results: pd.DataFrame,
    signal_concentration: dict[str, Any],
    flags: list[str],
    diagnosis: str,
    dataset_summary: dict[str, Any],
) -> None:
    """
    Print the final diagnostic report.
    """

    print(
        "\n"
        + "=" * 60
    )

    print(
        "EMPLOYEE ATTRITION — "
        "TARGET-FEATURE ALIGNMENT"
    )

    print(
        "=" * 60
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    print(
        "\n[DATASET]"
    )

    print(
        f"Rows:                 "
        f"{dataset_summary['rows']}"
    )

    print(
        f"Features:             "
        f"{dataset_summary['features']}"
    )

    print(
        f"Target prevalence:    "
        f"{dataset_summary['target_prevalence'] * 100:.2f}%"
    )

    print(
        f"Training rows:        "
        f"{dataset_summary['training_rows']}"
    )

    print(
        f"Holdout rows:         "
        f"{dataset_summary['holdout_rows']}"
    )

    # --------------------------------------------------------
    # NUMERICAL RELATIONSHIPS
    # --------------------------------------------------------

    print(
        "\n[TOP NUMERICAL RELATIONSHIPS]"
    )

    if numerical_relationships.empty:

        print(
            "No numerical relationships available."
        )

    else:

        display_columns = [
            "feature",
            "point_biserial_correlation",
            "p_value",
        ]

        print(
            numerical_relationships[
                display_columns
            ]
            .head(10)
            .round(4)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CATEGORICAL RELATIONSHIPS
    # --------------------------------------------------------

    print(
        "\n[TOP CATEGORICAL RELATIONSHIPS]"
    )

    if categorical_relationships.empty:

        print(
            "No categorical relationships available."
        )

    else:

        display = (
            categorical_relationships[
                [
                    "feature",
                    "category",
                    "attrition_rate",
                    "cramers_v",
                    "p_value",
                ]
            ]
            .drop_duplicates(
                subset=["feature"]
            )
            .head(10)
        )

        print(
            display
            .round(4)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # TRAIN / HOLDOUT ALIGNMENT
    # --------------------------------------------------------

    print(
        "\n[TRAIN / HOLDOUT NUMERICAL ALIGNMENT]"
    )

    if train_holdout_numerical.empty:

        print(
            "No numerical alignment results available."
        )

    else:

        print(
            train_holdout_numerical[
                [
                    "feature",
                    "train_correlation",
                    "holdout_correlation",
                    "correlation_delta",
                    "direction_consistent",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(
                index=False
            )
        )

    print(
        "\n[TRAIN / HOLDOUT CATEGORICAL ALIGNMENT]"
    )

    if train_holdout_categorical.empty:

        print(
            "No categorical alignment results available."
        )

    else:

        print(
            train_holdout_categorical[
                [
                    "feature",
                    "train_cramers_v",
                    "holdout_cramers_v",
                    "cramers_v_delta",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print(
        "\n[MODEL COMPARISON]"
    )

    if model_results.empty:

        print(
            "No model results available."
        )

    else:

        print(
            model_results[
                [
                    "rank",
                    "model",
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                    "predicted_positive_rate",
                ]
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # SIGNAL CONCENTRATION
    # --------------------------------------------------------

    print(
        "\n[SIGNAL CONCENTRATION]"
    )

    print(
        "Maximum numerical |r|: "
        f"{signal_concentration['maximum_numerical_absolute_correlation']:.4f}"
    )

    print(
        "Numerical features |r| >= 0.10: "
        f"{signal_concentration['numerical_features_absolute_correlation_ge_0.10']}"
    )

    print(
        "Numerical features |r| >= 0.05: "
        f"{signal_concentration['numerical_features_absolute_correlation_ge_0.05']}"
    )

    print(
        "Maximum Cramer's V:    "
        f"{signal_concentration['maximum_categorical_cramers_v']:.4f}"
    )

    print(
        "Maximum category rate deviation: "
        f"{signal_concentration['maximum_category_rate_deviation'] * 100:.2f} pp"
    )

    # --------------------------------------------------------
    # DIAGNOSTIC FLAGS
    # --------------------------------------------------------

    print(
        "\n[DIAGNOSTIC FLAGS]"
    )

    for flag in flags:

        print(
            f"- {flag}"
        )

    # --------------------------------------------------------
    # OVERALL DIAGNOSIS
    # --------------------------------------------------------

    print(
        "\n[OVERALL DIAGNOSIS]"
    )

    print(
        diagnosis
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        "\n[OUTPUT]"
    )

    print(
        f"Reports:              "
        f"{ALIGNMENT_REPORT_DIR}"
    )

    print(
        f"JSON report:          "
        f"{ALIGNMENT_REPORT_DIR / 'target_feature_alignment_report.json'}"
    )

    print(
        f"Summary report:       "
        f"{ALIGNMENT_REPORT_DIR / 'target_feature_alignment_summary.txt'}"
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "TARGET-FEATURE ALIGNMENT COMPLETE"
    )

    print(
        "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete target-feature alignment analysis.
    """

    print(
        "\nRunning target-feature alignment analysis..."
    )

    create_output_directories()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_dataset()

    target = prepare_target(
        df
    )

    numerical_features = (
        get_numerical_features(df)
    )

    categorical_features = (
        get_categorical_features(df)
    )

    print(
        f"Target prevalence:    "
        f"{target.mean() * 100:.2f}%"
    )

    print(
        "Running numerical target relationship analysis..."
    )

    numerical_relationships = (
        analyze_numerical_relationships(
            df,
            target,
        )
    )

    print(
        "Running categorical target relationship analysis..."
    )

    categorical_relationships = (
        analyze_categorical_relationships(
            df,
            target,
        )
    )

    print(
        "Running numerical target-rate analysis..."
    )

    numerical_target_rates = (
        analyze_numerical_target_rates(
            df,
            target,
        )
    )

    # --------------------------------------------------------
    # Fixed train / holdout split
    # --------------------------------------------------------

    print(
        "Creating fixed diagnostic train/holdout split..."
    )

    train_indices, holdout_indices = (
        train_test_split(
            np.arange(
                len(df)
            ),
            test_size=TEST_SIZE,
            stratify=target,
            random_state=RANDOM_STATE,
        )
    )

    train_df = (
        df.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    holdout_df = (
        df.iloc[
            holdout_indices
        ]
        .reset_index(
            drop=True
        )
    )

    train_target = (
        target.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    holdout_target = (
        target.iloc[
            holdout_indices
        ]
        .reset_index(
            drop=True
        )
    )

    print(
        "Comparing train/holdout feature-target relationships..."
    )

    train_holdout_numerical = (
        compare_numerical_relationships(
            train_df,
            holdout_df,
            train_target,
            holdout_target,
        )
    )

    train_holdout_categorical = (
        compare_categorical_relationships(
            train_df,
            holdout_df,
            train_target,
            holdout_target,
        )
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    print(
        "Comparing linear and nonlinear models..."
    )

    X, y = prepare_model_data(
        df
    )

    X_train = (
        X.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    X_holdout = (
        X.iloc[
            holdout_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_train = (
        y.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )

    y_holdout = (
        y.iloc[
            holdout_indices
        ]
        .reset_index(
            drop=True
        )
    )

    model_results = (
        evaluate_models(
            X_train,
            X_holdout,
            y_train,
            y_holdout,
        )
    )

    # --------------------------------------------------------
    # Signal concentration
    # --------------------------------------------------------

    print(
        "Calculating signal concentration..."
    )

    signal_concentration = (
        calculate_signal_concentration(
            numerical_relationships,
            categorical_relationships,
        )
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    train_prevalence = float(
        train_target.mean()
    )

    holdout_prevalence = float(
        holdout_target.mean()
    )

    dataset_summary = {
        "rows": int(
            len(df)
        ),
        "columns": int(
            len(df.columns)
        ),
        "features": int(
            len(
                numerical_features
                + categorical_features
            )
        ),
        "numerical_features": int(
            len(numerical_features)
        ),
        "categorical_features": int(
            len(categorical_features)
        ),
        "target_column": TARGET_COLUMN,
        "target_prevalence": float(
            target.mean()
        ),
        "attrition_yes": int(
            target.sum()
        ),
        "attrition_no": int(
            (target == 0).sum()
        ),
        "training_rows": int(
            len(train_df)
        ),
        "holdout_rows": int(
            len(holdout_df)
        ),
        "training_target_prevalence": (
            train_prevalence
        ),
        "holdout_target_prevalence": (
            holdout_prevalence
        ),
        "target_prevalence_difference": (
            holdout_prevalence
            - train_prevalence
        ),
    }

    # --------------------------------------------------------
    # Diagnostic flags
    # --------------------------------------------------------

    print(
        "Generating diagnostic flags..."
    )

    flags = generate_diagnostic_flags(
        numerical_relationships,
        categorical_relationships,
        train_holdout_numerical,
        train_holdout_categorical,
        model_results,
        train_prevalence,
        holdout_prevalence,
    )

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    print(
        "Generating overall diagnosis..."
    )

    diagnosis = (
        generate_overall_diagnosis(
            model_results,
            train_holdout_numerical,
            train_holdout_categorical,
        )
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    save_reports(
        numerical_relationships,
        categorical_relationships,
        numerical_target_rates,
        train_holdout_numerical,
        train_holdout_categorical,
        model_results,
        signal_concentration,
        flags,
        diagnosis,
        dataset_summary,
    )

    # --------------------------------------------------------
    # Print final report
    # --------------------------------------------------------

    print_report(
        numerical_relationships,
        categorical_relationships,
        train_holdout_numerical,
        train_holdout_categorical,
        model_results,
        signal_concentration,
        flags,
        diagnosis,
        dataset_summary,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()