"""
Feature Signal Quality Audit
============================

Purpose
-------
Diagnose whether the canonical employee attrition dataset contains:

1. Redundant / highly correlated features
2. Weak or noisy features
3. Features with meaningful marginal signal
4. Features with meaningful incremental predictive value
5. Signal concentrated in a small subset of variables
6. Differences between linear and nonlinear feature contribution

This is a diagnostic script.
It does NOT modify the dataset, final model, or production artifacts.

Canonical dataset:
    data/raw/employee_attrition_dataset_v2.csv

Outputs:
    reports/signal_analysis/feature_signal_quality/
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import pointbiserialr, spearmanr

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


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
    / "feature_signal_quality"
)

JSON_REPORT = OUTPUT_DIR / "feature_signal_quality_report.json"
SUMMARY_REPORT = OUTPUT_DIR / "feature_signal_quality_summary.txt"
REDUNDANCY_REPORT = OUTPUT_DIR / "feature_redundancy.csv"
INCREMENTAL_REPORT = OUTPUT_DIR / "feature_incremental_value.csv"


RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 3


# ============================================================
# HELPERS
# ============================================================

def clean_value(value: Any) -> Any:
    """
    Convert numpy / pandas / sklearn values into JSON-safe
    Python values.
    """

    if isinstance(value, dict):
        return {
            str(k): clean_value(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]

    if isinstance(value, np.ndarray):
        return [clean_value(v) for v in value.tolist()]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def save_json_report(report: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(JSON_REPORT, "w", encoding="utf-8") as f:
        json.dump(
            clean_value(report),
            f,
            indent=2,
            ensure_ascii=False,
        )


def save_text_summary(lines: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SUMMARY_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def make_one_hot_encoder():
    """
    Compatibility helper for different sklearn versions.
    """

    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if "Attrition" not in df.columns:
        raise ValueError(
            "Target column 'Attrition' was not found."
        )

    target = (
        df["Attrition"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            }
        )
    )

    if target.isna().any():
        raise ValueError(
            "Target column contains unsupported values."
        )

    X = df.drop(columns=["Attrition"]).copy()

    # Employee_ID is an identifier rather than predictive input.
    if "Employee_ID" in X.columns:
        X = X.drop(columns=["Employee_ID"])

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Features:             {X.shape[1]}")
    print(
        f"Target prevalence:    {target.mean() * 100:.2f}%"
    )

    return X, target


# ============================================================
# COLUMN TYPES
# ============================================================

def get_column_types(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:

    numerical_columns = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    return numerical_columns, categorical_columns


# ============================================================
# NUMERICAL SIGNAL
# ============================================================

def analyze_numerical_signal(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_columns: list[str],
) -> pd.DataFrame:

    rows = []

    for feature in numerical_columns:

        values = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        valid = values.notna() & y.notna()

        if valid.sum() < 10:
            continue

        x = values.loc[valid]
        target = y.loc[valid]

        try:
            point_corr, point_p = pointbiserialr(
                target,
                x,
            )
        except Exception:
            point_corr = np.nan
            point_p = np.nan

        try:
            spearman_corr, spearman_p = spearmanr(
                x,
                target,
            )
        except Exception:
            spearman_corr = np.nan
            spearman_p = np.nan

        rows.append(
            {
                "feature": feature,
                "point_biserial_correlation": point_corr,
                "point_biserial_p_value": point_p,
                "spearman_correlation": spearman_corr,
                "spearman_p_value": spearman_p,
                "absolute_point_biserial": abs(point_corr)
                if pd.notna(point_corr)
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_point_biserial",
            ascending=False,
        )

    return result


# ============================================================
# CATEGORICAL SIGNAL
# ============================================================

def analyze_categorical_signal(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
) -> pd.DataFrame:

    rows = []

    for feature in categorical_columns:

        temp = pd.DataFrame(
            {
                "feature": X[feature].astype(str),
                "target": y,
            }
        )

        rates = (
            temp.groupby("feature")["target"]
            .agg(["mean", "count"])
            .reset_index()
        )

        overall_rate = y.mean()

        if rates.empty:
            continue

        max_deviation = (
            rates["mean"] - overall_rate
        ).abs().max()

        # Approximate categorical effect strength.
        # This is deliberately diagnostic rather than inferential.
        weighted_variance = 0.0

        for _, row in rates.iterrows():
            weighted_variance += (
                row["count"]
                / len(temp)
                * (row["mean"] - overall_rate) ** 2
            )

        effect_strength = np.sqrt(
            weighted_variance
        )

        rows.append(
            {
                "feature": feature,
                "categories": int(rates.shape[0]),
                "max_target_rate_deviation": max_deviation,
                "categorical_effect_strength": effect_strength,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "categorical_effect_strength",
            ascending=False,
        )

    return result


# ============================================================
# NUMERICAL REDUNDANCY
# ============================================================

def analyze_numerical_redundancy(
    X: pd.DataFrame,
    numerical_columns: list[str],
) -> pd.DataFrame:

    if len(numerical_columns) < 2:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "pearson_correlation",
                "absolute_correlation",
                "redundancy_level",
            ]
        )

    corr = X[numerical_columns].corr(
        method="pearson"
    )

    rows = []

    for i, feature_a in enumerate(numerical_columns):

        for feature_b in numerical_columns[i + 1:]:

            value = corr.loc[
                feature_a,
                feature_b,
            ]

            if pd.isna(value):
                continue

            absolute_value = abs(value)

            if absolute_value >= 0.90:
                level = "very_high"

            elif absolute_value >= 0.80:
                level = "high"

            elif absolute_value >= 0.70:
                level = "moderate"

            else:
                level = "low"

            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "pearson_correlation": value,
                    "absolute_correlation": absolute_value,
                    "redundancy_level": level,
                }
            )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
        )

    return result


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    numerical_columns, categorical_columns = (
        get_column_types(X)
    )

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
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# CROSS-VALIDATED MODEL PERFORMANCE
# ============================================================

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
) -> tuple[float, float]:

    roc_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    pr_scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="average_precision",
        n_jobs=-1,
    )

    return (
        float(np.mean(roc_scores)),
        float(np.mean(pr_scores)),
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_model_families(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    print("Building baseline model comparison...")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(
            C=0.01,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []

    for name, estimator in models.items():

        print(f"Evaluating {name}...")

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    estimator,
                ),
            ]
        )

        roc_auc, pr_auc = evaluate_model(
            pipeline,
            X,
            y,
            cv,
        )

        rows.append(
            {
                "model": name,
                "roc_auc_mean": roc_auc,
                "pr_auc_mean": pr_auc,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    print("Calculating feature permutation importance...")

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    C=0.01,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X, y)

    importance = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    result = pd.DataFrame(
        {
            "feature": X.columns,
            "permutation_importance_mean":
                importance.importances_mean,
            "permutation_importance_std":
                importance.importances_std,
        }
    )

    result["absolute_importance"] = (
        result["permutation_importance_mean"]
        .abs()
    )

    result = result.sort_values(
        "absolute_importance",
        ascending=False,
    )

    return result


# ============================================================
# INCREMENTAL FEATURE GROUP ANALYSIS
# ============================================================

def evaluate_feature_subsets(
    X: pd.DataFrame,
    y: pd.Series,
    numerical_signal: pd.DataFrame,
    categorical_signal: pd.DataFrame,
) -> pd.DataFrame:

    print("Evaluating incremental feature subsets...")

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # Start with all features ranked by marginal signal.
    ranked_features: list[str] = []

    if not numerical_signal.empty:
        ranked_features.extend(
            numerical_signal[
                "feature"
            ].tolist()
        )

    if not categorical_signal.empty:
        ranked_features.extend(
            categorical_signal[
                "feature"
            ].tolist()
        )

    # Remove duplicates while preserving order.
    ranked_features = list(
        dict.fromkeys(ranked_features)
    )

    subsets = []

    # Full model.
    subsets.append(
        (
            "all_features",
            X.columns.tolist(),
        )
    )

    # Top 3 / 5 / 10 marginal features.
    for count in [3, 5, 10]:

        if len(ranked_features) >= count:

            subsets.append(
                (
                    f"top_{count}_marginal_features",
                    ranked_features[:count],
                )
            )

    rows = []

    for subset_name, features in subsets:

        if not features:
            continue

        X_subset = X[features].copy()

        preprocessor = build_preprocessor(
            X_subset
        )

        model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.01,
                        max_iter=5000,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        roc_scores = cross_val_score(
            model,
            X_subset,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

        pr_scores = cross_val_score(
            model,
            X_subset,
            y,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )

        rows.append(
            {
                "subset": subset_name,
                "feature_count": len(features),
                "roc_auc_mean": float(
                    np.mean(roc_scores)
                ),
                "roc_auc_std": float(
                    np.std(roc_scores)
                ),
                "pr_auc_mean": float(
                    np.mean(pr_scores)
                ),
                "pr_auc_std": float(
                    np.std(pr_scores)
                ),
                "features": ", ".join(features),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SIGNAL CONCENTRATION
# ============================================================

def calculate_signal_concentration(
    numerical_signal: pd.DataFrame,
    categorical_signal: pd.DataFrame,
    permutation: pd.DataFrame,
) -> dict[str, Any]:

    result: dict[str, Any] = {}

    if numerical_signal.empty:
        result["maximum_numerical_abs_correlation"] = 0.0
        result["numerical_features_abs_corr_ge_0_05"] = 0
        result["numerical_features_abs_corr_ge_0_10"] = 0

    else:
        abs_corr = numerical_signal[
            "absolute_point_biserial"
        ]

        result[
            "maximum_numerical_abs_correlation"
        ] = float(abs_corr.max())

        result[
            "numerical_features_abs_corr_ge_0_05"
        ] = int((abs_corr >= 0.05).sum())

        result[
            "numerical_features_abs_corr_ge_0_10"
        ] = int((abs_corr >= 0.10).sum())

    if categorical_signal.empty:
        result[
            "maximum_categorical_effect_strength"
        ] = 0.0

    else:
        result[
            "maximum_categorical_effect_strength"
        ] = float(
            categorical_signal[
                "categorical_effect_strength"
            ].max()
        )

    if not permutation.empty:

        positive_importance = permutation[
            "permutation_importance_mean"
        ].clip(lower=0)

        total = positive_importance.sum()

        if total > 0:

            top_5_share = (
                positive_importance.head(5).sum()
                / total
            )

            top_10_share = (
                positive_importance.head(10).sum()
                / total
            )

        else:
            top_5_share = 0.0
            top_10_share = 0.0

        result["top_5_positive_importance_share"] = (
            float(top_5_share)
        )

        result["top_10_positive_importance_share"] = (
            float(top_10_share)
        )

    else:

        result["top_5_positive_importance_share"] = 0.0
        result["top_10_positive_importance_share"] = 0.0

    return result


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_flags(
    numerical_signal: pd.DataFrame,
    categorical_signal: pd.DataFrame,
    redundancy: pd.DataFrame,
    permutation: pd.DataFrame,
    incremental: pd.DataFrame,
) -> list[str]:

    flags = []

    if not numerical_signal.empty:

        count_05 = int(
            (
                numerical_signal[
                    "absolute_point_biserial"
                ]
                >= 0.05
            ).sum()
        )

        count_10 = int(
            (
                numerical_signal[
                    "absolute_point_biserial"
                ]
                >= 0.10
            ).sum()
        )

        if count_10 > 0:
            flags.append(
                f"{count_10} numerical feature(s) "
                "show |point-biserial correlation| >= 0.10."
            )

        elif count_05 > 0:
            flags.append(
                f"{count_05} numerical feature(s) "
                "show |point-biserial correlation| >= 0.05."
            )

        else:
            flags.append(
                "Numerical marginal signal is weak "
                "across the available features."
            )

    if not categorical_signal.empty:

        strongest = float(
            categorical_signal[
                "categorical_effect_strength"
            ].max()
        )

        if strongest >= 0.10:
            flags.append(
                "At least one categorical feature "
                "shows a relatively strong target-rate effect."
            )

        else:
            flags.append(
                "Categorical feature effects are generally modest."
            )

    if not redundancy.empty:

        very_high = int(
            (
                redundancy[
                    "absolute_correlation"
                ]
                >= 0.90
            ).sum()
        )

        high = int(
            (
                redundancy[
                    "absolute_correlation"
                ]
                >= 0.80
            ).sum()
        )

        if very_high > 0:
            flags.append(
                f"{very_high} feature pair(s) have "
                "very high correlation (|r| >= 0.90)."
            )

        elif high > 0:
            flags.append(
                f"{high} feature pair(s) have "
                "high correlation (|r| >= 0.80)."
            )

        else:
            flags.append(
                "No severe numerical feature redundancy "
                "was detected."
            )

    if not permutation.empty:

        positive = permutation[
            permutation[
                "permutation_importance_mean"
            ] > 0
        ]

        if len(positive) <= 5:
            flags.append(
                "Predictive importance is concentrated "
                "in a small number of features."
            )

        negative_count = int(
            (
                permutation[
                    "permutation_importance_mean"
                ]
                < 0
            ).sum()
        )

        if negative_count > 0:
            flags.append(
                f"{negative_count} feature(s) have negative "
                "mean permutation importance in the fitted "
                "diagnostic model."
            )

    if not incremental.empty:

        full = incremental[
            incremental["subset"] == "all_features"
        ]

        top5 = incremental[
            incremental["subset"]
            == "top_5_marginal_features"
        ]

        if not full.empty and not top5.empty:

            full_auc = float(
                full.iloc[0]["roc_auc_mean"]
            )

            top5_auc = float(
                top5.iloc[0]["roc_auc_mean"]
            )

            if top5_auc >= full_auc - 0.01:
                flags.append(
                    "The top five marginal features perform "
                    "similarly to the full feature set, suggesting "
                    "substantial weak/noisy feature content."
                )

            elif full_auc - top5_auc >= 0.03:
                flags.append(
                    "The full feature set adds measurable "
                    "predictive value beyond the top five "
                    "marginal features."
                )

    return flags


# ============================================================
# OVERALL DIAGNOSIS
# ============================================================

def generate_diagnosis(
    numerical_signal: pd.DataFrame,
    categorical_signal: pd.DataFrame,
    redundancy: pd.DataFrame,
    permutation: pd.DataFrame,
    incremental: pd.DataFrame,
) -> str:

    max_corr = 0.0

    if not numerical_signal.empty:
        max_corr = float(
            numerical_signal[
                "absolute_point_biserial"
            ].max()
        )

    high_redundancy = 0

    if not redundancy.empty:
        high_redundancy = int(
            (
                redundancy[
                    "absolute_correlation"
                ]
                >= 0.80
            ).sum()
        )

    concentrated = False

    if not permutation.empty:

        positive = permutation[
            "permutation_importance_mean"
        ].clip(lower=0)

        total = positive.sum()

        if total > 0:

            top5_share = (
                positive.head(5).sum()
                / total
            )

            concentrated = top5_share >= 0.70

    if (
        max_corr < 0.05
        and high_redundancy == 0
    ):
        return (
            "The dataset shows weak marginal feature signal "
            "and no major numerical redundancy. This would "
            "suggest that the current performance ceiling is "
            "more likely related to limited predictive signal "
            "than to excessive feature duplication."
        )

    if high_redundancy > 0:
        return (
            "The dataset contains meaningful feature redundancy. "
            "Some predictive information may be duplicated across "
            "correlated variables, making coefficient estimates "
            "less efficient and potentially increasing model "
            "variance."
        )

    if concentrated:
        return (
            "Predictive information appears concentrated in a "
            "relatively small subset of features. The remaining "
            "features may contribute limited independent signal "
            "or introduce noise."
        )

    return (
        "The dataset contains measurable predictive structure, "
        "but the available feature set does not appear to contain "
        "a dominant high-strength signal source. Further model "
        "improvement should therefore focus on feature quality, "
        "interactions, and target construction rather than "
        "unrestricted hyperparameter tuning."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Running feature signal quality audit...")
    print()

    X, y = load_dataset()

    print()
    print("Analyzing numerical feature-target relationships...")

    numerical_columns, categorical_columns = (
        get_column_types(X)
    )

    numerical_signal = analyze_numerical_signal(
        X,
        y,
        numerical_columns,
    )

    print("Analyzing categorical feature-target relationships...")

    categorical_signal = analyze_categorical_signal(
        X,
        y,
        categorical_columns,
    )

    print("Analyzing numerical feature redundancy...")

    redundancy = analyze_numerical_redundancy(
        X,
        numerical_columns,
    )

    print("Comparing diagnostic model families...")

    model_comparison = compare_model_families(
        X,
        y,
    )

    permutation = calculate_permutation_importance(
        X,
        y,
    )

    incremental = evaluate_feature_subsets(
        X,
        y,
        numerical_signal,
        categorical_signal,
    )

    concentration = calculate_signal_concentration(
        numerical_signal,
        categorical_signal,
        permutation,
    )

    print("Generating diagnostic flags...")

    flags = generate_flags(
        numerical_signal,
        categorical_signal,
        redundancy,
        permutation,
        incremental,
    )

    diagnosis = generate_diagnosis(
        numerical_signal,
        categorical_signal,
        redundancy,
        permutation,
        incremental,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    redundancy.to_csv(
        REDUNDANCY_REPORT,
        index=False,
    )

    incremental.to_csv(
        INCREMENTAL_REPORT,
        index=False,
    )

    # --------------------------------------------------------
    # JSON REPORT
    # --------------------------------------------------------

    report = {
        "dataset": {
            "path": str(
                DATASET_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target": "Attrition",
            "target_prevalence": float(y.mean()),
        },
        "numerical_signal": (
            numerical_signal.to_dict(
                orient="records"
            )
        ),
        "categorical_signal": (
            categorical_signal.to_dict(
                orient="records"
            )
        ),
        "redundancy": (
            redundancy.to_dict(
                orient="records"
            )
        ),
        "model_comparison": (
            model_comparison.to_dict(
                orient="records"
            )
        ),
        "permutation_importance": (
            permutation.to_dict(
                orient="records"
            )
        ),
        "incremental_feature_value": (
            incremental.to_dict(
                orient="records"
            )
        ),
        "signal_concentration": concentration,
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    save_json_report(report)

    # --------------------------------------------------------
    # TERMINAL REPORT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("EMPLOYEE ATTRITION — FEATURE SIGNAL QUALITY")
    print("=" * 60)

    print()
    print("[DATASET]")
    print(f"Rows:                 {len(X)}")
    print(f"Features:             {X.shape[1]}")
    print(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )

    print()
    print("[TOP NUMERICAL SIGNAL]")

    if numerical_signal.empty:
        print("No numerical relationships available.")

    else:
        print(
            numerical_signal[
                [
                    "feature",
                    "point_biserial_correlation",
                    "point_biserial_p_value",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("[TOP CATEGORICAL SIGNAL]")

    if categorical_signal.empty:
        print("No categorical relationships available.")

    else:
        print(
            categorical_signal[
                [
                    "feature",
                    "categories",
                    "max_target_rate_deviation",
                    "categorical_effect_strength",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("[TOP NUMERICAL REDUNDANCY]")

    if redundancy.empty:
        print("No numerical feature pairs available.")

    else:
        print(
            redundancy[
                [
                    "feature_a",
                    "feature_b",
                    "pearson_correlation",
                    "redundancy_level",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("[MODEL COMPARISON]")
    print(
        model_comparison.to_string(
            index=False
        )
    )

    print()
    print("[TOP PERMUTATION IMPORTANCE]")

    if permutation.empty:
        print("No permutation results available.")

    else:
        print(
            permutation[
                [
                    "feature",
                    "permutation_importance_mean",
                    "permutation_importance_std",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print()
    print("[INCREMENTAL FEATURE VALUE]")

    if incremental.empty:
        print("No incremental feature results available.")

    else:
        print(
            incremental[
                [
                    "subset",
                    "feature_count",
                    "roc_auc_mean",
                    "roc_auc_std",
                    "pr_auc_mean",
                    "pr_auc_std",
                ]
            ]
            .to_string(index=False)
        )

    print()
    print("[SIGNAL CONCENTRATION]")

    print(
        f"Maximum numerical |r|: "
        f"{concentration.get('maximum_numerical_abs_correlation', 0):.4f}"
    )

    print(
        "Numerical features |r| >= 0.05: "
        f"{concentration.get('numerical_features_abs_corr_ge_0_05', 0)}"
    )

    print(
        "Numerical features |r| >= 0.10: "
        f"{concentration.get('numerical_features_abs_corr_ge_0_10', 0)}"
    )

    print(
        "Maximum categorical effect: "
        f"{concentration.get('maximum_categorical_effect_strength', 0):.4f}"
    )

    print(
        "Top-5 positive importance share: "
        f"{concentration.get('top_5_positive_importance_share', 0) * 100:.2f}%"
    )

    print(
        "Top-10 positive importance share: "
        f"{concentration.get('top_10_positive_importance_share', 0) * 100:.2f}%"
    )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            print(f"- {flag}")
    else:
        print("- No major diagnostic flags generated.")

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    print()
    print("[OUTPUT]")
    print(f"Reports:              {OUTPUT_DIR}")
    print(f"JSON report:          {JSON_REPORT}")
    print(f"Redundancy report:    {REDUNDANCY_REPORT}")
    print(f"Incremental report:   {INCREMENTAL_REPORT}")
    print(f"Summary report:       {SUMMARY_REPORT}")

    print()
    print("=" * 60)
    print("FEATURE SIGNAL QUALITY AUDIT COMPLETE")
    print("=" * 60)

    # --------------------------------------------------------
    # TEXT SUMMARY
    # --------------------------------------------------------

    summary_lines = [
        "EMPLOYEE ATTRITION — FEATURE SIGNAL QUALITY",
        "=" * 60,
        "",
        "[DATASET]",
        f"Rows: {len(X)}",
        f"Features: {X.shape[1]}",
        f"Target prevalence: {y.mean() * 100:.2f}%",
        "",
        "[MODEL COMPARISON]",
        model_comparison.to_string(index=False),
        "",
        "[SIGNAL CONCENTRATION]",
        (
            f"Maximum numerical |r|: "
            f"{concentration.get('maximum_numerical_abs_correlation', 0):.4f}"
        ),
        (
            "Numerical features |r| >= 0.05: "
            f"{concentration.get('numerical_features_abs_corr_ge_0_05', 0)}"
        ),
        (
            "Numerical features |r| >= 0.10: "
            f"{concentration.get('numerical_features_abs_corr_ge_0_10', 0)}"
        ),
        (
            "Maximum categorical effect: "
            f"{concentration.get('maximum_categorical_effect_strength', 0):.4f}"
        ),
        (
            "Top-5 importance share: "
            f"{concentration.get('top_5_positive_importance_share', 0) * 100:.2f}%"
        ),
        "",
        "[DIAGNOSTIC FLAGS]",
        *[
            f"- {flag}"
            for flag in flags
        ],
        "",
        "[OVERALL DIAGNOSIS]",
        diagnosis,
        "",
        "[OUTPUT]",
        str(JSON_REPORT),
        str(SUMMARY_REPORT),
        str(REDUNDANCY_REPORT),
        str(INCREMENTAL_REPORT),
    ]

    save_text_summary(summary_lines)


if __name__ == "__main__":
    main()