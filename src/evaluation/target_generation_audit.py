"""
Employee Attrition — Target Generation Audit

Purpose
-------
Investigate whether the synthetic Attrition target contains stable,
learnable relationships with the available employee features.

This module is diagnostic only.

It does NOT:
- modify the dataset
- modify the model
- tune the production threshold
- use the untouched holdout to tune anything
- declare the model production-ready

Run from the project root:

    python -m src.evaluation.target_generation_audit
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr

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
    / "target_generation_audit"
)

JSON_PATH = OUTPUT_DIR / "target_generation_audit_report.json"
SUMMARY_PATH = OUTPUT_DIR / "target_generation_audit_summary.txt"
UNIVARIATE_PATH = OUTPUT_DIR / "target_univariate_relationships.csv"
INTERACTION_PATH = OUTPUT_DIR / "target_interaction_analysis.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attrition"
RANDOM_STATE = 42

TARGET_POSITIVE = "Yes"
TARGET_NEGATIVE = "No"

TOP_NUMERICAL_FEATURES = 12

INTERACTION_PAIRS = [
    ("Job_Satisfaction", "Overtime"),
    ("Work_Life_Balance", "Average_Hours_Worked_Per_Week"),
    ("Distance_From_Home", "Job_Satisfaction"),
    ("Years_Since_Last_Promotion", "Job_Satisfaction"),
    ("Years_at_Company", "Job_Satisfaction"),
    ("Job_Involvement", "Overtime"),
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_output_directory() -> None:
    """Create the audit output directory if necessary."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        result = float(value)

        if np.isnan(result) or np.isinf(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """Load the V2 employee attrition dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    return df


# ============================================================
# BASIC TARGET AUDIT
# ============================================================

def audit_target_distribution(
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze the target distribution."""
    counts = df[TARGET_COLUMN].value_counts(dropna=False)

    total = len(df)

    positive_count = int(
        counts.get(TARGET_POSITIVE, 0)
    )

    negative_count = int(
        counts.get(TARGET_NEGATIVE, 0)
    )

    positive_rate = (
        positive_count / total
        if total > 0
        else 0.0
    )

    entropy = 0.0

    for count in counts.values:
        if count <= 0:
            continue

        probability = count / total
        entropy -= probability * np.log2(probability)

    return {
        "total_rows": total,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_rate,
        "entropy_bits": entropy,
        "unique_values": [
            str(value)
            for value in counts.index.tolist()
        ],
        "counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
    }


# ============================================================
# NUMERICAL RELATIONSHIPS
# ============================================================

def get_numeric_features(
    df: pd.DataFrame,
) -> list[str]:
    """Return usable numerical predictor columns."""
    excluded = {
        TARGET_COLUMN,
        "Employee_ID",
    }

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column not in excluded
    ]


def numerical_target_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate point-biserial relationships between numerical
    features and Attrition.
    """
    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .eq(TARGET_POSITIVE)
        .astype(int)
    )

    rows: list[dict[str, Any]] = []

    for feature in get_numeric_features(df):
        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid = x.notna() & y.notna()

        if valid.sum() < 3:
            continue

        if x[valid].nunique() < 2:
            continue

        try:
            correlation, p_value = pointbiserialr(
                y[valid],
                x[valid],
            )
        except Exception:
            correlation = np.nan
            p_value = np.nan

        positive_mean = (
            x[valid & y.eq(1)].mean()
            if (valid & y.eq(1)).sum() > 0
            else np.nan
        )

        negative_mean = (
            x[valid & y.eq(0)].mean()
            if (valid & y.eq(0)).sum() > 0
            else np.nan
        )

        rows.append(
            {
                "feature": feature,
                "point_biserial_correlation": safe_float(
                    correlation,
                    default=0.0,
                ),
                "absolute_correlation": abs(
                    safe_float(correlation)
                ),
                "p_value": safe_float(
                    p_value,
                    default=1.0,
                ),
                "mean_attrition_yes": safe_float(
                    positive_mean,
                    default=np.nan,
                ),
                "mean_attrition_no": safe_float(
                    negative_mean,
                    default=np.nan,
                ),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values(
        "absolute_correlation",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ============================================================
# CATEGORICAL RELATIONSHIPS
# ============================================================

def get_categorical_features(
    df: pd.DataFrame,
) -> list[str]:
    """Return categorical predictor columns."""
    excluded = {
        TARGET_COLUMN,
        "Employee_ID",
    }

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return [
        column
        for column in categorical_columns
        if column not in excluded
    ]


def cramers_v_from_table(
    table: pd.DataFrame,
) -> float:
    """Calculate Cramer's V."""
    if table.empty:
        return 0.0

    try:
        chi2, _, _, _ = chi2_contingency(
            table,
            correction=False,
        )
    except Exception:
        return 0.0

    n = table.to_numpy().sum()

    if n <= 0:
        return 0.0

    rows, columns = table.shape

    denominator = min(
        rows - 1,
        columns - 1,
    )

    if denominator <= 0:
        return 0.0

    phi2 = chi2 / n

    return float(
        np.sqrt(phi2 / denominator)
    )


def categorical_target_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze categorical feature relationships with Attrition."""
    rows: list[dict[str, Any]] = []

    for feature in get_categorical_features(df):
        working = df[
            [feature, TARGET_COLUMN]
        ].copy()

        working[feature] = working[feature].astype(str)

        contingency = pd.crosstab(
            working[feature],
            working[TARGET_COLUMN],
        )

        if contingency.empty:
            continue

        try:
            chi2, p_value, _, _ = chi2_contingency(
                contingency,
                correction=False,
            )
        except Exception:
            chi2 = 0.0
            p_value = 1.0

        v = cramers_v_from_table(
            contingency
        )

        total_by_category = (
            working.groupby(feature)
            .size()
        )

        positive_by_category = (
            working[
                working[TARGET_COLUMN].eq(
                    TARGET_POSITIVE
                )
            ]
            .groupby(feature)
            .size()
        )

        for category in total_by_category.index:
            total = int(
                total_by_category.loc[category]
            )

            positive = int(
                positive_by_category.get(
                    category,
                    0,
                )
            )

            attrition_rate = (
                positive / total
                if total > 0
                else 0.0
            )

            rows.append(
                {
                    "feature": feature,
                    "category": str(category),
                    "category_count": total,
                    "attrition_count": positive,
                    "attrition_rate": attrition_rate,
                    "cramers_v": v,
                    "p_value": safe_float(
                        p_value,
                        default=1.0,
                    ),
                    "chi_square": safe_float(
                        chi2,
                        default=0.0,
                    ),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result["rate_deviation_from_global"] = (
        result["attrition_rate"]
        - df[TARGET_COLUMN]
        .astype(str)
        .eq(TARGET_POSITIVE)
        .mean()
    )

    result["absolute_rate_deviation"] = (
        result["rate_deviation_from_global"]
        .abs()
    )

    result = result.sort_values(
        [
            "absolute_rate_deviation",
            "cramers_v",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


# ============================================================
# TARGET RATE BY NUMERICAL BINS
# ============================================================

def numerical_binned_target_rates(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """
    Bin numerical variables into quantiles and calculate
    attrition rates.

    This helps detect nonlinear target relationships that
    correlation alone can miss.
    """
    rows: list[dict[str, Any]] = []

    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .eq(TARGET_POSITIVE)
        .astype(int)
    )

    global_rate = y.mean()

    for feature in features:
        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid = x.notna()

        if valid.sum() < 20:
            continue

        if x[valid].nunique() < 4:
            continue

        try:
            bins = pd.qcut(
                x[valid],
                q=4,
                duplicates="drop",
            )
        except Exception:
            continue

        temp = pd.DataFrame(
            {
                "feature": feature,
                "value": x[valid],
                "target": y[valid],
                "bin": bins,
            }
        )

        grouped = (
            temp.groupby(
                "bin",
                observed=True,
            )
            .agg(
                count=("target", "size"),
                attrition_rate=("target", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rate = safe_float(
                row["attrition_rate"]
            )

            rows.append(
                {
                    "feature": feature,
                    "bin": str(row["bin"]),
                    "count": int(row["count"]),
                    "attrition_rate": rate,
                    "global_attrition_rate": global_rate,
                    "rate_difference": (
                        rate - global_rate
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# INTERACTION ANALYSIS
# ============================================================

def interaction_target_analysis(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Investigate pairwise feature interactions against Attrition.

    Numerical variables are discretized into low/high groups.
    Categorical variables are used directly.
    """
    rows: list[dict[str, Any]] = []

    target = (
        df[TARGET_COLUMN]
        .astype(str)
        .eq(TARGET_POSITIVE)
        .astype(int)
    )

    for feature_a, feature_b in INTERACTION_PAIRS:

        if (
            feature_a not in df.columns
            or feature_b not in df.columns
        ):
            continue

        working = pd.DataFrame(
            {
                "a": df[feature_a],
                "b": df[feature_b],
                "target": target,
            }
        )

        # ----------------------------------------------------
        # Numerical / categorical transformation
        # ----------------------------------------------------

        for column in ["a", "b"]:
            if pd.api.types.is_numeric_dtype(
                working[column]
            ):
                numeric = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )

                median = numeric.median()

                if pd.isna(median):
                    continue

                working[column] = np.where(
                    numeric <= median,
                    "Low",
                    "High",
                )
            else:
                working[column] = (
                    working[column]
                    .astype(str)
                )

        working = working.dropna()

        if working.empty:
            continue

        working["interaction"] = (
            working["a"].astype(str)
            + " | "
            + working["b"].astype(str)
        )

        grouped = (
            working.groupby("interaction")
            .agg(
                count=("target", "size"),
                attrition_rate=("target", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            rows.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "interaction_group": row[
                        "interaction"
                    ],
                    "count": int(row["count"]),
                    "attrition_rate": safe_float(
                        row["attrition_rate"]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SIGNAL CONCENTRATION
# ============================================================

def calculate_signal_concentration(
    numerical_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    global_rate: float,
) -> dict[str, Any]:
    """
    Determine whether target signal appears concentrated in
    a small number of variables.
    """
    result: dict[str, Any] = {}

    if numerical_results.empty:
        result["max_numeric_abs_correlation"] = 0.0
        result["numeric_features_above_0_10"] = 0
        result["numeric_features_above_0_05"] = 0
    else:
        abs_corr = numerical_results[
            "absolute_correlation"
        ]

        result["max_numeric_abs_correlation"] = (
            safe_float(abs_corr.max())
        )

        result["numeric_features_above_0_10"] = int(
            (abs_corr >= 0.10).sum()
        )

        result["numeric_features_above_0_05"] = int(
            (abs_corr >= 0.05).sum()
        )

    if categorical_results.empty:
        result["max_cramers_v"] = 0.0
    else:
        result["max_cramers_v"] = safe_float(
            categorical_results[
                "cramers_v"
            ].max()
        )

    if categorical_results.empty:
        result["max_category_rate_deviation"] = 0.0
    else:
        result["max_category_rate_deviation"] = safe_float(
            categorical_results[
                "absolute_rate_deviation"
            ].max()
        )

    result["global_attrition_rate"] = global_rate

    return result


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def build_diagnostic_flags(
    target_summary: dict[str, Any],
    numerical_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
) -> list[str]:
    """Build human-readable diagnostic flags."""
    flags: list[str] = []

    global_rate = target_summary[
        "positive_rate"
    ]

    # --------------------------------------------------------
    # Numerical signal
    # --------------------------------------------------------

    if numerical_results.empty:
        flags.append(
            "No usable numerical target relationships were detected."
        )
    else:
        max_corr = safe_float(
            numerical_results[
                "absolute_correlation"
            ].max()
        )

        if max_corr < 0.05:
            flags.append(
                "Numerical target relationships are extremely weak."
            )
        elif max_corr < 0.10:
            flags.append(
                "Numerical target relationships are weak, "
                "with no feature exceeding |r| = 0.10."
            )
        else:
            flags.append(
                "At least one numerical feature shows "
                "a potentially meaningful marginal relationship "
                "with Attrition."
            )

    # --------------------------------------------------------
    # Categorical signal
    # --------------------------------------------------------

    if categorical_results.empty:
        flags.append(
            "No usable categorical target relationships were detected."
        )
    else:
        max_v = safe_float(
            categorical_results[
                "cramers_v"
            ].max()
        )

        if max_v < 0.05:
            flags.append(
                "Categorical target relationships are very weak."
            )
        elif max_v < 0.10:
            flags.append(
                "Categorical relationships exist but remain modest."
            )
        else:
            flags.append(
                "At least one categorical feature shows "
                "a potentially meaningful association with Attrition."
            )

    # --------------------------------------------------------
    # Interaction signal
    # --------------------------------------------------------

    if interaction_results.empty:
        flags.append(
            "No interaction analysis results were available."
        )
    else:
        grouped = (
            interaction_results
            .groupby(
                ["feature_a", "feature_b"]
            )["attrition_rate"]
            .agg(["min", "max"])
            .reset_index()
        )

        if not grouped.empty:
            grouped["range"] = (
                grouped["max"]
                - grouped["min"]
            )

            max_range = safe_float(
                grouped["range"].max()
            )

            if max_range >= 0.20:
                flags.append(
                    "Some feature combinations show large "
                    "differences in observed attrition rates, "
                    "suggesting possible interaction effects."
                )
            elif max_range >= 0.10:
                flags.append(
                    "Some feature combinations show moderate "
                    "differences in observed attrition rates."
                )
            else:
                flags.append(
                    "Observed interaction groups do not show "
                    "large attrition-rate separation."
                )

    # --------------------------------------------------------
    # Class balance
    # --------------------------------------------------------

    if global_rate < 0.10:
        flags.append(
            "The target is strongly imbalanced toward No Attrition."
        )
    elif global_rate < 0.20:
        flags.append(
            "The target has moderate class imbalance."
        )
    elif global_rate > 0.40:
        flags.append(
            "The positive class is unusually prevalent."
        )
    else:
        flags.append(
            "The target prevalence is within a workable range "
            "for binary classification."
        )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = safe_float(
        target_summary["entropy_bits"]
    )

    if entropy < 0.50:
        flags.append(
            "Target entropy is low, indicating a highly "
            "imbalanced target distribution."
        )
    else:
        flags.append(
            "Target entropy does not indicate an extreme "
            "class-concentration problem."
        )

    return flags


# ============================================================
# OVERALL DIAGNOSIS
# ============================================================

def build_overall_diagnosis(
    numerical_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
) -> str:
    """Produce the overall target-generation diagnosis."""
    max_numeric = 0.0

    if not numerical_results.empty:
        max_numeric = safe_float(
            numerical_results[
                "absolute_correlation"
            ].max()
        )

    max_categorical = 0.0

    if not categorical_results.empty:
        max_categorical = safe_float(
            categorical_results[
                "cramers_v"
            ].max()
        )

    max_interaction_range = 0.0

    if not interaction_results.empty:
        grouped = (
            interaction_results
            .groupby(
                ["feature_a", "feature_b"]
            )["attrition_rate"]
            .agg(["min", "max"])
        )

        if not grouped.empty:
            max_interaction_range = safe_float(
                (
                    grouped["max"]
                    - grouped["min"]
                ).max()
            )

    # --------------------------------------------------------
    # Stronger evidence
    # --------------------------------------------------------

    if (
        max_numeric >= 0.10
        or max_categorical >= 0.10
        or max_interaction_range >= 0.20
    ):
        return (
            "The target contains identifiable relationships with "
            "the available employee features. The evidence suggests "
            "that the dataset contains genuine predictive structure, "
            "although the strength and generalization of that signal "
            "still require validation."
        )

    # --------------------------------------------------------
    # Moderate evidence
    # --------------------------------------------------------

    if (
        max_numeric >= 0.05
        or max_categorical >= 0.05
        or max_interaction_range >= 0.10
    ):
        return (
            "The target contains modest predictive structure, but "
            "the signal is relatively weak. This is consistent with "
            "the moderate cross-validated model performance observed "
            "earlier and suggests that model complexity alone may not "
            "solve the generalization problem."
        )

    # --------------------------------------------------------
    # Weak evidence
    # --------------------------------------------------------

    return (
        "The target shows weak marginal and interaction-level "
        "relationships with the available features. This suggests "
        "that the current dataset may contain insufficient "
        "learnable signal for reliable prediction, and that "
        "target-generation noise or weak feature construction "
        "should be investigated before further model optimization."
    )


# ============================================================
# SUMMARY REPORT
# ============================================================

def write_summary(
    target_summary: dict[str, Any],
    numerical_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
    flags: list[str],
    diagnosis: str,
) -> None:
    """Write the human-readable text report."""

    lines: list[str] = []

    lines.append(
        "============================================================"
    )
    lines.append(
        "EMPLOYEE ATTRITION — TARGET GENERATION AUDIT"
    )
    lines.append(
        "============================================================"
    )
    lines.append("")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    lines.append("[DATASET]")
    lines.append(
        f"Rows:                 {target_summary['total_rows']}"
    )
    lines.append(
        f"Positive target:      {target_summary['positive_count']}"
    )
    lines.append(
        f"Negative target:      {target_summary['negative_count']}"
    )
    lines.append(
        f"Attrition rate:       "
        f"{target_summary['positive_rate'] * 100:.2f}%"
    )
    lines.append(
        f"Target entropy:       "
        f"{target_summary['entropy_bits']:.4f} bits"
    )
    lines.append("")

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    lines.append("[TOP NUMERICAL RELATIONSHIPS]")

    if numerical_results.empty:
        lines.append(
            "No numerical relationships available."
        )
    else:
        top = numerical_results.head(
            TOP_NUMERICAL_FEATURES
        )

        for _, row in top.iterrows():
            lines.append(
                f"{str(row['feature']):35s}"
                f"r={row['point_biserial_correlation']:+.4f}  "
                f"p={row['p_value']:.4f}"
            )

    lines.append("")

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    lines.append("[TOP CATEGORICAL RELATIONSHIPS]")

    if categorical_results.empty:
        lines.append(
            "No categorical relationships available."
        )
    else:
        # Show the strongest feature/category combinations.
        top = categorical_results.head(10)

        for _, row in top.iterrows():
            lines.append(
                f"{str(row['feature']):25s}"
                f"{str(row['category']):18s}"
                f"rate={row['attrition_rate'] * 100:6.2f}%  "
                f"V={row['cramers_v']:.4f}  "
                f"p={row['p_value']:.4f}"
            )

    lines.append("")

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    lines.append("[INTERACTION ANALYSIS]")

    if interaction_results.empty:
        lines.append(
            "No interaction results available."
        )
    else:
        grouped = (
            interaction_results
            .groupby(
                ["feature_a", "feature_b"]
            )
            .agg(
                min_rate=("attrition_rate", "min"),
                max_rate=("attrition_rate", "max"),
                groups=("interaction_group", "count"),
            )
            .reset_index()
        )

        grouped["rate_range"] = (
            grouped["max_rate"]
            - grouped["min_rate"]
        )

        grouped = grouped.sort_values(
            "rate_range",
            ascending=False,
        )

        for _, row in grouped.iterrows():
            lines.append(
                f"{row['feature_a']} × "
                f"{row['feature_b']}: "
                f"min={row['min_rate'] * 100:.2f}%  "
                f"max={row['max_rate'] * 100:.2f}%  "
                f"range={row['rate_range'] * 100:.2f} pp"
            )

    lines.append("")

    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    lines.append("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        lines.append(f"- {flag}")

    lines.append("")

    # --------------------------------------------------------
    # Overall diagnosis
    # --------------------------------------------------------

    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(diagnosis)
    lines.append("")

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    lines.append("[OUTPUT]")
    lines.append(
        f"Reports:              {OUTPUT_DIR}"
    )
    lines.append(
        f"JSON report:          {JSON_PATH}"
    )
    lines.append(
        f"Univariate report:    {UNIVARIATE_PATH}"
    )
    lines.append(
        f"Interaction report:   {INTERACTION_PATH}"
    )
    lines.append("")

    lines.append(
        "============================================================"
    )
    lines.append(
        "TARGET GENERATION AUDIT COMPLETE"
    )
    lines.append(
        "============================================================"
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# JSON REPORT
# ============================================================

def write_json_report(
    target_summary: dict[str, Any],
    numerical_results: pd.DataFrame,
    categorical_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
    flags: list[str],
    diagnosis: str,
) -> None:
    """Write structured JSON audit results."""

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(numerical_results))
            if False
            else None,
        },
        "target": target_summary,
        "numerical_relationships": (
            numerical_results.to_dict(
                orient="records"
            )
            if not numerical_results.empty
            else []
        ),
        "categorical_relationships": (
            categorical_results.to_dict(
                orient="records"
            )
            if not categorical_results.empty
            else []
        ),
        "interaction_analysis": (
            interaction_results.to_dict(
                orient="records"
            )
            if not interaction_results.empty
            else []
        ),
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    # Correct the dataset row count explicitly.
    report["dataset"]["rows"] = (
        target_summary["total_rows"]
    )

    JSON_PATH.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete target-generation audit."""

    print("")
    print(
        "Running target generation audit..."
    )

    ensure_output_directory()

    print(
        "Loading dataset..."
    )

    df = load_dataset()

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 {len(df)}"
    )

    print(
        f"Columns:              {len(df.columns)}"
    )

    print(
        "Analyzing target distribution..."
    )

    target_summary = audit_target_distribution(
        df
    )

    print(
        "Running numerical target relationship analysis..."
    )

    numerical_results = numerical_target_relationships(
        df
    )

    print(
        "Running categorical target relationship analysis..."
    )

    categorical_results = categorical_target_relationships(
        df
    )

    print(
        "Running numerical target-rate analysis..."
    )

    numeric_features = get_numeric_features(
        df
    )

    binned_results = numerical_binned_target_rates(
        df,
        numeric_features,
    )

    print(
        "Running interaction analysis..."
    )

    interaction_results = interaction_target_analysis(
        df
    )

    print(
        "Calculating signal concentration..."
    )

    signal_concentration = calculate_signal_concentration(
        numerical_results,
        categorical_results,
        target_summary["positive_rate"],
    )

    print(
        "Generating diagnostic flags..."
    )

    flags = build_diagnostic_flags(
        target_summary,
        numerical_results,
        categorical_results,
        interaction_results,
    )

    print(
        "Generating overall diagnosis..."
    )

    diagnosis = build_overall_diagnosis(
        numerical_results,
        categorical_results,
        interaction_results,
    )

    # --------------------------------------------------------
    # Save CSV reports
    # --------------------------------------------------------

    if not numerical_results.empty:
        numerical_results.to_csv(
            UNIVARIATE_PATH,
            index=False,
        )
    else:
        pd.DataFrame().to_csv(
            UNIVARIATE_PATH,
            index=False,
        )

    # Append binned analysis to a separate internal report
    # while preserving the main univariate CSV.
    binned_path = (
        OUTPUT_DIR
        / "numerical_binned_target_rates.csv"
    )

    binned_results.to_csv(
        binned_path,
        index=False,
    )

    interaction_results.to_csv(
        INTERACTION_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    write_json_report(
        target_summary,
        numerical_results,
        categorical_results,
        interaction_results,
        flags,
        diagnosis,
    )

    # Add additional signal concentration information
    # directly into the JSON report.
    report_data = json.loads(
        JSON_PATH.read_text(
            encoding="utf-8"
        )
    )

    report_data[
        "signal_concentration"
    ] = signal_concentration

    report_data[
        "numerical_binned_target_rates"
    ] = (
        binned_results.to_dict(
            orient="records"
        )
        if not binned_results.empty
        else []
    )

    JSON_PATH.write_text(
        json.dumps(
            json_safe(report_data),
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary report
    # --------------------------------------------------------

    write_summary(
        target_summary,
        numerical_results,
        categorical_results,
        interaction_results,
        flags,
        diagnosis,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("")
    print(
        "============================================================"
    )
    print(
        "EMPLOYEE ATTRITION — TARGET GENERATION AUDIT"
    )
    print(
        "============================================================"
    )
    print("")

    print("[DATASET]")
    print(
        f"Rows:                 "
        f"{target_summary['total_rows']}"
    )
    print(
        f"Columns:              {len(df.columns)}"
    )
    print(
        f"Attrition Yes:        "
        f"{target_summary['positive_count']}"
    )
    print(
        f"Attrition No:         "
        f"{target_summary['negative_count']}"
    )
    print(
        f"Attrition rate:       "
        f"{target_summary['positive_rate'] * 100:.2f}%"
    )
    print(
        f"Target entropy:       "
        f"{target_summary['entropy_bits']:.4f} bits"
    )
    print("")

    print("[TOP NUMERICAL RELATIONSHIPS]")

    if numerical_results.empty:
        print(
            "No numerical relationships detected."
        )
    else:
        print(
            numerical_results[
                [
                    "feature",
                    "point_biserial_correlation",
                    "p_value",
                ]
            ]
            .head(TOP_NUMERICAL_FEATURES)
            .to_string(index=False)
        )

    print("")

    print("[TOP CATEGORICAL RELATIONSHIPS]")

    if categorical_results.empty:
        print(
            "No categorical relationships detected."
        )
    else:
        print(
            categorical_results[
                [
                    "feature",
                    "category",
                    "attrition_rate",
                    "cramers_v",
                    "p_value",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    print("")

    print("[INTERACTION ANALYSIS]")

    if interaction_results.empty:
        print(
            "No interaction results detected."
        )
    else:
        grouped = (
            interaction_results
            .groupby(
                ["feature_a", "feature_b"]
            )
            .agg(
                min_attrition_rate=(
                    "attrition_rate",
                    "min",
                ),
                max_attrition_rate=(
                    "attrition_rate",
                    "max",
                ),
            )
            .reset_index()
        )

        grouped["rate_range"] = (
            grouped["max_attrition_rate"]
            - grouped["min_attrition_rate"]
        )

        grouped = grouped.sort_values(
            "rate_range",
            ascending=False,
        )

        print(
            grouped.to_string(
                index=False
            )
        )

    print("")

    print("[SIGNAL CONCENTRATION]")

    print(
        f"Maximum numerical |r|: "
        f"{signal_concentration['max_numeric_abs_correlation']:.4f}"
    )

    print(
        f"Numerical features |r| >= 0.10: "
        f"{signal_concentration['numeric_features_above_0_10']}"
    )

    print(
        f"Numerical features |r| >= 0.05: "
        f"{signal_concentration['numeric_features_above_0_05']}"
    )

    print(
        f"Maximum Cramer's V:    "
        f"{signal_concentration['max_cramers_v']:.4f}"
    )

    print(
        f"Maximum category rate deviation: "
        f"{signal_concentration['max_category_rate_deviation'] * 100:.2f} pp"
    )

    print("")

    print("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(
            f"- {flag}"
        )

    print("")

    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    print("")

    print("[OUTPUT]")
    print(
        f"Reports:              {OUTPUT_DIR}"
    )
    print(
        f"JSON report:          {JSON_PATH}"
    )
    print(
        f"Univariate report:    {UNIVARIATE_PATH}"
    )
    print(
        f"Interaction report:   {INTERACTION_PATH}"
    )
    print(
        f"Binned numerical:     {binned_path}"
    )
    print(
        f"Summary report:       {SUMMARY_PATH}"
    )

    print("")
    print(
        "============================================================"
    )
    print(
        "TARGET GENERATION AUDIT COMPLETE"
    )
    print(
        "============================================================"
    )


if __name__ == "__main__":
    main()