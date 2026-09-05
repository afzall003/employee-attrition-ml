"""
Dataset Version Comparison
===========================

Purpose
-------
Compare all discovered employee attrition dataset versions and determine
whether differences in target prevalence, schema, rows, features, or target
labels explain the inconsistency observed across the evaluation pipeline.

This script is diagnostic only.
It DOES NOT modify any dataset.

Expected project structure
--------------------------
employee-attrition-ml/
├── data/
│   └── raw/
│       ├── employee_attrition_dataset.csv
│       └── employee_attrition_dataset_v2.csv
├── reports/
│   └── signal_analysis/
│       └── dataset_version_comparison/
└── src/
    └── evaluation/
        └── dataset_version_comparison.py
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import ks_2samp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "dataset_version_comparison"
)

EXPECTED_DATASETS = [
    "employee_attrition_dataset.csv",
    "employee_attrition_dataset_v2.csv",
]

TARGET_COLUMN_CANDIDATES = [
    "Attrition",
    "attrition",
    "ATTRITION",
    "Target",
    "target",
]

IDENTIFIER_CANDIDATES = [
    "Employee_ID",
    "Employee_ID",
    "employee_id",
    "EmployeeID",
    "employeeID",
    "ID",
    "id",
]


# Established values from the previous evaluation pipeline.
# These are diagnostic reference values only.
ESTABLISHED_ROWS = 1000
ESTABLISHED_COLUMNS = 26
ESTABLISHED_TARGET_PREVALENCE = 0.2360


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def safe_float(value: Any) -> float | None:
    """
    Convert values to JSON-safe Python floats.
    """
    if value is None:
        return None

    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    """
    Recursively convert pandas / NumPy values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Interval):
        return str(value)

    if isinstance(value, np.ndarray):
        return [
            json_safe(v)
            for v in value.tolist()
        ]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def dataframe_fingerprint(path: Path) -> str:
    """
    Calculate SHA-256 fingerprint of the actual CSV file bytes.
    """
    sha256 = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def dataframe_content_fingerprint(df: pd.DataFrame) -> str:
    """
    Fingerprint normalized dataframe content.
    """
    normalized = df.copy()

    normalized.columns = [
        str(column)
        for column in normalized.columns
    ]

    normalized = normalized.astype(str)

    csv_bytes = normalized.to_csv(
        index=False,
        lineterminator="\n",
    ).encode("utf-8")

    return hashlib.sha256(csv_bytes).hexdigest()


def discover_dataset_files() -> list[Path]:
    """
    Discover expected CSV files in data/raw.

    Expected names are prioritized, but additional CSV files are also
    reported so that hidden dataset versions are not silently ignored.
    """

    print("Discovering dataset files...")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {RAW_DATA_DIR}"
        )

    discovered = sorted(
        RAW_DATA_DIR.glob("*.csv")
    )

    if not discovered:
        raise FileNotFoundError(
            f"No CSV datasets found in: {RAW_DATA_DIR}"
        )

    # Expected files first.
    expected_paths = [
        RAW_DATA_DIR / name
        for name in EXPECTED_DATASETS
        if (RAW_DATA_DIR / name).exists()
    ]

    remaining = [
        path
        for path in discovered
        if path not in expected_paths
    ]

    ordered = expected_paths + sorted(remaining)

    print()
    print("Discovered dataset files:")

    for index, path in enumerate(ordered, start=1):
        print(
            f"  {index:2d}. "
            f"{path.relative_to(PROJECT_ROOT)}"
        )

    return ordered


# ============================================================
# TARGET DETECTION
# ============================================================

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect Attrition target column.
    """

    for candidate in TARGET_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate

    # Conservative fallback.
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in TARGET_COLUMN_CANDIDATES:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    raise ValueError(
        "Could not detect target column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_target_value(value: Any) -> int | None:
    """
    Normalize common binary target representations.

    Returns:
        1 for positive / Yes
        0 for negative / No
        None for missing / unknown
    """

    if pd.isna(value):
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "yes",
            "y",
            "true",
            "1",
            "attrition",
            "left",
        }:
            return 1

        if normalized in {
            "no",
            "n",
            "false",
            "0",
            "stay",
            "stayed",
        }:
            return 0

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)):
        if float(value) in (0.0, 1.0):
            return int(value)

    return None


def analyze_target(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze target distribution and encoding.
    """

    target_column = detect_target_column(df)

    raw_series = df[target_column]

    normalized = raw_series.map(
        normalize_target_value
    )

    positive_count = int(
        (normalized == 1).sum()
    )

    negative_count = int(
        (normalized == 0).sum()
    )

    missing_or_unknown = int(
        normalized.isna().sum()
    )

    valid_count = positive_count + negative_count

    prevalence = (
        positive_count / valid_count
        if valid_count > 0
        else None
    )

    raw_values = (
        raw_series
        .value_counts(dropna=False)
        .to_dict()
    )

    raw_value_counts = {
        str(key): int(value)
        for key, value in raw_values.items()
    }

    return {
        "column": target_column,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "missing_or_unknown_count": missing_or_unknown,
        "valid_count": valid_count,
        "prevalence": safe_float(prevalence),
        "raw_value_counts": raw_value_counts,
        "unique_values": [
            str(value)
            for value in raw_series.dropna().unique()
        ],
    }


# ============================================================
# SCHEMA ANALYSIS
# ============================================================

def analyze_schema(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict[str, Any]:

    columns_a = [
        str(column)
        for column in df_a.columns
    ]

    columns_b = [
        str(column)
        for column in df_b.columns
    ]

    set_a = set(columns_a)
    set_b = set(columns_b)

    only_a = [
        column
        for column in columns_a
        if column not in set_b
    ]

    only_b = [
        column
        for column in columns_b
        if column not in set_a
    ]

    common_columns = [
        column
        for column in columns_a
        if column in set_b
    ]

    order_matches = (
        columns_a == columns_b
    )

    dtype_differences = {}

    for column in common_columns:
        dtype_a = str(df_a[column].dtype)
        dtype_b = str(df_b[column].dtype)

        if dtype_a != dtype_b:
            dtype_differences[column] = {
                "dataset_a": dtype_a,
                "dataset_b": dtype_b,
            }

    return {
        "dataset_a_columns": columns_a,
        "dataset_b_columns": columns_b,
        "dataset_a_column_count": len(columns_a),
        "dataset_b_column_count": len(columns_b),
        "common_column_count": len(common_columns),
        "columns_only_in_a": only_a,
        "columns_only_in_b": only_b,
        "column_order_matches": order_matches,
        "dtype_differences": dtype_differences,
    }


# ============================================================
# DUPLICATE ANALYSIS
# ============================================================

def analyze_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze exact duplicate rows and likely identifiers.
    """

    duplicate_rows = int(
        df.duplicated().sum()
    )

    identifier_columns = [
        column
        for column in df.columns
        if str(column) in IDENTIFIER_CANDIDATES
        or "id" in str(column).lower()
    ]

    identifier_analysis = {}

    for column in identifier_columns:
        identifier_analysis[str(column)] = {
            "unique_count": int(
                df[column].nunique(dropna=True)
            ),
            "duplicate_count": int(
                df[column].duplicated().sum()
            ),
            "missing_count": int(
                df[column].isna().sum()
            ),
        }

    return {
        "exact_duplicate_rows": duplicate_rows,
        "identifier_columns": identifier_columns,
        "identifier_analysis": identifier_analysis,
    }


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================

def analyze_missingness(
    df: pd.DataFrame,
) -> dict[str, Any]:

    missing_by_column = (
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )

    columns_with_missing = {
        str(column): int(count)
        for column, count in missing_by_column.items()
        if count > 0
    }

    return {
        "rows_with_any_missing": int(
            df.isna().any(axis=1).sum()
        ),
        "columns_with_missing": columns_with_missing,
        "total_missing_cells": int(
            df.isna().sum().sum()
        ),
    }


# ============================================================
# ROW OVERLAP
# ============================================================

def determine_identifier(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> str | None:

    common_columns = [
        column
        for column in df_a.columns
        if column in df_b.columns
    ]

    for candidate in IDENTIFIER_CANDIDATES:
        if candidate in common_columns:
            return candidate

    for column in common_columns:
        column_lower = str(column).lower()

        if (
            "employee" in column_lower
            and "id" in column_lower
        ):
            return column

    for column in common_columns:
        column_lower = str(column).lower()

        if column_lower.endswith("_id"):
            return column

    return None


def analyze_row_overlap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict[str, Any]:

    identifier = determine_identifier(
        df_a,
        df_b,
    )

    if identifier is not None:

        ids_a = set(
            df_a[identifier]
            .dropna()
            .astype(str)
        )

        ids_b = set(
            df_b[identifier]
            .dropna()
            .astype(str)
        )

        intersection = ids_a & ids_b

        only_a = ids_a - ids_b
        only_b = ids_b - ids_a

        return {
            "method": "identifier",
            "identifier": identifier,
            "dataset_a_unique_ids": len(ids_a),
            "dataset_b_unique_ids": len(ids_b),
            "shared_ids": len(intersection),
            "only_in_a": len(only_a),
            "only_in_b": len(only_b),
            "overlap_rate_relative_to_a": safe_float(
                len(intersection) / len(ids_a)
                if ids_a
                else None
            ),
            "overlap_rate_relative_to_b": safe_float(
                len(intersection) / len(ids_b)
                if ids_b
                else None
            ),
        }

    # Fallback: compare complete rows.
    common_columns = [
        column
        for column in df_a.columns
        if column in df_b.columns
    ]

    if not common_columns:
        return {
            "method": "none",
            "reason": "No common columns available.",
        }

    rows_a = set(
        map(
            tuple,
            df_a[common_columns]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )

    rows_b = set(
        map(
            tuple,
            df_b[common_columns]
            .astype(str)
            .itertuples(index=False, name=None),
        )
    )

    shared = rows_a & rows_b

    return {
        "method": "complete_common_row",
        "identifier": None,
        "shared_rows": len(shared),
        "only_in_a": len(rows_a - rows_b),
        "only_in_b": len(rows_b - rows_a),
        "overlap_rate_relative_to_a": safe_float(
            len(shared) / len(rows_a)
            if rows_a
            else None
        ),
        "overlap_rate_relative_to_b": safe_float(
            len(shared) / len(rows_b)
            if rows_b
            else None
        ),
    }


# ============================================================
# TARGET LABEL COMPARISON
# ============================================================

def compare_target_labels(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict[str, Any]:

    target_a = detect_target_column(df_a)
    target_b = detect_target_column(df_b)

    identifier = determine_identifier(
        df_a,
        df_b,
    )

    if identifier is None:
        return {
            "method": "unavailable",
            "reason": "No shared identifier column found.",
        }

    left = df_a[
        [identifier, target_a]
    ].copy()

    right = df_b[
        [identifier, target_b]
    ].copy()

    left[identifier] = left[
        identifier
    ].astype(str)

    right[identifier] = right[
        identifier
    ].astype(str)

    merged = left.merge(
        right,
        on=identifier,
        how="inner",
        suffixes=("_a", "_b"),
    )

    if merged.empty:
        return {
            "method": "identifier",
            "identifier": identifier,
            "shared_rows": 0,
            "label_comparisons": 0,
            "label_disagreements": 0,
        }

    normalized_a = merged[
        f"{target_a}_a"
    ].map(normalize_target_value)

    normalized_b = merged[
        f"{target_b}_b"
    ].map(normalize_target_value)

    valid = (
        normalized_a.notna()
        & normalized_b.notna()
    )

    disagreements = (
        normalized_a[valid]
        != normalized_b[valid]
    )

    disagreement_count = int(
        disagreements.sum()
    )

    comparison_count = int(
        valid.sum()
    )

    return {
        "method": "identifier",
        "identifier": identifier,
        "shared_rows": int(len(merged)),
        "label_comparisons": comparison_count,
        "label_disagreements": disagreement_count,
        "label_disagreement_rate": safe_float(
            disagreement_count / comparison_count
            if comparison_count > 0
            else None
        ),
    }


# ============================================================
# NUMERICAL DISTRIBUTION COMPARISON
# ============================================================

def numerical_distribution_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> list[dict[str, Any]]:

    common_columns = [
        column
        for column in df_a.columns
        if column in df_b.columns
    ]

    results = []

    for column in common_columns:

        series_a = pd.to_numeric(
            df_a[column],
            errors="coerce",
        )

        series_b = pd.to_numeric(
            df_b[column],
            errors="coerce",
        )

        if (
            series_a.notna().sum() < 20
            or series_b.notna().sum() < 20
        ):
            continue

        values_a = series_a.dropna().to_numpy()
        values_b = series_b.dropna().to_numpy()

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))

        median_a = float(np.median(values_a))
        median_b = float(np.median(values_b))

        std_a = float(np.std(values_a))
        std_b = float(np.std(values_b))

        if SCIPY_AVAILABLE:
            ks_stat, ks_p = ks_2samp(
                values_a,
                values_b,
            )
        else:
            ks_stat = np.nan
            ks_p = np.nan

        results.append({
            "feature": str(column),
            "mean_a": safe_float(mean_a),
            "mean_b": safe_float(mean_b),
            "mean_delta": safe_float(
                mean_b - mean_a
            ),
            "median_a": safe_float(median_a),
            "median_b": safe_float(median_b),
            "median_delta": safe_float(
                median_b - median_a
            ),
            "std_a": safe_float(std_a),
            "std_b": safe_float(std_b),
            "std_delta": safe_float(
                std_b - std_a
            ),
            "ks_statistic": safe_float(ks_stat),
            "ks_p_value": safe_float(ks_p),
        })

    results.sort(
        key=lambda item: (
            item["ks_statistic"]
            if item["ks_statistic"] is not None
            else -1
        ),
        reverse=True,
    )

    return results


# ============================================================
# CATEGORICAL DISTRIBUTION COMPARISON
# ============================================================

def categorical_distribution_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> list[dict[str, Any]]:

    common_columns = [
        column
        for column in df_a.columns
        if column in df_b.columns
    ]

    results = []

    for column in common_columns:

        series_a = df_a[column]
        series_b = df_b[column]

        # Skip obviously numeric columns.
        numeric_a = pd.to_numeric(
            series_a,
            errors="coerce",
        )

        numeric_b = pd.to_numeric(
            series_b,
            errors="coerce",
        )

        if (
            numeric_a.notna().mean() > 0.95
            and numeric_b.notna().mean() > 0.95
        ):
            continue

        values = set(
            series_a.dropna().astype(str)
        ) | set(
            series_b.dropna().astype(str)
        )

        if not values:
            continue

        total_a = max(
            int(series_a.notna().sum()),
            1,
        )

        total_b = max(
            int(series_b.notna().sum()),
            1,
        )

        max_delta = 0.0

        for value in values:

            rate_a = (
                (series_a.astype(str) == value).sum()
                / total_a
            )

            rate_b = (
                (series_b.astype(str) == value).sum()
                / total_b
            )

            max_delta = max(
                max_delta,
                abs(rate_b - rate_a),
            )

        results.append({
            "feature": str(column),
            "unique_values_a": int(
                series_a.nunique(dropna=True)
            ),
            "unique_values_b": int(
                series_b.nunique(dropna=True)
            ),
            "max_category_proportion_delta": safe_float(
                max_delta
            ),
        })

    results.sort(
        key=lambda item: (
            item["max_category_proportion_delta"]
            if item["max_category_proportion_delta"] is not None
            else -1
        ),
        reverse=True,
    )

    return results


# ============================================================
# ROW-LEVEL FEATURE DIFFERENCE
# ============================================================

def compare_shared_rows(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> dict[str, Any]:

    identifier = determine_identifier(
        df_a,
        df_b,
    )

    if identifier is None:
        return {
            "method": "unavailable",
            "reason": "No shared identifier found.",
        }

    common_columns = [
        column
        for column in df_a.columns
        if column in df_b.columns
        and column != identifier
    ]

    if not common_columns:
        return {
            "method": "identifier",
            "identifier": identifier,
            "reason": "No comparable feature columns.",
        }

    left = df_a[
        [identifier] + common_columns
    ].copy()

    right = df_b[
        [identifier] + common_columns
    ].copy()

    left[identifier] = left[
        identifier
    ].astype(str)

    right[identifier] = right[
        identifier
    ].astype(str)

    merged = left.merge(
        right,
        on=identifier,
        how="inner",
        suffixes=("_a", "_b"),
    )

    feature_results = []

    for column in common_columns:

        column_a = f"{column}_a"
        column_b = f"{column}_b"

        if (
            column_a not in merged.columns
            or column_b not in merged.columns
        ):
            continue

        numeric_a = pd.to_numeric(
            merged[column_a],
            errors="coerce",
        )

        numeric_b = pd.to_numeric(
            merged[column_b],
            errors="coerce",
        )

        numeric_fraction_a = numeric_a.notna().mean()
        numeric_fraction_b = numeric_b.notna().mean()

        if (
            numeric_fraction_a > 0.95
            and numeric_fraction_b > 0.95
        ):

            valid = (
                numeric_a.notna()
                & numeric_b.notna()
            )

            if valid.sum() == 0:
                continue

            difference = (
                numeric_a[valid]
                != numeric_b[valid]
            )

            difference_count = int(
                difference.sum()
            )

        else:

            values_a = (
                merged[column_a]
                .astype(str)
            )

            values_b = (
                merged[column_b]
                .astype(str)
            )

            difference_count = int(
                (values_a != values_b).sum()
            )

        feature_results.append({
            "feature": str(column),
            "shared_rows": int(len(merged)),
            "different_values": difference_count,
            "difference_rate": safe_float(
                difference_count / len(merged)
                if len(merged) > 0
                else None
            ),
        })

    feature_results.sort(
        key=lambda item: (
            item["difference_rate"]
            if item["difference_rate"] is not None
            else -1
        ),
        reverse=True,
    )

    return {
        "method": "identifier",
        "identifier": identifier,
        "shared_rows": int(len(merged)),
        "feature_comparisons": feature_results,
    }


# ============================================================
# DATASET PROFILE
# ============================================================

def build_dataset_profile(
    path: Path,
) -> dict[str, Any]:

    print()
    print(
        f"Loading: "
        f"{path.relative_to(PROJECT_ROOT)}"
    )

    df = pd.read_csv(path)

    print(
        f"  Rows:    {len(df)}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    target = analyze_target(df)

    profile = {
        "filename": path.name,
        "relative_path": str(
            path.relative_to(PROJECT_ROOT)
        ),
        "absolute_path": str(path.resolve()),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "fingerprint": dataframe_fingerprint(path),
        "content_fingerprint": dataframe_content_fingerprint(df),
        "target": target,
        "duplicates": analyze_duplicates(df),
        "missingness": analyze_missingness(df),
        "column_names": [
            str(column)
            for column in df.columns
        ],
    }

    return profile, df


# ============================================================
# DIAGNOSTIC FLAGS
# ============================================================

def generate_diagnostic_flags(
    profiles: list[dict[str, Any]],
    schema: dict[str, Any] | None,
    row_overlap: dict[str, Any] | None,
    target_label_comparison: dict[str, Any] | None,
    numerical_comparison: list[dict[str, Any]],
    categorical_comparison: list[dict[str, Any]],
    shared_row_comparison: dict[str, Any] | None,
) -> list[str]:

    flags = []

    if len(profiles) < 2:
        flags.append(
            "Fewer than two dataset versions were available "
            "for direct comparison."
        )
        return flags

    prevalences = [
        profile["target"]["prevalence"]
        for profile in profiles
        if profile["target"]["prevalence"] is not None
    ]

    if prevalences:

        prevalence_range = (
            max(prevalences)
            - min(prevalences)
        )

        if prevalence_range >= 0.02:
            flags.append(
                "Dataset versions have materially different "
                "target prevalences."
            )

        if prevalence_range >= 0.04:
            flags.append(
                "The target prevalence difference exceeds "
                "4 percentage points and requires investigation "
                "of dataset version or target construction."
            )

    if schema is not None:

        if schema["columns_only_in_a"]:
            flags.append(
                "Dataset A contains columns absent from Dataset B."
            )

        if schema["columns_only_in_b"]:
            flags.append(
                "Dataset B contains columns absent from Dataset A."
            )

        if schema["dtype_differences"]:
            flags.append(
                "At least one common feature has different "
                "data types between dataset versions."
            )

        if not schema["column_order_matches"]:
            flags.append(
                "Column ordering differs between dataset versions."
            )

    if row_overlap is not None:

        overlap_rate_a = row_overlap.get(
            "overlap_rate_relative_to_a"
        )

        overlap_rate_b = row_overlap.get(
            "overlap_rate_relative_to_b"
        )

        if (
            overlap_rate_a is not None
            and overlap_rate_a < 0.95
        ):
            flags.append(
                "Dataset versions do not share at least "
                "95% of their rows relative to Dataset A."
            )

        if (
            overlap_rate_b is not None
            and overlap_rate_b < 0.95
        ):
            flags.append(
                "Dataset versions do not share at least "
                "95% of their rows relative to Dataset B."
            )

    if target_label_comparison is not None:

        disagreement_rate = target_label_comparison.get(
            "label_disagreement_rate"
        )

        if (
            disagreement_rate is not None
            and disagreement_rate > 0
        ):
            flags.append(
                "Shared employee records contain target-label "
                "disagreements between dataset versions."
            )

        if (
            disagreement_rate is not None
            and disagreement_rate >= 0.02
        ):
            flags.append(
                "Target-label disagreement is large enough to "
                "materially alter model evaluation."
            )

    if numerical_comparison:

        significant_numerical_drift = [
            item
            for item in numerical_comparison
            if (
                item["ks_p_value"] is not None
                and item["ks_p_value"] < 0.05
            )
        ]

        if significant_numerical_drift:
            flags.append(
                f"{len(significant_numerical_drift)} numerical "
                "features show statistically significant "
                "distribution differences between versions."
            )

    if categorical_comparison:

        large_categorical_drift = [
            item
            for item in categorical_comparison
            if (
                item["max_category_proportion_delta"] is not None
                and item["max_category_proportion_delta"] >= 0.05
            )
        ]

        if large_categorical_drift:
            flags.append(
                f"{len(large_categorical_drift)} categorical "
                "features show at least 5 percentage points "
                "of category-proportion difference."
            )

    if shared_row_comparison is not None:

        feature_comparisons = shared_row_comparison.get(
            "feature_comparisons",
            [],
        )

        materially_changed = [
            item
            for item in feature_comparisons
            if (
                item["difference_rate"] is not None
                and item["difference_rate"] >= 0.05
            )
        ]

        if materially_changed:
            flags.append(
                f"{len(materially_changed)} shared features have "
                "different values for at least 5% of shared rows."
            )

    return flags


def generate_overall_diagnosis(
    profiles: list[dict[str, Any]],
    flags: list[str],
    row_overlap: dict[str, Any] | None,
    target_label_comparison: dict[str, Any] | None,
    shared_row_comparison: dict[str, Any] | None,
) -> str:

    if len(profiles) < 2:
        return (
            "Only one dataset version was available. "
            "Dataset-version consistency cannot be established."
        )

    prevalence_values = [
        profile["target"]["prevalence"]
        for profile in profiles
        if profile["target"]["prevalence"] is not None
    ]

    prevalence_range = (
        max(prevalence_values)
        - min(prevalence_values)
        if prevalence_values
        else 0
    )

    disagreement_rate = None

    if target_label_comparison:
        disagreement_rate = target_label_comparison.get(
            "label_disagreement_rate"
        )

    shared_feature_changes = 0

    if shared_row_comparison:
        shared_feature_changes = sum(
            1
            for item in shared_row_comparison.get(
                "feature_comparisons",
                [],
            )
            if (
                item.get("difference_rate") is not None
                and item["difference_rate"] >= 0.05
            )
        )

    if prevalence_range >= 0.04:

        if (
            disagreement_rate is not None
            and disagreement_rate > 0
        ):
            return (
                "The dataset versions are materially inconsistent. "
                "Their target prevalences differ substantially and "
                "shared employee records contain target-label "
                "differences. The evaluation pipeline must "
                "standardize the canonical dataset before further "
                "model optimization."
            )

        if shared_feature_changes > 0:
            return (
                "The dataset versions are materially inconsistent. "
                "Their target prevalences differ substantially and "
                "shared rows also contain feature-value differences. "
                "This indicates that the versions are not simple "
                "target relabelings and must be investigated before "
                "further model optimization."
            )

        return (
            "The dataset versions have a substantial target "
            "prevalence discrepancy. The difference may arise from "
            "dataset versioning, target construction, row selection, "
            "or another data-generation process. The canonical "
            "dataset must be established before further model "
            "optimization."
        )

    if flags:
        return (
            "The dataset versions are not perfectly identical. "
            "Differences were detected that should be documented "
            "before using one version as the canonical evaluation "
            "dataset."
        )

    return (
        "The available dataset versions appear consistent across "
        "schema, target distribution, row overlap, and feature "
        "content. No major version inconsistency was detected."
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def save_json_report(
    report: dict[str, Any],
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "dataset_version_comparison_report.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            json_safe(report),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    return path


def save_summary_report(
    report: dict[str, Any],
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_DIR
        / "dataset_version_comparison_summary.txt"
    )

    profiles = report["datasets"]

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "EMPLOYEE ATTRITION — DATASET VERSION COMPARISON\n"
        )
        handle.write("=" * 60 + "\n\n")

        handle.write("[DATASETS]\n")

        for index, profile in enumerate(
            profiles,
            start=1,
        ):

            target = profile["target"]

            handle.write(
                f"Dataset {index}: "
                f"{profile['relative_path']}\n"
            )

            handle.write(
                f"Rows:                 "
                f"{profile['rows']}\n"
            )

            handle.write(
                f"Columns:              "
                f"{profile['columns']}\n"
            )

            handle.write(
                f"Target column:        "
                f"{target['column']}\n"
            )

            handle.write(
                f"Positive count:       "
                f"{target['positive_count']}\n"
            )

            handle.write(
                f"Negative count:       "
                f"{target['negative_count']}\n"
            )

            prevalence = target["prevalence"]

            handle.write(
                f"Target prevalence:    "
                f"{prevalence * 100:.2f}%\n"
                if prevalence is not None
                else
                "Target prevalence:    N/A\n"
            )

            handle.write(
                f"SHA-256:              "
                f"{profile['fingerprint']}\n"
            )

            handle.write("\n")

        handle.write("[ESTABLISHED REFERENCE]\n")

        handle.write(
            f"Rows:                 "
            f"{ESTABLISHED_ROWS}\n"
        )

        handle.write(
            f"Columns:              "
            f"{ESTABLISHED_COLUMNS}\n"
        )

        handle.write(
            f"Target prevalence:    "
            f"{ESTABLISHED_TARGET_PREVALENCE * 100:.2f}%\n"
        )

        handle.write("\n")

        schema = report.get("schema_comparison")

        if schema:

            handle.write("[SCHEMA COMPARISON]\n")

            handle.write(
                f"Common columns:       "
                f"{schema['common_column_count']}\n"
            )

            handle.write(
                f"Column order matches: "
                f"{schema['column_order_matches']}\n"
            )

            handle.write(
                "Only in Dataset A:    "
                f"{schema['columns_only_in_a']}\n"
            )

            handle.write(
                "Only in Dataset B:    "
                f"{schema['columns_only_in_b']}\n"
            )

            handle.write(
                "Dtype differences:    "
                f"{schema['dtype_differences']}\n"
            )

            handle.write("\n")

        row_overlap = report.get(
            "row_overlap"
        )

        if row_overlap:

            handle.write("[ROW OVERLAP]\n")

            for key, value in row_overlap.items():
                handle.write(
                    f"{key}: {value}\n"
                )

            handle.write("\n")

        label_comparison = report.get(
            "target_label_comparison"
        )

        if label_comparison:

            handle.write(
                "[TARGET LABEL COMPARISON]\n"
            )

            for key, value in label_comparison.items():
                handle.write(
                    f"{key}: {value}\n"
                )

            handle.write("\n")

        numerical = report.get(
            "numerical_distribution_comparison",
            [],
        )

        handle.write(
            "[TOP NUMERICAL DISTRIBUTION DIFFERENCES]\n"
        )

        for item in numerical[:10]:

            handle.write(
                f"{item['feature']:<40} "
                f"KS={item['ks_statistic']!s:<8} "
                f"p={item['ks_p_value']!s}\n"
            )

        handle.write("\n")

        categorical = report.get(
            "categorical_distribution_comparison",
            [],
        )

        handle.write(
            "[TOP CATEGORICAL DISTRIBUTION DIFFERENCES]\n"
        )

        for item in categorical[:10]:

            handle.write(
                f"{item['feature']:<40} "
                f"delta="
                f"{item['max_category_proportion_delta']}\n"
            )

        handle.write("\n")

        shared_rows = report.get(
            "shared_row_feature_comparison"
        )

        if shared_rows:

            handle.write(
                "[TOP SHARED-ROW FEATURE DIFFERENCES]\n"
            )

            for item in shared_rows.get(
                "feature_comparisons",
                [],
            )[:10]:

                handle.write(
                    f"{item['feature']:<40} "
                    f"difference_rate="
                    f"{item['difference_rate']}\n"
                )

            handle.write("\n")

        handle.write("[DIAGNOSTIC FLAGS]\n")

        flags = report.get(
            "diagnostic_flags",
            [],
        )

        if flags:

            for flag in flags:
                handle.write(
                    f"- {flag}\n"
                )

        else:
            handle.write(
                "- No major diagnostic flags.\n"
            )

        handle.write("\n")

        handle.write("[OVERALL DIAGNOSIS]\n")

        handle.write(
            report["overall_diagnosis"]
            + "\n"
        )

    return path


def save_csv_reports(
    numerical_comparison: list[dict[str, Any]],
    categorical_comparison: list[dict[str, Any]],
    shared_row_comparison: dict[str, Any] | None,
) -> list[Path]:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = []

    numerical_path = (
        OUTPUT_DIR
        / "numerical_distribution_comparison.csv"
    )

    pd.DataFrame(
        numerical_comparison
    ).to_csv(
        numerical_path,
        index=False,
    )

    paths.append(numerical_path)

    categorical_path = (
        OUTPUT_DIR
        / "categorical_distribution_comparison.csv"
    )

    pd.DataFrame(
        categorical_comparison
    ).to_csv(
        categorical_path,
        index=False,
    )

    paths.append(categorical_path)

    if shared_row_comparison:

        feature_comparisons = (
            shared_row_comparison.get(
                "feature_comparisons",
                [],
            )
        )

        shared_path = (
            OUTPUT_DIR
            / "shared_row_feature_comparison.csv"
        )

        pd.DataFrame(
            feature_comparisons
        ).to_csv(
            shared_path,
            index=False,
        )

        paths.append(shared_path)

    return paths


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_dataset_profile(
    index: int,
    profile: dict[str, Any],
) -> None:

    target = profile["target"]

    print()
    print(
        f"[DATASET {index}]"
    )

    print(
        f"Path:                 "
        f"{profile['relative_path']}"
    )

    print(
        f"Rows:                 "
        f"{profile['rows']}"
    )

    print(
        f"Columns:              "
        f"{profile['columns']}"
    )

    print(
        f"Target column:        "
        f"{target['column']}"
    )

    print(
        f"Positive count:       "
        f"{target['positive_count']}"
    )

    print(
        f"Negative count:       "
        f"{target['negative_count']}"
    )

    prevalence = target["prevalence"]

    if prevalence is not None:

        print(
            f"Target prevalence:    "
            f"{prevalence * 100:.2f}%"
        )

        difference = (
            prevalence
            - ESTABLISHED_TARGET_PREVALENCE
        )

        print(
            f"Established difference:"
            f" {difference * 100:+.2f} pp"
        )

    print(
        f"Exact duplicates:     "
        f"{profile['duplicates']['exact_duplicate_rows']}"
    )

    print(
        f"Missing cells:        "
        f"{profile['missingness']['total_missing_cells']}"
    )

    print(
        f"SHA-256:              "
        f"{profile['fingerprint'][:24]}..."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Running dataset version comparison..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_paths = discover_dataset_files()

    if len(dataset_paths) < 2:

        print()
        print(
            "ERROR: At least two dataset versions are required."
        )

        print(
            f"Expected files in: {RAW_DATA_DIR}"
        )

        raise SystemExit(1)

    print()
    print(
        "Loading dataset versions..."
    )

    profiles = []
    dataframes = []

    for path in dataset_paths:

        profile, df = build_dataset_profile(
            path
        )

        profiles.append(profile)
        dataframes.append(df)

    print_header(
        "EMPLOYEE ATTRITION — DATASET VERSION COMPARISON"
    )

    print_dataset_profile(
        1,
        profiles[0],
    )

    print_dataset_profile(
        2,
        profiles[1],
    )

    df_a = dataframes[0]
    df_b = dataframes[1]

    print()
    print(
        "Comparing dataset schemas..."
    )

    schema = analyze_schema(
        df_a,
        df_b,
    )

    print(
        "Comparing row overlap..."
    )

    row_overlap = analyze_row_overlap(
        df_a,
        df_b,
    )

    print(
        "Comparing target labels..."
    )

    target_label_comparison = compare_target_labels(
        df_a,
        df_b,
    )

    print(
        "Comparing numerical distributions..."
    )

    numerical_comparison = (
        numerical_distribution_comparison(
            df_a,
            df_b,
        )
    )

    print(
        "Comparing categorical distributions..."
    )

    categorical_comparison = (
        categorical_distribution_comparison(
            df_a,
            df_b,
        )
    )

    print(
        "Comparing shared-row feature values..."
    )

    shared_row_comparison = compare_shared_rows(
        df_a,
        df_b,
    )

    print(
        "Generating diagnostic flags..."
    )

    flags = generate_diagnostic_flags(
        profiles=profiles,
        schema=schema,
        row_overlap=row_overlap,
        target_label_comparison=target_label_comparison,
        numerical_comparison=numerical_comparison,
        categorical_comparison=categorical_comparison,
        shared_row_comparison=shared_row_comparison,
    )

    print(
        "Generating overall diagnosis..."
    )

    diagnosis = generate_overall_diagnosis(
        profiles=profiles,
        flags=flags,
        row_overlap=row_overlap,
        target_label_comparison=target_label_comparison,
        shared_row_comparison=shared_row_comparison,
    )

    report = {
        "analysis": "dataset_version_comparison",
        "established_reference": {
            "rows": ESTABLISHED_ROWS,
            "columns": ESTABLISHED_COLUMNS,
            "target_prevalence": ESTABLISHED_TARGET_PREVALENCE,
        },
        "datasets": profiles,
        "schema_comparison": schema,
        "row_overlap": row_overlap,
        "target_label_comparison": target_label_comparison,
        "numerical_distribution_comparison": numerical_comparison,
        "categorical_distribution_comparison": categorical_comparison,
        "shared_row_feature_comparison": shared_row_comparison,
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    json_path = save_json_report(
        report
    )

    summary_path = save_summary_report(
        report
    )

    csv_paths = save_csv_reports(
        numerical_comparison,
        categorical_comparison,
        shared_row_comparison,
    )

    # ========================================================
    # FINAL CONSOLE REPORT
    # ========================================================

    print_header(
        "EMPLOYEE ATTRITION — DATASET VERSION COMPARISON"
    )

    print(
        "[DATASET A]"
    )

    print(
        f"File:                 "
        f"{profiles[0]['relative_path']}"
    )

    print(
        f"Rows:                 "
        f"{profiles[0]['rows']}"
    )

    print(
        f"Columns:              "
        f"{profiles[0]['columns']}"
    )

    print(
        f"Target prevalence:    "
        f"{profiles[0]['target']['prevalence'] * 100:.2f}%"
    )

    print()

    print(
        "[DATASET B]"
    )

    print(
        f"File:                 "
        f"{profiles[1]['relative_path']}"
    )

    print(
        f"Rows:                 "
        f"{profiles[1]['rows']}"
    )

    print(
        f"Columns:              "
        f"{profiles[1]['columns']}"
    )

    print(
        f"Target prevalence:    "
        f"{profiles[1]['target']['prevalence'] * 100:.2f}%"
    )

    print()

    print(
        "[SCHEMA COMPARISON]"
    )

    print(
        f"Common columns:       "
        f"{schema['common_column_count']}"
    )

    print(
        f"Column order matches: "
        f"{schema['column_order_matches']}"
    )

    print(
        f"Only in Dataset A:    "
        f"{schema['columns_only_in_a']}"
    )

    print(
        f"Only in Dataset B:    "
        f"{schema['columns_only_in_b']}"
    )

    print(
        f"Dtype differences:    "
        f"{len(schema['dtype_differences'])}"
    )

    print()

    print(
        "[ROW OVERLAP]"
    )

    for key, value in row_overlap.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "[TARGET LABEL COMPARISON]"
    )

    for key, value in target_label_comparison.items():
        print(
            f"{key}: {value}"
        )

    print()

    print(
        "[TOP NUMERICAL DISTRIBUTION DIFFERENCES]"
    )

    for item in numerical_comparison[:10]:

        print(
            f"{item['feature']:<40} "
            f"KS={item['ks_statistic']:.4f}  "
            f"p={item['ks_p_value']:.4f}"
            if (
                item["ks_statistic"] is not None
                and item["ks_p_value"] is not None
            )
            else
            f"{item['feature']:<40} "
            f"KS=N/A"
        )

    print()

    print(
        "[TOP CATEGORICAL DISTRIBUTION DIFFERENCES]"
    )

    for item in categorical_comparison[:10]:

        delta = item[
            "max_category_proportion_delta"
        ]

        print(
            f"{item['feature']:<40} "
            f"max delta={delta * 100:.2f} pp"
            if delta is not None
            else
            f"{item['feature']:<40} "
            f"max delta=N/A"
        )

    print()

    if shared_row_comparison:

        print(
            "[TOP SHARED-ROW FEATURE DIFFERENCES]"
        )

        for item in shared_row_comparison.get(
            "feature_comparisons",
            [],
        )[:10]:

            difference_rate = item[
                "difference_rate"
            ]

            print(
                f"{item['feature']:<40} "
                f"different={difference_rate * 100:.2f}%"
                if difference_rate is not None
                else
                f"{item['feature']:<40} "
                f"different=N/A"
            )

        print()

    print(
        "[DIAGNOSTIC FLAGS]"
    )

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

    print(
        "[OVERALL DIAGNOSIS]"
    )

    print(
        diagnosis
    )

    print()

    print(
        "[OUTPUT]"
    )

    print(
        f"Reports:              "
        f"{OUTPUT_DIR}"
    )

    print(
        f"JSON report:          "
        f"{json_path}"
    )

    print(
        f"Summary report:       "
        f"{summary_path}"
    )

    for path in csv_paths:

        print(
            f"CSV report:           "
            f"{path}"
        )

    print()

    print(
        "=" * 60
    )

    print(
        "DATASET VERSION COMPARISON COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()