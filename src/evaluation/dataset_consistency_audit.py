"""
Dataset Consistency Audit
=========================

Employee Attrition ML Project

Purpose
-------
Audit whether the dataset used across the evaluation pipeline is consistent.

This diagnostic is intentionally READ-ONLY. It does not:
- modify the dataset
- modify the target
- fit or save a model
- change preprocessing
- change thresholds

The audit focuses on:
1. Dataset discovery and file identity
2. Row/column counts
3. Target column detection
4. Target encoding and prevalence
5. Missing values
6. Duplicate rows
7. Duplicate employee identifiers
8. Constant / near-constant columns
9. Column type consistency
10. Feature schema consistency
11. Target consistency after common filtering operations
12. Suspicious row loss
13. Basic numeric range consistency
14. Categorical value consistency
15. Reproducibility fingerprints
16. Comparison against the established 23.60% prevalence

Run
---
python -m src.evaluation.dataset_consistency_audit
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "dataset_consistency_audit"
)

EXPECTED_TARGET_PREVALENCE = 0.2360

TARGET_CANDIDATES = [
    "Attrition",
    "attrition",
    "Attrition_Flag",
    "attrition_flag",
    "Target",
    "target",
    "Exited",
    "exited",
    "Left",
    "left",
]

ID_CANDIDATES = [
    "EmployeeNumber",
    "Employee_ID",
    "EmployeeID",
    "Employee_Id",
    "employee_id",
    "employeeid",
    "ID",
    "Id",
    "id",
]

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "reports",
    ".pytest_cache",
    ".mypy_cache",
}

DATA_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".xls",
    ".parquet",
}


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def print_section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to a JSON-safe float.
    """
    if value is None:
        return None

    try:
        result = float(value)

        if not math.isfinite(result):
            return None

        return result

    except (TypeError, ValueError):
        return None


def json_safe(value: Any) -> Any:
    """
    Recursively convert numpy/pandas objects into JSON-safe objects.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)

        if not math.isfinite(value):
            return None

        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, pd.Interval):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def normalize_column_name(name: Any) -> str:
    """
    Normalize a column name for comparison.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Create a deterministic fingerprint for the dataframe.

    The fingerprint is based on:
    - column order
    - column names
    - dtypes
    - row values

    This helps determine whether two evaluation scripts are
    actually operating on the same dataset.
    """
    hash_obj = hashlib.sha256()

    schema_string = "|".join(
        f"{column}:{df[column].dtype}"
        for column in df.columns
    )

    hash_obj.update(schema_string.encode("utf-8"))

    try:
        hashed_values = pd.util.hash_pandas_object(
            df,
            index=True,
        ).values

        hash_obj.update(hashed_values.tobytes())

    except Exception:
        hash_obj.update(
            df.to_csv(index=True).encode("utf-8")
        )

    return hash_obj.hexdigest()


def file_sha256(path: Path) -> str:
    """
    Calculate SHA-256 hash of a file.
    """
    hash_obj = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hash_obj.update(chunk)

    return hash_obj.hexdigest()


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def discover_dataset_files() -> List[Path]:
    """
    Discover likely dataset files inside the project.

    Reports are deliberately excluded because generated CSV/JSON reports
    should never become accidental model datasets.
    """
    candidates: List[Path] = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        dirs[:] = [
            directory
            for directory in dirs
            if directory not in IGNORED_DIRECTORIES
        ]

        root_path = Path(root)

        for filename in files:
            path = root_path / filename

            if path.suffix.lower() not in DATA_EXTENSIONS:
                continue

            candidates.append(path)

    return sorted(candidates)


def dataset_priority(path: Path) -> Tuple[int, int, str]:
    """
    Give likely source datasets a higher priority.

    Lower score = higher priority.
    """
    path_text = str(path).lower()

    score = 100

    if "raw" in path_text:
        score -= 30

    if "data" in path_text:
        score -= 20

    if "dataset" in path_text:
        score -= 20

    if "employee" in path_text:
        score -= 15

    if "train" in path_text:
        score += 10

    if "test" in path_text:
        score += 20

    if "processed" in path_text:
        score += 5

    return (
        score,
        len(path.parts),
        str(path),
    )


def choose_dataset(candidates: List[Path]) -> Path:
    """
    Select the most likely source dataset.

    If several plausible datasets exist, print them all so that
    the audit makes the ambiguity visible.
    """
    if not candidates:
        raise FileNotFoundError(
            "No CSV/XLSX/XLS/Parquet dataset was found inside the project."
        )

    ranked = sorted(
        candidates,
        key=dataset_priority,
    )

    print()
    print("Discovered dataset files:")

    for index, path in enumerate(ranked, start=1):
        try:
            relative = path.relative_to(PROJECT_ROOT)
        except ValueError:
            relative = path

        print(f"  {index:>2}. {relative}")

    selected = ranked[0]

    print()
    print("Selected dataset:")
    print(f"  {selected}")

    if len(ranked) > 1:
        print()
        print(
            "WARNING: Multiple dataset files were found. "
            "The audit will inspect the highest-priority candidate."
        )

    return selected


# ============================================================================
# DATA LOADING
# ============================================================================

def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load CSV, Excel, or Parquet dataset.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)

    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)

    elif suffix == ".parquet":
        df = pd.read_parquet(path)

    else:
        raise ValueError(
            f"Unsupported dataset extension: {suffix}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


# ============================================================================
# TARGET DETECTION
# ============================================================================

def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Detect target column using known names first.
    """
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in TARGET_CANDIDATES:
        key = normalize_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def encode_binary_target(
    series: pd.Series,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Convert a binary target into 0/1 when possible.

    Returns:
        encoded_series
        metadata
    """
    original = series.copy()

    non_null = original.dropna()

    unique_values = list(
        pd.unique(non_null)
    )

    metadata: Dict[str, Any] = {
        "original_unique_values": [
            json_safe(value)
            for value in unique_values
        ],
        "mapping_method": None,
        "mapping": {},
    }

    if len(unique_values) != 2:
        return (
            pd.Series(
                np.nan,
                index=series.index,
                dtype=float,
            ),
            metadata,
        )

    # Boolean
    if pd.api.types.is_bool_dtype(series):
        encoded = series.astype(float)

        metadata["mapping_method"] = "boolean"

        return encoded, metadata

    # Numeric binary target
    if pd.api.types.is_numeric_dtype(series):
        numeric_values = sorted(
            float(value)
            for value in unique_values
        )

        if set(numeric_values).issubset({0.0, 1.0}):
            encoded = pd.to_numeric(
                series,
                errors="coerce",
            ).astype(float)

            metadata["mapping_method"] = "numeric_0_1"

            metadata["mapping"] = {
                str(value): int(value)
                for value in numeric_values
            }

            return encoded, metadata

    # String mapping
    normalized_values = {
        str(value).strip().lower(): value
        for value in unique_values
    }

    positive_tokens = {
        "yes",
        "y",
        "true",
        "1",
        "attrition",
        "left",
        "leave",
        "exited",
        "exit",
    }

    negative_tokens = {
        "no",
        "n",
        "false",
        "0",
        "stay",
        "stayed",
        "retained",
        "remain",
        "remaining",
    }

    positive_original = None
    negative_original = None

    for normalized, original_value in normalized_values.items():

        if normalized in positive_tokens:
            positive_original = original_value

        if normalized in negative_tokens:
            negative_original = original_value

    if (
        positive_original is not None
        and negative_original is not None
    ):
        encoded = pd.Series(
            np.nan,
            index=series.index,
            dtype=float,
        )

        encoded.loc[
            series == positive_original
        ] = 1.0

        encoded.loc[
            series == negative_original
        ] = 0.0

        metadata["mapping_method"] = "semantic_string"

        metadata["mapping"] = {
            str(negative_original): 0,
            str(positive_original): 1,
        }

        return encoded, metadata

    # Deterministic fallback.
    sorted_values = sorted(
        unique_values,
        key=lambda value: str(value),
    )

    mapping = {
        sorted_values[0]: 0,
        sorted_values[1]: 1,
    }

    encoded = series.map(mapping).astype(float)

    metadata["mapping_method"] = "deterministic_fallback"

    metadata["mapping"] = {
        str(key): value
        for key, value in mapping.items()
    }

    return encoded, metadata


# ============================================================================
# SCHEMA AUDIT
# ============================================================================

def audit_schema(
    df: pd.DataFrame,
    target_column: Optional[str],
) -> Dict[str, Any]:

    columns = list(df.columns)

    duplicate_columns = [
        column
        for column in pd.Series(columns)[
            pd.Series(columns).duplicated()
        ].tolist()
    ]

    dtype_counts = (
        df.dtypes
        .astype(str)
        .value_counts()
        .to_dict()
    )

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    return {
        "column_count": int(len(columns)),
        "feature_count": int(len(feature_columns)),
        "target_column": target_column,
        "duplicate_column_names": duplicate_columns,
        "dtype_counts": {
            str(key): int(value)
            for key, value in dtype_counts.items()
        },
        "columns": [
            {
                "name": str(column),
                "dtype": str(df[column].dtype),
                "missing": int(df[column].isna().sum()),
                "unique": int(df[column].nunique(dropna=True)),
            }
            for column in df.columns
        ],
    }


# ============================================================================
# TARGET AUDIT
# ============================================================================

def audit_target(
    df: pd.DataFrame,
    target_column: Optional[str],
) -> Dict[str, Any]:

    if target_column is None:
        return {
            "available": False,
            "reason": "Target column could not be detected.",
        }

    series = df[target_column]

    encoded, encoding_metadata = encode_binary_target(
        series
    )

    valid_target = encoded.dropna()

    positive_count = int(
        (valid_target == 1).sum()
    )

    negative_count = int(
        (valid_target == 0).sum()
    )

    missing_count = int(
        encoded.isna().sum()
    )

    total_valid = len(valid_target)

    prevalence = (
        positive_count / total_valid
        if total_valid > 0
        else None
    )

    expected_delta_pp = (
        (prevalence - EXPECTED_TARGET_PREVALENCE) * 100
        if prevalence is not None
        else None
    )

    return {
        "available": True,
        "column": target_column,
        "dtype": str(series.dtype),
        "row_count": int(len(series)),
        "missing_count": missing_count,
        "unique_count": int(series.nunique(dropna=True)),
        "unique_values": [
            json_safe(value)
            for value in pd.unique(
                series.dropna()
            )
        ],
        "encoding": encoding_metadata,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "valid_target_rows": int(total_valid),
        "positive_prevalence": safe_float(prevalence),
        "positive_prevalence_percent": (
            safe_float(prevalence * 100)
            if prevalence is not None
            else None
        ),
        "established_prevalence_percent": (
            EXPECTED_TARGET_PREVALENCE * 100
        ),
        "difference_from_established_pp": safe_float(
            expected_delta_pp
        ),
    }


# ============================================================================
# DUPLICATE AUDIT
# ============================================================================

def audit_duplicates(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    duplicate_rows = int(
        df.duplicated(keep=False).sum()
    )

    duplicate_row_groups = int(
        df.duplicated(keep=False).groupby(
            df.astype(str).agg(
                "||".join,
                axis=1,
            )
        ).any().sum()
    )

    detected_ids = []

    normalized_columns = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in ID_CANDIDATES:
        normalized = normalize_column_name(candidate)

        if normalized in normalized_columns:
            detected_ids.append(
                normalized_columns[normalized]
            )

    id_results = {}

    for column in detected_ids:
        non_null = df[column].dropna()

        duplicated_ids = int(
            non_null.duplicated(keep=False).sum()
        )

        unique_ids = int(
            non_null.nunique()
        )

        id_results[column] = {
            "non_null_rows": int(len(non_null)),
            "unique_values": unique_ids,
            "duplicated_id_rows": duplicated_ids,
        }

    return {
        "duplicate_row_count_keep_all": duplicate_rows,
        "duplicate_row_group_count": duplicate_row_groups,
        "detected_identifier_columns": detected_ids,
        "identifier_results": id_results,
    }


# ============================================================================
# MISSINGNESS AUDIT
# ============================================================================

def audit_missingness(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    missing = df.isna().sum()

    missing_percent = (
        missing / len(df) * 100
    )

    rows_with_missing = int(
        df.isna().any(axis=1).sum()
    )

    complete_rows = int(
        (~df.isna().any(axis=1)).sum()
    )

    top_missing = []

    for column in missing.sort_values(
        ascending=False
    ).index:

        count = int(missing[column])

        if count == 0:
            continue

        top_missing.append(
            {
                "feature": str(column),
                "missing_count": count,
                "missing_percent": safe_float(
                    missing_percent[column]
                ),
            }
        )

    return {
        "rows_with_any_missing": rows_with_missing,
        "complete_rows": complete_rows,
        "complete_row_percent": safe_float(
            complete_rows / len(df) * 100
        ),
        "columns_with_missing": int(
            (missing > 0).sum()
        ),
        "top_missing": top_missing[:20],
    }


# ============================================================================
# CONSTANT / LOW VARIANCE AUDIT
# ============================================================================

def audit_variance(
    df: pd.DataFrame,
) -> Dict[str, Any]:

    constant_columns = []
    near_constant_columns = []

    for column in df.columns:

        non_null = df[column].dropna()

        if len(non_null) == 0:
            continue

        unique_count = non_null.nunique()

        if unique_count <= 1:
            constant_columns.append(
                str(column)
            )
            continue

        frequencies = (
            non_null
            .value_counts(normalize=True)
        )

        top_frequency = float(
            frequencies.iloc[0]
        )

        if top_frequency >= 0.95:
            near_constant_columns.append(
                {
                    "feature": str(column),
                    "unique_values": int(unique_count),
                    "top_value_share": safe_float(
                        top_frequency
                    ),
                }
            )

    return {
        "constant_columns": constant_columns,
        "near_constant_columns": near_constant_columns,
    }


# ============================================================================
# RANGE AUDIT
# ============================================================================

def audit_numeric_ranges(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:

    results = []

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        results.append(
            {
                "feature": str(column),
                "min": safe_float(series.min()),
                "max": safe_float(series.max()),
                "mean": safe_float(series.mean()),
                "median": safe_float(series.median()),
                "std": safe_float(series.std()),
                "unique_values": int(
                    series.nunique()
                ),
            }
        )

    return results


# ============================================================================
# CATEGORICAL AUDIT
# ============================================================================

def audit_categorical_values(
    df: pd.DataFrame,
) -> List[Dict[str, Any]]:

    results = []

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    for column in categorical_columns:

        series = df[column]

        value_counts = (
            series
            .value_counts(dropna=False)
        )

        values = []

        for value, count in value_counts.head(20).items():

            if pd.isna(value):
                value_text = "<MISSING>"
            else:
                value_text = str(value)

            values.append(
                {
                    "value": value_text,
                    "count": int(count),
                    "share_percent": safe_float(
                        count / len(series) * 100
                    ),
                }
            )

        results.append(
            {
                "feature": str(column),
                "unique_values": int(
                    series.nunique(dropna=True)
                ),
                "values": values,
            }
        )

    return results


# ============================================================================
# FILTERING / ROW LOSS AUDIT
# ============================================================================

def audit_common_filters(
    df: pd.DataFrame,
    target_column: Optional[str],
) -> Dict[str, Any]:

    total_rows = len(df)

    results = []

    # ------------------------------------------------------------------
    # Complete-case filtering
    # ------------------------------------------------------------------

    complete_mask = ~df.isna().any(axis=1)

    complete_df = df.loc[
        complete_mask
    ].copy()

    results.append(
        {
            "operation": "drop_rows_with_any_missing",
            "rows_before": int(total_rows),
            "rows_after": int(len(complete_df)),
            "rows_removed": int(
                total_rows - len(complete_df)
            ),
            "retained_percent": safe_float(
                len(complete_df) / total_rows * 100
            ),
        }
    )

    # ------------------------------------------------------------------
    # Duplicate removal
    # ------------------------------------------------------------------

    deduplicated_df = df.drop_duplicates()

    results.append(
        {
            "operation": "drop_exact_duplicate_rows",
            "rows_before": int(total_rows),
            "rows_after": int(len(deduplicated_df)),
            "rows_removed": int(
                total_rows - len(deduplicated_df)
            ),
            "retained_percent": safe_float(
                len(deduplicated_df) / total_rows * 100
            ),
        }
    )

    # ------------------------------------------------------------------
    # Target-valid filtering
    # ------------------------------------------------------------------

    if target_column is not None:

        encoded_target, _ = encode_binary_target(
            df[target_column]
        )

        valid_mask = encoded_target.notna()

        target_valid_df = df.loc[
            valid_mask
        ].copy()

        results.append(
            {
                "operation": "keep_rows_with_valid_binary_target",
                "rows_before": int(total_rows),
                "rows_after": int(len(target_valid_df)),
                "rows_removed": int(
                    total_rows - len(target_valid_df)
                ),
                "retained_percent": safe_float(
                    len(target_valid_df) / total_rows * 100
                ),
            }
        )

        # --------------------------------------------------------------
        # Target prevalence after common filters
        # --------------------------------------------------------------

        filter_target_prevalence = {}

        for operation_name, filtered_df in [
            (
                "original",
                df,
            ),
            (
                "complete_case",
                complete_df,
            ),
            (
                "deduplicated",
                deduplicated_df,
            ),
            (
                "valid_target",
                target_valid_df,
            ),
        ]:

            if target_column not in filtered_df.columns:
                continue

            encoded, _ = encode_binary_target(
                filtered_df[target_column]
            )

            valid = encoded.dropna()

            if len(valid) == 0:
                prevalence = None
            else:
                prevalence = float(
                    (valid == 1).mean()
                )

            filter_target_prevalence[
                operation_name
            ] = {
                "rows": int(len(filtered_df)),
                "valid_target_rows": int(len(valid)),
                "positive_count": int(
                    (valid == 1).sum()
                ),
                "prevalence_percent": (
                    safe_float(prevalence * 100)
                    if prevalence is not None
                    else None
                ),
            }

        return {
            "operations": results,
            "target_prevalence_after_filters": (
                filter_target_prevalence
            ),
        }

    return {
        "operations": results,
        "target_prevalence_after_filters": {},
    }


# ============================================================================
# TARGET / FEATURE SCHEMA CHECK
# ============================================================================

def audit_target_position(
    df: pd.DataFrame,
    target_column: Optional[str],
) -> Dict[str, Any]:

    if target_column is None:
        return {
            "target_detected": False
        }

    target_index = list(
        df.columns
    ).index(target_column)

    feature_columns = [
        column
        for column in df.columns
        if column != target_column
    ]

    suspicious_target_names = [
        column
        for column in feature_columns
        if normalize_column_name(column)
        in {
            normalize_column_name(candidate)
            for candidate in TARGET_CANDIDATES
        }
    ]

    return {
        "target_detected": True,
        "target_column": target_column,
        "target_position_zero_based": int(
            target_index
        ),
        "target_is_last_column": bool(
            target_index == len(df.columns) - 1
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "other_target_like_columns": (
            suspicious_target_names
        ),
    }


# ============================================================================
# CROSS-DATASET COMPARISON
# ============================================================================

def inspect_all_candidate_datasets(
    candidates: List[Path],
    selected: Path,
) -> List[Dict[str, Any]]:

    results = []

    for path in candidates:

        try:
            df = load_dataset(path)

            target_column = detect_target_column(
                df
            )

            target_info = audit_target(
                df,
                target_column,
            )

            results.append(
                {
                    "path": str(path),
                    "relative_path": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "selected": bool(
                        path.resolve()
                        == selected.resolve()
                    ),
                    "rows": int(len(df)),
                    "columns": int(len(df.columns)),
                    "target_column": target_column,
                    "target_prevalence_percent": (
                        target_info.get(
                            "positive_prevalence_percent"
                        )
                    ),
                    "target_positive_count": (
                        target_info.get(
                            "positive_count"
                        )
                    ),
                    "fingerprint": dataframe_fingerprint(
                        df
                    ),
                    "file_sha256": file_sha256(
                        path
                    ),
                }
            )

        except Exception as exc:

            results.append(
                {
                    "path": str(path),
                    "relative_path": str(
                        path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "selected": bool(
                        path.resolve()
                        == selected.resolve()
                    ),
                    "error": str(exc),
                }
            )

    return results


# ============================================================================
# DIAGNOSTIC FLAGS
# ============================================================================

def generate_flags(
    dataset_info: Dict[str, Any],
    target_info: Dict[str, Any],
    duplicate_info: Dict[str, Any],
    missing_info: Dict[str, Any],
    variance_info: Dict[str, Any],
    filter_info: Dict[str, Any],
    candidate_datasets: List[Dict[str, Any]],
) -> List[str]:

    flags: List[str] = []

    rows = dataset_info["rows"]
    columns = dataset_info["columns"]

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    if not target_info.get("available", False):

        flags.append(
            "Target column could not be detected."
        )

    else:

        prevalence = target_info.get(
            "positive_prevalence"
        )

        if prevalence is not None:

            difference_pp = abs(
                prevalence
                - EXPECTED_TARGET_PREVALENCE
            ) * 100

            if difference_pp >= 1.0:

                flags.append(
                    "Target prevalence differs by "
                    f"{difference_pp:.2f} percentage points "
                    "from the established 23.60% dataset prevalence."
                )

            if difference_pp >= 3.0:

                flags.append(
                    "Target prevalence discrepancy is large enough "
                    "to strongly suggest a dataset-version, target-"
                    "construction, or row-filtering inconsistency."
                )

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    if rows != 1000:

        flags.append(
            f"Dataset contains {rows} rows instead of the established "
            "1000-row dataset."
        )

    if columns != 26:

        flags.append(
            f"Dataset contains {columns} columns instead of the "
            "previously observed 26-column raw dataset."
        )

    # ------------------------------------------------------------------
    # Duplicates
    # ------------------------------------------------------------------

    duplicate_rows = duplicate_info.get(
        "duplicate_row_count_keep_all",
        0,
    )

    if duplicate_rows > 0:

        flags.append(
            f"{duplicate_rows} rows participate in exact duplicate groups."
        )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    rows_with_missing = missing_info.get(
        "rows_with_any_missing",
        0,
    )

    if rows_with_missing > 0:

        flags.append(
            f"{rows_with_missing} rows contain at least one missing value; "
            "complete-case filtering would change the dataset."
        )

    # ------------------------------------------------------------------
    # Constant columns
    # ------------------------------------------------------------------

    constant_columns = variance_info.get(
        "constant_columns",
        [],
    )

    if constant_columns:

        flags.append(
            "Constant columns detected: "
            + ", ".join(constant_columns)
            + "."
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    prevalence_after_filters = filter_info.get(
        "target_prevalence_after_filters",
        {},
    )

    original = prevalence_after_filters.get(
        "original"
    )

    for operation_name, details in (
        prevalence_after_filters.items()
    ):

        if operation_name == "original":
            continue

        original_prevalence = (
            original.get("prevalence_percent")
            if original
            else None
        )

        filtered_prevalence = details.get(
            "prevalence_percent"
        )

        if (
            original_prevalence is not None
            and filtered_prevalence is not None
        ):

            delta = abs(
                filtered_prevalence
                - original_prevalence
            )

            if delta >= 1.0:

                flags.append(
                    f"The '{operation_name}' filtering operation changes "
                    f"target prevalence by {delta:.2f} percentage points."
                )

    # ------------------------------------------------------------------
    # Multiple candidate datasets
    # ------------------------------------------------------------------

    valid_candidates = [
        candidate
        for candidate in candidate_datasets
        if "error" not in candidate
    ]

    prevalences = []

    for candidate in valid_candidates:

        prevalence = candidate.get(
            "target_prevalence_percent"
        )

        if prevalence is not None:
            prevalences.append(
                (
                    candidate["relative_path"],
                    prevalence,
                )
            )

    if len(prevalences) > 1:

        unique_prevalences = {
            round(value, 4)
            for _, value in prevalences
        }

        if len(unique_prevalences) > 1:

            flags.append(
                "Multiple candidate dataset files have different "
                "target prevalences. Dataset selection must be "
                "standardized across evaluation scripts."
            )

    # ------------------------------------------------------------------
    # Feature count
    # ------------------------------------------------------------------

    if rows == 1000 and columns == 26:

        flags.append(
            "Dataset shape matches the established raw dataset "
            "shape of 1000 rows × 26 columns."
        )

    return flags


# ============================================================================
# OVERALL DIAGNOSIS
# ============================================================================

def generate_overall_diagnosis(
    target_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    flags: List[str],
) -> str:

    if not target_info.get("available", False):

        return (
            "The dataset cannot currently be considered consistent "
            "because the target column could not be reliably detected."
        )

    prevalence = target_info.get(
        "positive_prevalence"
    )

    if prevalence is None:

        return (
            "The target could not be encoded as a valid binary target. "
            "Target construction or dataset loading must be investigated "
            "before further model evaluation."
        )

    prevalence_delta_pp = abs(
        prevalence
        - EXPECTED_TARGET_PREVALENCE
    ) * 100

    rows = dataset_info["rows"]
    columns = dataset_info["columns"]

    if (
        prevalence_delta_pp >= 3.0
        or rows != 1000
        or columns != 26
    ):

        return (
            "The dataset is NOT consistent with the established evaluation "
            "dataset. The observed target prevalence and/or dataset shape "
            "indicates that a different dataset version, target-generation "
            "rule, row-filtering operation, or preprocessing stage is being "
            "used. This must be resolved before further model optimization."
        )

    if prevalence_delta_pp >= 1.0:

        return (
            "The dataset is broadly similar to the established dataset, "
            "but the target prevalence differs enough to warrant investigation "
            "before relying on evaluation results."
        )

    if flags:

        return (
            "The dataset is broadly consistent with the established "
            "dataset, but diagnostic issues remain that should be reviewed "
            "before interpreting model performance."
        )

    return (
        "The dataset is consistent with the established dataset shape, "
        "target prevalence, and basic structural checks."
    )


# ============================================================================
# REPORT WRITING
# ============================================================================

def save_json_report(
    report: Dict[str, Any],
) -> Path:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "dataset_consistency_audit_report.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            json_safe(report),
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def save_summary_report(
    report: Dict[str, Any],
) -> Path:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / "dataset_consistency_audit_summary.txt"
    )

    dataset = report["dataset"]
    target = report["target"]
    duplicates = report["duplicates"]
    missing = report["missingness"]
    variance = report["variance"]
    filters = report["filters"]

    lines: List[str] = []

    lines.append(
        "EMPLOYEE ATTRITION — DATASET CONSISTENCY AUDIT"
    )
    lines.append(
        "=" * 60
    )

    lines.append("")
    lines.append("[DATASET]")
    lines.append(
        f"Path:                 {dataset['path']}"
    )
    lines.append(
        f"Rows:                 {dataset['rows']}"
    )
    lines.append(
        f"Columns:              {dataset['columns']}"
    )
    lines.append(
        f"Fingerprint:          {dataset['fingerprint']}"
    )
    lines.append(
        f"File SHA-256:         {dataset['file_sha256']}"
    )

    lines.append("")
    lines.append("[TARGET]")

    if target.get("available"):

        lines.append(
            f"Column:               {target['column']}"
        )
        lines.append(
            f"Positive count:       {target['positive_count']}"
        )
        lines.append(
            f"Negative count:       {target['negative_count']}"
        )
        lines.append(
            f"Missing target:       {target['missing_count']}"
        )
        lines.append(
            f"Prevalence:            "
            f"{target['positive_prevalence_percent']:.2f}%"
        )
        lines.append(
            f"Established:           "
            f"{target['established_prevalence_percent']:.2f}%"
        )
        lines.append(
            f"Difference:            "
            f"{target['difference_from_established_pp']:+.2f} pp"
        )
        lines.append(
            f"Encoding:              "
            f"{target['encoding']['mapping_method']}"
        )

    else:

        lines.append(
            "Target could not be detected."
        )

    lines.append("")
    lines.append("[DUPLICATES]")
    lines.append(
        f"Duplicate row count:  "
        f"{duplicates['duplicate_row_count_keep_all']}"
    )
    lines.append(
        f"Identifier columns:   "
        f"{duplicates['detected_identifier_columns']}"
    )

    lines.append("")
    lines.append("[MISSINGNESS]")
    lines.append(
        f"Rows with missing:     "
        f"{missing['rows_with_any_missing']}"
    )
    lines.append(
        f"Complete rows:         "
        f"{missing['complete_rows']}"
    )
    lines.append(
        f"Columns with missing: "
        f"{missing['columns_with_missing']}"
    )

    lines.append("")
    lines.append("[VARIANCE]")
    lines.append(
        f"Constant columns:     "
        f"{variance['constant_columns']}"
    )
    lines.append(
        f"Near-constant count:  "
        f"{len(variance['near_constant_columns'])}"
    )

    lines.append("")
    lines.append("[FILTERING EFFECTS]")

    prevalence_after_filters = (
        filters["target_prevalence_after_filters"]
    )

    for operation, details in (
        prevalence_after_filters.items()
    ):

        prevalence = details.get(
            "prevalence_percent"
        )

        prevalence_text = (
            f"{prevalence:.2f}%"
            if prevalence is not None
            else "N/A"
        )

        lines.append(
            f"{operation:30s} "
            f"rows={details['rows']:4d} "
            f"prevalence={prevalence_text}"
        )

    lines.append("")
    lines.append("[DIAGNOSTIC FLAGS]")

    if report["diagnostic_flags"]:

        for flag in report["diagnostic_flags"]:
            lines.append(
                f"- {flag}"
            )

    else:

        lines.append(
            "No major consistency flags detected."
        )

    lines.append("")
    lines.append("[OVERALL DIAGNOSIS]")
    lines.append(
        report["overall_diagnosis"]
    )

    lines.append("")
    lines.append("[OTHER DATASET CANDIDATES]")

    for candidate in report[
        "candidate_datasets"
    ]:

        relative_path = candidate.get(
            "relative_path",
            candidate.get("path"),
        )

        prevalence = candidate.get(
            "target_prevalence_percent"
        )

        if prevalence is None:

            prevalence_text = "N/A"

        else:

            prevalence_text = (
                f"{prevalence:.2f}%"
            )

        lines.append(
            f"{relative_path} | "
            f"rows={candidate.get('rows', 'N/A')} | "
            f"columns={candidate.get('columns', 'N/A')} | "
            f"target={candidate.get('target_column', 'N/A')} | "
            f"prevalence={prevalence_text}"
        )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )

    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print(
        "Running dataset consistency audit..."
    )

    # ------------------------------------------------------------------
    # Discover datasets
    # ------------------------------------------------------------------

    print(
        "Discovering dataset files..."
    )

    candidates = discover_dataset_files()

    if not candidates:

        raise FileNotFoundError(
            "No dataset files were found."
        )

    selected_path = choose_dataset(
        candidates
    )

    # ------------------------------------------------------------------
    # Load selected dataset
    # ------------------------------------------------------------------

    print()
    print(
        "Loading selected dataset..."
    )

    df = load_dataset(
        selected_path
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

    # ------------------------------------------------------------------
    # Target detection
    # ------------------------------------------------------------------

    print(
        "Detecting target column..."
    )

    target_column = detect_target_column(
        df
    )

    if target_column is None:

        print(
            "WARNING: Target column could not be detected."
        )

    else:

        print(
            f"Target column:        {target_column}"
        )

    # ------------------------------------------------------------------
    # Core audits
    # ------------------------------------------------------------------

    print(
        "Auditing dataset schema..."
    )

    schema_info = audit_schema(
        df,
        target_column,
    )

    print(
        "Auditing target distribution..."
    )

    target_info = audit_target(
        df,
        target_column,
    )

    print(
        "Auditing duplicates..."
    )

    duplicate_info = audit_duplicates(
        df
    )

    print(
        "Auditing missing values..."
    )

    missing_info = audit_missingness(
        df
    )

    print(
        "Auditing constant and near-constant columns..."
    )

    variance_info = audit_variance(
        df
    )

    print(
        "Auditing numeric ranges..."
    )

    numeric_ranges = audit_numeric_ranges(
        df
    )

    print(
        "Auditing categorical values..."
    )

    categorical_values = (
        audit_categorical_values(
            df
        )
    )

    print(
        "Auditing target position and feature schema..."
    )

    target_position = audit_target_position(
        df,
        target_column,
    )

    print(
        "Testing common row-filtering effects..."
    )

    filter_info = audit_common_filters(
        df,
        target_column,
    )

    # ------------------------------------------------------------------
    # Compare all available datasets
    # ------------------------------------------------------------------

    print(
        "Comparing all discovered dataset candidates..."
    )

    candidate_datasets = (
        inspect_all_candidate_datasets(
            candidates,
            selected_path,
        )
    )

    # ------------------------------------------------------------------
    # Dataset identity
    # ------------------------------------------------------------------

    dataset_info = {
        "path": str(selected_path),
        "relative_path": str(
            selected_path.relative_to(
                PROJECT_ROOT
            )
        ),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "fingerprint": dataframe_fingerprint(
            df
        ),
        "file_sha256": file_sha256(
            selected_path
        ),
        "file_size_bytes": int(
            selected_path.stat().st_size
        ),
    }

    # ------------------------------------------------------------------
    # Diagnostic flags
    # ------------------------------------------------------------------

    print(
        "Generating diagnostic flags..."
    )

    flags = generate_flags(
        dataset_info=dataset_info,
        target_info=target_info,
        duplicate_info=duplicate_info,
        missing_info=missing_info,
        variance_info=variance_info,
        filter_info=filter_info,
        candidate_datasets=candidate_datasets,
    )

    # ------------------------------------------------------------------
    # Overall diagnosis
    # ------------------------------------------------------------------

    print(
        "Generating overall diagnosis..."
    )

    overall_diagnosis = (
        generate_overall_diagnosis(
            target_info=target_info,
            dataset_info=dataset_info,
            flags=flags,
        )
    )

    # ------------------------------------------------------------------
    # Final report object
    # ------------------------------------------------------------------

    report = {
        "audit": {
            "name": "dataset_consistency_audit",
            "version": "1.0",
            "expected_rows": 1000,
            "expected_raw_columns": 26,
            "expected_target_prevalence": 0.2360,
            "read_only": True,
        },
        "dataset": dataset_info,
        "schema": schema_info,
        "target": target_info,
        "target_position": target_position,
        "duplicates": duplicate_info,
        "missingness": missing_info,
        "variance": variance_info,
        "numeric_ranges": numeric_ranges,
        "categorical_values": categorical_values,
        "filters": filter_info,
        "candidate_datasets": candidate_datasets,
        "diagnostic_flags": flags,
        "overall_diagnosis": overall_diagnosis,
    }

    # ------------------------------------------------------------------
    # Save reports
    # ------------------------------------------------------------------

    json_path = save_json_report(
        report
    )

    summary_path = save_summary_report(
        report
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print_section(
        "EMPLOYEE ATTRITION — DATASET CONSISTENCY AUDIT"
    )

    print()
    print("[DATASET]")

    print(
        f"Path:                 "
        f"{dataset_info['relative_path']}"
    )

    print(
        f"Rows:                 "
        f"{dataset_info['rows']}"
    )

    print(
        f"Columns:              "
        f"{dataset_info['columns']}"
    )

    print(
        f"Fingerprint:          "
        f"{dataset_info['fingerprint'][:20]}..."
    )

    print()
    print("[TARGET]")

    if target_info.get("available"):

        print(
            f"Column:               "
            f"{target_info['column']}"
        )

        print(
            f"Positive count:       "
            f"{target_info['positive_count']}"
        )

        print(
            f"Negative count:       "
            f"{target_info['negative_count']}"
        )

        print(
            f"Missing target:       "
            f"{target_info['missing_count']}"
        )

        print(
            f"Target prevalence:    "
            f"{target_info['positive_prevalence_percent']:.2f}%"
        )

        print(
            f"Established:          "
            f"{target_info['established_prevalence_percent']:.2f}%"
        )

        print(
            f"Difference:           "
            f"{target_info['difference_from_established_pp']:+.2f} pp"
        )

        print(
            f"Encoding:             "
            f"{target_info['encoding']['mapping_method']}"
        )

        print(
            f"Mapping:              "
            f"{target_info['encoding']['mapping']}"
        )

    else:

        print(
            "Target unavailable."
        )

    print()
    print("[DUPLICATES]")

    print(
        f"Exact duplicate rows: "
        f"{duplicate_info['duplicate_row_count_keep_all']}"
    )

    print(
        f"Identifier columns:   "
        f"{duplicate_info['detected_identifier_columns']}"
    )

    print()
    print("[MISSINGNESS]")

    print(
        f"Rows with missing:     "
        f"{missing_info['rows_with_any_missing']}"
    )

    print(
        f"Complete rows:         "
        f"{missing_info['complete_rows']}"
    )

    print(
        f"Columns with missing: "
        f"{missing_info['columns_with_missing']}"
    )

    print()
    print("[VARIANCE]")

    print(
        f"Constant columns:     "
        f"{variance_info['constant_columns']}"
    )

    print(
        f"Near-constant count:  "
        f"{len(variance_info['near_constant_columns'])}"
    )

    print()
    print("[FILTERING EFFECTS]")

    prevalence_after_filters = (
        filter_info[
            "target_prevalence_after_filters"
        ]
    )

    for operation, details in (
        prevalence_after_filters.items()
    ):

        prevalence = details.get(
            "prevalence_percent"
        )

        if prevalence is None:
            prevalence_text = "N/A"
        else:
            prevalence_text = (
                f"{prevalence:.2f}%"
            )

        print(
            f"{operation:30s} "
            f"rows={details['rows']:4d} "
            f"prevalence={prevalence_text}"
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
            "No major consistency flags detected."
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(
        overall_diagnosis
    )

    print()
    print("[OUTPUT]")

    print(
        f"Reports:              "
        f"{REPORT_DIR}"
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
        "DATASET CONSISTENCY AUDIT COMPLETE"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()