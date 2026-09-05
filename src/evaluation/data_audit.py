from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "audit"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Attrition"
ID_COLUMN = "Employee_ID"

EXPECTED_ROWS = 1000

EXPECTED_ATTRITION_RATE = 0.189


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directory() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    return df


# ============================================================
# BASIC DATASET AUDIT
# ============================================================

def audit_basic_structure(
    df: pd.DataFrame,
) -> dict[str, object]:

    duplicate_rows = int(
        df.duplicated().sum()
    )

    duplicate_ids = 0

    if ID_COLUMN in df.columns:
        duplicate_ids = int(
            df[ID_COLUMN].duplicated().sum()
        )

    missing_cells = int(
        df.isna().sum().sum()
    )

    total_cells = int(
        df.shape[0] * df.shape[1]
    )

    missing_percentage = (
        missing_cells / total_cells * 100
        if total_cells > 0
        else 0.0
    )

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "missing_cells": missing_cells,
        "missing_percentage": missing_percentage,
    }


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target(
    df: pd.DataFrame,
) -> dict[str, object]:

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "not found."
        )

    counts = (
        df[TARGET_COLUMN]
        .value_counts(dropna=False)
    )

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        * 100
    )

    result = {
        "unique_values": [
            str(value)
            for value in counts.index.tolist()
        ],
        "counts": {
            str(key): int(value)
            for key, value in counts.items()
        },
        "percentages": {
            str(key): float(value)
            for key, value in percentages.items()
        },
    }

    if "Yes" in counts.index:
        result["positive_count"] = int(
            counts["Yes"]
        )

        result["positive_rate"] = float(
            percentages["Yes"] / 100
        )

    return result


# ============================================================
# COLUMN TYPE AUDIT
# ============================================================

def audit_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for column in df.columns:

        series = df[column]

        records.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "unique_values": int(
                    series.nunique(
                        dropna=True
                    )
                ),
                "missing_values": int(
                    series.isna().sum()
                ),
                "missing_percentage": float(
                    series.isna().mean() * 100
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by=[
                "missing_values",
                "unique_values",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# NUMERICAL DISTRIBUTION AUDIT
# ============================================================

def audit_numerical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    records = []

    for feature in numerical_columns:

        if feature == ID_COLUMN:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        q1 = float(
            values.quantile(0.25)
        )

        median = float(
            values.median()
        )

        q3 = float(
            values.quantile(0.75)
        )

        minimum = float(
            values.min()
        )

        maximum = float(
            values.max()
        )

        mean = float(
            values.mean()
        )

        std = float(
            values.std()
        )

        zero_count = int(
            (values == 0).sum()
        )

        negative_count = int(
            (values < 0).sum()
        )

        records.append(
            {
                "feature": feature,
                "minimum": minimum,
                "q1": q1,
                "median": median,
                "q3": q3,
                "maximum": maximum,
                "mean": mean,
                "std": std,
                "zero_count": zero_count,
                "negative_count": negative_count,
                "unique_values": int(
                    values.nunique()
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="unique_values",
            ascending=True,
        )
        .reset_index(drop=True)
    )


# ============================================================
# CATEGORICAL DISTRIBUTION AUDIT
# ============================================================

def audit_categorical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "string",
                "category",
            ]
        )
        .columns
        .tolist()
    )

    records = []

    for feature in categorical_columns:

        if feature == TARGET_COLUMN:
            continue

        series = df[feature].dropna()

        value_counts = (
            series.value_counts()
        )

        if value_counts.empty:
            continue

        most_common = int(
            value_counts.iloc[0]
        )

        least_common = int(
            value_counts.iloc[-1]
        )

        records.append(
            {
                "feature": feature,
                "categories": int(
                    series.nunique()
                ),
                "most_common_count": most_common,
                "least_common_count": least_common,
                "most_common_percentage": float(
                    most_common
                    / len(series)
                    * 100
                ),
                "least_common_percentage": float(
                    least_common
                    / len(series)
                    * 100
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="categories",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# TARGET ASSOCIATION AUDIT
# ============================================================

def audit_target_relationships(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if TARGET_COLUMN not in df.columns:
        return pd.DataFrame()

    target = (
        df[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    records = []

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for feature in numerical_columns:

        if feature in {
            ID_COLUMN,
            TARGET_COLUMN,
        }:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid = (
            values.notna()
            & target.notna()
        )

        x = values.loc[valid]
        y = target.loc[valid]

        if len(x) < 2:
            continue

        correlation = float(
            x.corr(y)
        )

        if np.isnan(correlation):
            correlation = 0.0

        records.append(
            {
                "feature": feature,
                "correlation_with_attrition": (
                    correlation
                ),
                "absolute_correlation": abs(
                    correlation
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# REALISM / CONSISTENCY CHECKS
# ============================================================

def audit_business_consistency(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    def check_greater(
        left: str,
        right: str,
        description: str,
    ) -> None:

        if (
            left not in df.columns
            or right not in df.columns
        ):
            return

        left_values = pd.to_numeric(
            df[left],
            errors="coerce",
        )

        right_values = pd.to_numeric(
            df[right],
            errors="coerce",
        )

        valid = (
            left_values.notna()
            & right_values.notna()
        )

        violations = (
            left_values[valid]
            > right_values[valid]
        )

        count = int(
            violations.sum()
        )

        records.append(
            {
                "check": description,
                "violations": count,
                "violation_rate": (
                    count
                    / int(valid.sum())
                    * 100
                    if valid.sum() > 0
                    else 0.0
                ),
            }
        )

    check_greater(
        "Years_in_Current_Role",
        "Years_at_Company",
        "Years_in_Current_Role > Years_at_Company",
    )

    check_greater(
        "Years_Since_Last_Promotion",
        "Years_at_Company",
        "Years_Since_Last_Promotion > Years_at_Company",
    )

    check_greater(
        "Years_at_Company",
        "Age",
        "Years_at_Company > Age",
    )

    check_greater(
        "Years_in_Current_Role",
        "Age",
        "Years_in_Current_Role > Age",
    )

    return pd.DataFrame(records)


# ============================================================
# SYNTHETIC / RANDOMNESS INDICATORS
# ============================================================

def calculate_randomness_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    for feature in numerical_columns:

        if feature in {
            ID_COLUMN,
            TARGET_COLUMN,
        }:
            continue

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        unique_ratio = (
            values.nunique()
            / len(values)
        )

        records.append(
            {
                "feature": feature,
                "unique_ratio": float(
                    unique_ratio
                ),
                "integer_like": bool(
                    np.allclose(
                        values,
                        np.round(values),
                    )
                ),
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std()
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            by="unique_ratio",
            ascending=True,
        )
        .reset_index(drop=True)
    )


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    basic: dict[str, object],
    target: dict[str, object],
    columns: pd.DataFrame,
    numerical: pd.DataFrame,
    categorical: pd.DataFrame,
    relationships: pd.DataFrame,
    consistency: pd.DataFrame,
    randomness: pd.DataFrame,
) -> None:

    columns.to_csv(
        REPORT_DIR / "column_audit.csv",
        index=False,
    )

    numerical.to_csv(
        REPORT_DIR / "numerical_distribution_audit.csv",
        index=False,
    )

    categorical.to_csv(
        REPORT_DIR / "categorical_distribution_audit.csv",
        index=False,
    )

    relationships.to_csv(
        REPORT_DIR / "target_relationship_audit.csv",
        index=False,
    )

    consistency.to_csv(
        REPORT_DIR / "business_consistency_audit.csv",
        index=False,
    )

    randomness.to_csv(
        REPORT_DIR / "randomness_indicators.csv",
        index=False,
    )

    summary = {
        "basic_structure": basic,
        "target": target,
        "expected_rows": EXPECTED_ROWS,
        "expected_attrition_rate": (
            EXPECTED_ATTRITION_RATE
        ),
    }

    with open(
        REPORT_DIR / "audit_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )


# ============================================================
# TERMINAL REPORT
# ============================================================

def print_report(
    basic: dict[str, object],
    target: dict[str, object],
    relationships: pd.DataFrame,
    consistency: pd.DataFrame,
    randomness: pd.DataFrame,
) -> None:

    print("\n" + "=" * 60)
    print(
        "EMPLOYEE ATTRITION — DATASET AUDIT"
    )
    print("=" * 60)

    print("\n[STRUCTURE]")

    print(
        f"Rows:                 "
        f"{basic['rows']}"
    )

    print(
        f"Columns:              "
        f"{basic['columns']}"
    )

    print(
        f"Duplicate rows:       "
        f"{basic['duplicate_rows']}"
    )

    print(
        f"Duplicate IDs:        "
        f"{basic['duplicate_ids']}"
    )

    print(
        f"Missing cells:        "
        f"{basic['missing_cells']}"
    )

    print(
        f"Missing percentage:   "
        f"{basic['missing_percentage']:.2f}%"
    )

    print("\n[TARGET]")

    print(
        f"Unique values:        "
        f"{target['unique_values']}"
    )

    print(
        f"Counts:               "
        f"{target['counts']}"
    )

    if "positive_rate" in target:

        print(
            f"Attrition rate:       "
            f"{target['positive_rate'] * 100:.2f}%"
        )

    print("\n[TOP TARGET RELATIONSHIPS]")

    if not relationships.empty:

        print(
            relationships[
                [
                    "feature",
                    "correlation_with_attrition",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

    print("\n[BUSINESS CONSISTENCY]")

    if not consistency.empty:

        print(
            consistency
            .round(2)
            .to_string(index=False)
        )

    print("\n[LOW UNIQUE-RATIO FEATURES]")

    if not randomness.empty:

        print(
            randomness[
                [
                    "feature",
                    "unique_ratio",
                    "integer_like",
                ]
            ]
            .head(10)
            .round(4)
            .to_string(index=False)
        )

    print("\n[OUTPUT]")

    print(
        f"Reports:              "
        f"{REPORT_DIR}"
    )

    print("\n" + "=" * 60)
    print(
        "DATASET AUDIT COMPLETE"
    )
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    create_output_directory()

    df = load_data()

    print(
        "\nRunning dataset structure audit..."
    )

    basic = audit_basic_structure(
        df
    )

    print(
        "Running target audit..."
    )

    target = audit_target(
        df
    )

    print(
        "Running column audit..."
    )

    columns = audit_columns(
        df
    )

    print(
        "Running numerical distribution audit..."
    )

    numerical = audit_numerical_features(
        df
    )

    print(
        "Running categorical distribution audit..."
    )

    categorical = audit_categorical_features(
        df
    )

    print(
        "Running target relationship audit..."
    )

    relationships = audit_target_relationships(
        df
    )

    print(
        "Running business consistency checks..."
    )

    consistency = audit_business_consistency(
        df
    )

    print(
        "Running randomness indicators..."
    )

    randomness = calculate_randomness_indicators(
        df
    )

    save_reports(
        basic,
        target,
        columns,
        numerical,
        categorical,
        relationships,
        consistency,
        randomness,
    )

    print_report(
        basic,
        target,
        relationships,
        consistency,
        randomness,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()