from pathlib import Path

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

REPORTS_DIR = PROJECT_ROOT / "reports"

PROFILE_REPORT_PATH = (
    REPORTS_DIR / "data_profile.csv"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw employee attrition dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    return df


# ============================================================
# NUMERICAL PROFILE
# ============================================================

def build_numerical_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a statistical profile of numerical columns.
    """

    numerical_df = df.select_dtypes(
        include="number"
    )

    profile = pd.DataFrame(
        {
            "dtype": numerical_df.dtypes.astype(str),
            "count": numerical_df.count(),
            "missing": numerical_df.isna().sum(),
            "unique": numerical_df.nunique(),
            "mean": numerical_df.mean(),
            "median": numerical_df.median(),
            "std": numerical_df.std(),
            "min": numerical_df.min(),
            "q25": numerical_df.quantile(0.25),
            "q75": numerical_df.quantile(0.75),
            "max": numerical_df.max(),
        }
    )

    return profile


# ============================================================
# CATEGORICAL PROFILE
# ============================================================

def build_categorical_profile(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a profile of categorical columns.
    """

    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns

    rows = []

    for column in categorical_columns:
        value_counts = (
            df[column]
            .value_counts(dropna=False)
        )

        total = len(df)

        for value, count in value_counts.items():
            percentage = (
                count / total
            ) * 100

            rows.append(
                {
                    "column": column,
                    "value": value,
                    "count": int(count),
                    "percentage": round(
                        percentage,
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# TARGET PROFILE
# ============================================================

def build_target_profile(
    df: pd.DataFrame,
    target: str = "Attrition",
) -> pd.DataFrame:
    """
    Build a detailed profile of the target variable.
    """

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    counts = (
        df[target]
        .value_counts(dropna=False)
    )

    percentages = (
        df[target]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        .mul(100)
        .round(2)
    )

    profile = pd.DataFrame(
        {
            "count": counts,
            "percentage": percentages,
        }
    )

    profile.index.name = target

    return profile


# ============================================================
# FEATURE SUMMARY
# ============================================================

def build_feature_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a high-level summary for every feature.
    """

    rows = []

    for column in df.columns:
        rows.append(
            {
                "column": column,
                "dtype": str(
                    df[column].dtype
                ),
                "missing": int(
                    df[column].isna().sum()
                ),
                "unique": int(
                    df[column].nunique(
                        dropna=False
                    )
                ),
                "unique_percentage": round(
                    (
                        df[column]
                        .nunique(
                            dropna=False
                        )
                        / len(df)
                    )
                    * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ATTRITION BY CATEGORICAL FEATURES
# ============================================================

def calculate_categorical_attrition_rates(
    df: pd.DataFrame,
    target: str = "Attrition",
) -> dict[str, pd.DataFrame]:
    """
    Calculate attrition counts and rates for each
    categorical feature.
    """

    categorical_columns = [
        column
        for column in df.select_dtypes(
            exclude="number"
        ).columns
        if column != target
    ]

    results = {}

    for column in categorical_columns:

        grouped = (
            df.groupby(column, dropna=False)[target]
            .agg(
                total_employees="count",
                attrition_count=lambda x: (
                    x == "Yes"
                ).sum(),
            )
        )

        grouped["attrition_rate"] = (
            grouped["attrition_count"]
            / grouped["total_employees"]
            * 100
        ).round(2)

        grouped = grouped.sort_values(
            "attrition_rate",
            ascending=False,
        )

        results[column] = grouped

    return results


# ============================================================
# ATTRITION BY NUMERICAL FEATURES
# ============================================================

def calculate_numerical_group_profiles(
    df: pd.DataFrame,
    target: str = "Attrition",
) -> dict[str, pd.DataFrame]:
    """
    Compare numerical features between employees
    who stayed and employees who left.
    """

    numerical_columns = [
        column
        for column in df.select_dtypes(
            include="number"
        ).columns
        if column != "Employee_ID"
    ]

    results = {}

    for column in numerical_columns:

        grouped = (
            df.groupby(target)[column]
            .agg(
                count="count",
                mean="mean",
                median="median",
                std="std",
                min="min",
                max="max",
            )
            .round(2)
        )

        results[column] = grouped

    return results


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_reports(
    df: pd.DataFrame,
) -> None:
    """
    Generate all initial profiling reports.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    numerical_profile = (
        build_numerical_profile(df)
    )

    categorical_profile = (
        build_categorical_profile(df)
    )

    target_profile = (
        build_target_profile(df)
    )

    feature_summary = (
        build_feature_summary(df)
    )

    categorical_attrition = (
        calculate_categorical_attrition_rates(
            df
        )
    )

    numerical_attrition = (
        calculate_numerical_group_profiles(
            df
        )
    )

    # --------------------------------------------------------
    # SAVE GENERAL REPORTS
    # --------------------------------------------------------

    numerical_profile.to_csv(
        REPORTS_DIR / "numerical_profile.csv"
    )

    categorical_profile.to_csv(
        REPORTS_DIR / "categorical_profile.csv",
        index=False,
    )

    target_profile.to_csv(
        REPORTS_DIR / "target_profile.csv"
    )

    feature_summary.to_csv(
        REPORTS_DIR / "feature_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SAVE CATEGORICAL ATTRITION REPORTS
    # --------------------------------------------------------

    categorical_dir = (
        REPORTS_DIR / "attrition_by_categorical"
    )

    categorical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column, result in (
        categorical_attrition.items()
    ):
        output_path = (
            categorical_dir
            / f"{column.lower()}_attrition.csv"
        )

        result.to_csv(output_path)

    # --------------------------------------------------------
    # SAVE NUMERICAL ATTRITION REPORTS
    # --------------------------------------------------------

    numerical_dir = (
        REPORTS_DIR / "attrition_by_numerical"
    )

    numerical_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for column, result in (
        numerical_attrition.items()
    ):
        output_path = (
            numerical_dir
            / f"{column.lower()}_attrition.csv"
        )

        result.to_csv(output_path)

    # --------------------------------------------------------
    # COMBINED PROFILE
    # --------------------------------------------------------

    feature_summary.to_csv(
        PROFILE_REPORT_PATH,
        index=False,
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print a concise summary of the profiling stage.
    """

    target_profile = (
        build_target_profile(df)
    )

    print("\n" + "=" * 60)
    print(
        "EMPLOYEE ATTRITION — DATA PROFILING"
    )
    print("=" * 60)

    print("\n[DATASET]")

    print(
        f"Rows:              {len(df)}"
    )

    print(
        f"Columns:           {len(df.columns)}"
    )

    print(
        f"Numerical columns: "
        f"{len(df.select_dtypes(include='number').columns)}"
    )

    print(
        f"Categorical columns: "
        f"{len(df.select_dtypes(exclude='number').columns)}"
    )

    print("\n[TARGET DISTRIBUTION]")

    print(
        target_profile.to_string()
    )

    print("\n[REPORTS]")

    print(
        f"Reports directory: "
        f"{REPORTS_DIR}"
    )

    print(
        f"Feature summary:   "
        f"{PROFILE_REPORT_PATH}"
    )

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

def main() -> None:
    """
    Run the complete profiling process.
    """

    df = load_dataset()

    generate_reports(df)

    print_summary(df)


if __name__ == "__main__":
    main()