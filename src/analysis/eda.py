from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr


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
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

NUMERICAL_FIGURES_DIR = FIGURES_DIR / "numerical_distributions"
CATEGORICAL_FIGURES_DIR = FIGURES_DIR / "categorical_distributions"
ATTRITION_FIGURES_DIR = FIGURES_DIR / "attrition_comparisons"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

TARGET_COLUMN = "Attrition"

IDENTIFIER_COLUMNS = [
    "Employee_ID",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Overtime",
]

ORDINAL_COLUMNS = [
    "Job_Level",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
]

NUMERICAL_COLUMNS = [
    "Age",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Training_Hours_Last_Year",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Distance_From_Home",
    "Number_of_Companies_Worked",
]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw employee attrition dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required for EDA outputs."""

    directories = [
        REPORTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        NUMERICAL_FIGURES_DIR,
        CATEGORICAL_FIGURES_DIR,
        ATTRITION_FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(df: pd.DataFrame) -> pd.Series:
    """
    Convert Attrition into a binary representation.

    No  -> 0
    Yes -> 1
    """

    mapping = {
        "No": 0,
        "Yes": 1,
    }

    encoded = df[TARGET_COLUMN].map(mapping)

    if encoded.isna().any():
        unexpected = (
            df.loc[encoded.isna(), TARGET_COLUMN]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unexpected target values found: {unexpected}"
        )

    return encoded


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def analyze_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate target counts and percentages."""

    counts = df[TARGET_COLUMN].value_counts().rename("count")

    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .rename("percentage")
    )

    result = pd.concat([counts, percentages], axis=1)
    result.index.name = TARGET_COLUMN

    return result


def plot_target_distribution(df: pd.DataFrame) -> None:
    """Generate target distribution chart."""

    counts = (
        df[TARGET_COLUMN]
        .value_counts()
        .reindex(["No", "Yes"])
    )

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        counts.index,
        counts.values,
    )

    plt.title("Employee Attrition Distribution")
    plt.xlabel("Attrition")
    plt.ylabel("Number of Employees")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, value in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def build_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate descriptive statistics for numerical features."""

    summary = df[NUMERICAL_COLUMNS].describe().T

    summary["median"] = df[NUMERICAL_COLUMNS].median()
    summary["missing"] = df[NUMERICAL_COLUMNS].isna().sum()
    summary["unique"] = df[NUMERICAL_COLUMNS].nunique()

    return summary


# ============================================================
# NUMERICAL DISTRIBUTION PLOTS
# ============================================================

def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """Generate individual distribution plots for numerical features."""

    for column in NUMERICAL_COLUMNS:
        plt.figure(figsize=(8, 6))

        plt.hist(
            df[column],
            bins=20,
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        output_path = (
            NUMERICAL_FIGURES_DIR
            / f"{column.lower()}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()


# ============================================================
# CATEGORICAL SUMMARY
# ============================================================

def build_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate counts and percentages for categorical variables."""

    rows = []

    for column in CATEGORICAL_COLUMNS:
        counts = df[column].value_counts(dropna=False)
        total = len(df)

        for value, count in counts.items():
            rows.append(
                {
                    "feature": column,
                    "value": value,
                    "count": int(count),
                    "percentage": round(
                        count / total * 100,
                        2,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CATEGORICAL DISTRIBUTION PLOTS
# ============================================================

def plot_categorical_distributions(df: pd.DataFrame) -> None:
    """Generate bar charts for categorical features."""

    for column in CATEGORICAL_COLUMNS:
        counts = (
            df[column]
            .value_counts()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 6))

        bars = plt.bar(
            counts.index.astype(str),
            counts.values,
        )

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Number of Employees")

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        for bar, value in zip(bars, counts.values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        output_path = (
            CATEGORICAL_FIGURES_DIR
            / f"{column.lower()}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()


# ============================================================
# CATEGORICAL ATTRITION ANALYSIS
# ============================================================

def analyze_categorical_attrition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate attrition rates for categorical features
    and perform chi-square tests.
    """

    rows = []

    for column in CATEGORICAL_COLUMNS:
        grouped = (
            df.groupby(
                column,
                dropna=False,
            )[TARGET_COLUMN]
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
        )

        contingency_table = pd.crosstab(
            df[column],
            df[TARGET_COLUMN],
        )

        chi2, p_value, degrees_of_freedom, _ = chi2_contingency(
            contingency_table
        )

        sample_size = len(df)
        min_dimension = min(contingency_table.shape)

        cramers_v = np.sqrt(
            (
                chi2
                / sample_size
            )
            / (
                min_dimension - 1
            )
        )

        for category, row in grouped.iterrows():
            rows.append(
                {
                    "feature": column,
                    "category": category,
                    "total_employees": int(
                        row["total_employees"]
                    ),
                    "attrition_count": int(
                        row["attrition_count"]
                    ),
                    "attrition_rate": round(
                        row["attrition_rate"],
                        2,
                    ),
                    "chi_square": round(
                        chi2,
                        4,
                    ),
                    "p_value": round(
                        p_value,
                        6,
                    ),
                    "degrees_of_freedom": (
                        degrees_of_freedom
                    ),
                    "cramers_v": round(
                        cramers_v,
                        4,
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# CATEGORICAL ATTRITION PLOTS
# ============================================================

def plot_categorical_attrition(df: pd.DataFrame) -> None:
    """Generate attrition-rate charts for categorical features."""

    for column in CATEGORICAL_COLUMNS:
        grouped = (
            df.groupby(
                column,
                dropna=False,
            )[TARGET_COLUMN]
            .apply(
                lambda x: (
                    x == "Yes"
                ).mean()
                * 100
            )
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(9, 6))

        bars = plt.bar(
            grouped.index.astype(str),
            grouped.values,
        )

        plt.title(f"Attrition Rate by {column}")
        plt.xlabel(column)
        plt.ylabel("Attrition Rate (%)")

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        for bar, value in zip(
            bars,
            grouped.values,
        ):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        output_path = (
            ATTRITION_FIGURES_DIR
            / f"{column.lower()}_attrition.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()


# ============================================================
# NUMERICAL ATTRITION ANALYSIS
# ============================================================

def analyze_numerical_attrition(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare numerical features between employees who stayed
    and employees who left.

    Mann-Whitney U is used because it does not require
    normally distributed numerical features.
    """

    rows = []

    target_encoded = encode_target(df)

    for column in NUMERICAL_COLUMNS:
        stayed = df.loc[
            df[TARGET_COLUMN] == "No",
            column,
        ]

        left = df.loc[
            df[TARGET_COLUMN] == "Yes",
            column,
        ]

        statistic, p_value = mannwhitneyu(
            stayed,
            left,
            alternative="two-sided",
        )

        correlation, correlation_p_value = pointbiserialr(
            target_encoded,
            df[column],
        )

        rows.append(
            {
                "feature": column,
                "stayed_mean": round(
                    stayed.mean(),
                    4,
                ),
                "left_mean": round(
                    left.mean(),
                    4,
                ),
                "stayed_median": round(
                    stayed.median(),
                    4,
                ),
                "left_median": round(
                    left.median(),
                    4,
                ),
                "mann_whitney_u": round(
                    statistic,
                    4,
                ),
                "p_value": round(
                    p_value,
                    6,
                ),
                "point_biserial_correlation": round(
                    correlation,
                    4,
                ),
                "correlation_p_value": round(
                    correlation_p_value,
                    6,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("p_value")
    )


# ============================================================
# NUMERICAL ATTRITION PLOTS
# ============================================================

def plot_numerical_attrition(df: pd.DataFrame) -> None:
    """Generate boxplots comparing numerical features across attrition classes."""

    for column in NUMERICAL_COLUMNS:
        stayed = df.loc[
            df[TARGET_COLUMN] == "No",
            column,
        ]

        left = df.loc[
            df[TARGET_COLUMN] == "Yes",
            column,
        ]

        plt.figure(figsize=(8, 6))

        plt.boxplot(
            [
                stayed,
                left,
            ],
            tick_labels=[
                "Stayed",
                "Left",
            ],
        )

        plt.title(f"{column} by Attrition")
        plt.xlabel("Attrition")
        plt.ylabel(column)

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        output_path = (
            ATTRITION_FIGURES_DIR
            / f"{column.lower()}_attrition.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

def calculate_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate point-biserial correlations between numerical
    features and binary attrition target.

    This is exploratory only and does not imply causation.
    """

    target_encoded = encode_target(df)

    rows = []

    for column in NUMERICAL_COLUMNS:
        correlation, p_value = pointbiserialr(
            target_encoded,
            df[column],
        )

        rows.append(
            {
                "feature": column,
                "correlation_with_attrition": round(
                    correlation,
                    4,
                ),
                "p_value": round(
                    p_value,
                    6,
                ),
                "absolute_correlation": round(
                    abs(correlation),
                    4,
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
    )


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate and save all statistical reports."""

    target_profile = analyze_target_distribution(df)

    numerical_summary = build_numerical_summary(df)

    categorical_summary = build_categorical_summary(df)

    categorical_tests = analyze_categorical_attrition(df)

    numerical_tests = analyze_numerical_attrition(df)

    correlations = calculate_correlations(df)

    target_profile.to_csv(
        METRICS_DIR / "eda_target_distribution.csv"
    )

    numerical_summary.to_csv(
        METRICS_DIR / "eda_numerical_summary.csv"
    )

    categorical_summary.to_csv(
        METRICS_DIR / "eda_categorical_summary.csv",
        index=False,
    )

    categorical_tests.to_csv(
        METRICS_DIR / "categorical_tests.csv",
        index=False,
    )

    numerical_tests.to_csv(
        METRICS_DIR / "numerical_tests.csv",
        index=False,
    )

    correlations.to_csv(
        METRICS_DIR / "correlations.csv",
        index=False,
    )

    return {
        "target_profile": target_profile,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary,
        "categorical_tests": categorical_tests,
        "numerical_tests": numerical_tests,
        "correlations": correlations,
    }


# ============================================================
# TERMINAL SUMMARY
# ============================================================

def print_analysis_summary(
    reports: dict[str, pd.DataFrame],
) -> None:
    """Print the most important EDA findings to the terminal."""

    categorical_tests = reports["categorical_tests"]
    numerical_tests = reports["numerical_tests"]
    correlations = reports["correlations"]

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION — EDA & STATISTICAL ANALYSIS")
    print("=" * 60)

    print("\n[TARGET]")

    print(
        reports["target_profile"].to_string()
    )

    print("\n[TOP CATEGORICAL ASSOCIATIONS]")

    categorical_display = (
        categorical_tests[
            [
                "feature",
                "category",
                "attrition_rate",
                "p_value",
                "cramers_v",
            ]
        ]
        .drop_duplicates(
            subset=["feature"]
        )
        .sort_values("p_value")
    )

    print(
        categorical_display.to_string(
            index=False
        )
    )

    print("\n[TOP NUMERICAL ASSOCIATIONS]")

    print(
        numerical_tests[
            [
                "feature",
                "stayed_mean",
                "left_mean",
                "p_value",
                "point_biserial_correlation",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n[TOP CORRELATIONS WITH ATTRITION]")

    print(
        correlations[
            [
                "feature",
                "correlation_with_attrition",
                "p_value",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\n[OUTPUT]")

    print(f"Figures: {FIGURES_DIR}")
    print(f"Metrics: {METRICS_DIR}")

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """Execute the complete EDA pipeline."""

    df = load_dataset()

    create_output_directories()

    plot_target_distribution(df)

    plot_numerical_distributions(df)

    plot_categorical_distributions(df)

    plot_categorical_attrition(df)

    plot_numerical_attrition(df)

    reports = save_reports(df)

    print_analysis_summary(reports)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()