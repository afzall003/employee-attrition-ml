from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr


# ============================================================
# EMPLOYEE ATTRITION — DATA / SIGNAL DIAGNOSIS
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
    / "diagnosis"
)

TARGET = "Attrition"
ID_COLUMN = "Employee_ID"

RANDOM_STATE = 42


def create_output_directory() -> None:
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Dataset is empty.")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    return df


def prepare_target(df: pd.DataFrame) -> pd.Series:
    target = df[TARGET].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    if target.isna().any():
        raise ValueError(
            f"Unexpected target values: "
            f"{df.loc[target.isna(), TARGET].unique().tolist()}"
        )

    return target.astype(int)


def numerical_diagnosis(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    records = []

    features = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    features = [
        c
        for c in features
        if c not in {ID_COLUMN, TARGET}
    ]

    for feature in features:
        x = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        mask = x.notna() & target.notna()

        x_valid = x.loc[mask]
        y_valid = target.loc[mask]

        if x_valid.nunique() <= 1:
            correlation = 0.0
            p_value = 1.0
        else:
            result = pointbiserialr(
                y_valid,
                x_valid,
            )
            correlation = float(result.statistic)
            p_value = float(result.pvalue)

        stayed = x_valid.loc[y_valid == 0]
        left = x_valid.loc[y_valid == 1]

        pooled_std = np.sqrt(
            (
                (len(stayed) - 1) * stayed.var()
                + (len(left) - 1) * left.var()
            )
            / max(
                len(stayed) + len(left) - 2,
                1,
            )
        )

        if pooled_std and np.isfinite(pooled_std):
            standardized_difference = (
                (left.mean() - stayed.mean())
                / pooled_std
            )
        else:
            standardized_difference = 0.0

        records.append(
            {
                "feature": feature,
                "stayed_mean": float(stayed.mean()),
                "attrition_mean": float(left.mean()),
                "mean_difference": float(
                    left.mean() - stayed.mean()
                ),
                "standardized_mean_difference": float(
                    standardized_difference
                ),
                "point_biserial_correlation": correlation,
                "absolute_correlation": abs(correlation),
                "p_value": p_value,
                "unique_values": int(x_valid.nunique()),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def categorical_diagnosis(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    records = []

    features = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    features = [
        c for c in features
        if c != TARGET
    ]

    for feature in features:
        working = pd.DataFrame(
            {
                "feature_value": df[feature],
                "target": target,
            }
        ).dropna()

        table = pd.crosstab(
            working["feature_value"],
            working["target"],
        )

        if table.shape[0] >= 2 and table.shape[1] >= 2:
            chi2, p_value, _, _ = chi2_contingency(table)

            n = table.to_numpy().sum()
            phi2 = chi2 / n if n else 0.0

            rows, columns = table.shape

            correction = (
                (columns - 1)
                * (rows - 1)
                / max(n - 1, 1)
            )

            corrected_phi2 = max(
                0.0,
                phi2 - correction,
            )

            corrected_rows = (
                rows
                - ((rows - 1) ** 2 / max(n - 1, 1))
            )

            corrected_columns = (
                columns
                - ((columns - 1) ** 2 / max(n - 1, 1))
            )

            denominator = min(
                corrected_rows - 1,
                corrected_columns - 1,
            )

            cramers_v = (
                np.sqrt(
                    corrected_phi2 / denominator
                )
                if denominator > 0
                else 0.0
            )
        else:
            p_value = 1.0
            cramers_v = 0.0

        overall_rate = target.mean() * 100

        category_rates = (
            working.groupby("feature_value")["target"]
            .agg(["count", "mean"])
            .reset_index()
        )

        category_rates["rate_percentage"] = (
            category_rates["mean"] * 100
        )

        rate_range = (
            category_rates["rate_percentage"].max()
            - category_rates["rate_percentage"].min()
        )

        records.append(
            {
                "feature": feature,
                "categories": int(
                    working["feature_value"].nunique()
                ),
                "overall_attrition_rate": float(
                    overall_rate
                ),
                "lowest_category_rate": float(
                    category_rates["rate_percentage"].min()
                ),
                "highest_category_rate": float(
                    category_rates["rate_percentage"].max()
                ),
                "category_rate_range": float(rate_range),
                "cramers_v": float(cramers_v),
                "p_value": float(p_value),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["cramers_v", "category_rate_range"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


def tenure_consistency(df: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}

    if {
        "Years_at_Company",
        "Years_in_Current_Role",
    }.issubset(df.columns):
        invalid_role = (
            df["Years_in_Current_Role"]
            > df["Years_at_Company"]
        )

        result["current_role_gt_company"] = int(
            invalid_role.sum()
        )

        result["current_role_gt_company_rate"] = float(
            invalid_role.mean() * 100
        )

    if {
        "Years_at_Company",
        "Years_Since_Last_Promotion",
    }.issubset(df.columns):
        invalid_promotion = (
            df["Years_Since_Last_Promotion"]
            > df["Years_at_Company"]
        )

        result["promotion_gt_company"] = int(
            invalid_promotion.sum()
        )

        result["promotion_gt_company_rate"] = float(
            invalid_promotion.mean() * 100
        )

    return result


def target_feature_overlap(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Compare target prevalence across simple business-relevant
    feature bands. This is diagnostic only and does not claim
    causal relationships.
    """

    numeric_features = [
        "Age",
        "Monthly_Income",
        "Years_at_Company",
        "Years_in_Current_Role",
        "Years_Since_Last_Promotion",
        "Training_Hours_Last_Year",
        "Average_Hours_Worked_Per_Week",
        "Absenteeism",
        "Distance_From_Home",
        "Number_of_Companies_Worked",
    ]

    records = []

    for feature in numeric_features:
        if feature not in df.columns:
            continue

        series = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        if series.nunique() < 4:
            continue

        try:
            bands = pd.qcut(
                series,
                q=4,
                duplicates="drop",
            )
        except ValueError:
            continue

        working = pd.DataFrame(
            {
                "band": bands,
                "target": target,
            }
        ).dropna()

        grouped = (
            working.groupby(
                "band",
                observed=True,
            )["target"]
            .agg(
                count="count",
                attrition_rate="mean",
            )
            .reset_index()
        )

        if grouped.empty:
            continue

        rate_range = (
            grouped["attrition_rate"].max()
            - grouped["attrition_rate"].min()
        ) * 100

        records.append(
            {
                "feature": feature,
                "quartile_rate_range_percentage_points": float(
                    rate_range
                ),
                "lowest_quartile_rate": float(
                    grouped["attrition_rate"].min() * 100
                ),
                "highest_quartile_rate": float(
                    grouped["attrition_rate"].max() * 100
                ),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            "quartile_rate_range_percentage_points",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def build_summary(
    df: pd.DataFrame,
    target: pd.Series,
    numerical: pd.DataFrame,
    categorical: pd.DataFrame,
    overlap: pd.DataFrame,
    tenure: dict[str, object],
) -> dict[str, object]:

    target_rate = float(target.mean())

    strongest_numeric = (
        numerical.iloc[0]["feature"]
        if not numerical.empty
        else None
    )

    strongest_categorical = (
        categorical.iloc[0]["feature"]
        if not categorical.empty
        else None
    )

    strongest_overlap = (
        overlap.iloc[0]["feature"]
        if not overlap.empty
        else None
    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target_rate": target_rate,
        "target_count_yes": int(target.sum()),
        "target_count_no": int((target == 0).sum()),
        "strongest_numeric_feature": strongest_numeric,
        "strongest_numeric_absolute_correlation": (
            float(numerical.iloc[0]["absolute_correlation"])
            if not numerical.empty
            else 0.0
        ),
        "strongest_categorical_feature": strongest_categorical,
        "strongest_categorical_cramers_v": (
            float(categorical.iloc[0]["cramers_v"])
            if not categorical.empty
            else 0.0
        ),
        "strongest_quartile_feature": strongest_overlap,
        "tenure_consistency": tenure,
        "diagnostic_note": (
            "These diagnostics identify potential signal and "
            "data-generation issues. They do not establish "
            "causality or production suitability."
        ),
    }


def print_report(
    summary: dict[str, object],
    numerical: pd.DataFrame,
    categorical: pd.DataFrame,
    overlap: pd.DataFrame,
) -> None:

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION — DATA / SIGNAL DIAGNOSIS")
    print("=" * 60)

    print("\n[DATASET]")
    print(f"Rows:                 {summary['rows']}")
    print(f"Columns:              {summary['columns']}")
    print(
        f"Attrition:            "
        f"{summary['target_count_yes']} Yes / "
        f"{summary['target_count_no']} No"
    )
    print(
        f"Target prevalence:    "
        f"{summary['target_rate'] * 100:.1f}%"
    )

    print("\n[NUMERICAL SIGNAL]")
    print(
        numerical[
            [
                "feature",
                "stayed_mean",
                "attrition_mean",
                "standardized_mean_difference",
                "point_biserial_correlation",
                "p_value",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[CATEGORICAL SIGNAL]")
    print(
        categorical[
            [
                "feature",
                "categories",
                "lowest_category_rate",
                "highest_category_rate",
                "category_rate_range",
                "cramers_v",
                "p_value",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[QUARTILE TARGET OVERLAP]")
    print(
        overlap[
            [
                "feature",
                "lowest_quartile_rate",
                "highest_quartile_rate",
                "quartile_rate_range_percentage_points",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[TENURE CONSISTENCY]")
    for key, value in summary["tenure_consistency"].items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    print("\n[DIAGNOSTIC INTERPRETATION]")
    print(
        "This stage is intended to determine whether the weak "
        "model performance comes from weak feature-target "
        "relationships, feature overlap, or possible data "
        "construction issues."
    )

    print(
        "\nStrongest numerical feature: "
        f"{summary['strongest_numeric_feature']} "
        f"(absolute r = "
        f"{summary['strongest_numeric_absolute_correlation']:.4f})"
    )

    print(
        "Strongest categorical feature: "
        f"{summary['strongest_categorical_feature']} "
        f"(Cramer's V = "
        f"{summary['strongest_categorical_cramers_v']:.4f})"
    )

    print(
        "Largest quartile attrition-rate range: "
        f"{summary['strongest_quartile_feature']}"
    )

    print("\n[OUTPUT]")
    print(f"Reports:              {REPORT_DIR}")

    print("\n" + "=" * 60)
    print("DATA / SIGNAL DIAGNOSIS COMPLETE")
    print("=" * 60)


def main() -> None:
    create_output_directory()

    df = load_data()
    target = prepare_target(df)

    print("\nRunning numerical target diagnostics...")
    numerical = numerical_diagnosis(
        df,
        target,
    )

    print("Running categorical target diagnostics...")
    categorical = categorical_diagnosis(
        df,
        target,
    )

    print("Running feature-band target overlap diagnostics...")
    overlap = target_feature_overlap(
        df,
        target,
    )

    print("Running tenure consistency diagnostics...")
    tenure = tenure_consistency(df)

    summary = build_summary(
        df,
        target,
        numerical,
        categorical,
        overlap,
        tenure,
    )

    numerical.to_csv(
        REPORT_DIR / "numerical_diagnosis.csv",
        index=False,
    )

    categorical.to_csv(
        REPORT_DIR / "categorical_diagnosis.csv",
        index=False,
    )

    overlap.to_csv(
        REPORT_DIR / "target_overlap_diagnosis.csv",
        index=False,
    )

    with open(
        REPORT_DIR / "diagnosis_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print_report(
        summary,
        numerical,
        categorical,
        overlap,
    )


if __name__ == "__main__":
    main()
