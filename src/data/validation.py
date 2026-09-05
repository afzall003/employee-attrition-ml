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


# ============================================================
# EXPECTED DATASET CONTRACT
# ============================================================

EXPECTED_COLUMNS = [
    "Employee_ID",
    "Age",
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Job_Level",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Training_Hours_Last_Year",
    "Overtime",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Distance_From_Home",
    "Number_of_Companies_Worked",
    "Attrition",
]


EXPECTED_CATEGORICAL_VALUES = {
    "Gender": {
        "Female",
        "Male",
    },
    "Marital_Status": {
        "Divorced",
        "Married",
        "Single",
    },
    "Department": {
        "Finance",
        "HR",
        "IT",
        "Marketing",
        "Sales",
    },
    "Job_Role": {
        "Analyst",
        "Assistant",
        "Executive",
        "Manager",
    },
    "Overtime": {
        "No",
        "Yes",
    },
    "Attrition": {
        "No",
        "Yes",
    },
}


RATING_COLUMNS = [
    "Job_Satisfaction",
    "Performance_Rating",
]


NON_NEGATIVE_COLUMNS = [
    "Age",
    "Job_Level",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Training_Hours_Last_Year",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Distance_From_Home",
    "Number_of_Companies_Worked",
]


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
# COLUMN AUDIT
# ============================================================

def audit_columns(df: pd.DataFrame) -> dict:
    """
    Verify that the supplied dataset contains the expected columns.
    """

    actual_columns = df.columns.tolist()

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    return {
        "expected_column_count": len(EXPECTED_COLUMNS),
        "actual_column_count": len(actual_columns),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "column_order_matches": (
            actual_columns == EXPECTED_COLUMNS
        ),
    }


# ============================================================
# STRUCTURE AUDIT
# ============================================================

def audit_structure(df: pd.DataFrame) -> dict:
    """
    Audit basic dataset structure and integrity.
    """

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "missing_values": int(
            df.isna().sum().sum()
        ),
    }


# ============================================================
# TARGET AUDIT
# ============================================================

def audit_target(
    df: pd.DataFrame,
    target: str = "Attrition",
) -> dict:
    """
    Audit the target variable.
    """

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    value_counts = df[target].value_counts()

    unexpected_values = sorted(
        set(df[target].dropna().unique())
        - EXPECTED_CATEGORICAL_VALUES[target]
    )

    return {
        "target": target,
        "unique_values": (
            df[target]
            .dropna()
            .unique()
            .tolist()
        ),
        "class_counts": value_counts.to_dict(),
        "class_percentages": (
            df[target]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
            .to_dict()
        ),
        "unexpected_values": unexpected_values,
    }


# ============================================================
# EMPLOYEE ID AUDIT
# ============================================================

def audit_employee_id(
    df: pd.DataFrame,
    employee_id: str = "Employee_ID",
) -> dict:
    """
    Check employee ID integrity.
    """

    if employee_id not in df.columns:
        return {
            "employee_id_column_found": False,
            "duplicate_employee_ids": None,
            "missing_employee_ids": None,
        }

    return {
        "employee_id_column_found": True,
        "duplicate_employee_ids": int(
            df[employee_id].duplicated().sum()
        ),
        "missing_employee_ids": int(
            df[employee_id].isna().sum()
        ),
    }


# ============================================================
# DATA TYPE AUDIT
# ============================================================

def audit_data_types(
    df: pd.DataFrame,
) -> dict:
    """
    Return the data type for every column.
    """

    return {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }


# ============================================================
# NUMERICAL RANGE AUDIT
# ============================================================

def audit_numerical_ranges(
    df: pd.DataFrame,
) -> dict:
    """
    Check for negative values and rating violations.

    These checks identify potential data-quality issues.
    They do not modify the dataset.
    """

    checks = {}

    for column in NON_NEGATIVE_COLUMNS:
        if column in df.columns:
            checks[
                f"negative_{column}"
            ] = int(
                (df[column] < 0).sum()
            )

    for column in RATING_COLUMNS:
        if column in df.columns:
            checks[
                f"{column}_outside_1_to_5"
            ] = int(
                (
                    (df[column] < 1)
                    | (df[column] > 5)
                ).sum()
            )

    return checks


# ============================================================
# CATEGORICAL VALUE AUDIT
# ============================================================

def audit_categorical_values(
    df: pd.DataFrame,
) -> dict:
    """
    Validate categorical values against the documented domains.
    """

    results = {}

    for column, expected_values in (
        EXPECTED_CATEGORICAL_VALUES.items()
    ):
        if column not in df.columns:
            continue

        observed_values = set(
            df[column]
            .dropna()
            .unique()
        )

        unexpected_values = sorted(
            observed_values - expected_values
        )

        results[column] = {
            "observed_values": sorted(
                observed_values
            ),
            "unexpected_values": (
                unexpected_values
            ),
        }

    return results


# ============================================================
# TENURE RELATIONSHIP AUDIT
# ============================================================

def audit_tenure_relationships(
    df: pd.DataFrame,
) -> dict:
    """
    Profile relationships between tenure-related fields.

    These are investigation indicators, NOT automatic data errors.

    The purpose is to understand the supplied data before deciding
    whether any transformation or treatment is appropriate.
    """

    return {
        "role_years_greater_than_company_years": int(
            (
                df["Years_in_Current_Role"]
                > df["Years_at_Company"]
            ).sum()
        ),
        "promotion_gap_greater_than_company_years": int(
            (
                df["Years_Since_Last_Promotion"]
                > df["Years_at_Company"]
            ).sum()
        ),
    }


# ============================================================
# COMPLETE AUDIT
# ============================================================

def run_audit() -> None:
    """
    Run the complete data-quality audit.
    """

    df = load_dataset()

    columns = audit_columns(df)
    structure = audit_structure(df)
    target = audit_target(df)
    employee_id = audit_employee_id(df)
    data_types = audit_data_types(df)
    numerical_checks = audit_numerical_ranges(df)
    categorical_checks = audit_categorical_values(df)
    tenure_checks = audit_tenure_relationships(df)

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION — DATA AUDIT")
    print("=" * 60)

    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    print("\n[COLUMN CONTRACT]")

    print(
        f"Expected columns:     "
        f"{columns['expected_column_count']}"
    )

    print(
        f"Actual columns:       "
        f"{columns['actual_column_count']}"
    )

    print(
        f"Missing columns:      "
        f"{columns['missing_columns']}"
    )

    print(
        f"Unexpected columns:   "
        f"{columns['unexpected_columns']}"
    )

    print(
        f"Column order matches: "
        f"{columns['column_order_matches']}"
    )

    # --------------------------------------------------------
    # STRUCTURE
    # --------------------------------------------------------

    print("\n[STRUCTURE]")

    print(
        f"Rows:             {structure['rows']}"
    )

    print(
        f"Columns:          {structure['columns']}"
    )

    print(
        f"Missing values:   {structure['missing_values']}"
    )

    print(
        f"Duplicate rows:   {structure['duplicate_rows']}"
    )

    # --------------------------------------------------------
    # EMPLOYEE ID
    # --------------------------------------------------------

    print("\n[EMPLOYEE ID]")

    print(
        f"Column found:     "
        f"{employee_id['employee_id_column_found']}"
    )

    print(
        f"Duplicate IDs:    "
        f"{employee_id['duplicate_employee_ids']}"
    )

    print(
        f"Missing IDs:      "
        f"{employee_id['missing_employee_ids']}"
    )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    print("\n[TARGET]")

    print(
        f"Target:           "
        f"{target['target']}"
    )

    print(
        f"Values:           "
        f"{target['unique_values']}"
    )

    print(
        f"Class counts:     "
        f"{target['class_counts']}"
    )

    print(
        f"Class percentages:"
        f"{target['class_percentages']}"
    )

    print(
        f"Unexpected values:"
        f"{target['unexpected_values']}"
    )

    # --------------------------------------------------------
    # DATA TYPES
    # --------------------------------------------------------

    print("\n[DATA TYPES]")

    for column, dtype in data_types.items():
        print(
            f"{column:<40} {dtype}"
        )

    # --------------------------------------------------------
    # NUMERICAL CHECKS
    # --------------------------------------------------------

    print("\n[NUMERICAL RANGE CHECKS]")

    for check, count in numerical_checks.items():
        print(
            f"{check:<55} {count}"
        )

    # --------------------------------------------------------
    # CATEGORICAL CHECKS
    # --------------------------------------------------------

    print("\n[CATEGORICAL VALUE CHECKS]")

    for column, result in (
        categorical_checks.items()
    ):
        print(
            f"{column:<30} "
            f"Unexpected: "
            f"{result['unexpected_values']}"
        )

    # --------------------------------------------------------
    # TENURE INVESTIGATION
    # --------------------------------------------------------

    print("\n[TENURE RELATIONSHIP INVESTIGATION]")

    print(
        "Years in current role > years at company: "
        f"{tenure_checks['role_years_greater_than_company_years']}"
    )

    print(
        "Years since last promotion > years at company: "
        f"{tenure_checks['promotion_gap_greater_than_company_years']}"
    )

    print(
        "Note: These are investigation indicators, "
        "not automatic data-quality errors."
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_audit()