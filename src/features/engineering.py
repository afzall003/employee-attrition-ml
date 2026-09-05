from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# COLUMN DEFINITIONS
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

NUMERICAL_COLUMNS = [
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
# EXPECTED COLUMNS
# ============================================================

EXPECTED_FEATURE_COLUMNS = (
    CATEGORICAL_COLUMNS
    + NUMERICAL_COLUMNS
)


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_data(
    path: Path = DATA_PATH,
) -> pd.DataFrame:
    """
    Load the raw employee attrition dataset.

    The raw CSV is kept untouched. All feature preparation
    happens downstream through the preprocessing pipeline.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    return df


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that all required columns are present.

    Employee_ID and Attrition are intentionally not included
    in the model feature matrix.
    """

    required_columns = (
        EXPECTED_FEATURE_COLUMNS
        + IDENTIFIER_COLUMNS
        + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# TARGET PREPARATION
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Convert Attrition from Yes/No into a binary target.

    No  -> 0
    Yes -> 1
    """

    target_mapping = {
        "No": 0,
        "Yes": 1,
    }

    y = df[TARGET_COLUMN].map(
        target_mapping
    )

    if y.isna().any():
        unexpected_values = (
            df.loc[
                y.isna(),
                TARGET_COLUMN,
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected Attrition values: "
            f"{unexpected_values}"
        )

    return y.astype(int)


# ============================================================
# FEATURE MATRIX PREPARATION
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the raw model feature matrix.

    Employee_ID is deliberately excluded because it is an
    identifier rather than a meaningful predictive feature.

    Attrition is also excluded because it is the target.
    """

    validate_input_columns(df)

    X = df[
        EXPECTED_FEATURE_COLUMNS
    ].copy()

    return X


# ============================================================
# NUMERICAL PIPELINE
# ============================================================

def build_numerical_pipeline() -> Pipeline:
    """
    Build preprocessing steps for numerical features.

    Steps:
        1. Median imputation
        2. Standard scaling

    Although the current dataset contains no missing values,
    imputation is included to make the pipeline robust to
    future client datasets.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )


# ============================================================
# CATEGORICAL PIPELINE
# ============================================================

def build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing steps for categorical features.

    Steps:
        1. Most-frequent imputation
        2. One-hot encoding

    handle_unknown="ignore" ensures that a previously unseen
    category in production does not crash prediction.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


# ============================================================
# COMPLETE PREPROCESSOR
# ============================================================

def build_preprocessor() -> ColumnTransformer:
    """
    Build the complete feature preprocessing pipeline.

    Numerical features:
        Imputation -> StandardScaler

    Categorical features:
        Imputation -> OneHotEncoder
    """

    numerical_pipeline = (
        build_numerical_pipeline()
    )

    categorical_pipeline = (
        build_categorical_pipeline()
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_COLUMNS,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# FEATURE NAME EXTRACTION
# ============================================================

def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
) -> list[str]:
    """
    Return the names of features produced by the fitted
    preprocessing pipeline.

    The preprocessor must be fitted before calling this
    function.
    """

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    return feature_names.tolist()


# ============================================================
# PREPROCESS DATA
# ============================================================

def fit_transform_features(
    X: pd.DataFrame,
) -> tuple[ColumnTransformer, object]:
    """
    Fit the preprocessing pipeline and transform the data.

    IMPORTANT:
    This function should only be used on training data during
    model development.

    For validation/test data, use transform_features() with
    the already-fitted preprocessor.
    """

    preprocessor = build_preprocessor()

    X_transformed = (
        preprocessor.fit_transform(X)
    )

    return (
        preprocessor,
        X_transformed,
    )


def transform_features(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer,
) -> object:
    """
    Transform data using an already-fitted preprocessor.

    This prevents test/validation information from influencing
    the preprocessing learned from training data.
    """

    return preprocessor.transform(X)


# ============================================================
# DATASET PREPARATION
# ============================================================

def prepare_model_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for model development.

    Returns:
        X -> model features
        y -> binary target
    """

    X = prepare_features(df)

    y = prepare_target(df)

    return X, y


# ============================================================
# FEATURE SUMMARY
# ============================================================

def print_feature_summary(
    X: pd.DataFrame,
) -> None:
    """
    Print a concise summary of the raw model features.
    """

    numerical_count = len(
        [
            column
            for column in X.columns
            if column in NUMERICAL_COLUMNS
        ]
    )

    categorical_count = len(
        [
            column
            for column in X.columns
            if column in CATEGORICAL_COLUMNS
        ]
    )

    print("\n" + "=" * 60)
    print("EMPLOYEE ATTRITION — FEATURE ENGINEERING")
    print("=" * 60)

    print("\n[MODEL INPUT]")

    print(
        f"Total features:       {len(X.columns)}"
    )

    print(
        f"Numerical features:   {numerical_count}"
    )

    print(
        f"Categorical features: {categorical_count}"
    )

    print(
        f"Rows:                 {len(X)}"
    )

    print("\n[EXCLUDED FROM MODEL]")

    for column in (
        IDENTIFIER_COLUMNS
        + [TARGET_COLUMN]
    ):
        print(
            f"- {column}"
        )

    print("\n[NUMERICAL FEATURES]")

    for column in NUMERICAL_COLUMNS:
        print(
            f"- {column}"
        )

    print("\n[CATEGORICAL FEATURES]")

    for column in CATEGORICAL_COLUMNS:
        print(
            f"- {column}"
        )

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run a standalone feature-engineering sanity check.

    No model is trained here.
    """

    df = load_raw_data()

    X, y = prepare_model_data(df)

    print_feature_summary(X)

    print("\n[TARGET DISTRIBUTION]")

    print(
        y.value_counts()
        .sort_index()
        .rename(
            index={
                0: "Stayed",
                1: "Left",
            }
        )
        .to_string()
    )

    print("\n[PREPROCESSING CHECK]")

    preprocessor = build_preprocessor()

    X_transformed = (
        preprocessor.fit_transform(X)
    )

    feature_names = (
        get_transformed_feature_names(
            preprocessor
        )
    )

    print(
        f"Raw feature count: "
        f"{X.shape[1]}"
    )

    print(
        f"Transformed feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Transformed matrix shape: "
        f"{X_transformed.shape}"
    )

    print("\nFEATURE ENGINEERING CHECK COMPLETE")


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()