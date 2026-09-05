from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.features.engineering import (
    build_preprocessor,
    prepare_model_data,
)


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

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

METRICS_DIR = REPORTS_DIR / "metrics"

FIGURES_DIR = REPORTS_DIR / "figures"

SIGNAL_REPORT_DIR = REPORTS_DIR / "signal_analysis"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_SPLITS = 5

CV_REPEATS = 3

PERMUTATION_REPEATS = 20

TARGET_COLUMN = "Attrition"

ID_COLUMN = "Employee_ID"


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_output_directories() -> None:
    """Create directories required for signal analysis."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SIGNAL_REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the employee attrition dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    return df


# ============================================================
# TARGET PREPARATION
# ============================================================

def prepare_target(
    df: pd.DataFrame,
) -> pd.Series:
    """Convert Attrition Yes/No into binary target values."""

    target = (
        df[TARGET_COLUMN]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    if target.isna().any():
        raise ValueError(
            "Unexpected values found in "
            f"{TARGET_COLUMN}."
        )

    return target.astype(int)


# ============================================================
# FEATURE GROUPS
# ============================================================

def get_numerical_features(
    df: pd.DataFrame,
) -> list[str]:
    """Return numerical predictor columns excluding ID and target."""

    numerical_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_columns = [
        column
        for column in numerical_columns
        if column != ID_COLUMN
        and column != TARGET_COLUMN
    ]

    return numerical_columns


def get_categorical_features(
    df: pd.DataFrame,
) -> list[str]:
    """Return categorical predictor columns."""

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in categorical_columns
        if column != TARGET_COLUMN
    ]

    return categorical_columns


# ============================================================
# NUMERICAL SIGNAL ANALYSIS
# ============================================================

def analyze_numerical_features(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Measure numerical feature relationships with attrition.

    Point-biserial correlation is used because the target is
    binary while the predictors are numerical.
    """

    records = []

    for feature in get_numerical_features(df):

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        valid_mask = (
            values.notna()
            & target.notna()
        )

        x = values.loc[valid_mask]
        y = target.loc[valid_mask]

        if x.nunique() <= 1:
            correlation = 0.0
            p_value = 1.0

        else:
            result = pointbiserialr(
                y,
                x,
            )

            correlation = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        stayed_mean = float(
            x[y == 0].mean()
        )

        attrition_mean = float(
            x[y == 1].mean()
        )

        records.append(
            {
                "feature": feature,
                "stayed_mean": stayed_mean,
                "attrition_mean": attrition_mean,
                "mean_difference": (
                    attrition_mean
                    - stayed_mean
                ),
                "point_biserial_correlation": (
                    correlation
                ),
                "absolute_correlation": abs(
                    correlation
                ),
                "p_value": p_value,
            }
        )

    result_df = pd.DataFrame(
        records
    )

    result_df = result_df.sort_values(
        by="absolute_correlation",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# CRAMER'S V
# ============================================================

def calculate_cramers_v(
    table: pd.DataFrame,
) -> tuple[float, float]:
    """
    Calculate Cramer's V and chi-square p-value
    for a categorical feature versus attrition.
    """

    if (
        table.shape[0] < 2
        or table.shape[1] < 2
    ):
        return 0.0, 1.0

    chi2, p_value, _, _ = chi2_contingency(
        table
    )

    n = table.to_numpy().sum()

    if n == 0:
        return 0.0, 1.0

    phi2 = chi2 / n

    rows, columns = table.shape

    if n <= 1:
        return 0.0, float(p_value)

    corrected_phi2 = max(
        0,
        phi2
        - (
            (columns - 1)
            * (rows - 1)
            / (n - 1)
        ),
    )

    corrected_rows = (
        rows
        - (
            (rows - 1) ** 2
            / (n - 1)
        )
    )

    corrected_columns = (
        columns
        - (
            (columns - 1) ** 2
            / (n - 1)
        )
    )

    denominator = min(
        corrected_columns - 1,
        corrected_rows - 1,
    )

    if denominator <= 0:
        return 0.0, float(p_value)

    cramers_v = np.sqrt(
        corrected_phi2
        / denominator
    )

    return (
        float(cramers_v),
        float(p_value),
    )


# ============================================================
# CATEGORICAL SIGNAL ANALYSIS
# ============================================================

def analyze_categorical_features(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """Measure categorical feature association with attrition."""

    records = []

    for feature in get_categorical_features(df):

        working = pd.DataFrame(
            {
                "feature": df[feature],
                "target": target,
            }
        ).dropna()

        contingency = pd.crosstab(
            working["feature"],
            working["target"],
        )

        cramers_v, p_value = (
            calculate_cramers_v(
                contingency
            )
        )

        for category in sorted(
            working["feature"].unique()
        ):

            category_mask = (
                working["feature"]
                == category
            )

            total = int(
                category_mask.sum()
            )

            attrition_count = int(
                (
                    working.loc[
                        category_mask,
                        "target",
                    ]
                    == 1
                ).sum()
            )

            attrition_rate = (
                attrition_count
                / total
                * 100
                if total > 0
                else 0.0
            )

            records.append(
                {
                    "feature": feature,
                    "category": category,
                    "total_employees": total,
                    "attrition_count": (
                        attrition_count
                    ),
                    "attrition_rate": (
                        attrition_rate
                    ),
                    "cramers_v": cramers_v,
                    "p_value": p_value,
                }
            )

    result_df = pd.DataFrame(
        records
    )

    result_df = result_df.sort_values(
        by=[
            "cramers_v",
            "attrition_rate",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def calculate_mutual_information(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Calculate mutual information between features
    and the binary attrition target.

    Mutual information can capture nonlinear dependencies
    that ordinary correlation may miss.
    """

    X = df.drop(
        columns=[
            TARGET_COLUMN,
            ID_COLUMN,
        ],
        errors="ignore",
    ).copy()

    y = target.copy()

    categorical_features = get_categorical_features(
        df
    )

    numerical_features = get_numerical_features(
        df
    )

    feature_names = (
        numerical_features
        + categorical_features
    )

    X_mi = X[
        feature_names
    ].copy()

    categorical_mask = []

    for feature in feature_names:

        if feature in categorical_features:
            X_mi[feature] = (
                X_mi[feature]
                .astype("category")
                .cat.codes
            )

            categorical_mask.append(
                True
            )

        else:
            X_mi[feature] = pd.to_numeric(
                X_mi[feature],
                errors="coerce",
            )

            categorical_mask.append(
                False
            )

    X_mi = X_mi.fillna(
        X_mi.median(
            numeric_only=True
        )
    )

    X_mi = X_mi.fillna(-999)

    mutual_information = (
        mutual_info_classif(
            X_mi,
            y,
            discrete_features=(
                categorical_mask
            ),
            random_state=RANDOM_STATE,
        )
    )

    result_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mutual_information": (
                mutual_information
            ),
        }
    )

    result_df = result_df.sort_values(
        by="mutual_information",
        ascending=False,
    ).reset_index(
        drop=True
    )

    return result_df


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_signal_model() -> Pipeline:
    """
    Build a Gradient Boosting pipeline for signal testing.

    This is not being selected as the final production model.
    It is used as a nonlinear signal detector.
    """

    from sklearn.ensemble import (
        GradientBoostingClassifier,
    )

    preprocessor = build_preprocessor()

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )


# ============================================================
# REPEATED CROSS-VALIDATION
# ============================================================

def repeated_cross_validation(
    df: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """
    Evaluate nonlinear predictive signal across
    repeated stratified folds.

    RepeatedStratifiedKFold cannot be used directly with
    cross_val_predict() because repeated CV evaluates each
    sample more than once.

    Each validation fold is therefore evaluated explicitly.
    """

    X, _ = prepare_model_data(
        df
    )

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )

    fold_roc_auc_scores = []
    fold_pr_auc_scores = []
    fold_positive_rates = []

    for train_indices, validation_indices in cv.split(
        X,
        target,
    ):

        X_train = X.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_train = target.iloc[
            train_indices
        ]

        y_validation = target.iloc[
            validation_indices
        ]

        model = build_signal_model()

        model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            model.predict_proba(
                X_validation
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        fold_roc_auc_scores.append(
            float(
                roc_auc_score(
                    y_validation,
                    probabilities,
                )
            )
        )

        fold_pr_auc_scores.append(
            float(
                average_precision_score(
                    y_validation,
                    probabilities,
                )
            )
        )

        fold_positive_rates.append(
            float(
                predictions.mean()
            )
        )

    return {
        "roc_auc": float(
            np.mean(
                fold_roc_auc_scores
            )
        ),
        "roc_auc_std": float(
            np.std(
                fold_roc_auc_scores,
                ddof=1,
            )
        ),
        "pr_auc": float(
            np.mean(
                fold_pr_auc_scores
            )
        ),
        "pr_auc_std": float(
            np.std(
                fold_pr_auc_scores,
                ddof=1,
            )
        ),
        "positive_rate": float(
            np.mean(
                fold_positive_rates
            )
        ),
        "positive_rate_std": float(
            np.std(
                fold_positive_rates,
                ddof=1,
            )
        ),
        "total_folds": int(
            len(
                fold_roc_auc_scores
            )
        ),
    }


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    df: pd.DataFrame,
    target: pd.Series,
) -> pd.DataFrame:
    """
    Calculate permutation importance on a held-out test set.

    Importance is calculated at the original feature level
    using the complete preprocessing pipeline.
    """

    X, y = prepare_model_data(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model = build_signal_model()

    model.fit(
        X_train,
        y_train,
    )

    baseline_auc = roc_auc_score(
        y_test,
        model.predict_proba(
            X_test
        )[:, 1],
    )

    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    feature_names = (
        X_test.columns.tolist()
    )

    result_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": (
                importance.importances_mean
            ),
            "importance_std": (
                importance.importances_std
            ),
        }
    )

    result_df[
        "absolute_importance"
    ] = result_df[
        "importance_mean"
    ].abs()

    result_df = result_df.sort_values(
        by="absolute_importance",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result_df.attrs[
        "baseline_auc"
    ] = float(
        baseline_auc
    )

    return result_df


# ============================================================
# SIGNAL INTERPRETATION
# ============================================================

def interpret_signal(
    repeated_cv_metrics: dict[str, float],
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    mutual_information_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    target: pd.Series,
) -> dict[str, object]:
    """
    Produce a conservative interpretation of the evidence.

    Target prevalence is calculated dynamically from the
    dataset rather than being hard-coded.
    """

    roc_auc = repeated_cv_metrics[
        "roc_auc"
    ]

    pr_auc = repeated_cv_metrics[
        "pr_auc"
    ]

    # Calculate actual target prevalence from the dataset.
    target_rate = float(
        target.mean()
    )

    numerical_max = (
        numerical_df[
            "absolute_correlation"
        ].max()
        if not numerical_df.empty
        else 0.0
    )

    categorical_max = (
        categorical_df[
            "cramers_v"
        ].max()
        if not categorical_df.empty
        else 0.0
    )

    mi_max = (
        mutual_information_df[
            "mutual_information"
        ].max()
        if not mutual_information_df.empty
        else 0.0
    )

    meaningful_auc_signal = (
        roc_auc >= 0.60
    )

    better_than_baseline = (
        pr_auc > target_rate
    )

    if meaningful_auc_signal:

        conclusion = (
            "Evidence of potentially useful "
            "predictive signal was detected. "
            "Proceed to controlled model "
            "optimization and validation."
        )

    elif better_than_baseline:

        conclusion = (
            "The model shows some ranking "
            "signal above the observed target "
            "prevalence, but the signal is not "
            "strong enough to call the model "
            "production-ready."
        )

    else:

        conclusion = (
            "No strong predictive signal was "
            "demonstrated. The available "
            "features appear insufficient for "
            "a reliable attrition prediction "
            "model without additional data, "
            "better labels, or further feature "
            "investigation."
        )

    return {
        "repeated_cv_roc_auc": roc_auc,
        "repeated_cv_pr_auc": pr_auc,
        "target_prevalence": target_rate,
        "maximum_numerical_absolute_correlation": (
            float(numerical_max)
        ),
        "maximum_categorical_cramers_v": (
            float(categorical_max)
        ),
        "maximum_mutual_information": (
            float(mi_max)
        ),
        "roc_auc_threshold_for_signal": 0.60,
        "meaningful_auc_signal_detected": (
            meaningful_auc_signal
        ),
        "pr_auc_above_target_prevalence": (
            better_than_baseline
        ),
        "conclusion": conclusion,
    }


# ============================================================
# SAVE REPORTS
# ============================================================

def save_reports(
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    mutual_information_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    repeated_cv_metrics: dict[str, float],
    interpretation: dict[str, object],
) -> None:
    """Save all signal-analysis artifacts."""

    numerical_df.to_csv(
        SIGNAL_REPORT_DIR
        / "numerical_signal.csv",
        index=False,
    )

    categorical_df.to_csv(
        SIGNAL_REPORT_DIR
        / "categorical_signal.csv",
        index=False,
    )

    mutual_information_df.to_csv(
        SIGNAL_REPORT_DIR
        / "mutual_information.csv",
        index=False,
    )

    permutation_df.to_csv(
        SIGNAL_REPORT_DIR
        / "permutation_importance.csv",
        index=False,
    )

    with open(
        SIGNAL_REPORT_DIR
        / "repeated_cv_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            repeated_cv_metrics,
            file,
            indent=4,
        )

    with open(
        SIGNAL_REPORT_DIR
        / "signal_conclusion.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            interpretation,
            file,
            indent=4,
        )


# ============================================================
# TERMINAL OUTPUT
# ============================================================

def print_report(
    numerical_df: pd.DataFrame,
    categorical_df: pd.DataFrame,
    mutual_information_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    repeated_cv_metrics: dict[str, float],
    interpretation: dict[str, object],
) -> None:
    """Print a concise signal investigation report."""

    print("\n" + "=" * 60)

    print(
        "EMPLOYEE ATTRITION — SIGNAL INVESTIGATION"
    )

    print("=" * 60)

    print("\n[DATASET]")

    print(
        "Rows:                 "
        f"{interpretation.get('dataset_rows', 'N/A')}"
    )

    print(
        f"Target:               {TARGET_COLUMN}"
    )

    print(
        "Target prevalence:    "
        f"{interpretation['target_prevalence'] * 100:.1f}%"
    )

    print("\n[NUMERICAL SIGNAL]")

    print(
        numerical_df[
            [
                "feature",
                "point_biserial_correlation",
                "p_value",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[CATEGORICAL SIGNAL]")

    categorical_display = (
        categorical_df[
            [
                "feature",
                "category",
                "attrition_rate",
                "cramers_v",
                "p_value",
            ]
        ]
        .drop_duplicates(
            subset=["feature"]
        )
        .head(10)
    )

    print(
        categorical_display
        .round(4)
        .to_string(index=False)
    )

    print("\n[MUTUAL INFORMATION]")

    print(
        mutual_information_df
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[PERMUTATION IMPORTANCE]")

    print(
        permutation_df[
            [
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )

    print("\n[REPEATED CROSS-VALIDATION]")

    print(
        "Folds:                "
        f"{CV_SPLITS}"
    )

    print(
        "Repeats:              "
        f"{CV_REPEATS}"
    )

    print(
        "ROC-AUC:              "
        f"{repeated_cv_metrics['roc_auc']:.4f} +/- "
        f"{repeated_cv_metrics['roc_auc_std']:.4f}"
    )

    print(
        "PR-AUC:               "
        f"{repeated_cv_metrics['pr_auc']:.4f} +/- "
        f"{repeated_cv_metrics['pr_auc_std']:.4f}"
    )

    print(
        "Predicted positive:   "
        f"{repeated_cv_metrics['positive_rate'] * 100:.2f}% +/- "
        f"{repeated_cv_metrics['positive_rate_std'] * 100:.2f}%"
    )

    print(
        "Validation folds:     "
        f"{repeated_cv_metrics['total_folds']}"
    )

    print("\n[SIGNAL SUMMARY]")

    print(
        "Maximum numerical |r|: "
        f"{interpretation['maximum_numerical_absolute_correlation']:.4f}"
    )

    print(
        "Maximum Cramer's V:    "
        f"{interpretation['maximum_categorical_cramers_v']:.4f}"
    )

    print(
        "Maximum mutual info:   "
        f"{interpretation['maximum_mutual_information']:.4f}"
    )

    print(
        "\n[CONCLUSION]"
    )

    print(
        interpretation[
            "conclusion"
        ]
    )

    print("\n[OUTPUT]")

    print(
        "Reports:              "
        f"{SIGNAL_REPORT_DIR}"
    )

    print("\n" + "=" * 60)

    print(
        "SIGNAL INVESTIGATION COMPLETE"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """Run the complete signal investigation."""

    create_output_directories()

    df = load_data()

    target = prepare_target(
        df
    )

    print(
        "\nRunning numerical signal analysis..."
    )

    numerical_df = (
        analyze_numerical_features(
            df,
            target,
        )
    )

    print(
        "Running categorical signal analysis..."
    )

    categorical_df = (
        analyze_categorical_features(
            df,
            target,
        )
    )

    print(
        "Running mutual information analysis..."
    )

    mutual_information_df = (
        calculate_mutual_information(
            df,
            target,
        )
    )

    print(
        "Running permutation importance..."
    )

    permutation_df = (
        calculate_permutation_importance(
            df,
            target,
        )
    )

    print(
        "Running repeated cross-validation..."
    )

    repeated_cv_metrics = (
        repeated_cross_validation(
            df,
            target,
        )
    )

    interpretation = interpret_signal(
        repeated_cv_metrics,
        numerical_df,
        categorical_df,
        mutual_information_df,
        permutation_df,
        target,
    )

    # Store dataset size for reporting.
    interpretation[
        "dataset_rows"
    ] = int(
        len(df)
    )

    save_reports(
        numerical_df,
        categorical_df,
        mutual_information_df,
        permutation_df,
        repeated_cv_metrics,
        interpretation,
    )

    print_report(
        numerical_df,
        categorical_df,
        mutual_information_df,
        permutation_df,
        repeated_cv_metrics,
        interpretation,
    )


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()