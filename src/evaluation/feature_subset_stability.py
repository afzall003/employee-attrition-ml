"""
Feature Subset Stability Analysis

Purpose
-------
Determine whether predictive performance is concentrated in a small,
stable subset of features or whether the model depends on many weak/noisy
features.

This is a diagnostic analysis only.

It does NOT modify the final model, final holdout, or operating threshold.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5

DATASET_PATH = Path("data/raw/employee_attrition_dataset_v2.csv")

OUTPUT_DIR = Path(
    "reports/signal_analysis/feature_subset_stability"
)

TARGET_COLUMN = "Attrition"
IDENTIFIER_COLUMNS = {"Employee_ID"}

SUBSET_SIZES = [3, 5, 7, 10, 15, 20, 24]


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    print("Loading canonical dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found."
        )

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(df)}")
    print(
        f"Features:             "
        f"{len([c for c in df.columns if c not in IDENTIFIER_COLUMNS | {TARGET_COLUMN}])}"
    )
    print(
        f"Target prevalence:    "
        f"{(df[TARGET_COLUMN] == 'Yes').mean():.2%}"
    )

    return df


# ============================================================
# TARGET ENCODING
# ============================================================

def encode_target(series: pd.Series) -> np.ndarray:
    mapping = {
        "No": 0,
        "Yes": 1,
        0: 0,
        1: 1,
        False: 0,
        True: 1,
    }

    encoded = series.map(mapping)

    if encoded.isna().any():
        unique_values = series.dropna().unique()
        raise ValueError(
            f"Unexpected target values: {unique_values}"
        )

    return encoded.astype(int).to_numpy()


# ============================================================
# FEATURE RANKING
# ============================================================

def rank_features(
    X: pd.DataFrame,
    y: np.ndarray,
) -> list[str]:
    """
    Rank features using training-data-only univariate signal.

    Numerical features:
        absolute point-biserial correlation.

    Categorical features:
        maximum absolute target-rate deviation.

    This ranking is performed separately inside each training fold
    to prevent validation leakage.
    """

    scores = {}

    for column in X.columns:

        series = X[column]

        if pd.api.types.is_numeric_dtype(series):

            clean = pd.DataFrame({
                "x": pd.to_numeric(series, errors="coerce"),
                "y": y,
            }).dropna()

            if len(clean) < 10 or clean["x"].nunique() < 2:
                scores[column] = 0.0
                continue

            correlation = np.corrcoef(
                clean["x"],
                clean["y"],
            )[0, 1]

            if np.isnan(correlation):
                correlation = 0.0

            scores[column] = abs(correlation)

        else:

            temp = pd.DataFrame({
                "feature": series.astype(str),
                "target": y,
            })

            overall_rate = temp["target"].mean()

            grouped = temp.groupby("feature")["target"].mean()

            if len(grouped) == 0:
                scores[column] = 0.0
            else:
                scores[column] = float(
                    (grouped - overall_rate).abs().max()
                )

    return sorted(
        scores,
        key=scores.get,
        reverse=True,
    )


# ============================================================
# PREPROCESSING
# ============================================================

def build_pipeline(
    X: pd.DataFrame,
    model,
) -> Pipeline:

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])

        transformers.append(
            (
                "numeric",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ])

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ============================================================
# MODELS
# ============================================================

def build_models():

    return {
        "Logistic Regression": LogisticRegression(
            C=0.01,
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("Running feature subset stability analysis...")

    df = load_dataset()

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
        and column != TARGET_COLUMN
    ]

    X = df[feature_columns].copy()
    y = encode_target(df[TARGET_COLUMN])

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    models = build_models()

    results = []

    total_splits = N_SPLITS * N_REPEATS

    print()
    print("Generating repeated validation splits...")
    print(f"Splits:               {total_splits}")
    print(f"Subset sizes:         {SUBSET_SIZES}")
    print()

    split_number = 0

    for train_index, validation_index in cv.split(X, y):

        split_number += 1

        print(
            f"Split {split_number}/{total_splits}"
        )

        X_train = X.iloc[train_index].copy()
        X_valid = X.iloc[validation_index].copy()

        y_train = y[train_index]
        y_valid = y[validation_index]

        ranking = rank_features(
            X_train,
            y_train,
        )

        available_subset_sizes = [
            size
            for size in SUBSET_SIZES
            if size <= len(ranking)
        ]

        for subset_size in available_subset_sizes:

            selected_features = ranking[:subset_size]

            for model_name, model in models.items():

                pipeline = build_pipeline(
                    X_train[selected_features],
                    model,
                )

                pipeline.fit(
                    X_train[selected_features],
                    y_train,
                )

                probabilities = pipeline.predict_proba(
                    X_valid[selected_features]
                )[:, 1]

                roc_auc = roc_auc_score(
                    y_valid,
                    probabilities,
                )

                pr_auc = average_precision_score(
                    y_valid,
                    probabilities,
                )

                results.append({
                    "split": split_number,
                    "model": model_name,
                    "subset_size": subset_size,
                    "features": "|".join(
                        selected_features
                    ),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                })

    results_df = pd.DataFrame(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = (
        results_df
        .groupby(
            ["model", "subset_size"],
            as_index=False,
        )
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            roc_auc_min=("roc_auc", "min"),
            roc_auc_max=("roc_auc", "max"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            pr_auc_min=("pr_auc", "min"),
            pr_auc_max=("pr_auc", "max"),
        )
    )

    # ========================================================
    # FEATURE FREQUENCY
    # ========================================================

    feature_frequency_records = []

    for subset_size in SUBSET_SIZES:

        relevant = results_df[
            results_df["subset_size"] == subset_size
        ]

        counts = {}

        for feature_string in relevant["features"]:

            for feature in feature_string.split("|"):

                counts[feature] = (
                    counts.get(feature, 0) + 1
                )

        total = len(relevant)

        for feature, count in counts.items():

            feature_frequency_records.append({
                "subset_size": subset_size,
                "feature": feature,
                "selection_frequency": (
                    count / total
                    if total
                    else 0
                ),
            })

    feature_frequency = pd.DataFrame(
        feature_frequency_records
    )

    # ========================================================
    # DIAGNOSTIC FLAGS
    # ========================================================

    flags = []

    for model_name in summary["model"].unique():

        model_summary = summary[
            summary["model"] == model_name
        ]

        best_row = model_summary.loc[
            model_summary["roc_auc_mean"].idxmax()
        ]

        full_row = model_summary[
            model_summary["subset_size"]
            == len(feature_columns)
        ].iloc[0]

        best_subset = int(
            best_row["subset_size"]
        )

        performance_difference = (
            best_row["roc_auc_mean"]
            - full_row["roc_auc_mean"]
        )

        if best_subset < len(feature_columns):
            flags.append(
                f"{model_name}: best mean ROC-AUC occurs "
                f"with {best_subset} features rather than "
                f"all {len(feature_columns)} features."
            )

        if performance_difference >= 0.02:
            flags.append(
                f"{model_name}: reduced feature subset "
                f"outperforms the full feature set by at "
                f"least 0.02 ROC-AUC."
            )

    # ========================================================
    # OVERALL DIAGNOSIS
    # ========================================================

    best_overall = summary.loc[
        summary["roc_auc_mean"].idxmax()
    ]

    full_results = summary[
        summary["subset_size"]
        == len(feature_columns)
    ]

    full_mean = full_results[
        "roc_auc_mean"
    ].max()

    best_mean = best_overall[
        "roc_auc_mean"
    ]

    if best_mean >= full_mean + 0.02:

        diagnosis = (
            "Predictive performance is concentrated in a "
            "smaller subset of features. The full feature "
            "set appears to introduce weak or noisy variables "
            "that reduce validation performance."
        )

    elif best_mean >= full_mean + 0.01:

        diagnosis = (
            "A smaller feature subset shows a modest "
            "generalization advantage over the full feature "
            "set. Feature reduction should be investigated "
            "before further model complexity is introduced."
        )

    else:

        diagnosis = (
            "No strong advantage for aggressive feature "
            "reduction was observed. Predictive performance "
            "appears distributed across a broader feature "
            "set, although individual features may still "
            "vary in stability."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIR
        / "feature_subset_stability_results.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "feature_subset_stability_summary.csv"
    )

    frequency_path = (
        OUTPUT_DIR
        / "feature_selection_frequency.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "feature_subset_stability_report.json"
    )

    text_path = (
        OUTPUT_DIR
        / "feature_subset_stability_summary.txt"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    feature_frequency.to_csv(
        frequency_path,
        index=False,
    )

    report = {
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": int(len(df)),
            "features": int(len(feature_columns)),
            "target_prevalence": float(y.mean()),
        },
        "validation": {
            "folds": N_SPLITS,
            "repeats": N_REPEATS,
            "total_splits": total_splits,
            "subset_sizes": SUBSET_SIZES,
        },
        "summary": summary.to_dict(
            orient="records"
        ),
        "feature_selection_frequency": (
            feature_frequency.to_dict(
                orient="records"
            )
        ),
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    with open(
        text_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EMPLOYEE ATTRITION — "
            "FEATURE SUBSET STABILITY ANALYSIS\n"
        )
        file.write("=" * 60 + "\n\n")

        file.write("[DATASET]\n")
        file.write(
            f"Rows:                 {len(df)}\n"
        )
        file.write(
            f"Features:             {len(feature_columns)}\n"
        )
        file.write(
            f"Target prevalence:    {y.mean():.2%}\n\n"
        )

        file.write("[VALIDATION DESIGN]\n")
        file.write(
            f"Folds per repeat:      {N_SPLITS}\n"
        )
        file.write(
            f"Repeats:               {N_REPEATS}\n"
        )
        file.write(
            f"Total validation:      {total_splits}\n\n"
        )

        file.write("[PERFORMANCE BY SUBSET]\n")

        for _, row in summary.iterrows():

            file.write(
                f"{row['model']:<24}"
                f"Features={int(row['subset_size']):>2} "
                f"ROC-AUC={row['roc_auc_mean']:.4f} "
                f"± {row['roc_auc_std']:.4f} "
                f"PR-AUC={row['pr_auc_mean']:.4f}\n"
            )

        file.write("\n[DIAGNOSTIC FLAGS]\n")

        if flags:

            for flag in flags:
                file.write(f"- {flag}\n")

        else:

            file.write(
                "- No major subset-stability flags detected.\n"
            )

        file.write("\n[OVERALL DIAGNOSIS]\n")
        file.write(diagnosis + "\n")

        file.write("\n[OUTPUT]\n")
        file.write(
            f"Results:              {results_path}\n"
        )
        file.write(
            f"Summary:              {summary_path}\n"
        )
        file.write(
            f"Feature frequency:    {frequency_path}\n"
        )
        file.write(
            f"JSON report:          {json_path}\n"
        )
        file.write(
            f"Summary report:       {text_path}\n"
        )

    # ========================================================
    # CONSOLE REPORT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — "
        "FEATURE SUBSET STABILITY ANALYSIS"
    )
    print("=" * 60)

    print()
    print("[DATASET]")
    print(f"Rows:                 {len(df)}")
    print(f"Features:             {len(feature_columns)}")
    print(f"Target prevalence:    {y.mean():.2%}")

    print()
    print("[VALIDATION DESIGN]")
    print(f"Folds per repeat:      {N_SPLITS}")
    print(f"Repeats:               {N_REPEATS}")
    print(f"Total validation:      {total_splits}")

    print()
    print("[PERFORMANCE BY SUBSET]")

    for _, row in summary.iterrows():

        print(
            f"{row['model']:<24}"
            f"Features={int(row['subset_size']):>2} "
            f"ROC-AUC={row['roc_auc_mean']:.4f} "
            f"Std={row['roc_auc_std']:.4f} "
            f"PR-AUC={row['pr_auc_mean']:.4f}"
        )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:

        for flag in flags:
            print(f"- {flag}")

    else:

        print(
            "- No major subset-stability flags detected."
        )

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    print()
    print("[OUTPUT]")
    print(f"Results:              {results_path}")
    print(f"Summary:              {summary_path}")
    print(f"Feature frequency:    {frequency_path}")
    print(f"JSON report:          {json_path}")
    print(f"Summary report:       {text_path}")

    print()
    print(
        "=" * 60
    )
    print(
        "FEATURE SUBSET STABILITY ANALYSIS COMPLETE"
    )
    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()