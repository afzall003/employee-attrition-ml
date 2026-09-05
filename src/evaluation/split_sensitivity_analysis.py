"""
Split Sensitivity Analysis
--------------------------

Determines whether the observed fixed-holdout performance is representative
of the canonical dataset or unusually dependent on the selected split.

This is a diagnostic analysis only.
It does NOT modify the final model or final holdout.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/raw/employee_attrition_dataset_v2.csv")

OUTPUT_DIR = Path(
    "reports/signal_analysis/split_sensitivity"
)

N_SPLITS = 50
TEST_SIZE = 0.20
BASE_SEED = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset():
    print("Loading canonical dataset...")

    df = pd.read_csv(DATA_PATH)

    target = "Attrition"

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found."
        )

    y = (
        df[target]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    if y.isna().any():
        raise ValueError(
            "Target contains values outside the expected No/Yes mapping."
        )

    X = df.drop(columns=[target])

    # Employee_ID is an identifier, not a predictive feature.
    if "Employee_ID" in X.columns:
        X = X.drop(columns=["Employee_ID"])

    return X, y.astype(int)


# ============================================================
# PREPROCESSING
# ============================================================

def build_preprocessor(X):

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )


# ============================================================
# MODEL FACTORIES
# ============================================================

def build_models(X):

    models = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X)
            ),
            (
                "model",
                LogisticRegression(
                    C=0.01,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )

    models["Gradient Boosting"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X)
            ),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=42
                ),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(X)
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("Running split sensitivity analysis...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X, y = load_dataset()

    print("Dataset loaded successfully.")
    print(f"Rows:                 {len(X)}")
    print(f"Features:             {X.shape[1]}")
    print(
        f"Target prevalence:    {y.mean() * 100:.2f}%"
    )

    print()
    print("Generating repeated stratified holdouts...")
    print(f"Splits:               {N_SPLITS}")
    print(f"Test size:            {TEST_SIZE:.0%}")

    splitter = StratifiedShuffleSplit(
        n_splits=N_SPLITS,
        test_size=TEST_SIZE,
        random_state=BASE_SEED,
    )

    models = build_models(X)

    results = []

    for split_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y),
        start=1
    ):

        print(
            f"Split {split_number}/{N_SPLITS}"
        )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        for model_name, model in models.items():

            model.fit(
                X_train,
                y_train
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            roc_auc = roc_auc_score(
                y_test,
                probabilities
            )

            pr_auc = average_precision_score(
                y_test,
                probabilities
            )

            results.append(
                {
                    "split": split_number,
                    "model": model_name,
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                    "train_prevalence": y_train.mean(),
                    "test_prevalence": y_test.mean(),
                    "prevalence_delta":
                        y_test.mean() - y_train.mean(),
                }
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "split_results.csv",
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for model_name, group in results_df.groupby("model"):

        auc = group["roc_auc"]

        pr = group["pr_auc"]

        summary_rows.append(
            {
                "model": model_name,

                "roc_auc_mean":
                    auc.mean(),

                "roc_auc_std":
                    auc.std(),

                "roc_auc_min":
                    auc.min(),

                "roc_auc_max":
                    auc.max(),

                "roc_auc_median":
                    auc.median(),

                "roc_auc_p10":
                    auc.quantile(0.10),

                "roc_auc_p90":
                    auc.quantile(0.90),

                "pr_auc_mean":
                    pr.mean(),

                "pr_auc_std":
                    pr.std(),

                "pr_auc_min":
                    pr.min(),

                "pr_auc_max":
                    pr.max(),

                "below_060_rate":
                    (auc < 0.60).mean(),

                "below_055_rate":
                    (auc < 0.55).mean(),

                "above_065_rate":
                    (auc >= 0.65).mean(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUTPUT_DIR / "split_sensitivity_summary.csv",
        index=False
    )

    # ========================================================
    # FIXED HOLDOUT REFERENCE
    # ========================================================

    fixed_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=42,
    )

    fixed_train_idx, fixed_test_idx = next(
        fixed_splitter.split(X, y)
    )

    fixed_reference = {}

    for model_name, model in models.items():

        model.fit(
            X.iloc[fixed_train_idx],
            y.iloc[fixed_train_idx]
        )

        probabilities = model.predict_proba(
            X.iloc[fixed_test_idx]
        )[:, 1]

        fixed_reference[model_name] = {
            "roc_auc": roc_auc_score(
                y.iloc[fixed_test_idx],
                probabilities
            ),
            "pr_auc": average_precision_score(
                y.iloc[fixed_test_idx],
                probabilities
            ),
        }

    # ========================================================
    # DIAGNOSTIC FLAGS
    # ========================================================

    flags = []

    for _, row in summary_df.iterrows():

        if row["roc_auc_std"] >= 0.04:
            flags.append(
                f'{row["model"]} shows high split sensitivity '
                f'(ROC-AUC std={row["roc_auc_std"]:.4f}).'
            )

        if row["below_060_rate"] >= 0.50:
            flags.append(
                f'{row["model"]} produces ROC-AUC below 0.60 '
                f'in at least half of repeated holdouts.'
            )

    for model_name, fixed in fixed_reference.items():

        row = summary_df[
            summary_df["model"] == model_name
        ].iloc[0]

        percentile = (
            results_df[
                results_df["model"] == model_name
            ]["roc_auc"] < fixed["roc_auc"]
        ).mean()

        if percentile < 0.10:
            flags.append(
                f'The fixed holdout is unusually difficult for '
                f'{model_name} relative to repeated splits.'
            )

        elif percentile > 0.90:
            flags.append(
                f'The fixed holdout is unusually favorable for '
                f'{model_name} relative to repeated splits.'
            )

    if not flags:
        flags.append(
            "No extreme split sensitivity pattern was detected."
        )

    # ========================================================
    # OVERALL DIAGNOSIS
    # ========================================================

    gb_row = summary_df[
        summary_df["model"] == "Gradient Boosting"
    ].iloc[0]

    lr_row = summary_df[
        summary_df["model"] == "Logistic Regression"
    ].iloc[0]

    rf_row = summary_df[
        summary_df["model"] == "Random Forest"
    ].iloc[0]

    if (
        gb_row["roc_auc_mean"] < 0.60
        and lr_row["roc_auc_mean"] < 0.60
        and rf_row["roc_auc_mean"] < 0.60
    ):
        diagnosis = (
            "Predictive performance is weak across repeated "
            "stratified holdouts. The fixed holdout is therefore "
            "unlikely to be the sole explanation for the weak "
            "generalization results."
        )

    elif (
        gb_row["roc_auc_mean"] >= 0.60
        or lr_row["roc_auc_mean"] >= 0.60
        or rf_row["roc_auc_mean"] >= 0.60
    ):
        diagnosis = (
            "Predictive performance varies substantially across "
            "splits. The fixed holdout may be unusually difficult, "
            "so conclusions should rely on repeated validation "
            "rather than the single holdout alone."
        )

    else:
        diagnosis = (
            "Repeated holdout performance requires further "
            "investigation before model optimization."
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "dataset": {
            "path": str(DATA_PATH),
            "rows": int(len(X)),
            "features": int(X.shape[1]),
            "target_prevalence": float(y.mean()),
        },

        "validation_design": {
            "splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "random_state": BASE_SEED,
        },

        "summary": summary_df.to_dict(
            orient="records"
        ),

        "fixed_holdout_reference":
            fixed_reference,

        "diagnostic_flags": flags,

        "overall_diagnosis": diagnosis,
    }

    with open(
        OUTPUT_DIR / "split_sensitivity_report.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — SPLIT SENSITIVITY ANALYSIS"
    )
    print("=" * 60)

    print()
    print("[VALIDATION DESIGN]")
    print(f"Repeated splits:      {N_SPLITS}")
    print(f"Test size:            {TEST_SIZE:.0%}")
    print("Stratification:       Yes")

    print()
    print("[REPEATED HOLDOUT PERFORMANCE]")

    for _, row in summary_df.iterrows():

        print(
            f'{row["model"]:<24} '
            f'Mean={row["roc_auc_mean"]:.4f} '
            f'Std={row["roc_auc_std"]:.4f} '
            f'Min={row["roc_auc_min"]:.4f} '
            f'Max={row["roc_auc_max"]:.4f}'
        )

    print()
    print("[FIXED HOLDOUT REFERENCE]")

    for model_name, metrics in fixed_reference.items():

        print(
            f'{model_name:<24} '
            f'ROC-AUC={metrics["roc_auc"]:.4f} '
            f'PR-AUC={metrics["pr_auc"]:.4f}'
        )

    print()
    print("[DIAGNOSTIC FLAGS]")

    for flag in flags:
        print(f"- {flag}")

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    print()
    print("[OUTPUT]")
    print(
        f"Reports:              "
        f"{OUTPUT_DIR.resolve()}"
    )
    print(
        f"JSON report:          "
        f"{(OUTPUT_DIR / 'split_sensitivity_report.json').resolve()}"
    )
    print(
        f"Split results:        "
        f"{(OUTPUT_DIR / 'split_results.csv').resolve()}"
    )
    print(
        f"Summary CSV:          "
        f"{(OUTPUT_DIR / 'split_sensitivity_summary.csv').resolve()}"
    )

    print()
    print("=" * 60)
    print(
        "SPLIT SENSITIVITY ANALYSIS COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()