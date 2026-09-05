from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
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

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "model_optimization"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CV_SPLITS = 5

DEFAULT_THRESHOLD = 0.50

TARGET_COLUMN = "Attrition"


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_output_directories() -> None:
    """
    Create all directories required for optimization reports.
    """

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the V2 employee attrition dataset.

    The project's engineering layer is responsible for
    separating features and target.

    IMPORTANT:
    prepare_model_data() is called with the dataframe only.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            "was not found in the dataset."
        )

    X, y = prepare_model_data(
        df
    )

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    X = X.reset_index(
        drop=True
    )

    y = y.reset_index(
        drop=True
    )

    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if y.nunique() != 2:
        raise ValueError(
            "The target must contain exactly two classes."
        )

    return X, y


# ============================================================
# PIPELINE
# ============================================================

def build_pipeline(
    estimator,
) -> Pipeline:
    """
    Build a fresh preprocessing + estimator pipeline.

    The current project implementation of build_preprocessor()
    accepts ZERO positional arguments.

    Therefore:

        build_preprocessor()

    is intentional.
    """

    preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                estimator,
            ),
        ]
    )

    return pipeline


# ============================================================
# OPTIMIZATION SEARCH SPACE
# ============================================================

def get_optimization_candidates() -> list[dict]:
    """
    Return the controlled optimization search space.

    The search is deliberately small and interpretable.

    Logistic Regression:
        - C controls regularization strength.
        - class_weight handles class imbalance.

    Gradient Boosting:
        - learning_rate
        - n_estimators
        - max_depth
        - min_samples_leaf

    We are intentionally NOT performing a huge grid search.
    The objective at this stage is controlled optimization
    against a reproducible baseline.
    """

    candidates = []

    # ========================================================
    # LOGISTIC REGRESSION
    # ========================================================

    logistic_configs = [
        {
            "C": 0.01,
            "class_weight": None,
        },
        {
            "C": 0.05,
            "class_weight": None,
        },
        {
            "C": 0.10,
            "class_weight": None,
        },
        {
            "C": 0.50,
            "class_weight": None,
        },
        {
            "C": 1.00,
            "class_weight": None,
        },
        {
            "C": 2.00,
            "class_weight": None,
        },
        {
            "C": 5.00,
            "class_weight": None,
        },
        {
            "C": 10.00,
            "class_weight": None,
        },
        {
            "C": 0.10,
            "class_weight": "balanced",
        },
        {
            "C": 0.50,
            "class_weight": "balanced",
        },
        {
            "C": 1.00,
            "class_weight": "balanced",
        },
        {
            "C": 2.00,
            "class_weight": "balanced",
        },
        {
            "C": 5.00,
            "class_weight": "balanced",
        },
    ]

    for config in logistic_configs:

        candidates.append(
            {
                "model_family": "Logistic Regression",
                "parameters": config,
            }
        )

    # ========================================================
    # GRADIENT BOOSTING
    # ========================================================

    gradient_configs = [
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 2,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 2,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 5,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_samples_leaf": 5,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_samples_leaf": 5,
        },
    ]

    for config in gradient_configs:

        candidates.append(
            {
                "model_family": "Gradient Boosting",
                "parameters": config,
            }
        )

    return candidates


# ============================================================
# ESTIMATOR FACTORY
# ============================================================

def create_estimator(
    model_family: str,
    parameters: dict,
):
    """
    Create the estimator corresponding to one candidate.
    """

    if model_family == "Logistic Regression":

        return LogisticRegression(
            C=float(
                parameters["C"]
            ),
            class_weight=parameters[
                "class_weight"
            ],
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    if model_family == "Gradient Boosting":

        return GradientBoostingClassifier(
            n_estimators=int(
                parameters["n_estimators"]
            ),
            learning_rate=float(
                parameters["learning_rate"]
            ),
            max_depth=int(
                parameters["max_depth"]
            ),
            min_samples_leaf=int(
                parameters["min_samples_leaf"]
            ),
            random_state=RANDOM_STATE,
        )

    raise ValueError(
        f"Unknown model family: {model_family}"
    )


# ============================================================
# SINGLE CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    candidate_number: int,
    total_candidates: int,
    model_family: str,
    parameters: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> dict:
    """
    Evaluate one optimization candidate using stratified
    cross-validation.

    Every fold receives a completely fresh pipeline.

    This prevents preprocessing leakage between folds.
    """

    roc_auc_scores = []

    pr_auc_scores = []

    f1_scores = []

    precision_scores = []

    recall_scores = []

    accuracy_scores = []

    predicted_positive_rates = []

    estimator = create_estimator(
        model_family=model_family,
        parameters=parameters,
    )

    print(
        f"Candidate {candidate_number}/{total_candidates}: "
        f"{model_family}"
    )

    print(
        f"  Parameters: {parameters}"
    )

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        cv.split(X, y),
        start=1,
    ):

        print(
            f"  fold {fold_number}/{CV_SPLITS}"
        )

        X_train = X.iloc[
            train_indices
        ]

        X_validation = X.iloc[
            validation_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_validation = y.iloc[
            validation_indices
        ]

        # ----------------------------------------------------
        # Fresh pipeline for every fold
        # ----------------------------------------------------

        pipeline = build_pipeline(
            estimator
        )

        # ----------------------------------------------------
        # Fit
        # ----------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Probability prediction
        # ----------------------------------------------------

        probabilities = (
            pipeline
            .predict_proba(
                X_validation
            )[:, 1]
        )

        # ----------------------------------------------------
        # Default threshold
        #
        # IMPORTANT:
        # Threshold optimization is deliberately separated
        # from model optimization.
        # ----------------------------------------------------

        predictions = (
            probabilities
            >= DEFAULT_THRESHOLD
        ).astype(int)

        # ----------------------------------------------------
        # ROC-AUC
        # ----------------------------------------------------

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        roc_auc_scores.append(
            float(roc_auc)
        )

        # ----------------------------------------------------
        # PR-AUC
        # ----------------------------------------------------

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        pr_auc_scores.append(
            float(pr_auc)
        )

        # ----------------------------------------------------
        # F1
        # ----------------------------------------------------

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1_scores.append(
            float(f1)
        )

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        precision_scores.append(
            float(precision)
        )

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall_scores.append(
            float(recall)
        )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_validation,
            predictions,
        )

        accuracy_scores.append(
            float(accuracy)
        )

        # ----------------------------------------------------
        # Predicted positive rate
        # ----------------------------------------------------

        positive_rate = (
            predictions.mean()
        )

        predicted_positive_rates.append(
            float(
                positive_rate
            )
        )

    # ========================================================
    # Aggregate metrics
    # ========================================================

    result = {
        "candidate": candidate_number,

        "model_family": model_family,

        "parameters": json.dumps(
            parameters,
            sort_keys=True,
        ),

        "roc_auc_mean": float(
            np.mean(
                roc_auc_scores
            )
        ),

        "roc_auc_std": float(
            np.std(
                roc_auc_scores,
                ddof=1,
            )
        ),

        "pr_auc_mean": float(
            np.mean(
                pr_auc_scores
            )
        ),

        "pr_auc_std": float(
            np.std(
                pr_auc_scores,
                ddof=1,
            )
        ),

        "f1_mean": float(
            np.mean(
                f1_scores
            )
        ),

        "precision_mean": float(
            np.mean(
                precision_scores
            )
        ),

        "recall_mean": float(
            np.mean(
                recall_scores
            )
        ),

        "accuracy_mean": float(
            np.mean(
                accuracy_scores
            )
        ),

        "predicted_positive_rate_mean": float(
            np.mean(
                predicted_positive_rates
            )
            * 100.0
        ),

        "cv_folds": CV_SPLITS,
    }

    return result


# ============================================================
# RANKING
# ============================================================

def rank_results(
    results: list[dict],
) -> pd.DataFrame:
    """
    Rank optimization candidates.

    Primary ranking:
        ROC-AUC

    Secondary ranking:
        PR-AUC

    This follows the model benchmark methodology.
    """

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(
        by=[
            "roc_auc_mean",
            "pr_auc_mean",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    result_df.insert(
        0,
        "rank",
        np.arange(
            1,
            len(result_df) + 1,
        ),
    )

    return result_df


# ============================================================
# BEST CANDIDATE
# ============================================================

def select_best_candidate(
    result_df: pd.DataFrame,
) -> pd.Series:
    """
    Select the best candidate using ROC-AUC first and PR-AUC
    second.
    """

    if result_df.empty:
        raise ValueError(
            "No optimization results were produced."
        )

    return result_df.iloc[0]


# ============================================================
# BASELINE COMPARISON
# ============================================================

def compare_against_baseline(
    best_candidate: pd.Series,
) -> dict:
    """
    Compare the optimized candidate against the established
    model benchmark baseline.

    Baseline values come from the completed benchmark:

        Logistic Regression
        ROC-AUC = 0.6258
        PR-AUC  = 0.3392
    """

    baseline_roc_auc = 0.6258

    baseline_pr_auc = 0.3392

    optimized_roc_auc = float(
        best_candidate[
            "roc_auc_mean"
        ]
    )

    optimized_pr_auc = float(
        best_candidate[
            "pr_auc_mean"
        ]
    )

    roc_auc_delta = (
        optimized_roc_auc
        - baseline_roc_auc
    )

    pr_auc_delta = (
        optimized_pr_auc
        - baseline_pr_auc
    )

    return {
        "baseline_model": (
            "Logistic Regression"
        ),

        "baseline_roc_auc": (
            baseline_roc_auc
        ),

        "baseline_pr_auc": (
            baseline_pr_auc
        ),

        "optimized_model": str(
            best_candidate[
                "model_family"
            ]
        ),

        "optimized_roc_auc": (
            optimized_roc_auc
        ),

        "optimized_pr_auc": (
            optimized_pr_auc
        ),

        "roc_auc_delta": (
            float(roc_auc_delta)
        ),

        "pr_auc_delta": (
            float(pr_auc_delta)
        ),

        "roc_auc_improved": (
            bool(
                optimized_roc_auc
                > baseline_roc_auc
            )
        ),

        "pr_auc_improved": (
            bool(
                optimized_pr_auc
                > baseline_pr_auc
            )
        ),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    result_df: pd.DataFrame,
    best_candidate: pd.Series,
    baseline_comparison: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Save optimization outputs.
    """

    # --------------------------------------------------------
    # Full optimization table
    # --------------------------------------------------------

    optimization_csv = (
        REPORTS_DIR
        / "optimization_results.csv"
    )

    result_df.to_csv(
        optimization_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Metrics copy
    # --------------------------------------------------------

    metrics_csv = (
        METRICS_DIR
        / "model_optimization.csv"
    )

    result_df.to_csv(
        metrics_csv,
        index=False,
    )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best_candidate_path = (
        REPORTS_DIR
        / "best_candidate.json"
    )

    best_payload = (
        best_candidate
        .to_dict()
    )

    best_payload[
        "baseline_comparison"
    ] = baseline_comparison

    with open(
        best_candidate_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            best_payload,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Complete summary
    # --------------------------------------------------------

    summary_path = (
        REPORTS_DIR
        / "model_optimization_summary.json"
    )

    metadata = {
        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "dataset": str(
            DATA_PATH
        ),

        "rows": int(
            len(X)
        ),

        "features": int(
            X.shape[1]
        ),

        "target_column": (
            TARGET_COLUMN
        ),

        "target_prevalence": float(
            y.mean()
        ),

        "cv_type": (
            "StratifiedKFold"
        ),

        "cv_folds": (
            CV_SPLITS
        ),

        "random_state": (
            RANDOM_STATE
        ),

        "classification_threshold": (
            DEFAULT_THRESHOLD
        ),

        "optimization_objective": (
            "ROC-AUC primary, PR-AUC secondary"
        ),
    }

    summary = {
        "metadata": metadata,

        "baseline": {
            "model": (
                "Logistic Regression"
            ),
            "roc_auc": 0.6258,
            "pr_auc": 0.3392,
        },

        "best_candidate": best_payload,

        "baseline_comparison": (
            baseline_comparison
        ),

        "all_candidates": (
            result_df.to_dict(
                orient="records"
            )
        ),
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_report(
    X: pd.DataFrame,
    y: pd.Series,
    result_df: pd.DataFrame,
    best_candidate: pd.Series,
    baseline_comparison: dict,
) -> None:
    """
    Print final optimization report.
    """

    print()
    print(
        "============================================================"
    )

    print(
        "EMPLOYEE ATTRITION — MODEL OPTIMIZATION"
    )

    print(
        "============================================================"
    )

    # ========================================================
    # DATASET
    # ========================================================

    print()
    print(
        "[DATASET]"
    )

    print(
        f"Rows:                 "
        f"{len(X)}"
    )

    print(
        f"Features:             "
        f"{X.shape[1]}"
    )

    print(
        f"Target prevalence:    "
        f"{y.mean() * 100:.2f}%"
    )

    # ========================================================
    # OPTIMIZATION RESULTS
    # ========================================================

    print()
    print(
        "[OPTIMIZATION RESULTS]"
    )

    display_columns = [
        "rank",
        "model_family",
        "parameters",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "accuracy_mean",
        "predicted_positive_rate_mean",
    ]

    display_df = result_df[
        display_columns
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    # ========================================================
    # BEST CANDIDATE
    # ========================================================

    print()
    print(
        "[BEST OPTIMIZED CANDIDATE]"
    )

    print(
        f"Model:                "
        f"{best_candidate['model_family']}"
    )

    print(
        f"Parameters:           "
        f"{best_candidate['parameters']}"
    )

    print(
        f"ROC-AUC:              "
        f"{best_candidate['roc_auc_mean']:.4f} "
        f"+/- {best_candidate['roc_auc_std']:.4f}"
    )

    print(
        f"PR-AUC:               "
        f"{best_candidate['pr_auc_mean']:.4f} "
        f"+/- {best_candidate['pr_auc_std']:.4f}"
    )

    print(
        f"F1 @ 0.50:            "
        f"{best_candidate['f1_mean']:.4f}"
    )

    print(
        f"Precision @ 0.50:     "
        f"{best_candidate['precision_mean']:.4f}"
    )

    print(
        f"Recall @ 0.50:        "
        f"{best_candidate['recall_mean']:.4f}"
    )

    print(
        f"Accuracy @ 0.50:     "
        f"{best_candidate['accuracy_mean']:.4f}"
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    print()
    print(
        "[BASELINE COMPARISON]"
    )

    print(
        "Baseline model:       "
        f"{baseline_comparison['baseline_model']}"
    )

    print(
        f"Baseline ROC-AUC:     "
        f"{baseline_comparison['baseline_roc_auc']:.4f}"
    )

    print(
        f"Optimized ROC-AUC:    "
        f"{baseline_comparison['optimized_roc_auc']:.4f}"
    )

    print(
        f"ROC-AUC delta:        "
        f"{baseline_comparison['roc_auc_delta']:+.4f}"
    )

    print(
        f"Baseline PR-AUC:      "
        f"{baseline_comparison['baseline_pr_auc']:.4f}"
    )

    print(
        f"Optimized PR-AUC:     "
        f"{baseline_comparison['optimized_pr_auc']:.4f}"
    )

    print(
        f"PR-AUC delta:         "
        f"{baseline_comparison['pr_auc_delta']:+.4f}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print()
    print(
        "[INTERPRETATION]"
    )

    print(
        "Optimization ranking uses ROC-AUC as the primary "
        "objective and PR-AUC as the secondary objective."
    )

    print(
        "The classification threshold remains fixed at "
        f"{DEFAULT_THRESHOLD:.2f} during model optimization."
    )

    print(
        "Threshold optimization is intentionally handled "
        "separately from model optimization."
    )

    if (
        baseline_comparison[
            "roc_auc_improved"
        ]
    ):
        print(
            "The optimized candidate improved ROC-AUC "
            "over the established benchmark."
        )

    else:
        print(
            "The optimized candidate did NOT improve "
            "ROC-AUC over the established benchmark."
        )

    if (
        baseline_comparison[
            "pr_auc_improved"
        ]
    ):
        print(
            "The optimized candidate improved PR-AUC "
            "over the established benchmark."
        )

    else:
        print(
            "The optimized candidate did NOT improve "
            "PR-AUC over the established benchmark."
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print()
    print(
        "[OUTPUT]"
    )

    print(
        f"Reports:              "
        f"{REPORTS_DIR}"
    )

    print()
    print(
        "============================================================"
    )

    print(
        "MODEL OPTIMIZATION COMPLETE"
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete controlled model optimization.
    """

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    create_output_directories()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    X, y = load_dataset()

    print(
        "Dataset loaded successfully."
    )

    print(
        f"Rows:                 "
        f"{len(X)}"
    )

    print(
        f"Features:             "
        f"{X.shape[1]}"
    )

    print(
        f"Target prevalence:    "
        f"{y.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print()
    print(
        "Starting controlled model optimization..."
    )

    print(
        f"Cross-validation folds: "
        f"{CV_SPLITS}"
    )

    print(
        f"Optimization threshold: "
        f"{DEFAULT_THRESHOLD:.2f}"
    )

    print(
        "Primary objective: ROC-AUC"
    )

    print(
        "Secondary objective: PR-AUC"
    )

    # --------------------------------------------------------
    # CV
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # Candidate search space
    # --------------------------------------------------------

    candidates = (
        get_optimization_candidates()
    )

    total_candidates = len(
        candidates
    )

    print()
    print(
        f"Optimization candidates: "
        f"{total_candidates}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = []

    for candidate_number, candidate in enumerate(
        candidates,
        start=1,
    ):

        result = evaluate_candidate(
            candidate_number=candidate_number,
            total_candidates=total_candidates,
            model_family=candidate[
                "model_family"
            ],
            parameters=candidate[
                "parameters"
            ],
            X=X,
            y=y,
            cv=cv,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    result_df = rank_results(
        results
    )

    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    best_candidate = (
        select_best_candidate(
            result_df
        )
    )

    # --------------------------------------------------------
    # Baseline comparison
    # --------------------------------------------------------

    baseline_comparison = (
        compare_against_baseline(
            best_candidate
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        result_df=result_df,
        best_candidate=best_candidate,
        baseline_comparison=baseline_comparison,
        X=X,
        y=y,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_report(
        X=X,
        y=y,
        result_df=result_df,
        best_candidate=best_candidate,
        baseline_comparison=baseline_comparison,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()