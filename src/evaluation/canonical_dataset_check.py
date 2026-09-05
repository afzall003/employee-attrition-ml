"""
Canonical Dataset Check
-----------------------
Verifies that the employee attrition project has one clearly defined
canonical dataset and that the dataset matches the established evaluation
specification.

This script is AUDIT-ONLY.
It does not modify, delete, rename, or overwrite datasets.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "canonical_dataset_check"
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ESTABLISHED DATASET SPECIFICATION
# ============================================================

CANONICAL_DATASET_NAME = "employee_attrition_dataset_v2.csv"

EXPECTED_ROWS = 1000
EXPECTED_COLUMNS = 26
EXPECTED_FEATURE_COUNT = 24

TARGET_COLUMN = "Attrition"

EXPECTED_POSITIVE_COUNT = 236
EXPECTED_NEGATIVE_COUNT = 764
EXPECTED_PREVALENCE = 0.236

EXPECTED_TARGET_VALUES = {"No", "Yes"}

IDENTIFIER_COLUMN = "Employee_ID"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def sha256_file(path: Path) -> str:
    """Return SHA-256 fingerprint for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy/path values into JSON-safe values."""

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {
            str(key): json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    return value


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV dataset."""

    return pd.read_csv(path)


def target_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate target statistics."""

    if TARGET_COLUMN not in df.columns:
        return {
            "target_present": False,
            "positive_count": None,
            "negative_count": None,
            "missing_count": None,
            "prevalence": None,
            "unique_values": [],
        }

    target = df[TARGET_COLUMN]

    positive_count = int((target == "Yes").sum())
    negative_count = int((target == "No").sum())
    missing_count = int(target.isna().sum())

    prevalence = (
        positive_count / len(df)
        if len(df) > 0
        else None
    )

    return {
        "target_present": True,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "missing_count": missing_count,
        "prevalence": prevalence,
        "unique_values": sorted(
            target.dropna().astype(str).unique().tolist()
        ),
    }


def inspect_dataset(path: Path) -> dict[str, Any]:
    """Inspect a dataset against the canonical specification."""

    df = load_dataset(path)

    stats = target_statistics(df)

    exact_duplicates = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    identifier_unique = None

    if IDENTIFIER_COLUMN in df.columns:
        identifier_unique = int(
            df[IDENTIFIER_COLUMN].nunique(dropna=False)
        )

    expected_columns_match = len(df.columns) == EXPECTED_COLUMNS

    feature_count = (
        len(df.columns) - 2
        if TARGET_COLUMN in df.columns
        and IDENTIFIER_COLUMN in df.columns
        else None
    )

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "absolute_path": str(path),
        "file_name": path.name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "feature_count": feature_count,
        "target": stats,
        "exact_duplicates": exact_duplicates,
        "missing_cells": missing_cells,
        "identifier_unique": identifier_unique,
        "sha256": sha256_file(path),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "column_order": df.columns.tolist(),
        "canonical_name_match": (
            path.name == CANONICAL_DATASET_NAME
        ),
        "shape_match": (
            len(df) == EXPECTED_ROWS
            and len(df.columns) == EXPECTED_COLUMNS
        ),
        "target_spec_match": (
            stats["target_present"]
            and stats["positive_count"] == EXPECTED_POSITIVE_COUNT
            and stats["negative_count"] == EXPECTED_NEGATIVE_COUNT
            and stats["missing_count"] == 0
            and set(stats["unique_values"])
            == EXPECTED_TARGET_VALUES
        ),
        "feature_count_match": (
            feature_count == EXPECTED_FEATURE_COUNT
        ),
        "identifier_integrity": (
            IDENTIFIER_COLUMN in df.columns
            and identifier_unique == len(df)
        ),
        "clean_dataset": (
            exact_duplicates == 0
            and missing_cells == 0
        ),
    }


def discover_datasets() -> list[Path]:
    """Discover CSV datasets under data/raw."""

    if not RAW_DATA_DIR.exists():
        return []

    return sorted(
        path
        for path in RAW_DATA_DIR.glob("*.csv")
        if path.is_file()
    )


def evaluate_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate whether a dataset satisfies the canonical specification."""

    checks = {
        "file_name": candidate["canonical_name_match"],
        "shape": candidate["shape_match"],
        "target": candidate["target_spec_match"],
        "feature_count": candidate["feature_count_match"],
        "identifier_integrity": candidate["identifier_integrity"],
        "clean_dataset": candidate["clean_dataset"],
    }

    passed = all(checks.values())

    return {
        "checks": checks,
        "passed": passed,
    }


def search_project_for_dataset_references() -> list[dict[str, Any]]:
    """
    Search source/config files for references to the two known dataset names.

    This is intentionally lightweight and excludes generated reports,
    virtual environments, caches, and binary files.
    """

    names = [
        "employee_attrition_dataset.csv",
        "employee_attrition_dataset_v2.csv",
    ]

    ignored_directories = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "reports",
        ".pytest_cache",
        ".mypy_cache",
    }

    extensions = {
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".md",
        ".cfg",
        ".ini",
    }

    references: list[dict[str, Any]] = []

    for path in PROJECT_ROOT.rglob("*"):

        if not path.is_file():
            continue

        if any(
            part in ignored_directories
            for part in path.parts
        ):
            continue

        if path.suffix.lower() not in extensions:
            continue

        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            for dataset_name in names:
                if dataset_name in line:
                    references.append(
                        {
                            "file": str(
                                path.relative_to(PROJECT_ROOT)
                            ),
                            "line": line_number,
                            "dataset": dataset_name,
                            "content": line.strip(),
                        }
                    )

    return references


def generate_flags(
    datasets: list[dict[str, Any]],
    selected_dataset: dict[str, Any] | None,
    references: list[dict[str, Any]],
) -> list[str]:
    """Generate diagnostic flags."""

    flags: list[str] = []

    if selected_dataset is None:
        flags.append(
            "The canonical dataset file was not found."
        )
        return flags

    if not selected_dataset["canonical_name_match"]:
        flags.append(
            "The selected dataset does not match the established "
            "canonical dataset name."
        )

    if not selected_dataset["shape_match"]:
        flags.append(
            "The selected dataset does not match the established "
            "1000-row × 26-column shape."
        )

    if not selected_dataset["target_spec_match"]:
        flags.append(
            "The selected dataset does not match the established "
            "target distribution of 236 Yes / 764 No."
        )

    if not selected_dataset["feature_count_match"]:
        flags.append(
            "The selected dataset does not contain the established "
            "24 model features."
        )

    if not selected_dataset["identifier_integrity"]:
        flags.append(
            "Employee_ID is missing or is not unique."
        )

    if not selected_dataset["clean_dataset"]:
        flags.append(
            "The selected dataset contains duplicates or missing cells."
        )

    discovered_names = {
        item["file_name"]
        for item in datasets
    }

    if len(datasets) > 1:
        flags.append(
            "Multiple dataset candidates exist under data/raw. "
            "The evaluation pipeline must use the canonical dataset "
            "explicitly rather than relying on discovery order."
        )

    if (
        "employee_attrition_dataset.csv" in discovered_names
        and CANONICAL_DATASET_NAME in discovered_names
    ):
        flags.append(
            "Both the legacy dataset and canonical dataset are present. "
            "The legacy dataset must not be used for evaluation."
        )

    legacy_references = [
        ref
        for ref in references
        if (
            ref["dataset"]
            == "employee_attrition_dataset.csv"
        )
    ]

    canonical_references = [
        ref
        for ref in references
        if (
            ref["dataset"]
            == CANONICAL_DATASET_NAME
        )
    ]

    if legacy_references:
        flags.append(
            f"{len(legacy_references)} source/config reference(s) "
            "still point to the legacy dataset."
        )

    if not canonical_references:
        flags.append(
            "No source/config reference to the canonical dataset "
            "was detected."
        )

    return flags


def overall_diagnosis(
    selected_dataset: dict[str, Any] | None,
    flags: list[str],
) -> tuple[str, bool]:
    """Produce final canonical dataset diagnosis."""

    if selected_dataset is None:
        return (
            "CANONICAL DATASET CHECK FAILED. "
            "The established canonical dataset could not be found.",
            False,
        )

    critical_checks = [
        selected_dataset["canonical_name_match"],
        selected_dataset["shape_match"],
        selected_dataset["target_spec_match"],
        selected_dataset["feature_count_match"],
        selected_dataset["identifier_integrity"],
        selected_dataset["clean_dataset"],
    ]

    if all(critical_checks):
        diagnosis = (
            "The canonical dataset itself matches the established "
            "evaluation specification. The pipeline can proceed only "
            "after confirming that evaluation scripts explicitly load "
            "this canonical dataset and do not silently select the "
            "legacy dataset."
        )

        return diagnosis, True

    return (
        "The canonical dataset check FAILED. The selected dataset does "
        "not fully match the established evaluation specification. "
        "Model optimization and validation should remain paused until "
        "the dataset inconsistency is resolved.",
        False,
    )


def save_json_report(report: dict[str, Any]) -> Path:
    """Save JSON report."""

    output = REPORT_DIR / "canonical_dataset_check_report.json"

    with output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(report),
            file,
            indent=2,
        )

    return output


def save_summary(
    report: dict[str, Any],
) -> Path:
    """Save human-readable summary."""

    output = (
        REPORT_DIR
        / "canonical_dataset_check_summary.txt"
    )

    selected = report.get("selected_dataset")

    lines = [
        "EMPLOYEE ATTRITION — CANONICAL DATASET CHECK",
        "=" * 60,
        "",
        "[CANONICAL SPECIFICATION]",
        f"Dataset:              {CANONICAL_DATASET_NAME}",
        f"Rows:                 {EXPECTED_ROWS}",
        f"Columns:              {EXPECTED_COLUMNS}",
        f"Features:             {EXPECTED_FEATURE_COUNT}",
        f"Target:               {TARGET_COLUMN}",
        f"Positive count:       {EXPECTED_POSITIVE_COUNT}",
        f"Negative count:       {EXPECTED_NEGATIVE_COUNT}",
        f"Prevalence:           {EXPECTED_PREVALENCE:.2%}",
        "",
    ]

    if selected:
        target = selected["target"]

        lines.extend(
            [
                "[SELECTED DATASET]",
                f"Path:                 {selected['path']}",
                f"Rows:                 {selected['rows']}",
                f"Columns:              {selected['columns']}",
                f"Features:             {selected['feature_count']}",
                f"Target positives:     {target['positive_count']}",
                f"Target negatives:     {target['negative_count']}",
                f"Target prevalence:    "
                f"{target['prevalence']:.2%}",
                f"SHA-256:              {selected['sha256']}",
                "",
            ]
        )

    lines.append("[CHECKS]")

    if selected:
        checks = evaluate_candidate(selected)["checks"]

        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            lines.append(
                f"{status:5} {name}"
            )

    lines.extend(
        [
            "",
            "[DATASET REFERENCES]",
        ]
    )

    for ref in report["source_references"]:
        lines.append(
            f"{ref['file']}:{ref['line']} "
            f"-> {ref['dataset']}"
        )

    if not report["source_references"]:
        lines.append(
            "No dataset references detected."
        )

    lines.extend(
        [
            "",
            "[DIAGNOSTIC FLAGS]",
        ]
    )

    if report["diagnostic_flags"]:
        for flag in report["diagnostic_flags"]:
            lines.append(f"- {flag}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "[OVERALL DIAGNOSIS]",
            report["overall_diagnosis"],
            "",
            f"CANONICAL DATASET STATUS: "
            f"{'PASS' if report['canonical_status'] else 'FAIL'}",
            "",
        ]
    )

    output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("Running canonical dataset check...")
    print("Discovering dataset files...")

    dataset_paths = discover_datasets()

    print()
    print("Discovered dataset files:")

    if not dataset_paths:
        print("   None")
    else:
        for index, path in enumerate(
            dataset_paths,
            start=1,
        ):
            print(
                f"   {index}. "
                f"{path.relative_to(PROJECT_ROOT)}"
            )

    print()
    print("Loading and validating dataset candidates...")

    datasets: list[dict[str, Any]] = []

    for path in dataset_paths:
        try:
            print()
            print(
                f"Loading: "
                f"{path.relative_to(PROJECT_ROOT)}"
            )

            candidate = inspect_dataset(path)

            datasets.append(candidate)

            print(
                f"  Rows:    {candidate['rows']}"
            )
            print(
                f"  Columns: {candidate['columns']}"
            )

        except Exception as exc:
            print(
                f"  ERROR: {type(exc).__name__}: {exc}"
            )

    selected_dataset = None

    for candidate in datasets:
        if candidate["file_name"] == CANONICAL_DATASET_NAME:
            selected_dataset = candidate
            break

    print()
    print("Selecting established canonical dataset...")

    if selected_dataset:
        print(
            f"Canonical dataset: "
            f"{selected_dataset['path']}"
        )
    else:
        print(
            "Canonical dataset NOT FOUND."
        )

    print()
    print("Checking project dataset references...")

    references = search_project_for_dataset_references()

    print(
        f"Dataset references found: {len(references)}"
    )

    for reference in references:
        print(
            f"  {reference['file']}:"
            f"{reference['line']} "
            f"-> {reference['dataset']}"
        )

    print()
    print("Generating diagnostic flags...")

    flags = generate_flags(
        datasets,
        selected_dataset,
        references,
    )

    diagnosis, canonical_status = overall_diagnosis(
        selected_dataset,
        flags,
    )

    report = {
        "canonical_specification": {
            "dataset_name": CANONICAL_DATASET_NAME,
            "rows": EXPECTED_ROWS,
            "columns": EXPECTED_COLUMNS,
            "feature_count": EXPECTED_FEATURE_COUNT,
            "target_column": TARGET_COLUMN,
            "positive_count": EXPECTED_POSITIVE_COUNT,
            "negative_count": EXPECTED_NEGATIVE_COUNT,
            "prevalence": EXPECTED_PREVALENCE,
        },
        "discovered_datasets": datasets,
        "selected_dataset": selected_dataset,
        "source_references": references,
        "diagnostic_flags": flags,
        "overall_diagnosis": diagnosis,
        "canonical_status": canonical_status,
    }

    json_path = save_json_report(report)
    summary_path = save_summary(report)

    print()
    print("=" * 60)
    print(
        "EMPLOYEE ATTRITION — CANONICAL DATASET CHECK"
    )
    print("=" * 60)

    print()
    print("[CANONICAL SPECIFICATION]")
    print(
        f"Dataset:              "
        f"{CANONICAL_DATASET_NAME}"
    )
    print(
        f"Rows:                 {EXPECTED_ROWS}"
    )
    print(
        f"Columns:              {EXPECTED_COLUMNS}"
    )
    print(
        f"Features:             {EXPECTED_FEATURE_COUNT}"
    )
    print(
        f"Target:               {TARGET_COLUMN}"
    )
    print(
        f"Positive count:       {EXPECTED_POSITIVE_COUNT}"
    )
    print(
        f"Negative count:       {EXPECTED_NEGATIVE_COUNT}"
    )
    print(
        f"Target prevalence:    "
        f"{EXPECTED_PREVALENCE:.2%}"
    )

    if selected_dataset:
        target = selected_dataset["target"]

        print()
        print("[SELECTED DATASET]")
        print(
            f"Path:                 "
            f"{selected_dataset['path']}"
        )
        print(
            f"Rows:                 "
            f"{selected_dataset['rows']}"
        )
        print(
            f"Columns:              "
            f"{selected_dataset['columns']}"
        )
        print(
            f"Features:             "
            f"{selected_dataset['feature_count']}"
        )
        print(
            f"Target positives:     "
            f"{target['positive_count']}"
        )
        print(
            f"Target negatives:     "
            f"{target['negative_count']}"
        )
        print(
            f"Target prevalence:    "
            f"{target['prevalence']:.2%}"
        )
        print(
            f"SHA-256:              "
            f"{selected_dataset['sha256']}"
        )

        checks = evaluate_candidate(
            selected_dataset
        )["checks"]

        print()
        print("[CANONICAL CHECKS]")

        for name, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            print(
                f"{status:5} {name}"
            )

    print()
    print("[DIAGNOSTIC FLAGS]")

    if flags:
        for flag in flags:
            print(f"- {flag}")
    else:
        print("- None")

    print()
    print("[OVERALL DIAGNOSIS]")
    print(diagnosis)

    print()
    print("[OUTPUT]")
    print(
        f"JSON report:          {json_path}"
    )
    print(
        f"Summary report:       {summary_path}"
    )

    print()
    print(
        f"CANONICAL DATASET STATUS: "
        f"{'PASS' if canonical_status else 'FAIL'}"
    )

    print()
    print("=" * 60)

    if canonical_status:
        print(
            "CANONICAL DATASET CHECK COMPLETE"
        )
    else:
        print(
            "CANONICAL DATASET CHECK FAILED"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()