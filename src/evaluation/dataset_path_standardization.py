"""
Dataset Path Standardization
=============================

Purpose
-------
Standardize the project so that the canonical dataset:

    data/raw/employee_attrition_dataset_v2.csv

is used consistently by the ML/evaluation pipeline.

This script:
1. Verifies that the canonical dataset exists.
2. Searches project source/config files for references to the
   legacy dataset.
3. Replaces legacy dataset references in operational pipeline files.
4. Preserves diagnostic/comparison scripts that intentionally need
   access to both dataset versions.
5. Creates backups before modifying files.
6. Re-scans the project after modification.
7. Produces a JSON and TXT audit report.

IMPORTANT
---------
This script does NOT modify or delete either dataset.
It only standardizes source-code references.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

LEGACY_DATASET = "employee_attrition_dataset.csv"
CANONICAL_DATASET = "employee_attrition_dataset_v2.csv"

LEGACY_PATH = DATA_DIR / LEGACY_DATASET
CANONICAL_PATH = DATA_DIR / CANONICAL_DATASET

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "signal_analysis"
    / "dataset_path_standardization"
)

JSON_REPORT = REPORT_DIR / "dataset_path_standardization_report.json"
SUMMARY_REPORT = (
    REPORT_DIR / "dataset_path_standardization_summary.txt"
)


# ============================================================
# FILE TYPES
# ============================================================

SOURCE_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".txt",
}


# ============================================================
# FILES THAT MUST RETAIN BOTH DATASET REFERENCES
# ============================================================
#
# These scripts are diagnostic/comparison utilities. They
# intentionally inspect both dataset versions and therefore
# should NOT be automatically rewritten.
#

EXCLUDED_FILES = {
    Path("src/evaluation/dataset_version_comparison.py"),
    Path("src/evaluation/dataset_consistency_audit.py"),
    Path("src/evaluation/canonical_dataset_check.py"),
    Path("src/evaluation/dataset_path_standardization.py"),
}


# ============================================================
# OPERATIONAL FILES THAT SHOULD USE CANONICAL DATASET
# ============================================================
#
# These are known ML/evaluation pipeline files.
#
# We explicitly list them so that diagnostic scripts that need
# the legacy dataset are not accidentally changed.
#

OPERATIONAL_FILES = {
    Path("src/analysis/eda.py"),
    Path("src/data/profile.py"),
    Path("src/data/validation.py"),
    Path("src/evaluation/data_signal_diagnosis.py"),
    Path("src/evaluation/feature_stability.py"),
    Path("src/evaluation/final_model_selection.py"),
    Path("src/evaluation/final_validation.py"),
    Path("src/evaluation/generalization_diagnosis.py"),
    Path("src/evaluation/model_benchmark.py"),
    Path("src/evaluation/model_optimization.py"),
    Path("src/evaluation/signal_analysis.py"),
    Path("src/evaluation/target_generation_audit.py"),
    Path("src/evaluation/target_feature_alignment.py"),
    Path("src/features/engineering.py"),
    Path("src/models/evaluate.py"),
    Path("src/models/train.py"),
}


# ============================================================
# HELPERS
# ============================================================

def relative_path(path: Path) -> str:
    """Return a project-relative path using forward slashes."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_text(path: Path) -> str:
    """Read a text file safely."""
    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def write_backup(path: Path, original_text: str) -> Path:
    """
    Create a timestamped backup next to the source file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup = path.with_name(
        f"{path.name}.backup_{timestamp}"
    )

    # Avoid accidental overwrite if two operations happen
    # within the same second.
    counter = 1

    while backup.exists():
        backup = path.with_name(
            f"{path.name}.backup_{timestamp}_{counter}"
        )
        counter += 1

    backup.write_text(
        original_text,
        encoding="utf-8",
    )

    return backup


def discover_source_files() -> list[Path]:
    """Discover project source/config files."""
    files: list[Path] = []

    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue

        # Ignore virtual environments and generated artifacts.
        parts = set(path.parts)

        if ".venv" in parts:
            continue

        if "__pycache__" in parts:
            continue

        if ".git" in parts:
            continue

        if "reports" in parts:
            continue

        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


def contains_legacy_reference(path: Path) -> bool:
    """Check whether a file references the legacy dataset."""
    try:
        text = read_text(path)
    except Exception:
        return False

    return LEGACY_DATASET in text


def standardize_reference(text: str) -> tuple[str, int]:
    """
    Replace legacy dataset references with the canonical dataset.

    Handles:
        employee_attrition_dataset.csv
        data/raw/employee_attrition_dataset.csv
        data\\raw\\employee_attrition_dataset.csv
    """

    replacements = 0

    patterns = [
        (
            r"employee_attrition_dataset\.csv",
            CANONICAL_DATASET,
        ),
    ]

    updated = text

    for pattern, replacement in patterns:
        updated, count = re.subn(
            pattern,
            replacement,
            updated,
        )
        replacements += count

    return updated, replacements


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_canonical_dataset() -> dict[str, Any]:
    """Validate that the canonical dataset exists and is usable."""

    result: dict[str, Any] = {
        "exists": CANONICAL_PATH.exists(),
        "path": relative_path(CANONICAL_PATH),
    }

    if not CANONICAL_PATH.exists():
        return result

    try:
        import pandas as pd

        df = pd.read_csv(CANONICAL_PATH)

        result.update(
            {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": list(df.columns),
                "target_exists": "Attrition" in df.columns,
                "target_positive_count": (
                    int((df["Attrition"] == "Yes").sum())
                    if "Attrition" in df.columns
                    else None
                ),
                "target_negative_count": (
                    int((df["Attrition"] == "No").sum())
                    if "Attrition" in df.columns
                    else None
                ),
                "missing_cells": int(df.isna().sum().sum()),
                "sha256": sha256_file(CANONICAL_PATH),
            }
        )

        if "Attrition" in df.columns:
            result["target_prevalence"] = round(
                float((df["Attrition"] == "Yes").mean()),
                6,
            )

    except Exception as exc:
        result["load_error"] = str(exc)

    return result


# ============================================================
# SCANNING
# ============================================================

def scan_project(files: list[Path]) -> dict[str, Any]:
    """
    Scan source/config files for legacy dataset references.
    """

    all_references: list[dict[str, Any]] = []
    operational_references: list[dict[str, Any]] = []
    excluded_references: list[dict[str, Any]] = []

    for path in files:
        if not contains_legacy_reference(path):
            continue

        rel = Path(relative_path(path))

        try:
            text = read_text(path)
        except Exception:
            continue

        count = text.count(LEGACY_DATASET)

        record = {
            "file": relative_path(path),
            "references": count,
        }

        all_references.append(record)

        if rel in EXCLUDED_FILES:
            excluded_references.append(record)

        elif rel in OPERATIONAL_FILES:
            operational_references.append(record)

    return {
        "all": all_references,
        "operational": operational_references,
        "excluded": excluded_references,
    }


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_operational_files() -> list[dict[str, Any]]:
    """
    Replace legacy references only in explicitly defined
    operational pipeline files.
    """

    changes: list[dict[str, Any]] = []

    for rel_path in sorted(OPERATIONAL_FILES):
        path = PROJECT_ROOT / rel_path

        if not path.exists():
            changes.append(
                {
                    "file": rel_path.as_posix(),
                    "status": "missing",
                    "replacements": 0,
                }
            )
            continue

        original = read_text(path)

        if LEGACY_DATASET not in original:
            changes.append(
                {
                    "file": rel_path.as_posix(),
                    "status": "already_canonical",
                    "replacements": 0,
                }
            )
            continue

        updated, replacement_count = standardize_reference(
            original
        )

        if updated == original:
            changes.append(
                {
                    "file": rel_path.as_posix(),
                    "status": "no_change",
                    "replacements": 0,
                }
            )
            continue

        backup = write_backup(path, original)

        path.write_text(
            updated,
            encoding="utf-8",
        )

        changes.append(
            {
                "file": rel_path.as_posix(),
                "status": "updated",
                "replacements": replacement_count,
                "backup": relative_path(backup),
            }
        )

    return changes


# ============================================================
# POST-CHECK
# ============================================================

def post_standardization_scan() -> dict[str, Any]:
    """
    Verify that operational files no longer contain the
    legacy dataset reference.
    """

    remaining_operational: list[str] = []

    for rel_path in sorted(OPERATIONAL_FILES):
        path = PROJECT_ROOT / rel_path

        if not path.exists():
            continue

        try:
            text = read_text(path)
        except Exception:
            continue

        if LEGACY_DATASET in text:
            remaining_operational.append(
                rel_path.as_posix()
            )

    return {
        "remaining_operational_legacy_references":
            remaining_operational,
        "pass": len(remaining_operational) == 0,
    }


# ============================================================
# REPORTING
# ============================================================

def build_summary(
    canonical_info: dict[str, Any],
    before_scan: dict[str, Any],
    changes: list[dict[str, Any]],
    after_scan: dict[str, Any],
) -> str:

    updated = [
        item
        for item in changes
        if item["status"] == "updated"
    ]

    lines = []

    lines.append("=" * 60)
    lines.append(
        "EMPLOYEE ATTRITION — DATASET PATH STANDARDIZATION"
    )
    lines.append("=" * 60)
    lines.append("")

    lines.append("[CANONICAL DATASET]")
    lines.append(
        f"File:                 {CANONICAL_DATASET}"
    )
    lines.append(
        f"Path:                 {canonical_info.get('path')}"
    )
    lines.append(
        f"Rows:                 {canonical_info.get('rows')}"
    )
    lines.append(
        f"Columns:              {canonical_info.get('columns')}"
    )
    lines.append(
        f"Target:               Attrition"
    )
    lines.append(
        f"Positive count:       "
        f"{canonical_info.get('target_positive_count')}"
    )
    lines.append(
        f"Negative count:       "
        f"{canonical_info.get('target_negative_count')}"
    )

    prevalence = canonical_info.get("target_prevalence")

    if prevalence is not None:
        lines.append(
            f"Target prevalence:    {prevalence:.2%}"
        )

    lines.append("")

    lines.append("[BEFORE STANDARDIZATION]")
    lines.append(
        f"Legacy references found: "
        f"{len(before_scan['all'])}"
    )
    lines.append(
        f"Operational references:  "
        f"{len(before_scan['operational'])}"
    )
    lines.append(
        f"Excluded diagnostic references: "
        f"{len(before_scan['excluded'])}"
    )

    lines.append("")

    lines.append("[FILES UPDATED]")
    lines.append(
        f"Files changed:         {len(updated)}"
    )

    if updated:
        for item in updated:
            lines.append(
                f"  PASS  {item['file']} "
                f"({item['replacements']} replacement(s))"
            )
    else:
        lines.append(
            "  No operational files required changes."
        )

    lines.append("")

    lines.append("[POST-STANDARDIZATION CHECK]")

    if after_scan["pass"]:
        lines.append(
            "PASS  No operational pipeline files "
            "reference the legacy dataset."
        )
    else:
        lines.append(
            "FAIL  Legacy dataset references remain:"
        )

        for file in after_scan[
            "remaining_operational_legacy_references"
        ]:
            lines.append(f"  - {file}")

    lines.append("")

    lines.append("[IMPORTANT]")
    lines.append(
        "Diagnostic comparison scripts may intentionally "
        "reference both dataset versions."
    )
    lines.append(
        "The canonical evaluation dataset is:"
    )
    lines.append(
        f"  {CANONICAL_DATASET}"
    )

    lines.append("")
    lines.append("[OVERALL STATUS]")

    if (
        canonical_info.get("exists")
        and after_scan["pass"]
    ):
        lines.append(
            "DATASET PATH STANDARDIZATION STATUS: PASS"
        )
    else:
        lines.append(
            "DATASET PATH STANDARDIZATION STATUS: FAIL"
        )

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Running dataset path standardization...")
    print()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Validating canonical dataset...")

    canonical_info = validate_canonical_dataset()

    if not canonical_info.get("exists"):
        print()
        print(
            "ERROR: Canonical dataset was not found:"
        )
        print(CANONICAL_PATH)
        raise SystemExit(1)

    print(
        f"Canonical dataset: {relative_path(CANONICAL_PATH)}"
    )

    if "rows" in canonical_info:
        print(
            f"Rows:              "
            f"{canonical_info['rows']}"
        )
        print(
            f"Columns:           "
            f"{canonical_info['columns']}"
        )

    print()
    print("Discovering project source/config files...")

    files = discover_source_files()

    print(
        f"Files scanned:      {len(files)}"
    )

    print()
    print("Scanning legacy dataset references...")

    before_scan = scan_project(files)

    print(
        f"Legacy references:  "
        f"{len(before_scan['all'])}"
    )
    print(
        f"Operational:        "
        f"{len(before_scan['operational'])}"
    )
    print(
        f"Diagnostic/excluded: "
        f"{len(before_scan['excluded'])}"
    )

    print()
    print("Standardizing operational dataset paths...")

    changes = standardize_operational_files()

    updated_count = sum(
        1
        for item in changes
        if item["status"] == "updated"
    )

    print(
        f"Files updated:      {updated_count}"
    )

    print()
    print("Running post-standardization verification...")

    after_scan = post_standardization_scan()

    print(
        "Remaining operational legacy references: "
        f"{len(after_scan['remaining_operational_legacy_references'])}"
    )

    summary = build_summary(
        canonical_info,
        before_scan,
        changes,
        after_scan,
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "canonical_dataset": canonical_info,
        "before_scan": before_scan,
        "changes": changes,
        "after_scan": after_scan,
        "status": (
            "PASS"
            if canonical_info.get("exists")
            and after_scan["pass"]
            else "FAIL"
        ),
    }

    JSON_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    SUMMARY_REPORT.write_text(
        summary,
        encoding="utf-8",
    )

    print()
    print(summary)

    print()
    print("[OUTPUT]")
    print(
        f"JSON report:       {JSON_REPORT}"
    )
    print(
        f"Summary report:    {SUMMARY_REPORT}"
    )

    if not after_scan["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()