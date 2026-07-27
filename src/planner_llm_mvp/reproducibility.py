"""Minimal checks for the autonomous phase/release evidence index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PHASE_IDS = tuple(f"P{i:02d}" for i in range(21))
MANDATORY_DECISION_GATES = {
    "G00_SCOPE",
    "G01_TRUST_AND_RESOURCES",
    "G06_STATISTICAL_IMPLEMENTATION_AUDIT",
    "G07_PLANNER_CONFIRMATORY_FREEZE",
    "G12_STAGE1A_CONFIRMATORY_FREEZE",
    "G16_STAGE1B_CONFIRMATORY_FREEZE",
    "G20_FINAL_ACCEPTANCE",
}
MANUAL_GATES = MANDATORY_DECISION_GATES


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def validate_release_index(data: dict[str, Any]) -> list[str]:
    """Validate final evidence index; All seven manual gate decisions are required."""
    errors: list[str] = []
    required = {"schema_version", "git_commit", "phase_reports", "decisions", "artifacts"}
    missing = sorted(required - data.keys())
    if missing:
        errors.append(f"Missing fields: {', '.join(missing)}")

    reports = data.get("phase_reports")
    if not isinstance(reports, dict) or tuple(sorted(reports)) != PHASE_IDS:
        errors.append("phase_reports must contain exactly P00 through P20")

    decisions = data.get("decisions")
    if not isinstance(decisions, dict):
        errors.append("decisions must be an object")
    else:
        keys = set(decisions)
        if not MANDATORY_DECISION_GATES.issubset(keys):
            errors.append("decisions must contain all seven mandatory manual gates")
        if not keys.issubset(MANUAL_GATES):
            errors.append("decisions contains unknown gates")

    commit = data.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40:
        errors.append("git_commit must be a 40-character commit hash")

    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
    return errors
