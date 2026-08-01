"""Cross-process canonical identity regression for quality v0.1."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


def test_independent_process_runs_have_stable_canonical_identity(tmp_path: Path) -> None:
    roots = [tmp_path / "first", tmp_path / "second"]
    environment = {**os.environ, "PYTHONHASHSEED": "random"}
    for root in roots:
        subprocess.run(
            [
                "python", "-m", "scripts.run_toy_quality_evaluation",
                "--output-dir", str(root), "--variants", "A2", "--seeds", "17",
                "--max-eval-tasks", "1",
            ],
            cwd=Path(__file__).parents[2], env=environment, check=True,
            capture_output=True, text=True,
        )
    assert json.loads((roots[0] / "evaluation-config.json").read_text()) == json.loads(
        (roots[1] / "evaluation-config.json").read_text()
    )
    assert (roots[0] / "replay-hash.txt").read_bytes() == (
        roots[1] / "replay-hash.txt"
    ).read_bytes()
    for name in [
        "task-results.jsonl", "per-seed-summary.json", "aggregate-summary.json",
        "paired-comparisons.json", "human-readable-examples.md",
    ]:
        assert (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes()


def _full_canonical_compact_exports_are_byte_identical(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    committed = json.loads(
        (repository / "docs/evaluations/data/a2_a3_a4_heldout_summary.json").read_text()
    )
    implementation = os.environ.get(
        "QUALITY_IMPLEMENTATION_COMMIT", committed["implementation_commit"]
    )
    roots = [tmp_path / "first", tmp_path / "second"]
    compact_roots = [tmp_path / "compact-first", tmp_path / "compact-second"]
    environment = {**os.environ, "PYTHONHASHSEED": "random"}
    for root, compact in zip(roots, compact_roots, strict=True):
        subprocess.run(
            [
                "python", "-m", "scripts.run_toy_quality_evaluation",
                "--output-dir", str(root), "--implementation-commit", implementation,
            ],
            cwd=repository, env=environment, check=True, capture_output=True, text=True,
        )
        subprocess.run(
            [
                "python", "-m", "scripts.run_toy_quality_evaluation",
                "--output-dir", str(root), "--validate-only", "--compact-dir", str(compact),
            ],
            cwd=repository, env=environment, check=True, capture_output=True, text=True,
        )
    for name in [
        "evaluation-config.json", "replay-hash.txt", "task-results.jsonl",
        "per-seed-summary.json", "aggregate-summary.json", "paired-comparisons.json",
        "human-readable-examples.md",
    ]:
        assert (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes()
    for name in [
        "data/a2_a3_a4_heldout_summary.json",
        "A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md",
    ]:
        assert (compact_roots[0] / name).read_bytes() == (
            compact_roots[1] / name
        ).read_bytes()


if os.environ.get("RUN_CANONICAL_QUALITY_DETERMINISM") == "1":
    test_full_canonical_compact_exports_are_byte_identical = pytest.mark.canonical_quality(
        _full_canonical_compact_exports_are_byte_identical
    )
