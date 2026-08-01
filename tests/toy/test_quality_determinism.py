"""Cross-process canonical identity regression for quality v0.1."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


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
