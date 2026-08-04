from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from scripts.run_canonical_training_probe import compare_probes

PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
}
PINNED_TORCH = torch.__version__.startswith("2.12.") and torch.__version__.endswith("+cpu")


def _run_probe(path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_canonical_training_probe",
            "run",
            "--mode",
            "canonical",
            "--output",
            str(path),
        ],
        env={**os.environ, **PINNED_ENV},
        check=True,
        text=True,
        capture_output=True,
    )


@pytest.mark.skipif(not PINNED_TORCH, reason="probe requires pinned PyTorch")
def test_minimal_training_probe_is_byte_identical_twice(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _run_probe(first)
    _run_probe(second)
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["probe_version"] == "toy-quality-canonical-training-probe/1.0"
    assert len(payload["updates"]) == 9


def _minimal_probe_fixture() -> dict:
    return {
        "probe_identity": "sha256:" + "0" * 64,
        "hardware_runtime_fingerprint": {
            "semantic_execution_identity_sha256": "sha256:" + "1" * 64
        },
        "initial_parameters": {"a": "initial"},
        "updates": [
            {
                "update": 1,
                "epoch": 1,
                "task_id": "bw-00000001",
                "encoded_task_sha256": "encoded",
                "forward_logits": {"action": "forward"},
                "loss_components": {"total": "loss"},
                "raw_gradients": {"a": "gradient"},
                "gradient_norm": "norm",
                "gradients_after_clipping": {"a": "clipped"},
                "adamw_exp_avg": {"a": "avg"},
                "adamw_exp_avg_sq": {"a": "avg_sq"},
                "parameters_after_optimizer_step": {"a": "parameter"},
            }
        ],
    }


def test_probe_comparison_detects_gradient_mutation() -> None:
    left = _minimal_probe_fixture()
    right = copy.deepcopy(left)
    right["updates"][0]["raw_gradients"]["a"] = "mutated"
    report = compare_probes(left, right)
    assert report["equal"] is False
    assert report["first_divergence"] == {
        "update": 1,
        "epoch": 1,
        "task_id": "bw-00000001",
        "stage": "raw_gradients",
        "name": "a",
    }
    assert report["first_parameter_divergence"]["parameter"] == "a"


def test_probe_comparison_detects_adamw_state_mutation() -> None:
    left = _minimal_probe_fixture()
    right = copy.deepcopy(left)
    right["updates"][0]["adamw_exp_avg_sq"]["a"] = "mutated"
    report = compare_probes(left, right)
    assert report["equal"] is False
    assert report["first_divergence"]["stage"] == "adamw_exp_avg_sq"
    assert report["first_parameter_divergence"] == {
        "update": 1,
        "epoch": 1,
        "task_id": "bw-00000001",
        "stage": "adamw_exp_avg_sq",
        "parameter": "a",
    }


def test_canonical_compare_remains_exact_without_tolerance() -> None:
    source = Path("scripts/run_canonical_training_probe.py").read_text(
        encoding="utf-8"
    )
    assert "isclose" not in source
    assert "tolerance" not in source
    assert "round(" not in source
    assert "quantiz" not in source


def test_frozen_quality_v0_1_artifacts_are_unchanged() -> None:
    expected = {
        "docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md": (
            "133b3be3ec0d1a9d1025c32ddd60413344cea03bd4acb84eb8b60cc0e3a46df1"
        ),
        "docs/evaluations/data/a2_a3_a4_heldout_summary.json": (
            "0a9c0cb0ee98fa4d458cb0a6fafa2e9e78bb0771a223d46a035f458c566df050"
        ),
        "docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md": (
            "a1392b21961c728c7b4fa8686d6d095a345dc319fdfeec2b1e47eaaccb9b754a"
        ),
    }
    for name, expected_sha256 in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_sha256


def test_frozen_quality_v0_1_artifacts_are_not_probe_outputs() -> None:
    source = Path("scripts/run_canonical_training_probe.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md",
        "a2_a3_a4_heldout_summary.json",
        "A2_A3_A4_V0_1_DECISION_RU.md",
    ):
        assert name not in source
