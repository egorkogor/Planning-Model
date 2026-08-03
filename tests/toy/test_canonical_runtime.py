from __future__ import annotations

import json
import os
import subprocess
import sys

PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def run_isolated(source: str, *, environment: dict[str, str] | None = None):
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, **PINNED_ENV, **(environment or {})},
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_canonical_cpu_runtime_fingerprint_and_idempotence() -> None:
    fingerprint = run_isolated(
        """
import json
from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
first = configure_canonical_cpu_runtime(17)
second = configure_canonical_cpu_runtime(17)
assert first == second
print(json.dumps(second, sort_keys=True))
"""
    )
    assert fingerprint == {
        "profile_version": "toy-quality-canonical-cpu-runtime/1.0",
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only_enabled": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        **PINNED_ENV,
    }


def test_incompatible_environment_fails_closed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from planner_toy.canonical_runtime import "
            "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        ],
        env={**os.environ, **PINNED_ENV, "OMP_NUM_THREADS": "2"},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_ENV_MISMATCH:OMP_NUM_THREADS" in result.stderr


def test_late_interop_configuration_fails_closed() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; torch.set_num_interop_threads(2); "
            "from planner_toy.canonical_runtime import "
            "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        ],
        env={**os.environ, **PINNED_ENV},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_CONFIGURATION_LATE" in result.stderr
