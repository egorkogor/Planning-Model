from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PINNED_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "default",
    "MKL_CBWR": "COMPATIBLE",
}
PINNED_TORCH = torch.__version__.startswith("2.12.") and torch.__version__.endswith("+cpu")


def run_isolated(
    source: str,
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
):
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={**os.environ, **PINNED_ENV, **(environment or {})},
        text=True,
        capture_output=True,
        check=check,
    )
    return result


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_canonical_cpu_runtime_fingerprint_and_idempotence() -> None:
    result = run_isolated(
        """
import json
from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
first = configure_canonical_cpu_runtime(17)
second = configure_canonical_cpu_runtime(17)
assert first == second
print(json.dumps(second, sort_keys=True))
"""
    )
    fingerprint = json.loads(result.stdout)
    assert fingerprint["profile_version"] == "toy-quality-canonical-cpu-runtime/1.1"
    assert fingerprint["deterministic_algorithms_enabled"] is True
    assert fingerprint["deterministic_warn_only_enabled"] is False
    assert fingerprint["torch_num_threads"] == 1
    assert fingerprint["torch_num_interop_threads"] == 1
    assert fingerprint["mkldnn_enabled"] is False
    assert fingerprint["cpu_dispatch_capability"] == "DEFAULT"
    assert fingerprint["mkl_available"] is True
    assert fingerprint["openmp_available"] is True
    assert fingerprint["ATEN_CPU_CAPABILITY"] == "default"
    assert fingerprint["MKL_CBWR"] == "COMPATIBLE"
    assert fingerprint["execution_sentinel_sha256"].startswith("sha256:")
    assert fingerprint["torch_build_config_sha256"].startswith("sha256:")


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_full_runtime_fingerprint_has_required_machine_fields() -> None:
    result = run_isolated(
        """
import json
from planner_toy.canonical_runtime import (
    configure_canonical_cpu_runtime,
    full_hardware_runtime_fingerprint,
)
configure_canonical_cpu_runtime(17)
print(json.dumps(full_hardware_runtime_fingerprint(), sort_keys=True))
"""
    )
    fingerprint = json.loads(result.stdout)
    assert fingerprint["fingerprint_version"] == (
        "toy-quality-cpu-hardware-fingerprint/1.0"
    )
    observed = fingerprint["observed_runtime_and_hardware"]
    assert set(observed) == {
        "os",
        "cpu",
        "runner",
        "python",
        "pytorch",
        "canonical_environment",
    }
    assert observed["os"]["system"]
    assert observed["os"]["release"]
    assert observed["os"]["machine"]
    assert observed["cpu"]["logical_cpu_count"] >= 1
    assert observed["cpu"]["flags_sha256"].startswith("sha256:")
    assert set(observed["cpu"]["capabilities"]) == {
        "sse2",
        "avx",
        "avx2",
        "avx512f",
        "avx512dq",
        "avx512bw",
        "avx512vl",
        "fma",
    }
    assert observed["pytorch"]["version"].startswith("2.12.")
    assert observed["pytorch"]["cpu_dispatch_capability"] == "DEFAULT"
    assert observed["canonical_environment"] == {
        key: PINNED_ENV[key] for key in sorted(PINNED_ENV)
    }


def test_semantic_identity_excludes_unstable_fields() -> None:
    source = Path("planner_toy/canonical_runtime.py").read_text(encoding="utf-8")
    forbidden = ("hostname", "gethostname", "process_id", "getpid", "timestamp")
    semantic_section = source.split("semantic_identity = {", 1)[1].split(
        "return {", 1
    )[0]
    for token in forbidden:
        assert token not in semantic_section


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_incompatible_environment_fails_closed() -> None:
    result = run_isolated(
        "from planner_toy.canonical_runtime import "
        "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        environment={"OMP_NUM_THREADS": "2"},
        check=False,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_ENV_MISMATCH:OMP_NUM_THREADS" in result.stderr


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_cpu_capability_mismatch_fails_closed() -> None:
    result = run_isolated(
        """
import torch
torch.backends.cpu.get_cpu_capability = lambda: "AVX2"
from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
configure_canonical_cpu_runtime()
""",
        check=False,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_PROFILE_DRIFT:cpu_dispatch_capability" in result.stderr


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_unsupported_blas_path_fails_closed() -> None:
    result = run_isolated(
        """
import torch
torch.backends.mkl.is_available = lambda: False
from planner_toy.canonical_runtime import configure_canonical_cpu_runtime
configure_canonical_cpu_runtime()
""",
        check=False,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_PROFILE_DRIFT:mkl_available" in result.stderr


@pytest.mark.skipif(not PINNED_TORCH, reason="canonical runtime requires pinned PyTorch")
def test_late_interop_configuration_fails_closed() -> None:
    result = run_isolated(
        "import torch; torch.set_num_interop_threads(2); "
        "from planner_toy.canonical_runtime import "
        "configure_canonical_cpu_runtime; configure_canonical_cpu_runtime()",
        check=False,
    )
    assert result.returncode != 0
    assert "CANONICAL_CPU_RUNTIME_CONFIGURATION_LATE" in result.stderr


def test_legacy_runtime_is_not_reinterpreted_as_current() -> None:
    from planner_toy.canonical_runtime import (
        LEGACY_CANONICAL_CPU_RUNTIME_VERSION,
        validate_canonical_cpu_runtime_fingerprint,
    )

    with pytest.raises(
        RuntimeError, match="CANONICAL_CPU_RUNTIME_PROFILE_DRIFT:profile_version"
    ):
        validate_canonical_cpu_runtime_fingerprint(
            {"profile_version": LEGACY_CANONICAL_CPU_RUNTIME_VERSION}
        )
