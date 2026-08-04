"""Fail-closed deterministic CPU profile for quality-v0.1 canonical runs."""
from __future__ import annotations

import os
import random
import threading

import numpy as np
import torch

CANONICAL_CPU_RUNTIME_VERSION = "toy-quality-canonical-cpu-runtime/1.0"
CANONICAL_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

_CONFIGURATION_LOCK = threading.Lock()
_CONFIGURED = False


def canonical_cpu_runtime_fingerprint() -> dict[str, object]:
    return {
        "profile_version": CANONICAL_CPU_RUNTIME_VERSION,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only_enabled": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        **{name: os.environ.get(name) for name in CANONICAL_THREAD_ENVIRONMENT},
    }


def _assert_canonical_profile() -> None:
    expected = {
        "profile_version": CANONICAL_CPU_RUNTIME_VERSION,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only_enabled": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        **CANONICAL_THREAD_ENVIRONMENT,
    }
    if canonical_cpu_runtime_fingerprint() != expected:
        raise RuntimeError("CANONICAL_CPU_RUNTIME_PROFILE_DRIFT")


def configure_canonical_cpu_runtime(seed: int | None = None) -> dict[str, object]:
    """Configure the process once, then optionally reset all supported RNGs."""
    global _CONFIGURED
    with _CONFIGURATION_LOCK:
        if not _CONFIGURED:
            for name, expected in CANONICAL_THREAD_ENVIRONMENT.items():
                actual = os.environ.get(name)
                if actual not in {None, expected}:
                    raise RuntimeError(f"CANONICAL_CPU_RUNTIME_ENV_MISMATCH:{name}")
                os.environ[name] = expected
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as error:
                raise RuntimeError("CANONICAL_CPU_RUNTIME_CONFIGURATION_LATE") from error
            torch.backends.mkldnn.enabled = False
            _CONFIGURED = True
        _assert_canonical_profile()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return canonical_cpu_runtime_fingerprint()
