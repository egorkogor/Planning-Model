from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_canonical_training_probe import (
    EXECUTION_CONTRACT_VERSION,
    PROBE_VERSION,
    compare_probes,
    compute_probe_identity,
)

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PROFILE_ENV = {
    "historical-default": {},
    "default-single-tensor": {
        "ATEN_CPU_CAPABILITY": "default",
        "MKL_CBWR": "COMPATIBLE",
    },
    "avx2-single-tensor": {
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_CBWR": "COMPATIBLE",
    },
}


def _environment(profile: str) -> dict[str, str]:
    environment = {**os.environ, **THREAD_ENV}
    environment.pop("ATEN_CPU_CAPABILITY", None)
    environment.pop("MKL_CBWR", None)
    environment.update(PROFILE_ENV[profile])
    return environment


def _run_probe(profile: str, path: Path, *extra: str) -> dict:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_canonical_training_probe",
            "run",
            "--profile",
            profile,
            *extra,
            "--output",
            str(path),
        ],
        env=_environment(profile),
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def repeated_profile_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, tuple[Path, Path]]:
    root = tmp_path_factory.mktemp("profiles")
    result = {}
    for profile in PROFILE_ENV:
        first = root / f"{profile}-first.json"
        second = root / f"{profile}-second.json"
        _run_probe(profile, first)
        _run_probe(profile, second)
        result[profile] = (first, second)
    return result


@pytest.mark.parametrize(
    "profile",
    (
        "historical-default",
        "default-single-tensor",
        "avx2-single-tensor",
    ),
)
def test_named_profile_is_byte_identical_twice_on_one_host(
    profile: str,
    repeated_profile_artifacts: dict[str, tuple[Path, Path]],
) -> None:
    first, second = repeated_profile_artifacts[profile]
    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["probe_version"] == PROBE_VERSION
    assert (
        payload["execution_contract"]["contract_version"]
        == EXECUTION_CONTRACT_VERSION
    )
    assert len(payload["updates"]) == 9


def test_controlled_profiles_use_explicit_single_tensor_flags(
    repeated_profile_artifacts: dict[str, tuple[Path, Path]],
) -> None:
    for profile in ("default-single-tensor", "avx2-single-tensor"):
        payload = json.loads(
            repeated_profile_artifacts[profile][0].read_text(
                encoding="utf-8"
            )
        )
        assert (
            payload["execution_contract"]["profile_kind"]
            == "controlled-investigation"
        )
        assert payload["execution_contract"]["foreach"] is False
        assert payload["execution_contract"]["fused"] is False


def test_historical_profile_preserves_optimizer_defaults(
    repeated_profile_artifacts: dict[str, tuple[Path, Path]],
) -> None:
    payload = json.loads(
        repeated_profile_artifacts["historical-default"][0].read_text(
            encoding="utf-8"
        )
    )
    contract = payload["execution_contract"]
    assert contract["profile_kind"] == "historical"
    assert contract["ATEN_CPU_CAPABILITY"] is None
    assert contract["MKL_CBWR"] is None
    assert contract["foreach"] is None
    assert contract["fused"] is None


def test_actual_dispatch_is_persisted_and_checked(
    repeated_profile_artifacts: dict[str, tuple[Path, Path]],
) -> None:
    default = json.loads(
        repeated_profile_artifacts["default-single-tensor"][0].read_text(
            encoding="utf-8"
        )
    )
    avx2 = json.loads(
        repeated_profile_artifacts["avx2-single-tensor"][0].read_text(
            encoding="utf-8"
        )
    )
    assert (
        default["execution_contract"]["actual_atten_cpu_capability"]
        == "DEFAULT"
    )
    assert (
        avx2["execution_contract"]["actual_atten_cpu_capability"]
        == "AVX2"
    )


def _contract(
    *,
    profile: str = "historical-default",
    aten: str | None = None,
    actual: str = "AVX2",
    mkl: str | None = None,
    foreach: bool | None = None,
    fused: bool | None = None,
) -> dict:
    return {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "profile": profile,
        "profile_kind": (
            "historical"
            if profile == "historical-default"
            else "controlled-investigation"
        ),
        "ATEN_CPU_CAPABILITY": aten,
        "actual_atten_cpu_capability": actual,
        "MKL_CBWR": mkl,
        "foreach": foreach,
        "fused": fused,
        "optimizer_class": "torch.optim.adamw.AdamW",
        "optimizer_hyperparameters": {
            "amsgrad": False,
            "betas": [0.9, 0.95],
            "capturable": False,
            "decoupled_weight_decay": True,
            "differentiable": False,
            "eps": 1e-8,
            "lr": 0.0003,
            "maximize": False,
            "weight_decay": 0.01,
        },
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "mkldnn_enabled": False,
        "deterministic_algorithms": {
            "enabled": True,
            "warn_only": False,
        },
    }


def _minimal_probe_fixture(contract: dict | None = None) -> dict:
    selected = copy.deepcopy(contract or _contract())
    contract_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            selected,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    payload = {
        "probe_version": PROBE_VERSION,
        "variant": "A2",
        "seed": 17,
        "epochs": 1,
        "ordered_train_task_ids": ["bw-00000001"],
        "parameter_names": ["a"],
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
                "parameters_after_optimizer_step": {
                    "a": "parameter"
                },
            }
        ],
        "execution_contract": selected,
        "execution_contract_sha256": contract_hash,
        "hardware_runtime_fingerprint": {
            "observation_identity_sha256": "sha256:" + "1" * 64
        },
    }
    payload["probe_identity"] = compute_probe_identity(payload)
    return payload


def _assert_incomparable(left: dict, right: dict) -> None:
    report = compare_probes(left, right)
    assert report["comparable"] is False
    assert report["reason"] == "EXECUTION_CONTRACT_MISMATCH"
    assert report["equal"] is None
    assert report["first_divergence"] is None


def test_avx2_and_default_artifacts_are_not_comparable() -> None:
    left = _minimal_probe_fixture(
        _contract(
            profile="avx2-single-tensor",
            aten="avx2",
            actual="AVX2",
            mkl="COMPATIBLE",
            foreach=False,
            fused=False,
        )
    )
    right = _minimal_probe_fixture(
        _contract(
            profile="default-single-tensor",
            aten="default",
            actual="DEFAULT",
            mkl="COMPATIBLE",
            foreach=False,
            fused=False,
        )
    )
    _assert_incomparable(left, right)


def test_foreach_false_and_none_are_not_comparable() -> None:
    left = _minimal_probe_fixture(_contract(foreach=False))
    right = _minimal_probe_fixture(_contract(foreach=None))
    _assert_incomparable(left, right)


def test_fused_false_and_none_are_not_comparable() -> None:
    left = _minimal_probe_fixture(_contract(fused=False))
    right = _minimal_probe_fixture(_contract(fused=None))
    _assert_incomparable(left, right)


def test_same_contract_and_same_numbers_are_equal_across_hardware() -> None:
    left = _minimal_probe_fixture()
    right = copy.deepcopy(left)
    right["hardware_runtime_fingerprint"][
        "observation_identity_sha256"
    ] = "sha256:" + "2" * 64
    report = compare_probes(left, right)
    assert report["comparable"] is True
    assert report["reason"] is None
    assert report["equal"] is True


def test_same_contract_and_numerical_mutation_are_different() -> None:
    left = _minimal_probe_fixture()
    right = copy.deepcopy(left)
    right["updates"][0]["adamw_exp_avg_sq"]["a"] = "mutated"
    report = compare_probes(left, right)
    assert report["comparable"] is True
    assert report["equal"] is False
    assert report["first_divergence"]["stage"] == "adamw_exp_avg_sq"


def test_execution_contract_enters_probe_identity() -> None:
    first = _minimal_probe_fixture()
    second = _minimal_probe_fixture(_contract(actual="DEFAULT"))
    assert first["probe_identity"] != second["probe_identity"]


def test_explicit_foreach_and_fused_cli_flags_are_persisted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "explicit.json"
    payload = _run_probe(
        "default-single-tensor",
        output,
        "--optimizer-foreach",
        "false",
        "--optimizer-fused",
        "false",
    )
    assert payload["execution_contract"]["foreach"] is False
    assert payload["execution_contract"]["fused"] is False


def test_historical_probe_matches_quality_training_update(
    tmp_path: Path,
) -> None:
    output = tmp_path / "parity.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_canonical_training_probe",
            "parity",
            "--output",
            str(output),
        ],
        env=_environment("historical-default"),
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["equal"] is True
    assert payload["profile"] == "historical-default"
    assert payload["quality_trace"] == payload["probe_trace"]


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
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == (
            expected_sha256
        )


def test_frozen_scope_files_are_not_probe_outputs() -> None:
    source = Path("scripts/run_canonical_training_probe.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "planner_toy/canonical_runtime.py",
        ".github/workflows/ci.yml",
        "A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md",
        "a2_a3_a4_heldout_summary.json",
        "A2_A3_A4_V0_1_DECISION_RU.md",
    ):
        assert name not in source
