from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts.canonical_training_probe_contract import (
    EXECUTION_CONTRACT_VERSION,
    PROBE_VERSION,
    _resolve_profile_before_torch_import,
    validate_execution_contract,
)
from scripts.canonical_training_probe_parity import _first_parity_difference
from scripts.run_canonical_training_probe import (
    compare_probes,
    compute_probe_identity,
    validate_probe_identity,
)


def _contract(profile: str = "historical-default") -> dict:
    controlled = profile != "historical-default"
    return {
        "contract_version": EXECUTION_CONTRACT_VERSION,
        "profile": profile,
        "profile_kind": "controlled-investigation" if controlled else "historical",
        "ATEN_CPU_CAPABILITY": (
            "default" if profile == "default-single-tensor" else "avx2" if controlled else None
        ),
        "actual_atten_cpu_capability": (
            "DEFAULT" if profile == "default-single-tensor" else "AVX2"
        ),
        "MKL_CBWR": "COMPATIBLE" if controlled else None,
        "foreach": False if controlled else None,
        "fused": False if controlled else None,
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
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
    }


def _hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _probe(contract: dict | None = None) -> dict:
    selected = copy.deepcopy(contract or _contract())
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
                "parameters_after_optimizer_step": {"a": "parameter"},
            }
        ],
        "execution_contract": selected,
        "execution_contract_sha256": _hash(selected),
    }
    payload["probe_identity"] = compute_probe_identity(payload)
    return payload


@pytest.mark.parametrize("field", sorted(_contract()))
def test_contract_rejects_missing_field_even_after_hash_recompute(field: str) -> None:
    contract = _contract()
    del contract[field]
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_FIELDS_MISMATCH"):
        validate_execution_contract(contract)


def test_contract_rejects_extra_field_even_after_hash_recompute() -> None:
    contract = _contract()
    contract["accepted_target"] = True
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_FIELDS_MISMATCH"):
        validate_execution_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    (("foreach", True), ("fused", True), ("MKL_CBWR", None)),
)
def test_controlled_profile_rejects_incoherent_field(field: str, value: object) -> None:
    contract = _contract("default-single-tensor")
    contract[field] = value
    with pytest.raises(ValueError, match="EXECUTION_CONTRACT_PROFILE_FIELD_MISMATCH"):
        validate_execution_contract(contract)


def test_contract_rejects_optimizer_hyperparameter_mutation() -> None:
    contract = _contract()
    contract["optimizer_hyperparameters"]["lr"] = 0.1
    with pytest.raises(ValueError, match="OPTIMIZER_HYPERPARAMETERS_MISMATCH"):
        validate_execution_contract(contract)


@pytest.mark.parametrize(
    ("profile", "foreach", "fused"),
    (
        ("historical-default", False, None),
        ("default-single-tensor", True, False),
        ("avx2-single-tensor", False, True),
    ),
)
def test_named_profile_flags_are_confirmations_not_overrides(
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    foreach: bool | None,
    fused: bool | None,
) -> None:
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.delenv("MKL_CBWR", raising=False)
    if profile == "default-single-tensor":
        monkeypatch.setenv("ATEN_CPU_CAPABILITY", "default")
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    elif profile == "avx2-single-tensor":
        monkeypatch.setenv("ATEN_CPU_CAPABILITY", "avx2")
        monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    with pytest.raises(RuntimeError, match="EXECUTION_PROFILE_OPTIMIZER_FLAG_MISMATCH"):
        _resolve_profile_before_torch_import(
            profile,
            optimizer_foreach=foreach,
            optimizer_fused=fused,
        )


def test_probe_rejects_stale_identity_after_numerical_mutation() -> None:
    payload = _probe()
    payload["updates"][0]["raw_gradients"]["a"] = "mutated"
    with pytest.raises(ValueError, match="PROBE_IDENTITY_HASH_MISMATCH"):
        validate_probe_identity(payload)


def test_compare_rejects_incompatible_contracts_before_stale_probe_identity() -> None:
    left = _probe(_contract("avx2-single-tensor"))
    right = _probe(_contract("default-single-tensor"))
    right["probe_identity"] = "sha256:" + "0" * 64
    result = compare_probes(left, right)
    assert result["comparable"] is False
    assert result["reason"] == "EXECUTION_CONTRACT_MISMATCH"
    assert result["equal"] is None


def test_same_contract_and_exact_numbers_compare_equal() -> None:
    left = _probe()
    right = copy.deepcopy(left)
    result = compare_probes(left, right)
    assert result["comparable"] is True
    assert result["equal"] is True


def test_same_contract_and_rehashed_mutation_compare_different() -> None:
    left = _probe()
    right = copy.deepcopy(left)
    right["updates"][0]["adamw_exp_avg_sq"]["a"] = "mutated"
    right["probe_identity"] = compute_probe_identity(right)
    result = compare_probes(left, right)
    assert result["comparable"] is True
    assert result["equal"] is False
    assert result["first_divergence"]["stage"] == "adamw_exp_avg_sq"


def test_execution_contract_enters_probe_identity() -> None:
    assert _probe(_contract())["probe_identity"] != _probe(
        _contract("default-single-tensor")
    )["probe_identity"]


@pytest.mark.parametrize(
    "field",
    (
        "parameter_names",
        "optimizer_defaults",
        "loss_components",
        "gradient_clip_max_norm",
        "update_events",
        "parameters_after_optimizer_step",
    ),
)
def test_parity_comparison_detects_semantic_mutation(field: str) -> None:
    baseline = {
        "parameter_names": ["a"],
        "optimizer_defaults": {"lr": 0.0003},
        "loss_components": {"total": "hash"},
        "gradient_clip_max_norm": 1.0,
        "update_events": ["zero_grad", "forward", "loss", "backward", "clip", "step"],
        "parameters_after_optimizer_step": {"a": "hash"},
    }
    mutated = copy.deepcopy(baseline)
    mutated[field] = "mutated"
    assert _first_parity_difference(baseline, mutated) == field
