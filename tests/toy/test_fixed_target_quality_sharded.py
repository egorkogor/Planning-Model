from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from scripts.fixed_target_quality_sharded import (
    EXECUTION_MODE,
    attempt_identity,
    unit_identity,
    validate_runtime11_optimizer,
)

SCHEMA = json.loads(
    (
        Path(__file__).parents[2] / "planner_toy/schemas/fixed_target_quality_unit.schema.json"
    ).read_text()
)
H = "sha256:" + "1" * 64


def manifest() -> dict:
    return {
        "unit_evidence_version": "toy-quality-fixed-target-sharded-unit/1.0",
        "attempt_identity_sha256": H,
        "implementation_commit": "1" * 40,
        "variant": "A2",
        "seed": 17,
        "dataset_hash": H,
        "ordered_train_task_ids": ["bw-00000001", "bw-00000002", "bw-00000003"],
        "ordered_eval_task_ids": ["bw-00000004", "bw-00000005"],
        "epochs": 3,
        "updates": 9,
        "training_execution_mode": EXECUTION_MODE,
        "optimizer": {
            "class": "AdamW",
            "learning_rate": 0.0003,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "observed_foreach": False,
            "observed_fused": False,
        },
        "runtime_contract_sha256": H,
        "target_observation_sha256": H,
        "source_inventory_sha256": H,
        "observation": {},
        "checkpoint_manifest_sha256": H,
        "task_results_sha256": H,
        "unit_manifest_sha256": H,
    }


def assert_schema_rejects(field: str, value) -> None:
    candidate = manifest()
    if field.startswith("optimizer."):
        candidate["optimizer"][field.split(".")[1]] = value
    else:
        candidate[field] = value
    assert list(Draft202012Validator(SCHEMA).iter_errors(candidate))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variant", "A1"),
        ("seed", 99),
        ("ordered_train_task_ids", []),
        ("ordered_eval_task_ids", []),
        ("epochs", 2),
        ("updates", 8),
        ("optimizer.observed_foreach", None),
        ("optimizer.observed_foreach", True),
        ("optimizer.observed_fused", None),
        ("optimizer.observed_fused", True),
        ("training_execution_mode", "REUSED_FROM_PRIOR_ATTEMPT"),
    ],
)
def test_unit_schema_rejects_contract_drift(field: str, value) -> None:
    assert_schema_rejects(field, value)


def test_unit_schema_is_closed() -> None:
    candidate = manifest()
    candidate["unexpected"] = True
    assert list(Draft202012Validator(SCHEMA).iter_errors(candidate))


def test_unit_manifest_reseal_changes_identity() -> None:
    candidate = manifest()
    first = unit_identity(candidate)
    candidate["variant"] = "A3"
    assert unit_identity(candidate) != first


def test_attempt_nonce_binds_units_but_can_be_excluded_from_numerical_claims() -> None:
    first = {"attempt_nonce": "Q1", "attempt_identity_sha256": ""}
    second = copy.deepcopy(first)
    second["attempt_nonce"] = "Q2"
    assert attempt_identity(first) != attempt_identity(second)


@pytest.mark.parametrize(
    ("foreach", "fused"), [(None, False), (True, False), (False, None), (False, True)]
)
def test_runtime11_optimizer_rejects_non_false(foreach, fused) -> None:
    with pytest.raises(ValueError, match="REQUIRED_FALSE"):
        validate_runtime11_optimizer(SimpleNamespace(defaults={"foreach": foreach, "fused": fused}))


def test_runtime11_optimizer_accepts_explicit_false() -> None:
    validate_runtime11_optimizer(SimpleNamespace(defaults={"foreach": False, "fused": False}))
