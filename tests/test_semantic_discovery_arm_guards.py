from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from planner_toy.model import LockedPlanner, canonical_task_encoding
from planner_toy.semantic_discovery_arms import (
    A5Unit,
    a5_mapping_manifest,
    construct_a5_derangement,
    state_dict_sha256,
)
from planner_toy.semantic_discovery_validation import (
    FrozenSameCheckpointA3,
    SemanticDiscoveryValidationError,
    select_frozen_wrong_semantic_donor,
    validate_a3r_training_equivalence,
    validate_normalized_float32_feedback,
)
from research_programs.planner.semantic_feedback_readiness import DonorUnit

GOLDEN = Path("tests/controls/a5_matching_golden.json")


def _training_config(variant: str) -> dict:
    config = {
        "variant": variant,
        "dataset_hash": "sha256:" + "1" * 64,
        "training_task_id": "task-01",
        "training_task_hash": "sha256:" + "2" * 64,
        "inventory_sha256": "sha256:" + "3" * 64,
        "task_encoding_sha256": "sha256:" + "4" * 64,
        "runtime": {"python": "3.12", "torch": "2.12.0+cpu"},
        "code_commit": "a" * 40,
        "training": {
            "steps": 9,
            "learning_rate": 0.001,
            "adamw_betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "semantic_loss_weight": 1.0,
        },
    }
    if variant == "A3r":
        config["a3r"] = {
            "codebook_id": "A3R-CODEBOOK-170029",
            "semantic_signature_sha256s": ["5" * 64],
        }
    return config


def _toy_inputs():
    row = {
        "blocks": ["A", "B"],
        "initial": [
            ["ON_TABLE", "A"],
            ["ON_TABLE", "B"],
            ["CLEAR", "A"],
            ["CLEAR", "B"],
            ["HAND_EMPTY"],
        ],
        "goal": [["ON", "A", "B"]],
    }
    encoded = canonical_task_encoding(row)
    return encoded, torch.tensor([[0, 3]]), torch.tensor([[0, 0]]), torch.tensor([[0, 1]])


def test_a3r_training_equivalence_allows_only_random_target_block():
    a3 = _training_config("A3")
    a3r = _training_config("A3r")
    digest = validate_a3r_training_equivalence(
        a3_config=a3,
        a3r_config=a3r,
        a3_training_row_sha256="sha256:" + "6" * 64,
        a3r_training_row_sha256="sha256:" + "6" * 64,
    )
    assert digest.startswith("sha256:")

    changed = copy.deepcopy(a3r)
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(SemanticDiscoveryValidationError, match="outside frozen"):
        validate_a3r_training_equivalence(
            a3_config=a3,
            a3r_config=changed,
            a3_training_row_sha256="sha256:" + "6" * 64,
            a3r_training_row_sha256="sha256:" + "6" * 64,
        )

    with pytest.raises(SemanticDiscoveryValidationError, match="examples/order"):
        validate_a3r_training_equivalence(
            a3_config=a3,
            a3r_config=a3r,
            a3_training_row_sha256="sha256:" + "6" * 64,
            a3r_training_row_sha256="sha256:" + "7" * 64,
        )


def test_foreign_semantic_feedback_requires_normalized_float32():
    good = torch.nn.functional.normalize(
        torch.ones((1, 2, 384), dtype=torch.float32), dim=-1
    )
    validate_normalized_float32_feedback(good)
    with pytest.raises(SemanticDiscoveryValidationError, match="float32"):
        validate_normalized_float32_feedback(good.double())
    with pytest.raises(SemanticDiscoveryValidationError, match="l2-normalized"):
        validate_normalized_float32_feedback(
            torch.ones((1, 2, 384), dtype=torch.float32)
        )


def test_validated_intervention_wrapper_has_exact_a3_state_and_rejects_bad_foreign_z():
    a3 = LockedPlanner(17, "A3")
    frozen = FrozenSameCheckpointA3(17)
    frozen.load_state_dict(a3.state_dict())
    assert state_dict_sha256(frozen.state_dict()) == state_dict_sha256(a3.state_dict())
    assert set(frozen.state_dict()) == set(a3.state_dict())

    encoded, actions, arg1, arg2 = _toy_inputs()
    feedback = torch.nn.functional.normalize(
        torch.ones((1, 2, 384), dtype=torch.float32), dim=-1
    )
    with pytest.raises(SemanticDiscoveryValidationError, match="l2-normalized"):
        frozen(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="FOREIGN",
            foreign_semantic_feedback=torch.ones((1, 2, 384), dtype=torch.float32),
        )


def test_wrong_semantic_donor_reuses_accepted_filter_order_and_null_state():
    target = DonorUnit(
        "target", "ep", "sig-a", 17, "dev", 3, "near", "empty", "n0", 1.0
    )
    far = DonorUnit(
        "far", "ep-2", "sig-b", 17, "dev", 3, "near", "empty", "n0", 1.2
    )
    near = DonorUnit(
        "near", "ep-3", "sig-c", 17, "dev", 3, "near", "empty", "n0", 1.1
    )
    donor, state = select_frozen_wrong_semantic_donor(target, [far, near])
    assert donor == near
    assert state == "EVALUATED"
    donor, state = select_frozen_wrong_semantic_donor(target, [])
    assert donor is None
    assert state == "NOT_EVALUATED_DONOR_UNAVAILABLE"


def test_frozen_a5_golden_fixture_replays_exactly():
    fixture = json.loads(GOLDEN.read_text(encoding="utf-8"))
    units = [A5Unit(**row) for row in fixture["units"]]
    mapping = construct_a5_derangement(units)
    assert mapping == fixture["expected_mapping"]
    manifest = a5_mapping_manifest(
        units=units,
        mapping=mapping,
        a3_checkpoint_sha256=fixture["a3_checkpoint_sha256"],
    )
    assert manifest["excluded_units"] == []
    assert len(manifest["mappings"]) == len(units)
