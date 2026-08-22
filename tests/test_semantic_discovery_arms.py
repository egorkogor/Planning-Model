from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from planner_toy.model import LockedPlanner, canonical_task_encoding
from planner_toy.semantic_discovery_arms import (
    A5Unit,
    SemanticArmError,
    a3r_codebook_identity_digest,
    a3r_targets,
    a5_mapping_digest,
    construct_a5_derangement,
    parameter_shape_manifest,
    state_dict_sha256,
    validate_a3r_checkpoint_independence,
    validate_exact_a3_checkpoint,
)


def _signature(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _row() -> dict:
    return {
        "blocks": ["A", "B"],
        "initial": [
            ["ON_TABLE", "A"],
            ["ON_TABLE", "B"],
            ["CLEAR", "A"],
            ["CLEAR", "B"],
            ["HAND_EMPTY"],
        ],
        "goal": [["ON", "A", "B"]],
        "oracle_work_plan": [["PICK_UP", "A"], ["STACK", "A", "B"]],
    }


def _inputs():
    row = _row()
    encoded = canonical_task_encoding(row)
    actions = torch.tensor([[0, 3]])
    arg1 = torch.tensor([[0, 0]])
    arg2 = torch.tensor([[0, 1]])
    feedback = torch.nn.functional.normalize(torch.ones((1, 2, 384)), dim=-1)
    foreign = torch.nn.functional.normalize(
        torch.cat([torch.ones((1, 2, 192)), -torch.ones((1, 2, 192))], dim=-1), dim=-1
    )
    return encoded, actions, arg1, arg2, feedback, foreign


def test_a3r_is_parameter_matched_a3_but_has_separate_identity():
    a3 = LockedPlanner(17, "A3")
    a3r = LockedPlanner(17, "A3r")
    assert a3.active_names == a3r.active_names
    assert parameter_shape_manifest(a3.state_dict()) == parameter_shape_manifest(a3r.state_dict())

    identity = validate_a3r_checkpoint_independence(
        a3_state=a3.state_dict(),
        a3r_state=a3r.state_dict(),
        a3_checkpoint_id="A3-checkpoint",
        a3r_checkpoint_id="A3r-checkpoint",
        require_trained_state_difference=False,
    )
    assert identity["a3_state_sha256"] == identity["a3r_state_sha256"]

    with pytest.raises(SemanticArmError, match="separate checkpoint"):
        validate_a3r_checkpoint_independence(
            a3_state=a3.state_dict(),
            a3r_state=a3r.state_dict(),
            a3_checkpoint_id="same",
            a3r_checkpoint_id="same",
            require_trained_state_difference=False,
        )

    with pytest.raises(SemanticArmError, match="reuse"):
        validate_a3r_checkpoint_independence(
            a3_state=a3.state_dict(),
            a3r_state=a3.state_dict(),
            a3_checkpoint_id="A3",
            a3r_checkpoint_id="A3r",
        )


def test_a3r_frozen_codebooks_are_exact_deterministic_and_distinct():
    signatures = [_signature("sig-a"), _signature("sig-b")]
    first = a3r_targets(signatures, codebook_id="A3R-CODEBOOK-170029")
    replay = a3r_targets(signatures, codebook_id="A3R-CODEBOOK-170029")
    second = a3r_targets(signatures, codebook_id="A3R-CODEBOOK-290043")
    assert torch.equal(first, replay)
    assert not torch.equal(first[:, :2], second[:, :2])
    assert first.shape == (1, 17, 384)
    assert a3r_codebook_identity_digest("A3R-CODEBOOK-170029").startswith("sha256:")
    with pytest.raises(Exception):
        a3r_targets(signatures, codebook_id="analyst-choice")


def test_exact_a3_checkpoint_guard_rejects_retraining_or_drift():
    state_hash = "sha256:" + "a" * 64
    validate_exact_a3_checkpoint(
        expected_a3_state_sha256=state_hash,
        arm_state_sha256=state_hash,
        retrained=False,
    )
    with pytest.raises(SemanticArmError, match="INVALID_CHECKPOINT_OR_RETRAINING"):
        validate_exact_a3_checkpoint(
            expected_a3_state_sha256=state_hash,
            arm_state_sha256="sha256:" + "b" * 64,
            retrained=False,
        )
    with pytest.raises(SemanticArmError, match="INVALID_CHECKPOINT_OR_RETRAINING"):
        validate_exact_a3_checkpoint(
            expected_a3_state_sha256=state_hash,
            arm_state_sha256=state_hash,
            retrained=True,
        )


def test_same_checkpoint_zero_matches_historical_a4_surface():
    encoded, actions, arg1, arg2, feedback, _ = _inputs()
    a3 = LockedPlanner(17, "A3")
    a4 = LockedPlanner(17, "A4")
    a4.load_state_dict(a3.state_dict())
    a3.eval()
    a4.eval()
    with torch.inference_mode():
        zero = a3(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="ZERO",
        )
        historical = a4(encoded, actions, arg1, arg2, semantic_feedback=feedback)
    assert torch.equal(zero.semantic_component, historical.semantic_component)
    assert torch.equal(zero.action, historical.action)
    assert torch.equal(zero.arg1, historical.arg1)
    assert torch.equal(zero.arg2, historical.arg2)


def test_foreign_and_wrong_donor_use_exact_a3_projection_surface():
    encoded, actions, arg1, arg2, feedback, foreign = _inputs()
    model = LockedPlanner(17, "A3").eval()
    with torch.inference_mode():
        foreign_out = model(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="FOREIGN",
            foreign_semantic_feedback=foreign,
        )
        wrong_out = model(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="WRONG_SEMANTIC_DONOR",
            foreign_semantic_feedback=foreign,
        )
        expected = model.project_semantic(foreign).clone()
        expected[:, 0] = 0
    assert torch.equal(foreign_out.semantic_component, expected)
    assert torch.equal(wrong_out.semantic_component, expected)

    with pytest.raises(ValueError, match="exact A3"):
        LockedPlanner(17, "A4")(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="ZERO",
        )
    with pytest.raises(ValueError, match="required"):
        model(
            encoded,
            actions,
            arg1,
            arg2,
            semantic_feedback=feedback,
            semantic_intervention="FOREIGN",
        )


def _a5_units() -> list[A5Unit]:
    return [
        A5Unit("u1", "task-a", 17, "dev", 2, "near", "empty", "sig-a", "intent-a", "a" * 64),
        A5Unit("u2", "task-b", 17, "dev", 2, "near", "empty", "sig-b", "intent-b", "b" * 64),
        A5Unit("u3", "task-c", 17, "dev", 2, "near", "empty", "sig-c", "intent-c", "c" * 64),
    ]


def test_a5_perfect_derangement_is_deterministic_complete_and_unique():
    units = _a5_units()
    mapping = construct_a5_derangement(units)
    replay = construct_a5_derangement(reversed(units))
    assert mapping == replay
    assert set(mapping) == {unit.unit_id for unit in units}
    assert set(mapping.values()) == {unit.unit_id for unit in units}
    assert all(source != donor for source, donor in mapping.items())
    assert a5_mapping_digest(mapping) == a5_mapping_digest(replay)


def test_a5_fails_closed_on_relaxed_or_impossible_construction():
    units = _a5_units()
    impossible = [copy.copy(units[0]), copy.copy(units[1])]
    impossible[1] = A5Unit(
        "u2",
        "task-b",
        17,
        "dev",
        2,
        "near",
        "empty",
        "sig-a",
        "intent-b",
        "b" * 64,
    )
    with pytest.raises(SemanticArmError, match="BLOCKED_CONTROL_CONSTRUCTION"):
        construct_a5_derangement(impossible)
    with pytest.raises(SemanticArmError, match="donor-unique"):
        a5_mapping_digest({"u1": "u2", "u3": "u2"})
    with pytest.raises(SemanticArmError, match="self-assignment"):
        a5_mapping_digest({"u1": "u1"})


def test_state_hash_changes_on_any_checkpoint_tensor_drift():
    model = LockedPlanner(17, "A3")
    original = state_dict_sha256(model.state_dict())
    changed = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    first_name = sorted(changed)[0]
    changed[first_name].view(-1)[0] += 1
    assert state_dict_sha256(changed) != original
