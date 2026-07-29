from __future__ import annotations

import torch
import yaml

from planner_toy.dataset import generate
from planner_toy.model import LockedA2, TaskEncoding, canonical_task_encoding
from planner_toy.training import labels, state_dict_sha256, train


def test_locked_inventory_and_initialization() -> None:
    first = LockedA2(17)
    assert len(first.state_dict()) == 177
    first_hash = state_dict_sha256(first.state_dict())
    del first
    second = LockedA2(17)
    assert len(second.state_dict()) == 177
    assert state_dict_sha256(second.state_dict()) == first_hash


def test_task_encoding_matches_locked_golden_vectors() -> None:
    spec = yaml.safe_load(open("docs/architecture/task_encoding_v1.yaml", encoding="utf-8"))
    for example in spec["golden_examples"]:
        encoded = canonical_task_encoding(
            {
                "blocks": example["ledger_refs"],
                "initial": example["initial"],
                "goal": example["goal"],
            }
        )
        assert encoded.token_ids[0].tolist() == example["expected_token_ids"]
        assert encoded.segment_ids[0].tolist() == example["expected_segment_ids"]
        assert (
            encoded.argument_position_ids[0].tolist() == example["expected_argument_position_ids"]
        )
        assert encoded.attention_mask[0].int().tolist() == example["expected_attention_mask"]
        assert (
            dict(zip(example["ledger_refs"], encoded.ref_slot_positions, strict=True))
            == example["expected_ref_slot_positions"]
        )


def test_decoder_cross_attention_cannot_observe_pad_states() -> None:
    row = next(row for row in generate()["train"] if len(row["oracle_work_plan"]) > 1)
    model = LockedA2(17).eval()
    encoded = canonical_task_encoding(row)
    changed = encoded.token_ids.clone()
    changed[~encoded.attention_mask] = 31
    adversarial = TaskEncoding(
        changed,
        encoded.segment_ids,
        encoded.argument_position_ids,
        encoded.attention_mask,
        encoded.ref_slot_positions,
    )
    action, arg1, arg2 = labels(row)
    with torch.no_grad():
        expected = model(encoded, action, arg1, arg2)
        actual = model(adversarial, action, arg1, arg2)
    assert torch.equal(expected.action, actual.action)
    assert torch.equal(expected.arg1, actual.arg1)
    assert torch.equal(expected.arg2, actual.arg2)


def test_real_training_active_and_dormant_policy(tmp_path) -> None:
    row = next(row for row in generate()["train"] if len(row["oracle_work_plan"]) > 1)
    _, report = train(row, tmp_path, steps=1)
    assert report["active_tensor_count"] == 140
    assert report["dormant_tensor_count"] == 37
    assert report["active_grad_count"] == 140
    assert report["active_changed_count"] == 140
    assert report["dormant_grad_none"] is True
    assert report["dormant_byte_equal"] is True
    assert report["optimizer_nonzero_state"] is True
    assert report["optimizer_betas"] == [0.9, 0.95]
    assert report["active_gradients_all_finite_nonzero"] is True
    assert report["optimizer_state_matches_active_set"] is True
    assert report["optimizer_state_all_finite_nonzero"] is True
