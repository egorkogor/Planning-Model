from __future__ import annotations

import copy

import pytest
import torch
import yaml

from planner_toy.dataset import generate
from planner_toy.e2e import validate_lineage
from planner_toy.model import LockedA2, canonical_task_encoding
from planner_toy.training import train


def test_locked_inventory_and_initialization() -> None:
    first = LockedA2(17)
    second = LockedA2(17)
    assert len(first.state_dict()) == 177
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()
    )


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


@pytest.mark.parametrize(
    "target,field,value",
    [
        ("plan", "steps", []),
        ("attempts", "attempts", []),
        ("manifest", "planner_seed", 42),
        ("evaluation", "success", False),
    ],
)
def test_real_artifact_mutations_fail_closed(e2e_artifacts, target, field, value) -> None:
    mutated = copy.deepcopy(e2e_artifacts)
    mutated[target][field] = value
    with pytest.raises(ValueError):
        validate_lineage(
            mutated["task"],
            mutated["request"],
            mutated["plan"],
            mutated["manifest"],
            mutated["attempts"],
            mutated["evaluation"],
        )
