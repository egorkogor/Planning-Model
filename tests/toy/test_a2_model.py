from __future__ import annotations

import copy

import pytest
import torch

from planner_toy.dataset import generate
from planner_toy.e2e import validate_lineage
from planner_toy.model import LockedA2
from planner_toy.training import train


def test_locked_inventory_and_initialization() -> None:
    first = LockedA2(17)
    second = LockedA2(17)
    assert len(first.state_dict()) == 177
    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items()
    )


def test_real_training_active_and_dormant_policy(tmp_path) -> None:
    row = next(
        row
        for row in generate()["train"] + generate()["validation"]
        if row["oracle_work_plan"] == [["END"]]
    )
    _, report = train(row, tmp_path, steps=1)
    assert report["active_tensor_count"] == 140
    assert report["dormant_tensor_count"] == 37
    assert report["active_grad_count"] == 140
    assert report["active_changed_count"] == 140
    assert report["dormant_grad_none"] is True
    assert report["dormant_byte_equal"] is True
    assert report["optimizer_nonzero_state"] is True


@pytest.mark.parametrize(
    "target,path",
    [
        ("attempts", ("plan_sha256",)),
        ("evaluation", ("attempt_log_sha256",)),
        ("manifest", ("seed",)),
    ],
)
def test_lineage_mutations_fail_closed(target, path) -> None:
    manifest = {"work_plan_sha256": "sha256:plan", "run_class": "DEVELOPMENT/TOY", "seed": 17}
    attempts = {"plan_sha256": "sha256:plan", "attempts": []}
    from planner_toy.canonical import sha256

    evaluation = {"attempt_log_sha256": sha256(attempts)}
    values = {"manifest": manifest, "attempts": attempts, "evaluation": evaluation}
    mutated = copy.deepcopy(values)
    mutated[target][path[0]] = "mutated" if path[0] != "seed" else 42
    with pytest.raises(ValueError):
        validate_lineage(mutated["manifest"], mutated["attempts"], mutated["evaluation"])
