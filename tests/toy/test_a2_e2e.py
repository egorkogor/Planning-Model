from __future__ import annotations

import pytest

from planner_toy.e2e import parse_work_plan


def test_two_clean_replays_are_recursively_byte_identical(replay_dirs) -> None:
    one, two, result_one, result_two = replay_dirs
    assert result_one == result_two
    relative = sorted(path.relative_to(one) for path in one.rglob("*") if path.is_file())
    assert relative
    assert all((one / name).read_bytes() == (two / name).read_bytes() for name in relative)


def test_multistep_plan_executes_and_changes_state(e2e_artifacts) -> None:
    plan = e2e_artifacts["plan"]
    attempts = e2e_artifacts["attempts"]["attempts"]
    assert len(plan["steps"]) == 5  # four actions plus END
    assert len(attempts) == 4
    assert all(item["before"] != item["after"] for item in attempts)
    assert e2e_artifacts["evaluation"]["success"] is True
    assert e2e_artifacts["manifest"]["planner_call_count"] == 1


def test_invalid_plan_is_not_repaired() -> None:
    with pytest.raises(ValueError):
        parse_work_plan([["PICK_UP", "@UNKNOWN"], ["END"]], ["@B0"])
