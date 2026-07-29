from __future__ import annotations

import json

import pytest

from planner_toy.e2e import parse_work_plan, run


def test_two_clean_replays_are_byte_identical(tmp_path) -> None:
    one, two = tmp_path / "one", tmp_path / "two"
    result_one = run(one)
    result_two = run(two)
    assert result_one == result_two
    names = [path.name for path in one.iterdir() if path.is_file()]
    assert names
    assert all((one / name).read_bytes() == (two / name).read_bytes() for name in names)
    assert json.loads((one / "episode-plan-manifest.json").read_text())["planner_call_count"] == 1


def test_invalid_plan_is_not_repaired() -> None:
    with pytest.raises(ValueError):
        parse_work_plan([["PICK_UP", "@UNKNOWN"], ["END"]], ["@B0"])
