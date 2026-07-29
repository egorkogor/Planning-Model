from __future__ import annotations

import json

import pytest

from planner_toy.dataset import generate
from planner_toy.e2e import run


@pytest.fixture(scope="session")
def replay_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("replays")
    one, two = root / "one", root / "two"
    return one, two, run(one), run(two)


@pytest.fixture(scope="session")
def e2e_artifacts(replay_dirs):
    one = replay_dirs[0]

    def load(name):
        return json.loads((one / name).read_bytes())

    request = load("planner-request.json")
    task = next(row for row in generate()["train"] if row["task_id"] == request["task_id"])
    return {
        "task": task,
        "request": request,
        "plan": load("work-plan.json"),
        "manifest": load("episode-plan-manifest.json"),
        "attempts": load("attempt-log.json"),
        "evaluation": load("evaluation-result.json"),
    }
