from __future__ import annotations

import json

import pytest

from planner_toy.dataset import generate
from planner_toy.e2e import run


@pytest.fixture(scope="module")
def replay_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("replays")
    one, two = root / "one", root / "two"
    return one, two, run(one), run(two)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def e2e_artifacts(replay_dirs):
    root = replay_dirs[0]

    def load(name):
        return json.loads((root / name).read_bytes())

    request = load("planner-request.json")
    task = next(row for row in generate()["train"] if row["task_id"] == request["task_id"])
    return {
        "root": root,
        "task": task,
        "request": request,
        "config": load("development-config.json"),
        "checkpoint": load("checkpoint-manifest.json"),
        "work_plan": load("results/development/plans/work-plan.json"),
        "manifest": load("episode-plan-manifest.json"),
        "attempts": load_jsonl(root / "attempt-log.jsonl"),
        "episode": load("episode-log.json"),
        "evaluation": load("evaluation-result.json"),
    }
