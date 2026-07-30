from __future__ import annotations

import json
import shutil

import pytest

from planner_toy.dataset import generate
from planner_toy.e2e import run, validate_run_directory


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_variant_e2e_frozen_execution_and_trace(tmp_path, variant) -> None:
    root = tmp_path / variant
    result = run(root, variant=variant)
    assert result["success"] is True
    assert result["planner_call_count"] == 1
    assert result["variant"] == variant
    trace = json.loads((root / "semantic-trace.json").read_text())
    plan = json.loads((root / "results/development/plans/work-plan.json").read_text())
    assert len(trace["steps"]) == len(plan["steps"]) > 2
    assert trace["feedback_applied"] is (variant == "A3")


def test_semantic_file_coherent_mutation_is_rejected(tmp_path) -> None:
    root = tmp_path / "original"
    run(root, variant="A3")
    changed = tmp_path / "changed"
    shutil.copytree(root, changed)
    latent = changed / "semantic-latents.f32"
    payload = bytearray(latent.read_bytes())
    payload[0] ^= 1
    latent.write_bytes(payload)
    trace_path = changed / "semantic-trace.json"
    trace = json.loads(trace_path.read_text())
    from planner_toy.e2e import file_hash

    trace["latent_file_sha256"] = file_hash(latent)  # coherent downstream rehash
    trace_path.write_text(json.dumps(trace, sort_keys=True, separators=(",", ":")) + "\n")
    request = json.loads((changed / "planner-request.json").read_text())
    task = next(row for row in generate()["train"] if row["task_id"] == request["task_id"])
    with pytest.raises(ValueError, match="SEMANTIC_STEP_HASH"):
        validate_run_directory(changed, task)


def test_variant_mismatch_fails_closed(tmp_path) -> None:
    root = tmp_path / "run"
    run(root, variant="A3")
    request_path = root / "planner-request.json"
    request = json.loads(request_path.read_text())
    request["variant"] = "A2"
    request_path.write_text(json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n")
    task = next(row for row in generate()["train"] if row["task_id"] == request["task_id"])
    with pytest.raises(ValueError, match="VARIANT_MISMATCH"):
        validate_run_directory(root, task)
