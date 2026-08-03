from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from planner_toy.canonical import canonical_bytes, sha256
from planner_toy.e2e import (
    A2Planner,
    file_hash,
    toy_hash,
    validate_frozen_plan_lineage_core,
    validate_persisted_quality_evidence,
)
from planner_toy.model import LockedPlanner
from planner_toy.quality import (
    MAPPING,
    _validate_optimizer_state,
    export_compact,
    paired,
    run,
    state_dict_sha256,
    summarize,
    validate_evaluation,
    validate_task_result_semantics,
)


@pytest.fixture(scope="session")
def canonical_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality-base") / "run"
    run(root, variants=("A2",), seeds=(17,), max_eval_tasks=1)
    return root


@pytest.fixture(scope="session")
def all_three_smoke(tmp_path_factory):
    root = tmp_path_factory.mktemp("quality-all-three") / "run"
    run(root, variants=("A2", "A3", "A4"), seeds=(17,), max_eval_tasks=1)
    return root


def copied_run(tmp_path, canonical_smoke):
    root = tmp_path / "run"
    shutil.copytree(canonical_smoke, root)
    return root


def rehash_run(root, name):
    manifest_path = root / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib
    manifest["artifact_hashes"][name] = (
        "sha256:" + hashlib.sha256((root / name).read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    (root / "replay-hash.txt").write_text(sha256(manifest) + "\n")


def rehash_evidence_and_result(root: Path, variant: str) -> None:
    result_path = root / "task-results.jsonl"
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    row = next(item for item in rows if item["variant"] == variant)
    evidence = root / row["evidence_root"]
    row["evidence_hash"] = toy_hash(
        "quality_evidence",
        {path.name: file_hash(path) for path in sorted(evidence.iterdir()) if path.is_file()},
    )
    result_path.write_bytes(b"".join(canonical_bytes(item) + b"\n" for item in rows))
    manifest_path = root / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in sorted(evidence.iterdir()):
        if path.is_file():
            manifest["evidence_artifact_hashes"][str(path.relative_to(root))] = file_hash(path)
    manifest["artifact_hashes"]["task-results.jsonl"] = file_hash(result_path)
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    (root / "replay-hash.txt").write_text(sha256(manifest) + "\n")


def validate_one_persisted_evidence(root: Path, variant: str) -> None:
    from planner_toy.dataset import generate

    evidence = next((root / f"evidence/{variant}/seed-17").iterdir())
    task_id = evidence.name
    task = next(row for row in generate(17)["validation"] if row["task_id"] == task_id)
    checkpoint = json.loads(
        (root / f"training-runs/{variant}/seed-17/checkpoint-manifest.json").read_text()
    )
    validate_persisted_quality_evidence(root=evidence, task=task, checkpoint=checkpoint)


@pytest.fixture
def nonempty_quality_evidence(tmp_path):
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = generate(17)["validation"][0]

    class FixedPlanner:
        model = SimpleNamespace(variant="A2")
        calls = 0
        model_forward_count = 0
        semantic_steps = None
        semantic_audit = None

        def plan(self, _row):
            self.calls += 1
            self.model_forward_count = len(task["oracle_work_plan"])
            return task["oracle_work_plan"]

    checkpoint = {
        "trained_state_dict_sha256": "sha256:" + "1" * 64,
        "trained_file_sha256": "sha256:" + "2" * 64,
        "variant_identity": {"implementation_variant": "A2"},
    }
    root = tmp_path / "evidence"
    evaluate_frozen_plan(
        row=task, planner=FixedPlanner(), output=root, checkpoint_binding=checkpoint,
    )
    return root, task, checkpoint


def rewrite_quality_lineage_hashes(root: Path) -> None:
    evaluation_path = root / "evaluation-result.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["attempt_log_hash"] = file_hash(root / "attempt-log.jsonl")
    evaluation["episode_log_hash"] = file_hash(root / "episode-log.json")
    evaluation_path.write_bytes(canonical_bytes(evaluation) + b"\n")


def test_common_initialization_is_byte_identical() -> None:
    hashes = [state_dict_sha256(LockedPlanner(29, variant).state_dict()) for variant in MAPPING]
    assert len(set(hashes)) == 1


def test_handcrafted_metrics_and_paired_matrix() -> None:
    base = {
        "predicted_plan_length": 2, "gold_plan_length": 1, "plan_generation_success": True,
        "terminal_end_produced": True, "action_parse_valid": True,
        "action_applicability": [True, True], "full_plan_executable": True,
        "exact_plan_match": False, "failure_code": None,
        "action_attempt_count": 2, "applicable_action_count": 2,
        "nonempty_plan": True, "end_only_plan": False,
        "execution_completed_without_precondition_failure": True,
    }
    rows = [
        {**base, "variant": "A3", "seed": 17, "task_id": "x", "goal_reached": True},
        {**base, "variant": "A2", "seed": 17, "task_id": "x", "goal_reached": False,
         "failure_code": "GOAL_NOT_ACHIEVED"},
    ]
    assert summarize(rows[:1])["mean_absolute_plan_length_difference"] == 1
    assert paired(rows, "A3", "A2")["only_first_succeeds"] == 1


def test_deterministic_smoke_and_public_mutation_rejection(tmp_path, canonical_smoke) -> None:
    assert (canonical_smoke / "replay-hash.txt").is_file()  # run() already validated it
    damaged = tmp_path / "damaged"
    shutil.copytree(canonical_smoke, damaged)
    row = json.loads((damaged / "task-results.jsonl").read_text())
    row["goal_reached"] = not row["goal_reached"]
    (damaged / "task-results.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError):
        validate_evaluation(damaged)


def test_runtime_fingerprint_mutation_is_rejected(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    config_path = root / "evaluation-config.json"
    config = json.loads(config_path.read_text())
    config["canonical_cpu_runtime"]["torch_num_threads"] = 2
    config_path.write_bytes(canonical_bytes(config) + b"\n")
    with pytest.raises(ValueError, match="CANONICAL_CPU_RUNTIME_PROFILE_MISMATCH"):
        validate_evaluation(root)


def test_training_configures_runtime_before_model_construction() -> None:
    import inspect

    from planner_toy.quality import _train

    source = inspect.getsource(_train)
    assert source.index("configure_canonical_cpu_runtime(seed)") < source.index(
        "LockedPlanner(seed, variant)"
    )


def test_split_overlap_fails_closed(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    config_path = root / "evaluation-config.json"
    config = json.loads(config_path.read_text())
    config["train_task_ids"].append(config["eval_task_ids"][0])
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_evaluation(root)


class _ScriptedModel:
    def __init__(self, actions: list[int], variant: str = "A2"):
        self.actions, self.variant, self.index = actions, variant, 0

    def __call__(self, *args, **kwargs):
        action = torch.full((1, 17, 5), -10.0)
        action[0, self.index, self.actions[min(self.index, len(self.actions) - 1)]] = 10.0
        self.index += 1
        z = (
            torch.nn.functional.normalize(torch.ones(1, 17, 384), dim=-1)
            if self.variant != "A2"
            else None
        )
        return SimpleNamespace(
            action=action,
            arg1=torch.zeros(1, 17, 3),
            arg2=torch.zeros(1, 17, 3),
            z_semantic=z,
            projected_semantic=(torch.ones(1, 17, 256) if z is not None else None),
            semantic_component=(
                torch.zeros(1, 17, 256)
                if self.variant == "A4"
                else torch.ones(1, 17, 256)
                if z is not None
                else None
            ),
        )


@pytest.mark.parametrize("actions,expected", [([4], 1), ([0, 4], 2), ([0, 2, 4], 3)])
def test_model_forward_count(actions, expected) -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel(actions))
    planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == expected


@pytest.mark.parametrize("variant", ["A2", "A3", "A4"])
def test_forward_count_policy_same_for_variants(variant) -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel([4], variant))
    planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == 1


def test_no_end_counts_all_17_forwards() -> None:
    from planner_toy.dataset import generate
    planner = A2Planner(_ScriptedModel([0]))
    with pytest.raises(ValueError, match="PLAN_NO_END"):
        planner.plan(generate()["validation"][0])
    assert planner.model_forward_count == 17


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_reached", True), ("terminal_end_produced", False),
        ("generated_action_count", 1), ("action_attempt_count", 1),
        ("applicable_action_count", 1), ("model_forward_count", 99),
        ("planner_call_count", 2), ("replanning_count", 1),
        ("experimental_arm", "A3a-zero"), ("architecture_stage", "STAGE_2A"),
        ("target_type", "DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK"),
        ("seed", 43), ("task_id", "unknown"),
    ],
)
def test_semantic_task_result_mutations_fail(canonical_smoke, field, value) -> None:
    reproduced = json.loads((canonical_smoke / "task-results.jsonl").read_text())
    persisted = dict(reproduced)
    persisted[field] = value
    with pytest.raises(ValueError, match="SEMANTIC_REPLAY_MISMATCH"):
        validate_task_result_semantics(persisted, reproduced)


@pytest.mark.parametrize("name", ["initialization.pt", "trained.pt", "training-report.json"])
def test_missing_training_lineage_fails(tmp_path, canonical_smoke, name) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    (root / "training-runs/A2/seed-17" / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_evaluation(root)


@pytest.mark.parametrize("field", ["dataset_manifest_hash", "train_task_ids", "eval_task_ids"])
def test_canonical_dataset_mutations_fail(tmp_path, canonical_smoke, field) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "evaluation-config.json"
    config = json.loads(path.read_text())
    config[field] = (
        "sha256:" + "0" * 64
        if field == "dataset_manifest_hash"
        else list(reversed(config[field])) + (["unknown"] if len(config[field]) == 1 else [])
    )
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        validate_evaluation(root)


def test_all_three_variant_real_smoke(all_three_smoke) -> None:
    root = all_three_smoke
    config = json.loads((root / "evaluation-config.json").read_text())
    assert config["diagnostic_complete"] is False
    rows = [json.loads(line) for line in (root / "task-results.jsonl").read_text().splitlines()]
    assert {row["variant"] for row in rows} == {"A2", "A3", "A4"}
    assert all(row["planner_call_count"] == 1 and row["replanning_count"] == 0 for row in rows)
    assert not (root / rows[0]["evidence_root"] / "semantic-trace.json").exists()
    for row in rows[1:]:
        trace = json.loads((root / row["evidence_root"] / "semantic-trace.json").read_text())
        assert trace["feedback_source"] == "predicted"
        assert trace["feedback_application_count"] == 0
        assert trace["nonzero_feedback_application_count"] == 0
        assert trace["nonzero_downstream_semantic_component_count"] == 0
        assert trace["downstream_semantic_component_observed"] is False
        assert trace["feedback_mode_enabled"] == (row["variant"] == "A3")
        assert trace["compute_then_zero"] == (row["variant"] == "A4")


def test_incomplete_export_rejected(tmp_path, canonical_smoke) -> None:
    with pytest.raises(ValueError, match="COMPLETE_CANONICAL"):
        export_compact(canonical_smoke, tmp_path / "docs")


def test_reuse_rejects_incomplete_source(tmp_path, canonical_smoke) -> None:
    reused = tmp_path / "reused"
    with pytest.raises(ValueError, match="COMPLETE_CANONICAL_SOURCE"):
        run(
            reused, variants=("A2",), seeds=(17,), max_eval_tasks=1,
            reuse_checkpoint_root=canonical_smoke,
        )


def test_self_consistent_markdown_claim_rejected(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "human-readable-examples.md"
    path.write_text(path.read_text().replace("Goal reached: `false`", "Goal reached: `true`"))
    rehash_run(root, "human-readable-examples.md")
    with pytest.raises(ValueError, match="HUMAN_EXAMPLES"):
        validate_evaluation(root)


def test_self_consistent_dataset_manifest_rejected(tmp_path, canonical_smoke) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    path = root / "dataset-manifest.json"
    value = json.loads(path.read_text())
    value["dataset_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(value) + "\n")
    rehash_run(root, "dataset-manifest.json")
    with pytest.raises(ValueError, match="DATASET_MANIFEST"):
        validate_evaluation(root)


def test_plan_no_end_retains_partial_output() -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import PlannerGenerationFailure
    planner = A2Planner(_ScriptedModel([0]))
    with pytest.raises(PlannerGenerationFailure) as caught:
        planner.plan(generate()["validation"][0])
    assert len(caught.value.partial_raw_output) == 17
    assert caught.value.model_forward_count == 17


def test_plan_no_end_persists_generation_failure_without_workplan(tmp_path) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan
    planner = A2Planner(_ScriptedModel([0]))
    result = evaluate_frozen_plan(
        row=generate()["validation"][0], planner=planner, output=tmp_path / "evidence",
        checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "0" * 64,
                            "trained_file_sha256": "sha256:" + "1" * 64},
    )
    assert result["failure_code"] == "PLAN_NO_END"
    assert len(result["raw_output"]) == 17
    assert result["model_forward_count"] == 17
    assert not (tmp_path / "evidence/work-plan.json").exists()
    assert (tmp_path / "evidence/attempt-log.jsonl").read_text() == ""


def test_quality_adapter_cannot_bypass_shared_lineage_core(tmp_path, monkeypatch) -> None:
    from planner_toy import e2e
    from planner_toy.dataset import generate
    called = []
    original = e2e.validate_frozen_plan_lineage_core
    def spy(**kwargs):
        called.append(kwargs["variant"])
        return original(**kwargs)
    monkeypatch.setattr(e2e, "validate_frozen_plan_lineage_core", spy)
    e2e.evaluate_frozen_plan(
        row=generate()["validation"][0], planner=A2Planner(_ScriptedModel([4])),
        output=tmp_path / "evidence",
        checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "0" * 64,
                            "trained_file_sha256": "sha256:" + "1" * 64},
    )
    assert called == ["A2"]


def test_task_result_reordering_rejected(tmp_path, all_three_smoke) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    path = root / "task-results.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n")
    rehash_run(root, "task-results.jsonl")
    with pytest.raises(ValueError, match="CANONICAL_ORDER"):
        validate_evaluation(root)


def test_zero_max_eval_tasks_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        run(tmp_path / "run", max_eval_tasks=0)


def test_cli_rejects_noncanonical_seed(tmp_path) -> None:
    import subprocess
    result = subprocess.run(
        ["python", "-m", "scripts.run_toy_quality_evaluation", "--output-dir",
         str(tmp_path / "run"), "--seeds", "99"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_missing_implementation_commit_rejected_for_complete(tmp_path, monkeypatch) -> None:
    from planner_toy import quality
    monkeypatch.setattr(quality, "implementation_provenance", lambda commit: (_ for _ in ()).throw(
        AssertionError("provenance must not be resolved before explicit commit check")
    ))
    with pytest.raises(ValueError, match="requires --implementation-commit"):
        run(tmp_path / "run", implementation_commit=None)


@pytest.mark.parametrize("commit", ["not-a-sha", "0" * 40], ids=["malformed", "missing"])
def test_invalid_implementation_commit_rejected(commit) -> None:
    from planner_toy.quality import implementation_provenance
    with pytest.raises(ValueError, match="existing commit"):
        implementation_provenance(commit)


def test_requirements_hash_is_read_from_implementation_tree() -> None:
    import hashlib
    import subprocess

    from planner_toy.quality import implementation_provenance
    expected = subprocess.run(
        ["git", "show", "HEAD:requirements.lock"], capture_output=True, check=True,
    ).stdout
    provenance = implementation_provenance("HEAD")
    expected_hash = "sha256:" + hashlib.sha256(expected).hexdigest()
    assert provenance["requirements_lock_sha256"] == expected_hash
    assert set(provenance["runtime_versions"]) == {
        "python", "torch", "numpy", "cuda_version", "cuda_available", "execution_device",
    }


def test_ci_uses_full_history_and_explicit_provenance_preflight() -> None:
    workflow = (Path(__file__).parents[2] / ".github/workflows/ci.yml").read_text()
    assert "fetch-depth: 0" in workflow
    assert 'git cat-file -e "${IMPLEMENTATION_SHA}^{commit}"' in workflow
    assert 'git merge-base --is-ancestor "$IMPLEMENTATION_SHA" HEAD' in workflow
    assert 'git show "$IMPLEMENTATION_SHA:requirements.lock"' in workflow


def test_shared_lineage_core_rejects_reordered_attempts() -> None:
    from planner_toy.e2e import validate_frozen_plan_lineage_core

    plan = {"steps": [
        {"step_index": 0, "action": "PICK_UP", "args": ["A"]},
        {"step_index": 1, "action": "PUT_DOWN", "args": ["A"]},
        {"step_index": 2, "action": "END", "args": []},
    ]}
    attempts = [
        {"step_index": 1, "candidate_action": ["PUT_DOWN", "A"],
         "state_before_hash": "one", "state_after_hash": "two", "status": "APPLIED"},
        {"step_index": 0, "candidate_action": ["PICK_UP", "A"],
         "state_before_hash": "two", "state_after_hash": "three", "status": "APPLIED"},
    ]
    with pytest.raises(ValueError, match="frozen WorkPlan"):
        validate_frozen_plan_lineage_core(
            variant="A2", planner_calls=1, replanning_count=0,
            work_plan=plan, attempts=attempts, evaluation={"replanning_count": 0},
            semantic_trace=None,
        )


def test_shared_lineage_core_rejects_broken_state_chain() -> None:
    from planner_toy.e2e import validate_frozen_plan_lineage_core

    plan = {"steps": [
        {"step_index": 0, "action": "PICK_UP", "args": ["A"]},
        {"step_index": 1, "action": "PUT_DOWN", "args": ["A"]},
        {"step_index": 2, "action": "END", "args": []},
    ]}
    attempts = [
        {"step_index": 0, "candidate_action": ["PICK_UP", "A"],
         "state_before_hash": "one", "state_after_hash": "two", "status": "APPLIED"},
        {"step_index": 1, "candidate_action": ["PUT_DOWN", "A"],
         "state_before_hash": "not-two", "state_after_hash": "three", "status": "APPLIED"},
    ]
    with pytest.raises(ValueError, match="state transition"):
        validate_frozen_plan_lineage_core(
            variant="A2", planner_calls=1, replanning_count=0,
            work_plan=plan, attempts=attempts, evaluation={"replanning_count": 0},
            semantic_trace=None,
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda trace: trace.update(control_audit=[]), "QUALITY_SEMANTIC_LENGTH_MISMATCH"),
        (
            lambda trace: trace["steps"][0].update(previous_z_sha256="sha256:" + "0" * 64),
            "QUALITY_LATENT_BLOCK_MISMATCH",
        ),
        (
            lambda trace: trace["steps"][0].update(latent_norm=0.25),
            "QUALITY_LATENT_BLOCK_MISMATCH",
        ),
        (
            lambda trace: trace.update(feedback_source="foreign"),
            "QUALITY_SCHEMA_INVALID:semantic-trace.json",
        ),
    ],
    ids=["empty-audit", "previous-z", "latent-norm", "foreign-source"],
)
def test_persisted_semantic_mutations_fail_before_replay(
    tmp_path, all_three_smoke, mutation, error,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    trace_path = next((root / "evidence/A3/seed-17").glob("*/semantic-trace.json"))
    trace = json.loads(trace_path.read_text())
    mutation(trace)
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    rehash_evidence_and_result(root, "A3")
    with pytest.raises(ValueError, match=error):
        validate_one_persisted_evidence(root, "A3")


@pytest.mark.parametrize("suffix", [b"\x00", b"\x00" * 1536], ids=["one-byte", "one-block"])
def test_persisted_latent_extra_bytes_fail_before_replay(
    tmp_path, all_three_smoke, suffix,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    latent_path = next((root / "evidence/A3/seed-17").glob("*/semantic-latents.f32"))
    latent_path.write_bytes(latent_path.read_bytes() + suffix)
    trace_path = latent_path.with_name("semantic-trace.json")
    trace = json.loads(trace_path.read_text())
    trace["latent_file_sha256"] = file_hash(latent_path)
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    rehash_evidence_and_result(root, "A3")
    with pytest.raises(ValueError, match="QUALITY_LATENT_FILE_MISMATCH"):
        validate_one_persisted_evidence(root, "A3")


def test_persisted_latent_truncation_fails_before_replay(
    tmp_path, all_three_smoke,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    latent_path = next((root / "evidence/A3/seed-17").glob("*/semantic-latents.f32"))
    latent_path.write_bytes(latent_path.read_bytes()[:-4])
    trace_path = latent_path.with_name("semantic-trace.json")
    trace = json.loads(trace_path.read_text())
    trace["latent_file_sha256"] = file_hash(latent_path)
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    rehash_evidence_and_result(root, "A3")
    with pytest.raises(ValueError, match="QUALITY_LATENT_FILE_MISMATCH"):
        validate_one_persisted_evidence(root, "A3")


def test_persisted_a4_nonzero_downstream_fails_before_replay(
    tmp_path, all_three_smoke,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    trace_path = next((root / "evidence/A4/seed-17").glob("*/semantic-trace.json"))
    trace = json.loads(trace_path.read_text())
    trace["control_audit"][0]["downstream_component_zero"] = False
    trace["control_audit"][0]["downstream_semantic_component_norm"] = 1.0
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    rehash_evidence_and_result(root, "A4")
    with pytest.raises(ValueError, match="QUALITY_DOWNSTREAM_ZERO_FLAG_MISMATCH"):
        validate_one_persisted_evidence(root, "A4")


@pytest.mark.parametrize(
    "action",
    [
        ["END", "@B0"], ["PICK_UP"], ["PICK_UP", "@B0", "@B1"],
        ["STACK", "@B0"], ["STACK", "@B0", "@B1", "@B2"],
    ],
    ids=["end-extra", "pickup-missing", "pickup-extra", "stack-missing", "stack-extra"],
)
def test_work_plan_wrong_arity_rejected(action) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    steps = [{"step_index": 0, "action": action[0], "args": action[1:]}]
    if action[0] != "END":
        steps.append({"step_index": 1, "action": "END", "args": []})
    work_plan = {
        "schema_version": "toy-quality-work-plan/1.0", "variant": "A2",
        "task_id": "task", "state_hash": "sha256:" + "0" * 64,
        "steps": steps, "plan_content_hash": "sha256:" + "1" * 64,
    }
    schema = json.loads(
        (Path(__file__).parents[2] / "planner_toy/schemas/"
         "toy_quality_work_plan.schema.json").read_text()
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(work_plan)
    with pytest.raises(ValueError, match="ACTION_ARITY"):
        validate_frozen_plan_lineage_core(
            variant="A2", planner_calls=1, replanning_count=0,
            work_plan=work_plan, attempts=[],
            evaluation={"replanning_count": 0}, semantic_trace=None,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"plan_status": "READY", "failure_code": "PLAN_NO_END"},
        {"plan_status": "READY", "work_plan_path": "foreign.json"},
        {"plan_status": "READY", "work_plan_hash": None},
        {"plan_status": "FAILED", "work_plan_path": "work-plan.json"},
        {"plan_status": "FAILED", "work_plan_hash": "sha256:" + "1" * 64},
        {"plan_status": "FAILED", "failure_code": "EXECUTOR_PRECONDITION_FAILED"},
    ],
    ids=["ready-failure", "ready-path", "ready-null-hash", "failed-path",
         "failed-hash", "failed-executor-code"],
)
def test_episode_manifest_conditionals_are_schema_enforced(changes) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads(
        (Path(__file__).parents[2] / "planner_toy/schemas/"
         "toy_quality_episode_plan_manifest.schema.json").read_text()
    )
    manifest = {
        "schema_version": "toy-quality-episode-plan-manifest/1.0",
        "variant": "A2",
        "planner_call_count": 1, "replanning_count": 0, "model_forward_count": 1,
        "plan_status": "READY", "work_plan_path": "work-plan.json",
        "work_plan_hash": "sha256:" + "0" * 64, "failure_code": None,
        "partial_raw_output": None,
    }
    manifest.update(changes)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(manifest)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("self-consistent-state", "QUALITY_EXECUTOR_APPLIED_MISMATCH"),
        ("changed-action", "attempt does not match frozen WorkPlan"),
        ("false-failed", "QUALITY_EXECUTOR_APPLIED_MISMATCH"),
        ("wrong-error", "QUALITY_EXECUTOR_APPLIED_MISMATCH"),
        ("continue-after-failed", "QUALITY_EXECUTOR_APPLIED_MISMATCH"),
    ],
)
def test_persisted_attempts_are_domain_replayed(
    nonempty_quality_evidence, mutation, error,
) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    attempts_path = root / "attempt-log.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()]
    episode_path = root / "episode-log.json"
    episode = json.loads(episode_path.read_text())
    if mutation == "self-consistent-state":
        fake = "sha256:" + "f" * 64
        attempts[0]["state_after_hash"] = fake
        attempts[1]["state_before_hash"] = fake
    elif mutation == "changed-action":
        attempts[0]["candidate_action"] = ["PICK_UP", "@B0"]
    elif mutation == "false-failed":
        attempts[0].update(status="FAILED", error="PRECONDITION_CLEAR")
        attempts = attempts[:1]
        episode.update(attempts_total=1, executed_length=0,
                       final_state_hash=attempts[0]["state_after_hash"])
    elif mutation == "wrong-error":
        attempts[0].update(status="FAILED", error="WRONG")
        attempts = attempts[:1]
        episode.update(attempts_total=1, executed_length=0,
                       final_state_hash=attempts[0]["state_after_hash"])
    else:
        attempts[0].update(status="FAILED", error="WRONG")
    attempts_path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in attempts))
    episode_path.write_bytes(canonical_bytes(episode) + b"\n")
    rewrite_quality_lineage_hashes(root)
    with pytest.raises(ValueError, match=error):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("delete-first", "attempt does not match frozen WorkPlan"),
        ("delete-middle", "attempt does not match frozen WorkPlan"),
        ("reorder", "attempt does not match frozen WorkPlan"),
        ("args", "attempt does not match frozen WorkPlan"),
        ("step-index", "attempt does not match frozen WorkPlan"),
        ("extra", "attempt count exceeds frozen WorkPlan"),
        ("truncate", "execution stopped before frozen WorkPlan"),
        ("initial-state", "QUALITY_EXECUTOR_STATE_BEFORE_MISMATCH"),
    ],
)
def test_persisted_attempt_sequence_mutations_fail(
    nonempty_quality_evidence, mutation, error,
) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    path = root / "attempt-log.jsonl"
    attempts = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "delete-first":
        attempts.pop(0)
    elif mutation == "delete-middle":
        attempts.pop(len(attempts) // 2)
    elif mutation == "reorder":
        attempts[0], attempts[1] = attempts[1], attempts[0]
    elif mutation == "args":
        attempts[0]["candidate_action"][-1] = "@B2"
    elif mutation == "step-index":
        attempts[0]["step_index"] = 9
    elif mutation == "extra":
        attempts.append(dict(attempts[-1], step_index=len(attempts)))
    elif mutation == "truncate":
        attempts.pop()
    else:
        attempts[0]["state_before_hash"] = "sha256:" + "f" * 64
    path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in attempts))
    rewrite_quality_lineage_hashes(root)
    with pytest.raises(ValueError, match=error):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "artifact,field,value,error",
    [
        ("episode-log.json", "attempts_total", 99, "QUALITY_EPISODE_COUNTS_MISMATCH"),
        ("episode-log.json", "executed_length", 99, "QUALITY_EPISODE_COUNTS_MISMATCH"),
        ("episode-log.json", "final_state_hash", "sha256:" + "f" * 64,
         "QUALITY_EPISODE_COUNTS_MISMATCH"),
        ("episode-log.json", "goal_success", False, "QUALITY_EPISODE_COUNTS_MISMATCH"),
        ("evaluation-result.json", "failure_code", "GOAL_NOT_ACHIEVED",
         "QUALITY_FAILURE_LINEAGE_MISMATCH"),
        ("episode-log.json", "planner_calls", 2, "QUALITY_SCHEMA_INVALID"),
        ("evaluation-result.json", "model_forward_count", 99,
         "QUALITY_MODEL_FORWARD_COUNT_MISMATCH"),
        ("evaluation-result.json", "attempt_log_hash", "sha256:" + "f" * 64,
         "QUALITY_ATTEMPT_LOG_HASH_MISMATCH"),
    ],
)
def test_episode_evaluation_mutations_fail_before_replay(
    nonempty_quality_evidence, artifact, field, value, error,
) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    path = root / artifact
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_bytes(canonical_bytes(payload) + b"\n")
    if artifact == "episode-log.json":
        evaluation_path = root / "evaluation-result.json"
        evaluation = json.loads(evaluation_path.read_text())
        evaluation["episode_log_hash"] = file_hash(path)
        evaluation_path.write_bytes(canonical_bytes(evaluation) + b"\n")
    with pytest.raises(ValueError, match=error):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


def test_semantically_equivalent_checkpoint_reserialization_preserves_canonical_identity(
    tmp_path, canonical_smoke,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(canonical_smoke, root)
    original_replay = (root / "replay-hash.txt").read_bytes()
    checkpoint_dir = root / "training-runs/A2/seed-17"
    checkpoint_path = checkpoint_dir / "checkpoint-manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    raw_hash_changes = []
    for filename, hash_field in [
        ("initialization.pt", "initialization_file_sha256"),
        ("trained.pt", "trained_file_sha256"),
        ("optimizer-state.pt", "optimizer_state_file_sha256"),
    ]:
        path = checkpoint_dir / filename
        value = torch.load(path, map_location="cpu", weights_only=True)
        before_raw = file_hash(path)
        torch.save(value, path, _use_new_zipfile_serialization=False)
        raw_hash_changes.append((before_raw, file_hash(path)))
        checkpoint[hash_field] = file_hash(path)
        if filename != "optimizer-state.pt":
            assert state_dict_sha256(value) == checkpoint[
                "initialization_state_dict_sha256"
                if filename == "initialization.pt" else "trained_state_dict_sha256"
            ]
    checkpoint_path.write_bytes(canonical_bytes(checkpoint) + b"\n")
    manifest_path = root / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path in checkpoint_dir.iterdir():
        if path.is_file():
            manifest["checkpoint_manifest_hashes"][str(path.relative_to(root))] = file_hash(path)
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    assert all(before != after for before, after in raw_hash_changes)
    assert (root / "replay-hash.txt").read_bytes() == original_replay
    assert validate_evaluation(root)["replay_hash"] == original_replay.decode().strip()


def test_raw_checkpoint_tamper_still_fails_local_integrity(tmp_path, canonical_smoke) -> None:
    root = tmp_path / "run"
    shutil.copytree(canonical_smoke, root)
    path = root / "training-runs/A2/seed-17/trained.pt"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="CHECKPOINT_FILE_HASH_MISMATCH"):
        validate_evaluation(root)


def test_executor_failure_after_goal_is_not_success(tmp_path) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = generate(17)["validation"][0]
    output = [*task["oracle_work_plan"][:-1], ["PICK_UP", "@B2"], ["END"]]

    class GoalThenFailPlanner:
        model = SimpleNamespace(variant="A2")
        calls = 0
        model_forward_count = 0
        semantic_steps = None
        semantic_audit = None

        def plan(self, _row):
            self.calls += 1
            self.model_forward_count = len(output)
            return output

    checkpoint = {
        "trained_state_dict_sha256": "sha256:" + "1" * 64,
        "trained_file_sha256": "sha256:" + "2" * 64,
        "variant_identity": {"implementation_variant": "A2"},
    }
    root = tmp_path / "evidence"
    result = evaluate_frozen_plan(
        row=task, planner=GoalThenFailPlanner(), output=root,
        checkpoint_binding=checkpoint,
    )
    assert result["failure_code"] == "EXECUTOR_PRECONDITION_FAILED"
    assert result["goal_reached"] is False
    validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)

    episode_path = root / "episode-log.json"
    episode = json.loads(episode_path.read_text())
    episode["goal_success"] = True
    episode_path.write_bytes(canonical_bytes(episode) + b"\n")
    evaluation_path = root / "evaluation-result.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["success"] = True
    evaluation["episode_log_hash"] = file_hash(episode_path)
    evaluation_path.write_bytes(canonical_bytes(evaluation) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_EPISODE_COUNTS_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "target,manifest_count,evaluation_count",
    [
        ("both", 99, 99),
        ("manifest-only", 99, None),
        ("evaluation-only", None, 99),
    ],
)
def test_ready_forward_counts_are_bound_to_frozen_plan(
    nonempty_quality_evidence, target, manifest_count, evaluation_count,
) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    if manifest_count is not None:
        path = root / "episode-plan-manifest.json"
        value = json.loads(path.read_text())
        value["model_forward_count"] = manifest_count
        path.write_bytes(canonical_bytes(value) + b"\n")
    if evaluation_count is not None:
        path = root / "evaluation-result.json"
        value = json.loads(path.read_text())
        value["model_forward_count"] = evaluation_count
        path.write_bytes(canonical_bytes(value) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_MODEL_FORWARD_COUNT_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_failed_semantic_plan_no_end_has_complete_forward_evidence(tmp_path, variant) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = generate(17)["validation"][0]
    checkpoint = {
        "trained_state_dict_sha256": "sha256:" + "1" * 64,
        "trained_file_sha256": "sha256:" + "2" * 64,
        "variant_identity": {"implementation_variant": variant},
    }
    root = tmp_path / variant
    result = evaluate_frozen_plan(
        row=task, planner=A2Planner(_ScriptedModel([0], variant)), output=root,
        checkpoint_binding=checkpoint,
    )
    trace = json.loads((root / "semantic-trace.json").read_text())
    manifest = json.loads((root / "episode-plan-manifest.json").read_text())
    assert result["failure_code"] == "PLAN_NO_END"
    assert result["model_forward_count"] == 17
    assert manifest["partial_raw_output"] == result["raw_output"]
    assert len(trace["steps"]) == len(trace["control_audit"]) == 17
    assert not (root / "work-plan.json").exists()
    assert (root / "attempt-log.jsonl").read_text() == ""
    validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)

    manifest["model_forward_count"] = 16
    manifest_path = root / "episode-plan-manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    evaluation_path = root / "evaluation-result.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["model_forward_count"] = 16
    evaluation_path.write_bytes(canonical_bytes(evaluation) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_MODEL_FORWARD_COUNT_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "variant,failure", [("A3", "LATENT_NONFINITE"), ("A4", "LATENT_DIMENSION")]
)
def test_semantic_pre_forward_failure_has_empty_trace(tmp_path, variant, failure) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = generate(17)["validation"][0]
    checkpoint = {
        "trained_state_dict_sha256": "sha256:" + "1" * 64,
        "trained_file_sha256": "sha256:" + "2" * 64,
        "variant_identity": {"implementation_variant": variant},
    }
    planner = A2Planner(_ScriptedModel([4], variant), injected_failure=failure)
    root = tmp_path / variant
    result = evaluate_frozen_plan(
        row=task, planner=planner, output=root, checkpoint_binding=checkpoint,
    )
    trace = json.loads((root / "semantic-trace.json").read_text())
    assert result["model_forward_count"] == 0
    assert trace["steps"] == trace["control_audit"] == []
    validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


def _rewrite_checkpoint_hash_chain(root: Path, checkpoint_path: Path) -> None:
    manifest_path = root / "evaluation-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    training_dir = checkpoint_path.parent
    for path in training_dir.iterdir():
        if path.is_file():
            manifest["checkpoint_manifest_hashes"][str(path.relative_to(root))] = file_hash(path)
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("missing-state", "OPTIMIZER_TOP_LEVEL_STRUCTURE_MISMATCH"),
        ("extra-state", "OPTIMIZER_STATE_PARAMETER_SET_MISMATCH"),
        ("duplicate-id", "OPTIMIZER_PARAMETER_ID_DUPLICATE"),
        ("reordered-ids", "OPTIMIZER_PARAMETER_ID_ORDER_MISMATCH"),
        ("unknown-id", "OPTIMIZER_PARAMETER_ID_ORDER_MISMATCH"),
        ("group-extra-field", "OPTIMIZER_PARAMETER_GROUP_FIELDS_MISMATCH"),
        ("group-wrong-config", "OPTIMIZER_PARAMETER_GROUP_CONFIG_MISMATCH"),
        ("missing-exp-avg", "OPTIMIZER_PARAMETER_STATE_FIELDS_MISMATCH"),
        ("missing-exp-avg-sq", "OPTIMIZER_PARAMETER_STATE_FIELDS_MISMATCH"),
        ("extra-field", "OPTIMIZER_PARAMETER_STATE_FIELDS_MISMATCH"),
        ("wrong-shape", "OPTIMIZER_MOMENT_SHAPE_MISMATCH"),
        ("wrong-dtype", "OPTIMIZER_MOMENT_DTYPE_MISMATCH"),
        ("nan", "OPTIMIZER_MOMENT_NONFINITE"),
        ("positive-inf", "OPTIMIZER_MOMENT_NONFINITE"),
        ("negative-inf", "OPTIMIZER_MOMENT_NONFINITE"),
        ("wrong-step", "OPTIMIZER_STEP_MISMATCH"),
    ],
)
def test_optimizer_state_semantic_mutations_are_rejected(
    tmp_path, canonical_smoke, mutation, error,
) -> None:
    del tmp_path
    run_dir = canonical_smoke / "training-runs/A2/seed-17"
    state_path = run_dir / "optimizer-state.pt"
    optimizer = torch.load(state_path, map_location="cpu", weights_only=True)
    parameter_id = optimizer["param_groups"][0]["params"][0]
    entry = optimizer["state"][parameter_id]
    if mutation == "missing-state":
        del optimizer["state"]
    elif mutation == "extra-state":
        optimizer["state"][999999] = dict(entry)
    elif mutation == "duplicate-id":
        optimizer["param_groups"][0]["params"].append(parameter_id)
    elif mutation == "reordered-ids":
        optimizer["param_groups"][0]["params"][:2] = reversed(
            optimizer["param_groups"][0]["params"][:2]
        )
    elif mutation == "unknown-id":
        optimizer["param_groups"][0]["params"][0] = 999999
    elif mutation == "group-extra-field":
        optimizer["param_groups"][0]["foreign"] = True
    elif mutation == "group-wrong-config":
        optimizer["param_groups"][0]["lr"] = 1.0
    elif mutation == "missing-exp-avg":
        del entry["exp_avg"]
    elif mutation == "missing-exp-avg-sq":
        del entry["exp_avg_sq"]
    elif mutation == "extra-field":
        entry["foreign"] = torch.tensor(0)
    elif mutation == "wrong-shape":
        entry["exp_avg"] = entry["exp_avg"].reshape(-1)[:1]
    elif mutation == "wrong-dtype":
        entry["exp_avg"] = entry["exp_avg"].to(torch.float64)
    elif mutation in {"nan", "positive-inf", "negative-inf"}:
        value = {"nan": float("nan"), "positive-inf": float("inf"),
                 "negative-inf": -float("inf")}[mutation]
        entry["exp_avg"].reshape(-1)[0] = value
    elif mutation == "wrong-step":
        entry["step"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match=error):
        _validate_optimizer_state(optimizer, LockedPlanner(17, "A2").cpu(), 9)


@pytest.mark.parametrize(
    "update_semantic_hash", [False, True], ids=["raw-only", "raw-and-semantic"]
)
def test_changed_trained_tensor_cannot_be_hidden_by_rehash(
    tmp_path, canonical_smoke, update_semantic_hash,
) -> None:
    root = copied_run(tmp_path, canonical_smoke)
    run_dir = root / "training-runs/A2/seed-17"
    trained_path = run_dir / "trained.pt"
    checkpoint_path = run_dir / "checkpoint-manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    state = torch.load(trained_path, map_location="cpu", weights_only=True)
    active_name = checkpoint["active_parameter_names"][0]
    state[active_name] = state[active_name].clone()
    state[active_name].view(-1)[0] += 1.0
    torch.save(state, trained_path)
    checkpoint["trained_file_sha256"] = file_hash(trained_path)
    if update_semantic_hash:
        checkpoint["trained_state_dict_sha256"] = state_dict_sha256(state)
    checkpoint_path.write_bytes(canonical_bytes(checkpoint) + b"\n")
    _rewrite_checkpoint_hash_chain(root, checkpoint_path)
    expected = (
        "CHECKPOINT_CANONICAL_NUMERIC_HASH_MISMATCH"
        if update_semantic_hash else "CHECKPOINT_STATE_HASH_MISMATCH"
    )
    with pytest.raises(ValueError, match=expected):
        validate_evaluation(root, force_training_replay=True)


@pytest.mark.parametrize("variant", ["A3", "A4"])
def test_ready_semantic_forward_count_mutation_is_independently_rejected(
    tmp_path, all_three_smoke, variant,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    evidence = next((root / f"evidence/{variant}/seed-17").iterdir())
    for name in ("episode-plan-manifest.json", "evaluation-result.json"):
        path = evidence / name
        value = json.loads(path.read_text())
        value["model_forward_count"] = 99
        path.write_bytes(canonical_bytes(value) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_MODEL_FORWARD_COUNT_MISMATCH"):
        validate_one_persisted_evidence(root, variant)


@pytest.mark.parametrize(
    "variant,failure", [("A2", "PLAN_NO_END"), ("A3", "PLAN_NO_END"),
                        ("A4", "LATENT_NONFINITE")],
)
def test_generation_failure_cannot_succeed_for_initially_satisfied_goal(
    tmp_path, variant, failure,
) -> None:
    from planner_toy.canonical import canonical_task_hash
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = dict(generate(17)["validation"][0])
    task["goal"] = [task["initial"][0]]
    task["canonical_task_hash"] = canonical_task_hash({
        "domain_id": "blocks_world_v1", "blocks": task["blocks"],
        "initial": task["initial"], "goal": task["goal"],
    })
    planner = A2Planner(
        _ScriptedModel([0] if failure == "PLAN_NO_END" else [4], variant),
        injected_failure=None if failure == "PLAN_NO_END" else failure,
    )
    checkpoint = {
        "trained_state_dict_sha256": "sha256:" + "1" * 64,
        "trained_file_sha256": "sha256:" + "2" * 64,
        "variant_identity": {"implementation_variant": variant},
    }
    root = tmp_path / variant
    result = evaluate_frozen_plan(
        row=task, planner=planner, output=root, checkpoint_binding=checkpoint,
    )
    assert result["goal_reached"] is False
    validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)
    episode_path = root / "episode-log.json"
    episode = json.loads(episode_path.read_text())
    episode["goal_success"] = True
    episode_path.write_bytes(canonical_bytes(episode) + b"\n")
    evaluation_path = root / "evaluation-result.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["success"] = True
    evaluation["episode_log_hash"] = file_hash(episode_path)
    evaluation_path.write_bytes(canonical_bytes(evaluation) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_EPISODE_COUNTS_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize("failure", ["LATENT_NONFINITE", "LATENT_ZERO_NORM", "LATENT_DIMENSION"])
def test_a2_latent_generation_failure_is_rejected(tmp_path, failure) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads((
        Path(__file__).parents[2] / "planner_toy/schemas/"
        "toy_quality_episode_plan_manifest.schema.json"
    ).read_text())
    manifest = {
        "schema_version": "toy-quality-episode-plan-manifest/1.0", "variant": "A2",
        "planner_call_count": 1, "replanning_count": 0, "model_forward_count": 0,
        "plan_status": "FAILED", "work_plan_path": None, "work_plan_hash": None,
        "failure_code": failure, "partial_raw_output": [],
    }
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(manifest)


def test_unknown_planner_value_error_is_not_persisted(tmp_path) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    class BrokenPlanner:
        model = SimpleNamespace(variant="A2")
        calls = 0
        model_forward_count = 0
        semantic_steps = None
        semantic_audit = None

        def plan(self, _row):
            self.calls += 1
            raise ValueError("RANDOM_INTERNAL_ERROR")

    root = tmp_path / "evidence"
    with pytest.raises(ValueError, match="QUALITY_UNKNOWN_GENERATION_EXCEPTION"):
        evaluate_frozen_plan(
            row=generate(17)["validation"][0], planner=BrokenPlanner(), output=root,
            checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "1" * 64,
                                "trained_file_sha256": "sha256:" + "2" * 64},
        )
    assert not (root / "episode-plan-manifest.json").exists()


@pytest.mark.parametrize("length", [18, 100])
def test_work_plan_schema_rejects_decoding_budget_excess(length) -> None:
    from jsonschema import Draft202012Validator, ValidationError

    schema = json.loads((
        Path(__file__).parents[2] / "planner_toy/schemas/toy_quality_work_plan.schema.json"
    ).read_text())
    steps = [
        {"step_index": index, "action": "PICK_UP", "args": ["@B0"]}
        for index in range(length - 1)
    ] + [{"step_index": length - 1, "action": "END", "args": []}]
    value = {"schema_version": "toy-quality-work-plan/1.0", "variant": "A2",
             "task_id": "x", "state_hash": "sha256:" + "0" * 64,
             "steps": steps, "plan_content_hash": "sha256:" + "1" * 64}
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(value)


@pytest.mark.parametrize("mutation", ["trace-action", "trace-args", "trace-delete", "partial"])
def test_failed_semantic_trace_is_bound_to_partial_output(tmp_path, mutation) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    task = generate(17)["validation"][0]
    checkpoint = {"trained_state_dict_sha256": "sha256:" + "1" * 64,
                  "trained_file_sha256": "sha256:" + "2" * 64,
                  "variant_identity": {"implementation_variant": "A3"}}
    root = tmp_path / "evidence"
    evaluate_frozen_plan(row=task, planner=A2Planner(_ScriptedModel([0], "A3")),
                         output=root, checkpoint_binding=checkpoint)
    trace_path = root / "semantic-trace.json"
    trace = json.loads(trace_path.read_text())
    manifest_path = root / "episode-plan-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "trace-action":
        trace["steps"][0]["action"] = "PUT_DOWN"
    elif mutation == "trace-args":
        trace["steps"][0]["args"] = ["@B1"]
    elif mutation == "trace-delete":
        trace["steps"].pop()
        trace["control_audit"].pop()
    else:
        manifest["partial_raw_output"][0] = ["PUT_DOWN", "@B0"]
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_(PARTIAL_TRACE|SEMANTIC_LENGTH)_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize("mutation", ["input-norm", "projected-norm", "projected-byte", "truncate"])
def test_feedback_tensor_evidence_mutations_are_rejected(
    tmp_path, all_three_smoke, mutation,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    evidence = next((root / "evidence/A3/seed-17").iterdir())
    trace_path = evidence / "semantic-trace.json"
    trace = json.loads(trace_path.read_text())
    projected = evidence / "projected-feedback.f32"
    if mutation == "input-norm":
        trace["control_audit"][0]["input_feedback_norm"] = 1.0
    elif mutation == "projected-norm":
        trace["control_audit"][0]["projected_feedback_norm"] = 1.0
    else:
        payload = bytearray(projected.read_bytes())
        if mutation == "projected-byte":
            payload[0] ^= 1
        else:
            payload = payload[:-4]
        projected.write_bytes(payload)
        trace["projected_file_sha256"] = file_hash(projected)
    trace_path.write_bytes(canonical_bytes(trace) + b"\n")
    expected = (
        "QUALITY_INPUT_FEEDBACK_NORM_MISMATCH"
        if mutation == "input-norm" else "QUALITY_PROJECTED_FEEDBACK_MISMATCH"
    )
    with pytest.raises(ValueError, match=expected):
        validate_one_persisted_evidence(root, "A3")


def test_work_plan_variant_is_bound_to_request(nonempty_quality_evidence) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    plan_path = root / "work-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["variant"] = "A3"
    plan["plan_content_hash"] = toy_hash(
        "quality_frozen_plan", {k: v for k, v in plan.items() if k != "plan_content_hash"}
    )
    plan_path.write_bytes(canonical_bytes(plan) + b"\n")
    manifest_path = root / "episode-plan-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["work_plan_hash"] = plan["plan_content_hash"]
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_VARIANT_BINDING_MISMATCH"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize("variant,foreign", [("A3", "A4"), ("A4", "A3")])
def test_semantic_trace_variant_is_bound_to_request(
    tmp_path, all_three_smoke, variant, foreign,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    evidence = next((root / f"evidence/{variant}/seed-17").iterdir())
    path = evidence / "semantic-trace.json"
    trace = json.loads(path.read_text())
    trace["variant"] = foreign
    path.write_bytes(canonical_bytes(trace) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_VARIANT_BINDING_MISMATCH"):
        validate_one_persisted_evidence(root, variant)


@pytest.mark.parametrize(
    "action",
    [
        ["PICK_UP", "@UNKNOWN"], ["PUT_DOWN", "@UNKNOWN"],
        ["UNSTACK", "@B0", "@UNKNOWN"], ["STACK", "@UNKNOWN", "@B1"],
        ["STACK", "@B0", "@UNKNOWN"],
    ],
)
def test_ready_work_plan_unknown_refs_rejected_before_executor(
    nonempty_quality_evidence, action,
) -> None:
    root, task, checkpoint = nonempty_quality_evidence
    plan_path = root / "work-plan.json"
    plan = json.loads(plan_path.read_text())
    plan["steps"][0] = {"step_index": 0, "action": action[0], "args": action[1:]}
    plan["plan_content_hash"] = toy_hash(
        "quality_frozen_plan", {k: v for k, v in plan.items() if k != "plan_content_hash"}
    )
    plan_path.write_bytes(canonical_bytes(plan) + b"\n")
    manifest_path = root / "episode-plan-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["work_plan_hash"] = plan["plan_content_hash"]
    manifest_path.write_bytes(canonical_bytes(manifest) + b"\n")
    with pytest.raises(ValueError, match="QUALITY_WORK_PLAN_UNKNOWN_REF"):
        validate_persisted_quality_evidence(root=root, task=task, checkpoint=checkpoint)


@pytest.mark.parametrize(
    "variant,field,value,error",
    [
        ("A3", "projected_feedback_present", False, "QUALITY_PROJECTED_FEEDBACK_MISMATCH"),
        ("A4", "projected_feedback_present", False, "QUALITY_PROJECTED_FEEDBACK_MISMATCH"),
        ("A3", "downstream_component_zero", False,
         "QUALITY_DOWNSTREAM_ZERO_FLAG_MISMATCH"),
        ("A4", "downstream_component_zero", False,
         "QUALITY_DOWNSTREAM_ZERO_FLAG_MISMATCH"),
    ],
)
def test_semantic_audit_flags_are_derived(
    tmp_path, all_three_smoke, variant, field, value, error,
) -> None:
    root = tmp_path / "run"
    shutil.copytree(all_three_smoke, root)
    evidence = next((root / f"evidence/{variant}/seed-17").iterdir())
    path = evidence / "semantic-trace.json"
    trace = json.loads(path.read_text())
    trace["control_audit"][0][field] = value
    path.write_bytes(canonical_bytes(trace) + b"\n")
    with pytest.raises(ValueError, match=error):
        validate_one_persisted_evidence(root, variant)


def test_unexpected_runtime_error_never_creates_manifest(tmp_path) -> None:
    from planner_toy.dataset import generate
    from planner_toy.e2e import evaluate_frozen_plan

    class BrokenPlanner:
        model = SimpleNamespace(variant="A2")
        calls = 0
        model_forward_count = 0

        def plan(self, _row):
            self.calls += 1
            raise RuntimeError("unexpected model failure")

    root = tmp_path / "evidence"
    with pytest.raises(RuntimeError, match="unexpected model failure"):
        evaluate_frozen_plan(
            row=generate(17)["validation"][0], planner=BrokenPlanner(), output=root,
            checkpoint_binding={"trained_state_dict_sha256": "sha256:" + "1" * 64,
                                "trained_file_sha256": "sha256:" + "2" * 64},
        )
    assert not (root / "episode-plan-manifest.json").exists()
