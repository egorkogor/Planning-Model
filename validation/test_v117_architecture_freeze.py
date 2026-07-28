from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from validation.full_plan_lineage_validator import validate_lineage_index
from validation.hashing import hash_json
from validation.test_v115_launch_minimum import _planner_lineage_fixture

ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_a1_and_step_arms_share_one_85_position_decoder_inventory() -> None:
    architecture = yaml.safe_load((ROOT / "docs/architecture/planner_architecture_v1.yaml").read_text())
    grammar = yaml.safe_load((ROOT / "docs/architecture/a1_token_grammar_v1.yaml").read_text())
    decoder = architecture["planner_decoder"]
    assert decoder["max_decoder_positions"] == 85
    assert decoder["variant_position_limits"]["A1_token_grammar"] == 85
    assert decoder["variant_position_limits"]["A2_A2b_A2c_A3_A3r_A4_A5_step_level"] == 17
    assert architecture["heads"]["token_grammar"] == "linear_256_to_24"
    assert grammar["architecture"]["common_decoder_max_positions"] == 85
    assert grammar["architecture"]["step_level_active_positions"] == 17


def test_loss_masks_have_no_duplicate_end_loss_and_define_empty_batch_behavior() -> None:
    contract = yaml.safe_load((ROOT / "docs/training/planner_training_contract_v1.yaml").read_text())
    losses = contract["losses"]
    assert "end_ce" not in losses["common"]
    assert "including terminal END" in losses["masking"]["action_ce"]
    assert losses["masking"]["arg2_pointer_ce"] == "UNSTACK and STACK positions only"
    assert losses["reduction"]["per_head"] == "mean over valid targets for that head"
    assert losses["contrastive_edge_case"]["batch_without_eligible_anchors"] == "contrastive loss is exact zero with no gradient"
    assert losses["A1"] == {"token_grammar_ce": 1.0}
    assert losses["active_heads_by_trainable_arm"]["A1"] == ["token_grammar_ce"]
    assert losses["active_heads_by_trainable_arm"]["A3"][-2:] == ["cosine", "supervised_multi_positive_contrastive"]


def test_size_ood_blocks_are_forbidden_in_training_everywhere() -> None:
    corpus = yaml.safe_load((ROOT / "docs/domain/training_corpus_contract_v1.yaml").read_text())
    assert corpus["off_policy_expansion"]["block_count_7_8"] == "forbidden_in_training"
    spec = (ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.18.md").read_text()
    assert "для n=7,8 — 8 walks" not in spec
    assert "deterministic deviations/walks для n=7,8" not in spec
    assert "n=7,8 полностью запрещены в training/development expansion" in spec


def test_flops_sensitivity_is_train_flops_matched_not_inference_retraining() -> None:
    scientific = yaml.safe_load((ROOT / "docs/architecture/planner_scientific_contract_v1.yaml").read_text())
    training = yaml.safe_load((ROOT / "docs/training/planner_training_contract_v1.yaml").read_text())
    regimes = scientific["compute_matching"]["regimes"]
    assert "inference_FLOPs_matched" not in regimes
    assert "measured_train_flops_matched" in regimes
    assert training["compute_comparisons"]["sensitivity"]["name"] == "measured_train_flops_matched"
    assert scientific["compute_matching"]["inference_flops"].startswith("guardrail_only")


def test_failed_plan_artifact_cannot_be_reused_for_another_seed(tmp_path: Path) -> None:
    index = _planner_lineage_fixture(tmp_path)
    assert not validate_lineage_index(tmp_path, index, expected_stage="PLANNER")

    rows = [row for row in index["records"] if row["arm"] == "PLANNER_A4_RAW"]
    row_101 = next(row for row in rows if row["planner_seed"] == 101)
    row_202 = next(row for row in rows if row["planner_seed"] == 202)

    manifest_path = tmp_path / row_101["episode_plan_manifest"]
    episode_path = tmp_path / row_101["episode_log"]
    attempts_path = tmp_path / row_101["attempt_log"]
    manifest = json.loads(manifest_path.read_text())
    episode = json.loads(episode_path.read_text())

    manifest.update(
        planner_seed=101,
        plan_status="FAILED",
        work_plan_content_hash=None,
        work_plan_artifact_hash=None,
        work_plan_path=None,
        generation_failure_code="SEMANTIC_UNRESOLVED",
    )
    manifest["manifest_hash"] = hash_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    manifest_path.write_text(json.dumps(manifest))
    episode.update(
        planner_seed=101,
        episode_plan_manifest_hash=manifest["manifest_hash"],
        plan_generation_status="FAILED",
        goal_success=False,
        terminal_error="SEMANTIC_UNRESOLVED",
        attempts_total=0,
        steps_accepted=0,
        executed_length=0,
        plan_positions_consumed=0,
    )
    episode_path.write_text(json.dumps(episode))
    attempts_path.write_text("")
    row_101["work_plan"] = None

    # Reuse the exact same FAILED artifact paths as the alleged result for seed 202.
    row_202.update(
        episode_plan_manifest=row_101["episode_plan_manifest"],
        work_plan=None,
        episode_log=row_101["episode_log"],
        attempt_log=row_101["attempt_log"],
    )
    index["index_hash"] = hash_json({k: v for k, v in index.items() if k != "index_hash"})
    errors = validate_lineage_index(tmp_path, index, expected_stage="PLANNER")
    assert any("EpisodePlanManifest planner_seed" in error for error in errors)
    assert any("EpisodeLog planner_seed" in error for error in errors)
