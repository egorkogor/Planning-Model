from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import yaml

from analysis.sample_size import _draw_requirement
from validation import phase_check_runner

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(rel: str) -> dict:
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_phase_lock_checker_uses_declared_argument_order() -> None:
    source = inspect.getsource(phase_check_runner.core_check)
    assert 'verify_lock("scientific", lock, report.get("run_id"))' in source
    assert 'verify_lock("implementation", lock, report.get("run_id"))' in source
    assert 'verify_lock(lock,"scientific")' not in source


def test_p02_bundle_check_runs_static_validation_and_pytest() -> None:
    contract = load_yaml("docs/operator/phase_check_contract_v1.yaml")
    assert contract["core_checks"]["P02_exec_02"] == "bundle_tests"
    source = inspect.getsource(phase_check_runner.core_check)
    assert '"validation/validate_bundle.py", "--skip-nested-pytest"' in source
    assert '"validation/run_test_suite.py"' in source


def test_numpy_pin_is_single_and_byte_reproducible() -> None:
    generator = load_yaml("docs/domain/generator_contract_v1.yaml")
    runtime = load_yaml("docs/infrastructure/runtime_dependency_contract_v1.yaml")
    locked = next(line for line in (ROOT / "requirements.lock").read_text().splitlines() if line.startswith("numpy=="))
    assert generator["numpy_requirement"] == locked
    assert str(runtime["validation_environment"]["required_packages"]["numpy"]) == locked.split("==", 1)[1]


def test_hyperparameter_selection_is_identical_in_both_contracts() -> None:
    search = load_yaml("docs/training/hyperparameter_search_v1.yaml")["selection"]
    training = load_yaml("docs/training/planner_training_contract_v1.yaml")["hyperparameter_selection_policy"]
    for key in (
        "per_arm_validation_composite",
        "arms_ranked",
        "primary_score",
        "ranking_tie_policy",
        "eligibility_floors",
        "tie_breaks",
    ):
        assert search[key] == training[key]


def test_parameter_matching_is_exact_everywhere() -> None:
    a1 = load_yaml("docs/architecture/a1_token_grammar_v1.yaml")
    arch = load_yaml("docs/architecture/planner_architecture_v1.yaml")
    sci = load_yaml("docs/architecture/planner_scientific_contract_v1.yaml")
    assert a1["compute_accounting"]["parameter_tolerance_fraction"] == 0.0
    assert arch["parameter_matching"]["total_parameter_count_tolerance_fraction"] == 0.0
    assert sci["compute_matching"]["parameter_count_tolerance_fraction"] == 0.0


def _pair_rows() -> list[dict]:
    pattern = [(0, 1), (0, 0), (1, 1), (1, 0)] * 10
    return [
        {"pair_id": f"p{i}", "left": left, "right": right, "difference": left - right}
        for i, (left, right) in enumerate(pattern)
    ]


def test_sample_size_generator_stays_in_paired_binary_support() -> None:
    requirement = {
        "comparison": "E1-E0",
        "analysis_type": "TASK_PAIRED",
        "complete_pair_count": 40,
        "pairs": _pair_rows(),
    }
    drawn = _draw_requirement(requirement, 500, np.random.default_rng(17), 0.075)
    assert set(drawn["pairs"]) <= {-1.0, 0.0, 1.0}
    assert all(value != 0.075 for value in drawn["pairs"])


def test_stage1a_reserve_and_hidden_stage1b_certification_are_precommitted() -> None:
    split = load_yaml("docs/data/dataset_split_contract_v1.yaml")
    assert split["partitions"]["stage1a_confirmatory_reserve"]["target_base_tasks"] >= 4000
    sealing = load_yaml("docs/controls/confirmatory_sealing_contract_v1.yaml")
    cert = sealing["sealer_protocol"]["stage1b_hidden_control_certification"]
    assert cert["owner"] == "DATA_SEALER"
    assert cert["selection_basis"] == "TASK_AND_DOMAIN_METADATA_ONLY"
    assert cert["degeneracy_exclusion_count"] == 0
    schema = json.loads((ROOT / "docs/schemas/sealer_manifest.schema.json").read_text())
    assert "control_certification" in schema["properties"]
    assert any("control_certification" in branch.get("then", {}).get("required", []) for branch in schema["allOf"])


def test_few_shot_candidates_define_exact_chat_message_sequence() -> None:
    for candidate_id in ("C01", "C02", "C03", "C04"):
        candidate = load_yaml(f"docs/prompt/candidates/{candidate_id}.yaml")
        sequence = candidate["message_sequence_template_exact"]
        assert sequence[0]["role"] == "system"
        assert sequence[-1] == {"role": "user", "content_source": "RUNTIME_USER_PROMPT_UTF8"}
        if candidate_id in {"C02", "C04"}:
            assert [row["role"] for row in sequence] == ["system", "user", "assistant", "user", "assistant", "user"]
        else:
            assert [row["role"] for row in sequence] == ["system", "user"]


def test_intent_labeler_is_scientifically_locked_and_not_patchable() -> None:
    scientific = set(load_yaml("docs/operator/scientific_lock_v1.yaml")["protected_paths"])
    allowed = set(load_yaml("docs/operator/implementation_lock_v1.yaml")["pre_lock_patch_window"]["allowed_path_globs"])
    assert "docs/domain/**" in scientific
    assert "docs/domain/**" not in allowed
    spec = (ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.20.md").read_text(encoding="utf-8")
    assert "не могут изменяться implementation-only patch" in spec


def test_stage1b_sealer_semantics_fail_closed_on_counts_and_unbound_contracts() -> None:
    from validation.sealer_manifest_validator import validate_sealer_manifest_semantics
    h = lambda c: "sha256:" + c * 64
    obj = {
        "stage": "STAGE1B", "task_count": 100,
        "selected_task_manifest_path": "sealed/stage1b-confirmatory/selected-task-manifest.json",
        "selected_task_manifest_sha256": h("e"),
        "contract_hashes": {"eligibility": h("a"), "split": h("b"), "generator": h("c")},
        "control_certification": {
            "candidate_task_count": 120, "eligible_task_count": 110, "selected_task_count": 100,
            "selection_basis": "TASK_AND_DOMAIN_METADATA_ONLY",
            "planner_output_used_for_selection": False, "llm_output_used_for_selection": False,
            "arm_outcome_used_for_selection": False, "plan_or_control_degeneracy_exclusion_count": 0,
            "plan_generation_failure_policy": "RETAIN_AS_ZERO_SUCCESS_PAIRED_OUTCOME",
            "control_degeneracy_policy": "RETAIN_AS_ZERO_SUCCESS_PAIRED_OUTCOME",
            "all_selected_tasks_task_only_eligible": True,
            "certification_completed_before_outcome_access": True,
            "eligibility_contract_sha256": h("a"), "split_contract_sha256": h("b"),
            "generator_contract_sha256": h("c"),
            "task_only_selection_manifest_sha256": h("e"),
        },
    }
    assert not validate_sealer_manifest_semantics(obj)
    obj["control_certification"]["eligible_task_count"] = 90
    assert any("candidate >= eligible >= selected" in e for e in validate_sealer_manifest_semantics(obj))
    obj["control_certification"]["eligible_task_count"] = 110
    obj["control_certification"]["planner_output_used_for_selection"] = True
    assert any("planner_output_used_for_selection" in e for e in validate_sealer_manifest_semantics(obj))
    obj["control_certification"]["planner_output_used_for_selection"] = False
    obj["control_certification"]["generator_contract_sha256"] = h("d")
    assert any("not bound" in e for e in validate_sealer_manifest_semantics(obj))
