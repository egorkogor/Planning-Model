from __future__ import annotations

import yaml

from analysis.decision_gates import evaluate_gate, paired_tost
from analysis.sample_size import (
    _draw_requirement,
    calculate_components_from_structured_requirements,
    paired_sd,
)
from validation.code_fingerprint import analysis_code_digest
from validation.statistics_validator import (
    summarize_comparison,
    validate_analysis_input,
    validate_sample_size_report,
    validate_scientific_decision,
)

REQS = ("primary_ci", "primary_power", "current_vs_shuffled_power", "random_code_power", "structured_noninferiority_power", "self_plan_power", "flops_direction_power")

COMPARISONS = {
    "PLANNER": {"primary_ci":"A3-A1_HORIZON","primary_power":"A3-A1_HORIZON","current_vs_shuffled_power":"A3-A4_HORIZON"},
    "STAGE1A": {"primary_ci":"I1-I0","primary_power":"I1-I0","current_vs_shuffled_power":"I1-I2"},
    "STAGE1B": {"primary_ci":"E1-E0","primary_power":"E1-E0","current_vs_shuffled_power":"E1-E2","random_code_power":"E1-E3","structured_noninferiority_power":"E1-E4","self_plan_power":"E1-E5","flops_direction_power":"E1_FLOPS-E0_FLOPS"},
}


def _pair_rows(name: str, values: list[float]) -> list[dict]:
    rows = []
    for i, x in enumerate(values):
        if x == 1:
            left, right = 1, 0
        elif x == -1:
            left, right = 0, 1
        elif x == 0:
            left = right = i % 2
        else:
            # AnalysisInput supports fractional task-level metrics; sample-size
            # fixtures use only paired-binary rows.
            left, right = max(float(x), 0.0), max(-float(x), 0.0)
        rows.append({"pair_id": f"{name}-{i}", "left": left, "right": right, "difference": float(x)})
    return rows


def _paired_requirement(name: str, values: list[float], stage: str = "STAGE1B") -> dict:
    return {
        "comparison": COMPARISONS.get(stage, {}).get(name, name),
        "analysis_type": "TASK_PAIRED",
        "complete_pair_count": len(values),
        "pairs": _pair_rows(name, values),
    }


def test_sample_size_is_recomputed_from_estimator_matched_inputs():
    values = [-1, 0, 0, 1] * 5
    requirements = {k: _paired_requirement(k, values) for k in REQS}
    inputs = {
        "pilot_sd_by_requirement": {k: paired_sd(v) for k, v in requirements.items()},
        "target_effect": 0.5, "minimum_effect_gate": 0.1,
        "target_ci_half_width": 0.5, "target_power": 0.10,
        "round_up_multiple": 10, "minimum_n": 20, "simulation_count": 4,
        "ci_resamples_per_simulation": 30, "maximum_n": 500,
        "power_confidence_level": 0.51, "power_confirmation_points": 1,
    }
    components = calculate_components_from_structured_requirements(
        requirements, simulations=4, ci_resamples=30, target_power=0.10,
        minimum_n=20, design_effect=0.5, minimum_effect_gate=0.1,
        half_width=0.5, round_multiple=10, maximum_n=500,
        power_confidence_level=0.51, power_confirmation_points=1,
    )
    selected = components.pop("selected_n")
    sample_input = {"stage": "STAGE1B", "requirements": requirements}
    obj = {
        "stage": "STAGE1B", "method": "estimator_matched_paired_binary_simulation_v5",
        "inputs": inputs, "component_requirements": components, "selected_n": selected,
        "reserve_n": selected + 10, "status": "PASS", "random_seed": 7302,
        "analysis_code_sha256": analysis_code_digest(),
    }
    assert not validate_sample_size_report(obj, sample_input)
    obj["selected_n"] -= 10
    assert validate_sample_size_report(obj, sample_input)


def test_sample_size_rejects_wrong_estimator_and_fabricated_sd():
    values = [-1, 0, 0, 1] * 5
    requirements = {k: _paired_requirement(k, values) for k in REQS}
    sample_input = {"stage": "STAGE1B", "requirements": requirements}
    obj = {
        "stage": "STAGE1B", "method": "wrong-estimator",
        "inputs": {"pilot_sd_by_requirement": {k: 0.9 for k in REQS}},
        "component_requirements": {}, "selected_n": 20, "reserve_n": 20,
        "status": "PASS", "random_seed": 7302, "analysis_code_sha256": analysis_code_digest(),
    }
    errors = validate_sample_size_report(obj, sample_input)
    assert any("method differs" in x for x in errors)
    assert any("pilot_sd" in x for x in errors)




def _comparison(kind: str, values: list[float]):
    if kind == "TASK_PAIRED":
        return {"analysis_type": kind, "complete_pair_count": len(values), "pairs": _pair_rows("t", values)}
    if kind == "SCALAR_RATE":
        return {"analysis_type": kind, "complete_pair_count": len(values),
                "units": [{"unit_id": f"t-{i}", "value": float(v)} for i, v in enumerate(values)]}
    raise AssertionError(kind)

def test_scientific_decision_diagnostic_control_cannot_veto_core_go():
    analysis_input = {
        "stage": "STAGE1A",
        "comparisons": {
            "I1-I0": _comparison("TASK_PAIRED", [0.1] * 40),
            "I1-I2": _comparison("TASK_PAIRED", [0.1] * 40),
            "I2-I0": _comparison("TASK_PAIRED", [0.1] * 40),
            "I1_PROGRESS_RATE": _comparison("SCALAR_RATE", [1.0] * 28 + [0.0] * 12),
            "I_PARSE_FAILURE_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
            "I_INFRASTRUCTURE_FAILURE_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
            "I_INCOMPLETE_PAIR_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
            "I_CONTRACT_VIOLATION_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
            "I_HASH_VIOLATION_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
            "I_DETERMINISM_VIOLATION_RATE": _comparison("SCALAR_RATE", [0.0] * 40),
        },
        "raw_result_manifest_sha256": "sha256:" + "a" * 64,
        "expected_task_ids": [f"t-{i}" for i in range(40)],
        "expected_task_count": 40,
        "expected_task_set_sha256": __import__("validation.hashing", fromlist=["hash_json"]).hash_json({"task_ids": sorted([f"t-{i}" for i in range(40)])}),
    }
    contract = yaml.safe_load(open("docs/statistics/statistics_contract_v1.yaml"))
    row = contract["stage_gates"]["STAGE1A"]
    defs = ([{**g, "gate_group": "INTERFACE"} for g in row["core_gates"]]
            + [{**g, "gate_group": "DIAGNOSTIC"} for g in row["diagnostic_gates"]])
    gates=[]
    for definition in defs:
        est,lo,hi=summarize_comparison(analysis_input["comparisons"][definition["comparison"]])
        gates.append({**definition,"estimate":est,"ci_low":lo,"ci_high":hi,
                      "pass":evaluate_gate(definition["rule"],est,lo,hi,definition["threshold"])})
    assert next(g for g in gates if g["gate_group"]=="DIAGNOSTIC")["pass"] is False
    upstream={"decision":"GO_PLANNER_STAGE1B_ELIGIBLE"}
    obj={"stage":"STAGE1A","planner_stage1b_eligible":True,
         "upstream_decision_sha256":"sha256:"+"b"*64,
         "raw_result_manifest_sha256":"sha256:"+"a"*64,
         "gates":gates,"decision":"GO_INTERFACE_STAGE1B_ELIGIBLE"}
    assert not validate_scientific_decision(obj,analysis_input,upstream)


def test_confirmatory_contract_forbids_tost_and_separates_design_effect_from_go_boundary():
    contract=yaml.safe_load(open("docs/statistics/statistics_contract_v1.yaml"))
    assert all("equivalence_TOST_power" not in rows for rows in contract["sample_size"]["component_set_by_stage"].values())
    for stage in ("STAGE1A","STAGE1B"):
        rules=[g["rule"] for key in ("core_gates","diagnostic_gates") for g in contract["stage_gates"][stage].get(key,[])]
        assert "paired_tost" not in rules
    semantics=contract["sample_size"]["component_decision_semantics"]["primary_power"]
    assert semantics["design_effect"] > semantics["decision_boundary"]

def test_analysis_input_rejects_fabricated_differences_duplicate_ids_and_incomplete_seed_pairing():
    pair = lambda i, l, r, d: {"pair_id": i, "left": l, "right": r, "difference": d}
    obj = {"expected_task_ids": ["t1", "t2"], "expected_task_count": 2,
           "expected_task_set_sha256": __import__("validation.hashing", fromlist=["hash_json"]).hash_json({"task_ids": ["t1", "t2"]}),
           "comparisons": {
        "bad-difference": {"analysis_type": "TASK_PAIRED", "complete_pair_count": 2, "pairs": [pair("t1", 1, 0, 0), pair("t2", 0, 1, -1)]},
        "duplicate": {"analysis_type": "TASK_PAIRED", "complete_pair_count": 2, "pairs": [pair("x", 1, 0, 1), pair("x", 0, 1, -1)]},
        "seed-missing": {"analysis_type": "PLANNER_HIERARCHICAL", "complete_pair_count": 2, "seed_groups": {
            "101": [pair("a", 1, 0, 1), pair("b", 0, 1, -1)],
            "202": [pair("a", 1, 0, 1), pair("c", 0, 1, -1)],
        }},
    }}
    errors = validate_analysis_input(obj)
    assert any("difference is not left-right" in x for x in errors)
    assert any("duplicate pair_id" in x for x in errors)
    assert any("identical base_task_id sets" in x for x in errors)


def test_zero_discordant_tost_and_scalar_wilson_are_deterministic():
    result = paired_tost([0.0] * 40, 0.02)
    assert result["pass"] is True and result["ci_low"] == 0 and result["ci_high"] == 0
    comp = _comparison("SCALAR_RATE", [1.0] * 20)
    estimate, lo, hi = summarize_comparison(comp)
    assert estimate == 1.0 and 0.0 < lo < 1.0 and hi == 1.0
    bad = {"expected_task_ids": ["t-0", "t-1", "t-2"], "expected_task_count": 3,
           "expected_task_set_sha256": __import__("validation.hashing", fromlist=["hash_json"]).hash_json({"task_ids": ["t-0", "t-1", "t-2"]}),
           "comparisons": {"rate": _comparison("SCALAR_RATE", [0.0, 0.5, 1.0])}}
    assert any("binary values" in x for x in validate_analysis_input(bad))


def test_sample_size_simulation_never_invents_fractional_binary_outcomes():
    import numpy as np
    rows = _pair_rows("binary", [-1, 0, 0, 1] * 10)
    requirement = _paired_requirement("binary", [-1, 0, 0, 1] * 10)
    drawn = _draw_requirement(requirement, 250, np.random.default_rng(7), 0.075)
    assert set(drawn["pairs"]) <= {-1.0, 0.0, 1.0}
    assert 0.075 not in drawn["pairs"]
