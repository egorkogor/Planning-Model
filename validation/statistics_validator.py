from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

from analysis.decision_gates import (
    clustered_stage1a_bootstrap,
    evaluate_gate,
    hierarchical_planner_bootstrap,
    paired_task_bootstrap,
    positive_seed_count,
    stage_decision,
    wilson_binary_rate_ci,
)
from analysis.sample_size import calculate_components_from_structured_requirements, paired_sd
from validation.code_fingerprint import analysis_code_digest
from validation.hashing import hash_json

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "docs/statistics/statistics_contract_v1.yaml"


def _contract() -> dict:
    return yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


def _definitions(stage: str) -> list[dict]:
    row = _contract()["stage_gates"][stage]
    if stage == "PLANNER":
        out=[]
        for g in row["architecture_gates"]:
            out.append({**g,"gate_group":"ARCHITECTURE"})
        for g in row["stage1b_eligibility_gates"]:
            out.append({**g,"gate_group":"STAGE1B_ELIGIBILITY"})
        return out
    group = "INTERFACE" if stage == "STAGE1A" else "END_TO_END"
    if "core_gates" in row:
        return ([{**g, "gate_group": group} for g in row["core_gates"]]
                + [{**g, "gate_group": "DIAGNOSTIC"} for g in row.get("diagnostic_gates", [])])
    return [{**g,"gate_group":group} for g in row["gates"]]


def _pair_values(rows: list[dict]) -> list[float]:
    return [float(row["difference"]) for row in rows]


def _raw_differences(comp: dict) -> list[float] | None:
    kind=comp["analysis_type"]
    if kind=="TASK_PAIRED": return _pair_values(comp["pairs"])
    if kind=="STAGE1A_CLUSTERED": return [fmean(_pair_values(rows)) for rows in comp["task_clusters"].values()]
    if kind=="SCALAR_RATE": return [float(row["value"]) for row in comp["units"]]
    return None


def _validate_pair_rows(name: str, rows: list[dict]) -> list[str]:
    errors=[]; ids=[]
    for row in rows:
        ids.append(row.get("pair_id"))
        expected=float(row.get("left",0))-float(row.get("right",0))
        if not _close(row.get("difference",999),expected): errors.append(f"{name}: difference is not left-right for {row.get('pair_id')}")
    if len(ids)!=len(set(ids)): errors.append(f"{name}: duplicate pair_id")
    return errors


def validate_analysis_input(obj: dict, selected_task_manifest: dict | None = None) -> list[str]:
    errors=[]
    expected_ids = [str(x) for x in obj.get("expected_task_ids", [])]
    expected_set = set(expected_ids)
    if len(expected_ids) != len(expected_set):
        errors.append("expected_task_ids: duplicate task_id")
    if obj.get("expected_task_count") != len(expected_set):
        errors.append("expected_task_count does not equal exact expected_task_ids set")
    if obj.get("expected_task_set_sha256") != hash_json({"task_ids": sorted(expected_set)}):
        errors.append("expected_task_set_sha256 does not match canonical expected task set")
    selected_path_value = obj.get("selected_task_manifest_path")
    selected_sha = obj.get("selected_task_manifest_sha256")
    canonical_selected_paths = {
        "PLANNER": "sealed/planner-confirmatory/selected-task-manifest.json",
        "STAGE1A": "sealed/stage1a-confirmatory/selected-task-manifest.json",
        "STAGE1B": "sealed/stage1b-confirmatory/selected-task-manifest.json",
    }
    if selected_path_value is not None and selected_path_value != canonical_selected_paths.get(obj.get("stage")):
        errors.append("selected task manifest path differs from canonical stage path")
    if selected_path_value is not None or selected_sha is not None or selected_task_manifest is not None:
        try:
            if selected_task_manifest is None:
                if not selected_path_value or not selected_sha:
                    raise ValueError("selected task manifest path and sha256 are both required")
                selected_path = (ROOT / str(selected_path_value)).resolve()
                if selected_path != ROOT.resolve() and ROOT.resolve() not in selected_path.parents:
                    raise ValueError("selected task manifest path escapes repository")
                if not selected_path.is_file():
                    raise ValueError("selected task manifest file is missing")
                actual_sha = "sha256:" + hashlib.sha256(selected_path.read_bytes()).hexdigest()
                if actual_sha != selected_sha:
                    raise ValueError("selected task manifest file hash mismatch")
                selected_task_manifest = json.loads(selected_path.read_text(encoding="utf-8"))
            payload = dict(selected_task_manifest)
            declared_manifest_hash = payload.pop("manifest_hash", None)
            if declared_manifest_hash != hash_json(payload):
                errors.append("selected task manifest canonical hash mismatch")
            if selected_task_manifest.get("run_id") != obj.get("run_id") or selected_task_manifest.get("stage") != obj.get("stage"):
                errors.append("selected task manifest run_id/stage differs from AnalysisInput")
            selected_ids = {str(x) for x in selected_task_manifest.get("task_ids", [])}
            if selected_task_manifest.get("task_count") != len(selected_ids):
                errors.append("selected task manifest task_count mismatch")
            if selected_ids != expected_set:
                errors.append("AnalysisInput expected_task_ids differ from selected task manifest")
        except Exception as exc:
            errors.append(f"selected task manifest validation failed: {exc}")
    stage1a_snapshot_sets: dict[str, dict[str, frozenset[str]]] = {}
    for name,comp in obj.get("comparisons",{}).items():
        kind=comp.get("analysis_type"); count=comp.get("complete_pair_count")
        if kind=="PLANNER_HIERARCHICAL":
            groups=comp.get("seed_groups",{})
            expected_seeds={"101", "202", "303", "404", "505"}
            if set(str(seed) for seed in groups) != expected_seeds:
                errors.append(f"{name}: exactly the five locked final seed groups are required")
            lengths={len(v) for v in groups.values()}
            if len(lengths)!=1: errors.append(f"{name}: planner seed groups must contain the same paired task count")
            elif lengths and count!=next(iter(lengths)): errors.append(f"{name}: complete_pair_count mismatch")
            id_sets=[]
            for seed,rows in groups.items():
                errors.extend(_validate_pair_rows(f"{name}/{seed}",rows)); id_sets.append({r.get("pair_id") for r in rows})
            if id_sets and any(ids!=id_sets[0] for ids in id_sets[1:]): errors.append(f"{name}: planner seed groups must use identical base_task_id sets")
            if id_sets and id_sets[0] != expected_set: errors.append(f"{name}: task set differs from expected_task_ids")
        elif kind=="STAGE1A_CLUSTERED":
            clusters=comp.get("task_clusters",{})
            if count!=len(clusters): errors.append(f"{name}: complete_pair_count mismatch")
            snapshot_sets: dict[str, frozenset[str]] = {}
            for task,rows in clusters.items():
                errors.extend(_validate_pair_rows(f"{name}/{task}",rows))
                snapshot_sets[str(task)] = frozenset(str(row.get("pair_id")) for row in rows)
            stage1a_snapshot_sets[name] = snapshot_sets
            if set(clusters) != expected_set: errors.append(f"{name}: task set differs from expected_task_ids")
        elif kind=="TASK_PAIRED":
            rows=comp.get("pairs",[])
            if count!=len(rows): errors.append(f"{name}: complete_pair_count mismatch")
            errors.extend(_validate_pair_rows(name,rows))
            if {r.get("pair_id") for r in rows} != expected_set: errors.append(f"{name}: task set differs from expected_task_ids")
        elif kind=="SCALAR_RATE":
            rows=comp.get("units",[])
            if count!=len(rows): errors.append(f"{name}: complete_pair_count mismatch")
            ids=[r.get("unit_id") for r in rows]
            if len(ids)!=len(set(ids)): errors.append(f"{name}: duplicate unit_id")
            if set(ids) != expected_set: errors.append(f"{name}: task set differs from expected_task_ids")
            if any(float(r.get("value",-1)) not in (0.0,1.0) for r in rows): errors.append(f"{name}: SCALAR_RATE requires unit-level binary values")
        else: errors.append(f"{name}: unknown analysis_type")
    if obj.get("stage") == "STAGE1A" and stage1a_snapshot_sets:
        reference_name = sorted(stage1a_snapshot_sets)[0]
        reference = stage1a_snapshot_sets[reference_name]
        for name, snapshot_sets in sorted(stage1a_snapshot_sets.items()):
            if snapshot_sets != reference:
                errors.append(
                    f"{name}: Stage 1A snapshot pair_id sets must exactly match {reference_name} for every task"
                )
    return errors


def summarize_comparison(comp: dict) -> tuple[float,float,float]:
    kind=comp["analysis_type"]
    if kind=="PLANNER_HIERARCHICAL":
        return hierarchical_planner_bootstrap({seed:_pair_values(rows) for seed,rows in comp["seed_groups"].items()})
    if kind=="STAGE1A_CLUSTERED":
        return clustered_stage1a_bootstrap({task:_pair_values(rows) for task,rows in comp["task_clusters"].items()})
    if kind=="TASK_PAIRED":
        return paired_task_bootstrap(_pair_values(comp["pairs"]))
    if kind=="SCALAR_RATE":
        return wilson_binary_rate_ci([float(row["value"]) for row in comp["units"]])
    raise ValueError(f"unknown analysis_type: {kind}")


def _validate_sample_size_requirement(name: str, row: dict, stage: str) -> list[str]:
    errors: list[str] = []
    kind = row.get("analysis_type")
    expected_kind = {"PLANNER": "PLANNER_HIERARCHICAL", "STAGE1A": "STAGE1A_CLUSTERED", "STAGE1B": "TASK_PAIRED"}[stage]
    if kind != expected_kind:
        return [f"{name}: analysis_type {kind} != locked stage estimator {expected_kind}"]
    if kind == "PLANNER_HIERARCHICAL":
        groups = row.get("seed_groups", {})
        if len(groups) != 5:
            errors.append(f"{name}: exactly five final seed groups required")
        lengths = {len(rows) for rows in groups.values()}
        if len(lengths) != 1:
            errors.append(f"{name}: seed groups must have identical paired task counts")
        id_sets = []
        for seed, rows in groups.items():
            errors.extend(_validate_pair_rows(f"sample-size/{name}/{seed}", rows))
            id_sets.append({r.get("pair_id") for r in rows})
        if id_sets and any(ids != id_sets[0] for ids in id_sets[1:]):
            errors.append(f"{name}: seed groups must use identical pair ids")
        unit_count = next(iter(lengths)) if len(lengths) == 1 else 0
    elif kind == "STAGE1A_CLUSTERED":
        clusters = row.get("task_clusters", {})
        for task, rows in clusters.items():
            errors.extend(_validate_pair_rows(f"sample-size/{name}/{task}", rows))
        unit_count = len(clusters)
    else:
        rows = row.get("pairs", [])
        errors.extend(_validate_pair_rows(f"sample-size/{name}", rows))
        unit_count = len(rows)
    if row.get("complete_pair_count") != unit_count:
        errors.append(f"{name}: complete_pair_count mismatch")
    if unit_count < 20:
        errors.append(f"{name}: at least 20 pilot analysis units required")
    return errors


def validate_sample_size_report(obj: dict, sample_size_input: dict | None = None) -> list[str]:
    errors: list[str] = []
    if obj.get("method") != "estimator_matched_paired_binary_simulation_v5":
        errors.append("sample-size method differs from statistics contract")
    if obj.get("analysis_code_sha256") != analysis_code_digest():
        errors.append("analysis_code_sha256 does not match the locked analysis implementation")
    if sample_size_input is None:
        errors.append("sample_size_input is required for locked recomputation")
        return errors
    stage = obj.get("stage")
    if sample_size_input.get("stage") != stage:
        errors.append("sample_size_input stage mismatch")
    requirements = sample_size_input.get("requirements", {})
    comparison_map = _contract().get("sample_size", {}).get("component_comparison_by_stage", {}).get(stage, {})
    expected_by_stage = {
        "PLANNER": {"primary_ci", "primary_power", "current_vs_shuffled_power"},
        "STAGE1A": {"primary_ci", "primary_power", "current_vs_shuffled_power"},
        "STAGE1B": {"primary_ci", "primary_power", "current_vs_shuffled_power",
                    "random_code_power", "structured_noninferiority_power",
                    "self_plan_power", "flops_direction_power"},
    }
    expected_names = expected_by_stage.get(stage, set())
    if set(requirements) != expected_names:
        errors.append(f"sample-size requirement set mismatch for {stage}: {set(requirements)} != {expected_names}")
        return errors
    expected_sds: dict[str, float] = {}
    for name in sorted(expected_names):
        row = requirements[name]
        expected_comparison = comparison_map.get(name)
        if row.get("comparison") != expected_comparison:
            errors.append(f"{name}: comparison {row.get('comparison')} != locked {expected_comparison}")
        errors.extend(_validate_sample_size_requirement(name, row, stage))
        try:
            expected_sds[name] = paired_sd(row)
        except Exception as exc:
            errors.append(f"{name}: invalid estimator-matched pilot input: {exc}")
    inputs = obj.get("inputs", {})
    reported_sds = inputs.get("pilot_sd_by_requirement", {})
    for name, expected in expected_sds.items():
        if name not in reported_sds or not _close(reported_sds[name], expected, tol=1e-12):
            errors.append(f"{name}: pilot_sd is not derived from final-estimator analysis units")
    if errors:
        return errors
    try:
        expected_components = calculate_components_from_structured_requirements(
            requirements,
            design_effect=inputs["target_effect"],
            minimum_effect_gate=inputs["minimum_effect_gate"],
            half_width=inputs["target_ci_half_width"],
            target_power=inputs["target_power"],
            simulations=inputs["simulation_count"],
            ci_resamples=inputs["ci_resamples_per_simulation"],
            seed=obj["random_seed"],
            minimum_n=inputs["minimum_n"],
            round_multiple=inputs["round_up_multiple"],
            maximum_n=inputs["maximum_n"],
            power_confidence_level=inputs["power_confidence_level"],
            power_confirmation_points=inputs["power_confirmation_points"],
        )
    except Exception as exc:
        errors.append(f"locked estimator-matched sample-size simulation failed: {exc}")
        return errors
    expected_selected = expected_components.pop("selected_n")
    if obj.get("component_requirements") != expected_components:
        errors.append("component_requirements do not match locked estimator-matched simulation")
    if obj.get("selected_n") != expected_selected:
        errors.append(f"selected_n {obj.get('selected_n')} != locked result {expected_selected}")
    expected_status = "PASS" if obj.get("reserve_n", 0) >= expected_selected else "BLOCKED_INSUFFICIENT_RESERVE"
    if obj.get("status") != expected_status:
        errors.append(f"status {obj.get('status')} != {expected_status}")
    return errors


def validate_scientific_decision(obj: dict, analysis_input: dict | None = None, upstream_decision: dict | None = None) -> list[str]:
    errors=[]; stage=obj["stage"]
    expected_defs=_definitions(stage)
    actual={g["gate_id"]:g for g in obj["gates"]}
    expected_ids=[g["gate_id"] for g in expected_defs]
    if set(actual)!=set(expected_ids):
        errors.append(f"gate set mismatch: expected={expected_ids}, actual={sorted(actual)}")
        return errors
    if analysis_input is not None:
        if analysis_input.get("stage")!=stage: errors.append("analysis_input stage mismatch")
        if analysis_input.get("raw_result_manifest_sha256")!=obj.get("raw_result_manifest_sha256"):
            errors.append("raw result lineage differs between AnalysisInput and decision")
        errors.extend(validate_analysis_input(analysis_input))
    if stage=="PLANNER":
        if obj.get("upstream_decision_sha256") is not None: errors.append("PLANNER decision must not have upstream decision")
    elif upstream_decision is None:
        errors.append("upstream scientific decision is required")
    elif stage=="STAGE1A":
        expected_eligible=upstream_decision.get("decision")=="GO_PLANNER_STAGE1B_ELIGIBLE"
        if obj.get("planner_stage1b_eligible") is not expected_eligible: errors.append("planner_stage1b_eligible differs from upstream Planner decision")
    elif stage=="STAGE1B":
        if upstream_decision.get("decision")!="GO_INTERFACE_STAGE1B_ELIGIBLE": errors.append("Stage 1B requires eligible upstream interface decision")
    recomputed=[]
    for definition in expected_defs:
        gate=actual[definition["gate_id"]]
        for key in ("comparison","rule","threshold","gate_group"):
            if gate.get(key)!=definition.get(key): errors.append(f"gate {gate['gate_id']} {key} differs from locked contract")
        raw=None
        if analysis_input is not None:
            comp=analysis_input["comparisons"].get(definition["comparison"])
            if comp is None:
                errors.append(f"missing analysis comparison {definition['comparison']}"); continue
            raw=_raw_differences(comp)
            if definition["rule"]=="paired_tost":
                errors.append(f"gate {gate['gate_id']} uses forbidden confirmatory paired_tost rule")
                continue
            elif definition["rule"]=="minimum_positive_seed_count":
                if comp.get("analysis_type") != "PLANNER_HIERARCHICAL":
                    errors.append(f"gate {gate['gate_id']} requires PLANNER_HIERARCHICAL input"); continue
                estimate=float(positive_seed_count({seed:_pair_values(rows) for seed,rows in comp["seed_groups"].items()}))
                lo=hi=estimate
            else:
                estimate,lo,hi=summarize_comparison(comp)
            for key,expected in (("estimate",estimate),("ci_low",lo),("ci_high",hi)):
                if not _close(gate[key],expected): errors.append(f"gate {gate['gate_id']} {key} not recomputed from AnalysisInput")
        try:
            passed=evaluate_gate(definition["rule"],gate["estimate"],gate["ci_low"],gate["ci_high"],definition["threshold"],raw_differences=raw)
        except Exception as exc:
            errors.append(f"gate {gate['gate_id']} evaluation failed: {exc}"); continue
        if gate["pass"] is not passed: errors.append(f"gate {gate['gate_id']} pass mismatch")
        recomputed.append({**gate,"pass":passed})
    if obj["decision"]!="INVALID_RUN" and len(recomputed)==len(expected_defs):
        expected_decision=stage_decision(stage,recomputed,planner_stage1b_eligible=obj.get("planner_stage1b_eligible"))
        if obj["decision"]!=expected_decision: errors.append(f"decision {obj['decision']} != {expected_decision}")
    return errors
