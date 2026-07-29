from __future__ import annotations

import argparse
import sys
import hashlib
import fnmatch
import json
import shlex
import subprocess
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.verify_lock import DEFAULT_LOCKS, verify as verify_lock
from validation.role_validator import validate_role_independence, canonical_identity_hash
from validation.resource_plan_validator import validate_resource_plan_semantics
from validation.statistical_audit_validator import validate_statistical_audit
from validation.implementation_audit_validator import validate_implementation_audit
from validation.implementation_candidate_validator import validate_candidate
from validation.signature_validator import verify_signed_manifest
from validation.hashing import hash_json, decision_record_hash
from validation.code_fingerprint import analysis_code_digest
from validation.operator_decision_validator import (
    verify_decision_history,
    verify_operator_decision,
    verify_run_decision_history,
)
from validation.confirmatory_lineage_validator import (
    validate_approved_pointer,
    validate_dispatch,
    validate_evaluator_manifest,
    validate_experiment_freeze,
)
from validation.phase_check_runner import core_check as run_core_phase_check
from validation.trust_topology_validator import verify as verify_trust_topology


@lru_cache(maxsize=1)
def schema_bundle():
    reg=Registry(); objs={}
    for p in sorted((ROOT/'docs/schemas').glob('*.json')):
        obj=json.loads(p.read_text(encoding='utf-8')); objs[p.name]=obj
        reg=reg.with_resource(obj['$id'],Resource.from_contents(obj))
    return objs,reg


def digest(path: Path)->str:
    return 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace('Z','+00:00'))


def verify_phase_timing(report: dict, decision: dict | None = None) -> list[str]:
    errors: list[str] = []
    try:
        started=parse_timestamp(report['started_at'])
        finished=parse_timestamp(report['finished_at']) if report.get('finished_at') else None
        if report.get('status') in {'PASS','FAIL','BLOCKED','WAITING_APPROVAL','SKIPPED_BY_CONTRACT'} and finished is None:
            errors.append('completed phase status requires finished_at')
        if finished is not None and finished < started:
            errors.append('phase finished_at predates started_at')
        if decision is not None:
            decided=parse_timestamp(decision['timestamp'])
            if decided < started:
                errors.append('DecisionRecord timestamp predates phase start')
            if finished is not None and decided > finished:
                errors.append('DecisionRecord timestamp is later than phase finish')
    except Exception as exc:
        errors.append(f'invalid phase timing: {exc}')
    return errors


def validate(obj, schema_name: str):
    schemas,reg=schema_bundle(); errors=list(Draft202012Validator(schemas[schema_name],registry=reg,format_checker=FormatChecker()).iter_errors(obj))
    if errors: raise ValueError('; '.join(e.message for e in errors))


def safe_path(rel: str) -> Path:
    p=(ROOT/rel).resolve()
    if ROOT.resolve() not in p.parents and p!=ROOT.resolve():
        raise ValueError(f'path escapes repository: {rel}')
    return p


def expected_checks(phase: dict, *, include_post: bool, outcome: str | None = None) -> dict[str,dict]:
    rows=[]
    rows.extend(phase.get('pre_gate_checks',[]) or [])
    rows.extend(phase.get('execution_checks',[]) or [])
    if include_post:
        rows.extend(phase.get('post_gate_checks',[]) or [])
        rows.extend((phase.get('post_gate_checks_by_outcome',{}) or {}).get(outcome,[]) or [])
    return {row['check_id']:row for row in rows}


def _registry_rule(rel: str) -> dict | None:
    registry=yaml.safe_load((ROOT/'docs/operator/report_registry_v1.yaml').read_text(encoding='utf-8'))
    for row in registry['rules']:
        if fnmatch.fnmatch(rel,row['path_pattern']): return row
    return None

def verify_report_artifact_constraints(report: dict) -> list[str]:
    errors: list[str] = []
    artifact_paths = [a.get("path") for a in report.get("artifacts", [])]
    if len(artifact_paths) != len(set(artifact_paths)):
        errors.append("duplicate artifact path in phase report")
    own_rel = f"reports/phase-{report.get('phase_id')}.json"
    if own_rel in set(artifact_paths):
        errors.append("phase report must not contain a self-referential artifact hash")
    return errors


def verify_registered_artifact(rel: str) -> list[str]:
    errors=[]; path=safe_path(rel); rule=_registry_rule(rel)
    if path.suffix not in {'.json','.jsonl'}: return errors
    if rule is None: return [f'unregistered JSON artifact: {rel}']
    try:
        if 'schema' in rule:
            obj=load_json(path)
            validate(obj,Path(rule['schema']).name)
            if rel.endswith('/sealer-manifest.json'):
                errors.extend(verify_signed_manifest(obj,'DATA_SEALER'))
            if rel.endswith('/evaluator-result-manifest.json'):
                errors.extend(verify_signed_manifest(obj,'EVALUATION_RUNNER'))
            if rel=='locks/trust-topology.lock.json':
                errors.extend(verify_trust_topology(path))
            if rel=='freezes/implementation-lock.candidate.json':
                errors.extend(validate_candidate(obj))
            if rel=='reports/statistical-implementation-audit.json':
                errors.extend(verify_signed_manifest(obj,'STATISTICAL_REVIEWER',hash_field='report_hash'))
                errors.extend(validate_statistical_audit(obj, load_json(ROOT/'reports/resource-plan.json')))
            if rel=='reports/independent-implementation-audit.json':
                errors.extend(verify_signed_manifest(obj,'AUDITOR',hash_field='report_hash'))
                errors.extend(validate_implementation_audit(obj, load_json(ROOT/'reports/resource-plan.json')))
            if rel=='reports/final-audit.json':
                errors.extend(verify_signed_manifest(obj,'AUDITOR',hash_field='report_hash'))
        else:
            for i,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
                if line.strip(): validate(json.loads(line),Path(rule['schema_per_line']).name)
    except Exception as exc: errors.append(f'artifact schema validation failed {rel}: {exc}')
    return errors

def verify_required_outputs(phase: dict, report: dict, phase_report_path: Path, *, include_post: bool) -> list[str]:
    errors=[]; arts={a['path']:a for a in report['artifacts']}
    own_rel=f"reports/phase-{phase['phase_id']}.json"
    required=list(phase.get('required_outputs',[]))
    required.extend(phase.get('pre_gate_required_outputs',[]) or [])
    if include_post:
        required.extend(phase.get('post_gate_required_outputs',[]) or [])
        required.extend((phase.get('post_gate_required_outputs_by_outcome',{}) or {}).get(report.get('outcome'),[]) or [])
    for rel in required:
        path=phase_report_path if rel==own_rel else safe_path(rel.rstrip('/'))
        if not path.exists():
            errors.append(f'required output missing: {rel}'); continue
        if path.is_file():
            if rel!=own_rel:
                row=arts.get(rel)
                if row is None: errors.append(f'required output absent from phase artifacts: {rel}')
                elif digest(path)!=row['sha256']: errors.append(f'required output hash mismatch: {rel}')
                errors.extend(verify_registered_artifact(rel))
        elif path.is_dir():
            files=[f for f in path.rglob('*') if f.is_file() and '.git' not in f.parts and '__pycache__' not in f.parts]
            if not files:
                errors.append(f'required output directory empty: {rel}')
                continue
            for file_path in sorted(files):
                if file_path.is_symlink():
                    errors.append(f'required output directory contains symlink: {file_path.relative_to(ROOT).as_posix()}')
                    continue
                file_rel=file_path.relative_to(ROOT).as_posix()
                row=arts.get(file_rel)
                if row is None:
                    errors.append(f'required output file absent from phase artifacts: {file_rel}')
                elif digest(file_path)!=row['sha256']:
                    errors.append(f'required output file hash mismatch: {file_rel}')
                errors.extend(verify_registered_artifact(file_rel))
    return errors


def verify_required_inputs(phase: dict, report: dict) -> list[str]:
    errors: list[str] = []
    declared = report.get("input_hashes", {})
    for rel in phase.get("required_inputs", []) or []:
        try:
            path = safe_path(rel)
        except ValueError as exc:
            errors.append(str(exc)); continue
        if not path.is_file():
            errors.append(f"required input missing: {rel}")
        elif declared.get(rel) != digest(path):
            errors.append(f"required input absent from input_hashes or hash mismatch: {rel}")
    return errors


def verify_checks(phase: dict, report: dict, *, include_post: bool, blocked: bool) -> list[str]:
    errors=[]; expected=expected_checks(phase,include_post=include_post,outcome=report.get('outcome'))
    actual_rows=report['checks']; ids=[x['check_id'] for x in actual_rows]
    if len(ids)!=len(set(ids)): errors.append('duplicate check_id in phase report')
    actual={x['check_id']:x for x in actual_rows}
    if not blocked and set(actual)!=set(expected):
        errors.append(f'check set mismatch: missing={sorted(set(expected)-set(actual))}, extra={sorted(set(actual)-set(expected))}')
        return errors
    if blocked:
        if not any(x['status']=='FAIL' for x in actual_rows): errors.append('BLOCKED/FAIL report requires at least one FAIL check')
        if not set(actual).issubset(set(expected)): errors.append('blocked report contains unknown check_id')
    else:
        for check_id,row in actual.items():
            if row['status']!='PASS': errors.append(f'non-PASS required check: {check_id}')
            if not row['evidence']: errors.append(f'check without evidence: {check_id}')
            verifier=expected[check_id].get('verifier')
            if verifier:
                command=shlex.split(verifier)
                phase_contract=yaml.safe_load((ROOT/'docs/operator/phase_check_contract_v1.yaml').read_text(encoding='utf-8'))
                core_kind=phase_contract.get('core_checks',{}).get(check_id)
                canonical_prefix=['python','validation/phase_check_runner.py','--phase',phase['phase_id'],'--check',check_id,'--report',f"reports/phase-{phase['phase_id']}.json"]
                if core_kind and command==canonical_prefix:
                    check_errors=run_core_phase_check(core_kind,phase['phase_id'],check_id,report)
                    if check_errors: errors.append(f"locked verifier failed for {check_id}: {'; '.join(check_errors)}")
                else:
                    cp=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
                    if cp.returncode: errors.append(f'locked verifier failed for {check_id}: {verifier}')
    return errors


def transition_for(sm: dict, phase_id: str, outcome: str | None):
    if outcome is None: return None
    return sm['transitions'][phase_id].get(outcome)


def verify_next_phase(sm: dict, report: dict) -> list[str]:
    trans=transition_for(sm,report['phase_id'],report['outcome'])
    if report['outcome'] is None: return []
    if trans is None: return ['outcome missing from state machine']
    errors=[]
    expected=trans.get('next') or trans.get('repeat')
    if report.get('next_phase')!=expected:
        errors.append(f'next_phase {report.get("next_phase")} != locked transition {expected}')
    if trans.get('requires_flags') or trans.get('set_flags'):
        try:
            status=load_json(ROOT/'RUN_STATUS.json'); validate(status,'run_status.schema.json')
            flags=status.get('flags',{})
            for key,value in (trans.get('requires_flags') or {}).items():
                if flags.get(key) is not value:
                    errors.append(f'state-machine required flag {key}={value} is not persisted in RUN_STATUS.json')
            for key,value in (trans.get('set_flags') or {}).items():
                if flags.get(key) is not value:
                    errors.append(f'state-machine set flag {key}={value} is not persisted in RUN_STATUS.json')
        except Exception as exc:
            errors.append(f'cannot verify state-machine flags: {exc}')
    return errors


def verify_git_commit(commit: str | None) -> bool:
    if commit is None: return False
    cp=subprocess.run(['git','cat-file','-e',f'{commit}^{{commit}}'],cwd=ROOT,capture_output=True)
    return cp.returncode==0



def _artifact_paths(report: dict) -> set[str]:
    return {a['path'] for a in report.get('artifacts',[])}


def verify_commands_and_evidence(report: dict, *, blocked: bool) -> list[str]:
    errors=[]; artifact_paths=_artifact_paths(report)
    for check in report.get('checks',[]):
        for rel in check.get('evidence',[]):
            try: path=safe_path(rel)
            except ValueError as exc: errors.append(str(exc)); continue
            if rel not in artifact_paths: errors.append(f'check evidence absent from phase artifacts: {rel}')
            if not path.is_file(): errors.append(f'check evidence file missing: {rel}')
    for i,row in enumerate(report.get('commands',[])):
        if report.get('status') in {'PASS','WAITING_APPROVAL'} and row['exit_code']!=0:
            errors.append(f'PASS phase command[{i}] has non-zero exit_code')
        for field in ('stdout_artifact','stderr_artifact'):
            rel=row[field]
            if not rel:
                errors.append(f'command[{i}] {field} is empty'); continue
            try: path=safe_path(rel)
            except ValueError as exc: errors.append(str(exc)); continue
            if rel not in artifact_paths: errors.append(f'command[{i}] {field} absent from phase artifacts')
            if not path.is_file(): errors.append(f'command[{i}] {field} missing')
    if blocked:
        expected_self=f"reports/self-review-{report['phase_id']}.json"
        if expected_self not in artifact_paths: errors.append('BLOCKED/FAIL report requires self-review artifact')
    return errors


def verify_commit_lineage(before: str | None, implementation: str | None) -> list[str]:
    if not before or not implementation: return ['commit lineage is incomplete']
    if not verify_git_commit(before) or not verify_git_commit(implementation): return ['commit lineage references a non-commit']
    cp=subprocess.run(['git','merge-base','--is-ancestor',before,implementation],cwd=ROOT,capture_output=True)
    return [] if cp.returncode==0 else ['implementation_commit does not descend from git_commit_before']


def _load_artifact_by_sha(report: dict, wanted: str) -> tuple[str,dict] | None:
    for art in report.get('artifacts',[]):
        if art.get('sha256')==wanted and art['path'].endswith('.json'):
            try: return art['path'],load_json(safe_path(art['path']))
            except Exception: return None
    return None


def verify_declared_self_hash(rel: str, obj: dict) -> list[str]:
    field=None; kind=None
    if fnmatch.fnmatch(rel,'analysis/inputs/*.json'): field,kind='analysis_input_hash','analysis_input'
    elif fnmatch.fnmatch(rel,'analysis/sample-size-inputs/*.json'): field,kind='input_hash','sample_size_input'
    elif fnmatch.fnmatch(rel,'reports/*-decision.json'): field,kind='decision_hash','scientific_decision'
    elif fnmatch.fnmatch(rel,'reports/sample-size-*.json'): field,kind='report_hash','sample_size_report'
    if field is None: return []
    payload=dict(obj); declared=payload.pop(field,None)
    expected=hash_json({'schema':'work-planner-hash/1.0','kind':kind,'value':payload})
    return [] if declared==expected else [f'{rel}: {field} mismatch']

def verify_semantic_report(rel: str, obj: dict, report: dict) -> list[str]:
    errors=verify_declared_self_hash(rel,obj)
    if "run_id" in obj and obj.get("run_id") != report.get("run_id"):
        errors.append(f"{rel}: run_id differs from phase report")
    artifact_map={a['path']:a['sha256'] for a in report.get('artifacts',[])}
    if fnmatch.fnmatch(rel, "freezes/*.candidate.json"):
        errors.extend(validate_experiment_freeze(obj, expected_run_id=report.get("run_id")))
    if fnmatch.fnmatch(rel, "freezes/*.approved.json"):
        errors.extend(validate_approved_pointer(obj, expected_run_id=report.get("run_id")))
    if fnmatch.fnmatch(rel, "dispatch/*.json"):
        errors.extend(validate_dispatch(obj, expected_run_id=report.get("run_id")))
    if rel.endswith('/sealer-manifest.json'):
        try:
            from validation.sealer_manifest_validator import validate_sealer_manifest_semantics
            errors.extend(validate_sealer_manifest_semantics(obj))
            plan=load_json(ROOT/'reports/resource-plan.json'); identity=plan['data_sealer']
            if obj.get('sealer_identity_sha256')!=canonical_identity_hash(identity): errors.append('sealer identity differs from locked resource plan')
            if obj.get('public_key_id')!=identity['public_signing_key_id']: errors.append('sealer public key differs from locked identity')
            envelope_path=obj['auditor_seed_envelope_artifact']; envelope_hash=obj['auditor_seed_envelope_sha256']
            if artifact_map.get(envelope_path)!=envelope_hash:
                errors.append(f'auditor seed envelope not present with declared hash: {envelope_path}')
            elif not safe_path(envelope_path).is_file() or digest(safe_path(envelope_path))!=envelope_hash:
                errors.append(f'auditor seed envelope file mismatch: {envelope_path}')
            if obj.get('stage') == 'STAGE1B':
                cert = obj.get('control_certification', {})
                for field in ('task_only_selection_manifest_sha256', 'preoutcome_artifact_manifest_sha256', 'public_exclusion_manifest_sha256'):
                    expected = cert.get(field)
                    if expected not in artifact_map.values():
                        errors.append(f'Stage1B task-only/pre-outcome certification artifact hash absent from phase evidence: {field}')
            evidence=obj['deletion_evidence']
            for path_field,hash_field in (('workspace_scan_artifact','workspace_scan_sha256'),('process_log_artifact','process_log_sha256'),('volume_destroy_artifact','volume_destroy_sha256')):
                ep=evidence[path_field]; expected=evidence[hash_field]
                if artifact_map.get(ep)!=expected: errors.append(f'sealer deletion evidence not present with declared hash: {ep}')
                elif not safe_path(ep).is_file() or digest(safe_path(ep))!=expected: errors.append(f'sealer deletion evidence file mismatch: {ep}')
        except Exception as exc: errors.append(f'sealer semantic validation failed: {exc}')
    if rel.endswith('/evaluator-result-manifest.json'):
        errors.extend(validate_evaluator_manifest(obj, report))
        try:
            plan=load_json(ROOT/'reports/resource-plan.json'); identity=plan['evaluator']
            if obj.get('evaluator_identity_sha256')!=canonical_identity_hash(identity): errors.append('evaluator identity differs from locked resource plan')
            if obj.get('public_key_id')!=identity['public_signing_key_id']: errors.append('evaluator public key differs from locked identity')
            if not verify_git_commit(obj.get('git_commit')): errors.append('evaluator git_commit is not present in repository')
            artifact_map={a['path']:a['sha256'] for a in report.get('artifacts',[]) if a['path']!=rel}
            if any(artifact_map.get(path)!=sha for path,sha in obj.get('raw_artifacts',{}).items()): errors.append('evaluator raw artifacts absent from phase evidence or path/hash mismatch')
        except Exception as exc: errors.append(f'evaluator semantic validation failed: {exc}')
    if rel=='reports/final-audit.json':
        try:
            plan=load_json(ROOT/'reports/resource-plan.json'); identity=plan['auditor']
            if obj.get('auditor')!=identity: errors.append('final audit identity differs from locked resource plan')
            if obj.get('auditor_identity_sha256')!=canonical_identity_hash(identity): errors.append('final audit identity hash mismatch')
            if obj.get('public_key_id')!=identity['public_signing_key_id']: errors.append('final audit public key mismatch')
            expected_env='sha256:'+hashlib.sha256(identity['environment_identity'].encode('utf-8')).hexdigest()
            if obj.get('auditor_environment_sha256')!=expected_env: errors.append('final audit environment differs from locked resource plan')
            if not verify_git_commit(obj.get('clean_checkout_commit')): errors.append('final audit clean checkout commit is unavailable')
            candidate=load_json(ROOT/'freezes/implementation-lock.candidate.json')
            if obj.get('clean_checkout_commit')!=candidate.get('reviewed_commit'): errors.append('final audit did not reproduce the approved implementation commit')
            required_checks=set(yaml.safe_load((ROOT/'docs/audit/independent_audit_contract_v1.yaml').read_text(encoding='utf-8'))['checks'])
            rows=obj.get('checks',[])
            ids=[row.get('check_id') for row in rows]
            if set(ids)!=required_checks or len(ids)!=len(required_checks): errors.append('final audit check set differs from independent audit contract')
            if obj.get('status')=='PASS' and any(row.get('status')!='PASS' for row in rows): errors.append('final audit cannot PASS with a failed check')
            if obj.get('status')=='PASS' and obj.get('mismatches'): errors.append('final audit cannot PASS with mismatches')
            evidence_hashes={a['sha256'] for a in report.get('artifacts',[])}
            for row in rows:
                if not set(row.get('evidence_hashes',[])).issubset(evidence_hashes): errors.append(f"final audit evidence absent from P19 artifacts: {row.get('check_id')}")
            status=load_json(ROOT/'RUN_STATUS.json')
            flags=status.get('flags',{})
            stage1a_executed=(ROOT/'results/stage1a-confirmatory/evaluator-result-manifest.json').is_file()
            stage1b_executed=(ROOT/'results/stage1b-confirmatory/evaluator-result-manifest.json').is_file()
            if stage1b_executed:
                expected_profile='END_TO_END'
            elif flags.get('planner_architecture_go') is False:
                expected_profile='PLANNER_DIAGNOSTIC_ONLY'
            elif stage1a_executed:
                expected_profile='INTERFACE_STOP'
            else:
                expected_profile='PLANNER_STOP'
            if obj.get('audit_profile')!=expected_profile: errors.append('final audit profile differs from executed stages and locked state-machine flags')
        except Exception as exc: errors.append(f'final audit semantic validation failed: {exc}')
    if fnmatch.fnmatch(rel,'reports/sample-size-*.json'):
        found=_load_artifact_by_sha(report,obj.get('sample_size_input_sha256'))
        if found is None: return ['sample-size input artifact missing from same phase evidence']
        from validation.statistics_validator import validate_sample_size_report
        errors.extend(validate_sample_size_report(obj,found[1]))
    if fnmatch.fnmatch(rel,'analysis/sample-size-inputs/*.json'):
        try:
            raw_path=safe_path(obj['pilot_result_manifest_path'])
            if not raw_path.is_file() or digest(raw_path)!=obj['pilot_result_manifest_sha256']:
                errors.append('SampleSizeInput pilot result manifest path/hash mismatch')
            builder=ROOT/'analysis/build_sample_size_input.py'
            if not builder.is_file() or digest(builder)!=obj['input_builder_sha256']:
                errors.append('SampleSizeInput builder code hash mismatch')
            else:
                cp=subprocess.run([sys.executable,str(builder),'--verify-existing',str(safe_path(rel)),'--pilot-manifest',str(raw_path),'--stage',obj['stage']],cwd=ROOT,text=True,capture_output=True)
                if cp.returncode: errors.append('SampleSizeInput cannot be reproduced from pilot results by locked builder')
        except Exception as exc: errors.append(f'SampleSizeInput lineage validation failed: {exc}')
    if fnmatch.fnmatch(rel,'analysis/inputs/*.json'):
        try:
            raw_path=safe_path(obj['raw_result_manifest_path'])
            if not raw_path.is_file() or digest(raw_path)!=obj['raw_result_manifest_sha256']:
                errors.append('AnalysisInput raw result manifest path/hash mismatch')
            builder=ROOT/'analysis/build_analysis_input.py'
            if not builder.is_file() or digest(builder)!=obj['analysis_builder_sha256']:
                errors.append('AnalysisInput builder code hash mismatch')
            else:
                cp=subprocess.run([sys.executable,str(builder),'--verify-existing',str(safe_path(rel)),'--raw-manifest',str(raw_path),'--stage',obj['stage']],cwd=ROOT,text=True,capture_output=True)
                if cp.returncode: errors.append('AnalysisInput cannot be reproduced from raw results by locked builder')
        except Exception as exc: errors.append(f'AnalysisInput lineage validation failed: {exc}')
    if fnmatch.fnmatch(rel,'reports/*-decision.json'):
        found=_load_artifact_by_sha(report,obj.get('analysis_input_sha256'))
        if found is None: return errors+['AnalysisInput artifact missing from same phase evidence']
        upstream=None
        if obj.get('stage')=='STAGE1A': upstream_path='reports/planner-decision.json'
        elif obj.get('stage')=='STAGE1B': upstream_path='reports/stage1a-decision.json'
        else: upstream_path=None
        if upstream_path:
            up=safe_path(upstream_path)
            if not up.is_file() or digest(up)!=obj.get('upstream_decision_sha256'):
                errors.append('upstream scientific decision hash/path mismatch')
            else:
                upstream=load_json(up); validate(upstream,'scientific_decision.schema.json')
        if obj.get('analysis_code_sha256')!=analysis_code_digest(): errors.append('analysis_code_sha256 mismatch')
        if obj.get('statistics_contract_sha256')!=digest(ROOT/'docs/statistics/statistics_contract_v1.yaml'): errors.append('statistics_contract_sha256 mismatch')
        if obj.get('raw_result_manifest_sha256') not in {a['sha256'] for a in report.get('artifacts',[])}:
            errors.append('raw result manifest absent from phase artifacts')
        from validation.statistics_validator import validate_scientific_decision
        errors.extend(validate_scientific_decision(obj,found[1],upstream))
    return errors


def verify_decision_ledgers(report: dict, phase: dict, decision: dict, decision_path: Path) -> list[str]:
    errors=[]; artifacts={a['path']:a['sha256'] for a in report.get('artifacts',[])}
    try: rel_decision=decision_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception: return ['DecisionRecord path escapes repository']
    if artifacts.get(rel_decision)!=digest(decision_path): errors.append('DecisionRecord file absent from phase artifacts or hash mismatch')
    required={'reports/gate-ledger.jsonl','reports/decision-log.jsonl'}
    for rel in required:
        path=ROOT/rel
        if artifacts.get(rel)!=(digest(path) if path.is_file() else None): errors.append(f'{rel} absent from phase artifacts or hash mismatch')
        if not path.is_file(): continue
        errors.extend(verify_registered_artifact(rel))
    if errors: return errors
    gate_rows=[json.loads(x) for x in (ROOT/'reports/gate-ledger.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    decision_rows=[json.loads(x) for x in (ROOT/'reports/decision-log.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    nonnull_gate_ids=[x.get('decision_id') for x in gate_rows if x.get('decision_id')]
    log_ids=[x.get('decision_id') for x in decision_rows]
    if len(nonnull_gate_ids)!=len(set(nonnull_gate_ids)): errors.append('gate ledger decision_id values must be globally unique')
    if len(log_ids)!=len(set(log_ids)): errors.append('decision log decision_id values must be globally unique')
    record_hashes=[x.get('record_hash') for x in decision_rows]
    if len(record_hashes)!=len(set(record_hashes)): errors.append('decision log record_hash values must be globally unique')
    gates=[x for x in gate_rows if x.get('run_id')==report['run_id'] and x.get('gate_id')==phase.get('manual_gate_id') and x.get('decision_id')==decision['decision_id']]
    logs=[x for x in decision_rows if x.get('run_id')==report['run_id'] and x.get('decision_id')==decision['decision_id']]
    if len(gates)!=1: errors.append('gate ledger must contain exactly one matching decision entry')
    else:
        expected_status={'APPROVE':'APPROVED','REJECT':'REJECTED','STOP':'STOPPED'}[decision['decision']]
        row=gates[0]
        if row.get('phase_id')!=report['phase_id'] or row.get('status')!=expected_status or row.get('target_hash')!=decision['target_artifact_hash'] or row.get('timestamp')!=decision['timestamp']:
            errors.append('gate ledger entry does not match DecisionRecord')
    if len(logs)!=1: errors.append('decision log must contain exactly one matching entry')
    else:
        row=logs[0]
        if row.get('gate_id')!=decision['gate_id'] or row.get('decision')!=decision['decision'] or row.get('record_hash')!=decision['decision_hash'] or row.get('timestamp')!=decision['timestamp']:
            errors.append('decision log entry does not match DecisionRecord')
    same_gate=[x for x in decision_rows if x.get('run_id')==report['run_id'] and x.get('gate_id')==decision['gate_id']]
    if same_gate:
        latest=max(same_gate,key=lambda x:(x.get('timestamp',''),x.get('decision_id','')))
        if latest.get('decision_id')!=decision['decision_id']:
            errors.append('DecisionRecord is not the latest decision for this run/gate')
    prior_rejects=sum(1 for x in same_gate if x.get('timestamp','') < decision['timestamp'] and x.get('decision')=='REJECT')
    if decision.get('resubmission_index') != prior_rejects:
        errors.append(f"DecisionRecord resubmission_index {decision.get('resubmission_index')} != prior reject count {prior_rejects}")
    return errors


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('phase_report',type=Path); ap.add_argument('--decision',type=Path); ap.add_argument('--allow-dirty',action='store_true'); args=ap.parse_args()
    if subprocess.run(['python',str(ROOT/'validation/verify_release_manifest.py')],cwd=ROOT,capture_output=True).returncode:
        print('bootstrap release manifest mismatch'); return 2
    try:
        report=load_json(args.phase_report); validate(report,'phase_report.schema.json')
    except Exception as e: print(e); return 2
    reg=yaml.safe_load((ROOT/'docs/operator/phase_registry_v1.yaml').read_text(encoding='utf-8'))
    sm=yaml.safe_load((ROOT/'docs/operator/phase_state_machine_v1.yaml').read_text(encoding='utf-8'))
    phase=next((p for p in reg['phases'] if p['phase_id']==report['phase_id']),None)
    if phase is None: print('unknown phase'); return 2
    errors=verify_report_artifact_constraints(report)
    errors.extend(verify_phase_timing(report))
    if report.get('execution_role')!=phase.get('execution_role'):
        errors.append(f"execution_role {report.get('execution_role')} != locked role {phase.get('execution_role')}")
    # Every listed artifact is safe and hashed. Phase report itself is sealed by the evidence commit and cannot self-hash.
    for art in report['artifacts']:
        try: p=safe_path(art['path'])
        except ValueError as exc: errors.append(str(exc)); continue
        if not p.is_file() or digest(p)!=art['sha256']:
            errors.append(f'artifact hash mismatch: {art["path"]}')
            continue
        errors.extend(verify_registered_artifact(art['path']))
        if p.suffix=='.json':
            try: errors.extend(verify_semantic_report(art['path'],load_json(p),report))
            except Exception as exc: errors.append(f'semantic artifact validation failed {art["path"]}: {exc}')
    for rel,expected_hash in report.get('input_hashes',{}).items():
        try: ip=safe_path(rel)
        except ValueError as exc: errors.append(str(exc)); continue
        if not ip.is_file(): errors.append(f'input_hash path missing: {rel}')
        elif digest(ip)!=expected_hash: errors.append(f'input_hash mismatch: {rel}')
    if report['outcome'] is not None and report['outcome'] not in phase['allowed_outcomes']:
        errors.append('outcome not allowed by phase registry')
    errors.extend(verify_next_phase(sm,report))
    gate_required=bool(phase.get('manual_gate_id'))
    current_phase_num=int(report['phase_id'][1:])
    expected_approved_gates={
        row['manual_gate_id']
        for row in reg['phases']
        if row.get('manual_gate_id') and int(row['phase_id'][1:]) < current_phase_num
    }
    if gate_required and report.get('status')=='PASS' and str(report.get('outcome', '')).startswith('APPROVE_'):
        expected_approved_gates.add(phase['manual_gate_id'])
    # Role and independent-judgment checks are semantic, not merely schema checks.
    if report['phase_id']=='P01' and (ROOT/'reports/resource-plan.json').exists():
        try:
            resource_plan=load_json(ROOT/'reports/resource-plan.json'); validate(resource_plan,'resource_plan.schema.json')
            key_registry=load_json(ROOT/'locks/public-keys.json')
            validate(key_registry,'public_key_registry.schema.json')
            errors.extend(validate_role_independence(resource_plan,key_registry))
            errors.extend(validate_resource_plan_semantics(resource_plan))
        except Exception as exc: errors.append(f'invalid resource plan: {exc}')
        errors.extend(verify_trust_topology(expected_run_id=report.get("run_id")))
    if report['phase_id']!='P00' and (ROOT/'reports/resource-plan.json').exists() and phase.get('execution_role')!='OPERATOR':
        try:
            resource_plan_for_role=load_json(ROOT/'reports/resource-plan.json')
            role_key={'BUILDER':'builder','DATA_SEALER':'data_sealer','EVALUATION_RUNNER':'evaluator','AUDITOR':'auditor','STATISTICAL_REVIEWER':'statistical_reviewer'}[phase['execution_role']]
            expected_identity=canonical_identity_hash(resource_plan_for_role[role_key])
            if report.get('executor_identity_sha256')!=expected_identity:
                errors.append('executor_identity_sha256 differs from locked execution role')
        except Exception as exc: errors.append(f'cannot verify execution role identity: {exc}')
    if phase.get('execution_role')=='OPERATOR':
        try:
            trust=load_json(ROOT/'locks/trust-topology.lock.json')
            if report.get('executor_identity_sha256')!=trust.get('operator_public_key_sha256'):
                errors.append('OPERATOR executor_identity_sha256 must equal the external operator key fingerprint')
        except Exception as exc:
            errors.append(f'cannot verify OPERATOR execution identity: {exc}')
    if report['phase_id']=='P06':
        try:
            resource_plan=load_json(ROOT/'reports/resource-plan.json')
            for rel,schema,validator,label in (
                ('reports/statistical-implementation-audit.json','statistical_audit.schema.json',validate_statistical_audit,'statistical'),
                ('reports/independent-implementation-audit.json','implementation_audit.schema.json',validate_implementation_audit,'implementation'),
            ):
                path=ROOT/rel
                if not path.exists():
                    continue
                audit=load_json(path); validate(audit,schema)
                errors.extend(validator(audit,resource_plan))
                if audit.get('run_id')!=report.get('run_id'): errors.append(f'{label} audit run_id differs from phase report')
                if audit.get('reviewed_commit')!=report.get('implementation_commit'): errors.append(f'{label} audit reviewed_commit differs from phase implementation_commit')
        except Exception as exc: errors.append(f'invalid G06 audit package: {exc}')
    decision=None
    if args.decision is not None:
        try:
            decision=load_json(args.decision); validate(decision,'decision_record.schema.json')
            errors.extend(verify_operator_decision(
                decision,
                require_trust_lock=report.get('phase_id') != 'P00',
                expected_run_id=report.get('run_id'),
            ))
            errors.extend(verify_decision_history(
                decision,
                require_trust_lock=report.get('phase_id') != 'P00',
                root=ROOT,
            ))
        except Exception as exc: errors.append(f'invalid DecisionRecord: {exc}')
    if gate_required and report['status']=='WAITING_APPROVAL' and decision is not None:
        errors.append('WAITING_APPROVAL report cannot already contain a decision')
    if gate_required and report['status']=='PASS' and decision is None:
        errors.append('manual gate PASS requires DecisionRecord')
    if decision is not None:
        errors.extend(verify_phase_timing(report, decision))
        if decision['gate_id']!=phase['manual_gate_id'] or decision['phase_outcome'] not in phase['allowed_outcomes']:
            errors.append('decision/gate/outcome mismatch')
        if report['outcome']!=decision['phase_outcome']: errors.append('phase report outcome conflicts with DecisionRecord')
        if decision.get('decision_hash')!=decision_record_hash(decision): errors.append('DecisionRecord decision_hash mismatch')
        artifacts_by_path={a['path']:a['sha256'] for a in report['artifacts']}
        artifact_hashes=set(artifacts_by_path.values())
        if not set(decision.get('evidence_hashes',[])).issubset(artifact_hashes): errors.append('DecisionRecord evidence_hashes absent from phase artifacts')
        if decision.get('target_artifact_hash') not in set(decision.get('evidence_hashes',[])):
            errors.append('DecisionRecord evidence_hashes must include the approval target hash')
        target_path=phase.get('approval_target_output')
        if not target_path: errors.append('manual gate lacks locked approval_target_output')
        elif artifacts_by_path.get(target_path)!=decision['target_artifact_hash']:
            errors.append('decision target hash does not match locked approval target')
        if args.decision is not None:
            errors.extend(verify_decision_ledgers(report,phase,decision,args.decision))
    errors.extend(verify_run_decision_history(
        report.get('run_id', ''),
        require_trust_lock=current_phase_num >= 1,
        expected_approved_gates=expected_approved_gates,
        root=ROOT,
    ))
    blocked=report['status'] in {'BLOCKED','FAIL'} or report['outcome'] in {'BLOCKED','INVALID_RUN'}
    include_post=bool(decision) or (phase.get('approval_mode')=='auto' and report['status']=='PASS')
    errors.extend(verify_checks(phase,report,include_post=include_post,blocked=blocked))
    errors.extend(verify_commands_and_evidence(report,blocked=blocked))
    if not blocked:
        errors.extend(verify_required_inputs(phase, report))
        errors.extend(verify_required_outputs(phase,report,args.phase_report,include_post=include_post))
    # Trust Topology is active from P01; Scientific lock from P02; Implementation lock after approved G06.
    phase_num=current_phase_num
    if phase_num>=1:
        errors.extend(verify_trust_topology(expected_run_id=report.get('run_id')))
    if phase_num>=2:
        errors.extend(verify_lock('scientific',DEFAULT_LOCKS['scientific'],report.get('run_id')))
    implementation_active = phase_num>6 or (report['phase_id']=='P06' and report.get('outcome')=='APPROVE_G06_AUDITS' and report['status']=='PASS')
    if implementation_active:
        errors.extend(verify_lock('implementation',DEFAULT_LOCKS['implementation'],report.get('run_id')))
    if not blocked:
        expected_trust = 'VERIFIED' if phase_num >= 1 else {'NOT_CREATED', 'NOT_APPLICABLE'}
        expected_scientific = 'VERIFIED' if phase_num >= 2 else {'NOT_CREATED', 'NOT_APPLICABLE'}
        expected_implementation = 'VERIFIED' if implementation_active else {'NOT_CREATED', 'NOT_APPLICABLE'}
        for field, expected in (
            ('trust_topology_lock_status', expected_trust),
            ('scientific_lock_status', expected_scientific),
            ('implementation_lock_status', expected_implementation),
        ):
            actual = report.get(field)
            allowed = {expected} if isinstance(expected, str) else expected
            if actual not in allowed:
                errors.append(f'{field}={actual} inconsistent with active lock state; expected one of {sorted(allowed)}')
    if report['status'] in {'PASS','WAITING_APPROVAL'} and report['phase_id']!='P00':
        errors.extend(verify_commit_lineage(report.get('git_commit_before'),report.get('implementation_commit')))
    if not args.allow_dirty:
        cp=subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True)
        if cp.returncode or cp.stdout.strip(): errors.append('working tree not clean')
        try:
            rel_report=args.phase_report.resolve().relative_to(ROOT.resolve()).as_posix()
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            committed=subprocess.check_output(['git','show',f'HEAD:{rel_report}'],cwd=ROOT)
            if committed!=args.phase_report.read_bytes(): errors.append('phase report is not sealed in current HEAD')
            impl=report.get('implementation_commit')
            if impl and impl==head: errors.append('evidence-seal HEAD must differ from implementation_commit')
            if impl and subprocess.run(['git','merge-base','--is-ancestor',impl,head],cwd=ROOT).returncode:
                errors.append('evidence-seal HEAD does not descend from implementation_commit')
        except Exception as exc: errors.append(f'cannot verify evidence-seal commit: {exc}')
    if errors:
        for e in errors: print(e)
        return 2
    print('GATE_EVIDENCE_VALID'); return 0


if __name__=='__main__': raise SystemExit(main())
