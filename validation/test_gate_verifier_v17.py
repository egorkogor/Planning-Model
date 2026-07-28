from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
import pytest

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from validation.role_validator import canonical_identity_hash, canonical_registry_hash, role_challenge_bytes
from validation.operator_decision_validator import sign_record

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT


@pytest.fixture(autouse=True)
def isolated_repository(tmp_path: Path):
    """Run each gate test in a fresh repository to prevent cross-test trust state."""
    global ROOT
    workspace=tmp_path/'repo'
    shutil.copytree(
        SOURCE_ROOT,
        workspace,
        ignore=shutil.ignore_patterns('.git','__pycache__','.pytest_cache','.pytest-tmp','.ruff_cache','.mypy_cache'),
    )
    subprocess.run(['git','init','-q'],cwd=workspace,check=True)
    subprocess.run(['git','config','user.email','protocol-tests@example.invalid'],cwd=workspace,check=True)
    subprocess.run(['git','config','user.name','Protocol Tests'],cwd=workspace,check=True)
    subprocess.run(['git','add','-A'],cwd=workspace,check=True)
    subprocess.run(['git','commit','-q','-m','fixture baseline'],cwd=workspace,check=True)
    ROOT=workspace
    import validation.verify_gate as gate_module
    import validation.trust_topology_validator as trust_module
    import validation.confirmatory_lineage_validator as lineage_module
    import validation.operator_decision_validator as decision_module
    import validation.code_fingerprint as fingerprint_module
    originals=(
        gate_module.ROOT,
        (trust_module.ROOT,trust_module.POLICY,trust_module.DEFAULT_LOCK),
        lineage_module.ROOT,
        decision_module.ROOT,
        fingerprint_module.ROOT,
    )
    gate_module.ROOT=workspace; gate_module.schema_bundle.cache_clear()
    trust_module.ROOT=workspace
    trust_module.POLICY=workspace/'docs/operator/trust_topology_lock_v1.yaml'
    trust_module.DEFAULT_LOCK=workspace/'locks/trust-topology.lock.json'
    lineage_module.ROOT=workspace; lineage_module._schemas.cache_clear()
    decision_module.ROOT=workspace; decision_module._decision_schema.cache_clear()
    fingerprint_module.ROOT=workspace
    yield
    gate_module.ROOT=originals[0]; gate_module.schema_bundle.cache_clear()
    trust_module.ROOT,trust_module.POLICY,trust_module.DEFAULT_LOCK=originals[1]
    lineage_module.ROOT=originals[2]; lineage_module._schemas.cache_clear()
    decision_module.ROOT=originals[3]; decision_module._decision_schema.cache_clear()
    fingerprint_module.ROOT=originals[4]
    ROOT=SOURCE_ROOT


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def role(role_name: str, principal: str, family: str = 'builder-family') -> dict:
    reviewer='SERVICE_PROCESS' if role_name in {'DATA_SEALER','EVALUATION_RUNNER'} else 'MODEL'
    if role_name in {'STATISTICAL_REVIEWER','AUDITOR'}: family='independent-family'
    return {
      'role':role_name,'reviewer_type':reviewer,'agent_provider':'test-provider','model_family':family,
      'model_revision':'test-rev','system_prompt_hash':'sha256:'+'1'*64,'environment_identity':principal+'-env',
      'credential_principal':principal,'public_signing_key_id':principal+'-key',
    }


def resource_plan(approval: bool) -> dict:
    return {
        "schema_version":"work-planner-infra/1.0","run_id":"run-gate-test",
        "builder":role('BUILDER','builder'),"data_sealer":role('DATA_SEALER','sealer'),
        "evaluator":role('EVALUATION_RUNNER','evaluator'),"auditor":role('AUDITOR','auditor','audit-family'),
        "statistical_reviewer":role('STATISTICAL_REVIEWER','stat-review','stat-family'),
        "machine_checks":{k:"PASS" for k in ("cpu","ram","disk","gpu","credentials","workspace","role_separation")},
        "estimated_cost":1 if approval else 0,"currency":"USD","requires_operator_budget_approval":approval,
        "capacity_limits":{"maximum_gpu_seconds":1000000.0,"maximum_storage_bytes":1000000000,"gpu_hour_cost":0.0},
        "plan_hash":"sha256:"+"0"*64,
    }


def _phase() -> dict:
    reg=yaml.safe_load((ROOT/'docs/operator/phase_registry_v1.yaml').read_text())
    return next(x for x in reg['phases'] if x['phase_id']=='P01')


def write_required_outputs(plan: dict) -> list[dict]:
    rows=[]
    scope=ROOT/'artifacts/scope.md'
    scope.parent.mkdir(parents=True,exist_ok=True)
    scope.write_text('work-planner/1.18 runbook 2.18 locked scope\n')
    for rel in _phase()['required_outputs']:
        p=ROOT/rel.rstrip('/'); p.parent.mkdir(parents=True,exist_ok=True)
        if rel in {'reports/resource-plan.json','locks/infrastructure-plan.json'}:
            p.write_text(json.dumps(plan))
        elif rel=='locks/public-keys.json':
            role_rows={
              'sealer':plan['data_sealer'],'evaluator':plan['evaluator'],
              'auditor':plan['auditor'],'stat-review':plan['statistical_reviewer'],
            }
            role_names={'sealer':'DATA_SEALER','evaluator':'EVALUATION_RUNNER','auditor':'AUDITOR','stat-review':'STATISTICAL_REVIEWER'}
            key_rows=[]
            for name,identity in role_rows.items():
                private=Ed25519PrivateKey.generate()
                public=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
                nonce=hashlib.sha256(f'nonce:{name}'.encode()).digest()
                row={
                  'key_id':f'{name}-key','role':role_names[name],'algorithm':'ed25519',
                  'public_key_b64':base64.b64encode(public).decode(),
                  'identity_sha256':canonical_identity_hash(identity),
                  'challenge_nonce_b64':base64.b64encode(nonce).decode(),
                  'attested_at':'2026-07-25T00:00:00Z',
                  'environment_identity_sha256':'sha256:'+hashlib.sha256(identity['environment_identity'].encode()).hexdigest(),
                  'credential_principal_sha256':'sha256:'+hashlib.sha256(identity['credential_principal'].encode()).hexdigest(),
                }
                row['challenge_signature_b64']=base64.b64encode(private.sign(role_challenge_bytes('run-gate-test',row,identity))).decode()
                key_rows.append(row)
            registry={
              'schema_version':'work-planner-keys/1.0','run_id':'run-gate-test',
              'keys':key_rows,
              'created_at':'2026-07-25T00:00:00Z','registry_hash':'sha256:'+'0'*64,
            }
            registry['registry_hash']=canonical_registry_hash(registry)
            p.write_text(json.dumps(registry))
        elif rel=='locks/trust-topology.lock.json':
            private=Ed25519PrivateKey.generate()
            private_raw=private.private_bytes(serialization.Encoding.Raw,serialization.PrivateFormat.Raw,serialization.NoEncryption())
            public_raw=private.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw)
            private_path=ROOT.parent/'operator-private.b64'; public_path=ROOT.parent/'operator-public.b64'
            private_path.write_bytes(base64.b64encode(private_raw)); public_path.write_bytes(base64.b64encode(public_raw))
            import validation.trust_topology_validator as trust_module
            trust_module.create(
              'run-gate-test','test-operator',private_path,public_path,
              ROOT/'locks/trust-topology.lock.json',
            )
        elif rel=='reports/self-review-P01.json':
            p.write_text(json.dumps({
              'schema_version':'work-planner-agent/1.2','run_id':'run-gate-test','phase_id':'P01','cycle_count':1,
              'test_evidence_hashes':['sha256:'+'3'*64],'findings':[],'clean_restore_status':'NOT_APPLICABLE',
              'final_status':'PASS','report_hash':'sha256:'+'4'*64,
            }))
        elif rel=='RUN_STATUS.json':
            p.write_text(json.dumps({
              'schema_version':'work-planner-agent/1.2','run_id':'run-gate-test','phase_id':'P01','status':'PASS',
              'progress_percent':5,'latest_action':'resource planning','latest_check':'P01_pre_04',
              'trust_topology_lock_status':'VERIFIED','scientific_lock_status':'NOT_CREATED','implementation_lock_status':'NOT_CREATED','blindness_status':'NOT_APPLICABLE',
              'flags':{'planner_architecture_go':None,'stage1b_eligible':None,'interface_go':None,'end_to_end_go':None},
              'resources':{},'cost':{'estimated':0,'currency':'USD'},'eta_minutes':None,'risks':[],
              'operator_action':None,'updated_at':'2026-07-25T00:01:00Z',
            }))
        elif rel=='RUN_STATUS.md':
            p.write_text('# RUN STATUS\n\nP01 PASS\n')
        elif rel=='reports/phase-P01.json': continue
        else: p.write_text('{}\n')
        rows.append({'path':rel,'sha256':sha(p),'schema_id':None})
    unsigned={
      'schema_version':'work-planner-agent/1.3','decision_id':'D-0001','run_id':'run-gate-test',
      'gate_id':'G00_SCOPE','timestamp':'2026-07-24T23:59:00Z','decision':'APPROVE',
      'phase_outcome':'APPROVE_SCOPE','target_artifact_hash':sha(scope),'evidence_hashes':[sha(scope)],
      'operator_note':'fixture scope approval','resubmission_index':0,'previous_decision_hash':None,
      'decision_hash':'sha256:'+'0'*64,'operator_key_id':'test-operator',
      'operator_public_key_sha256':'sha256:'+'0'*64,'signature_algorithm':'ed25519','signature':'',
    }
    unsigned_path=ROOT.parent/'unsigned-p00.json'; unsigned_path.write_text(json.dumps(unsigned))
    decisions=ROOT/'decisions'; decisions.mkdir(exist_ok=True)
    sign_record(unsigned_path,decisions/'D-0001.json',ROOT.parent/'operator-private.b64',ROOT.parent/'operator-public.b64',root=ROOT)
    return rows


def phase_report(outcome: str | None, plan: dict, status: str = 'PASS') -> dict:
    artifacts=write_required_outputs(plan)
    checks=[]
    for section in ('pre_gate_checks','execution_checks'):
        for row in _phase()[section]: checks.append({'check_id':row['check_id'],'status':'PASS','evidence':[artifacts[0]['path']]})
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    report={
      'schema_version':'work-planner-agent/1.2','run_id':'run-gate-test','phase_id':'P01','execution_role':'BUILDER','executor_identity_sha256':canonical_identity_hash(plan['builder']),'status':status,
      'started_at':'2026-07-25T00:00:00Z','finished_at':'2026-07-25T00:01:00Z','input_hashes':{},
      'trust_topology_lock_status':'VERIFIED','scientific_lock_status':'NOT_CREATED','implementation_lock_status':'NOT_CREATED','git_commit_before':head,'implementation_commit':head,
      'commands':[],'checks':checks,'artifacts':artifacts,
      'resource_usage':{'elapsed_minutes':1,'cpu_peak_percent':1,'ram_peak_gb':1,'gpu_hours':None,'disk_peak_gb':1,'estimated_cost':0,'currency':'USD'},
      'deviations':[],'outcome':outcome,'next_phase':('P02' if outcome else None),'operator_action_required':None,
    }
    phase_path=ROOT/'reports/phase-P01.json'; phase_path.write_text(json.dumps(report))
    # The phase report is verified by its evidence-seal commit, not by a self-referential artifact row.
    return report


def run_verifier(report_path: Path) -> subprocess.CompletedProcess[str]:
    # File-backed capture avoids pipe inheritance by verifier grandchildren,
    # which can otherwise block communicate() after the parent exits.
    import tempfile
    env={**__import__('os').environ,'PLANNER_OPERATOR_TRUST_PUBLIC_KEY':str(ROOT.parent/'operator-public.b64')}
    with tempfile.TemporaryFile(mode='w+',encoding='utf-8') as stdout, tempfile.TemporaryFile(mode='w+',encoding='utf-8') as stderr:
        cp=subprocess.run(
            [sys.executable,str(ROOT/'validation/verify_gate.py'),str(report_path),'--allow-dirty'],
            cwd=ROOT,text=True,stdout=stdout,stderr=stderr,env=env,timeout=30,
        )
        stdout.seek(0); stderr.seek(0)
        return subprocess.CompletedProcess(cp.args,cp.returncode,stdout.read(),stderr.read())


def test_p01_waiting_approval_path_needs_no_decision(tmp_path: Path) -> None:
    plan=resource_plan(False); report=phase_report(None,plan,status='WAITING_APPROVAL')
    path=tmp_path/'phase.json'; path.write_text(json.dumps(report))
    result=run_verifier(path)
    assert result.returncode==0,result.stdout+result.stderr


def test_phase_report_cannot_misstate_active_lock_status(tmp_path: Path) -> None:
    plan=resource_plan(False); report=phase_report(None,plan,status='WAITING_APPROVAL')
    report['trust_topology_lock_status']='NOT_CREATED'
    path=tmp_path/'phase.json'; path.write_text(json.dumps(report))
    result=run_verifier(path)
    assert result.returncode!=0
    assert 'trust_topology_lock_status=NOT_CREATED inconsistent with active lock state' in result.stdout


def test_p01_paid_path_cannot_pass_without_decision(tmp_path: Path) -> None:
    plan=resource_plan(True); report=phase_report('APPROVE_TRUST_AND_RESOURCES',plan)
    path=tmp_path/'phase.json'; path.write_text(json.dumps(report))
    result=run_verifier(path)
    assert result.returncode!=0
    assert 'manual gate PASS requires DecisionRecord' in result.stdout


def test_gate_verifier_rejects_empty_checks_and_missing_outputs(tmp_path: Path) -> None:
    plan=resource_plan(False); report=phase_report(None,plan,status='WAITING_APPROVAL')
    report['checks']=[]; report['artifacts']=report['artifacts'][:1]
    path=tmp_path/'phase.json'; path.write_text(json.dumps(report))
    result=run_verifier(path)
    assert result.returncode!=0
    assert 'check set mismatch' in result.stdout or 'required output' in result.stdout


def test_p03_and_p07_empty_evidence_are_fail_closed():
    from validation.verify_gate import verify_checks, verify_required_outputs
    reg=yaml.safe_load((ROOT/'docs/operator/phase_registry_v1.yaml').read_text())
    for pid in ('P03','P07'):
        phase=next(x for x in reg['phases'] if x['phase_id']==pid)
        report={'checks':[],'artifacts':[]}
        check_errors=verify_checks(phase,report,include_post=False,blocked=False)
        output_errors=verify_required_outputs(phase,report,ROOT/f'reports/phase-{pid}.json',include_post=False)
        assert any('check set mismatch' in x for x in check_errors)
        assert output_errors


def test_gate_verifier_rejects_bad_evidence_command_and_input_hash(tmp_path: Path) -> None:
    plan=resource_plan(False); report=phase_report(None,plan,status='WAITING_APPROVAL')
    report['checks'][0]['evidence']=['reports/not-present.log']
    report['commands']=[{'command':'false','exit_code':1,'stdout_artifact':'logs/no.stdout','stderr_artifact':'logs/no.stderr'}]
    report['input_hashes']={'README.md':'sha256:'+'f'*64}
    path=tmp_path/'phase.json'; path.write_text(json.dumps(report))
    result=run_verifier(path)
    assert result.returncode!=0
    assert 'check evidence' in result.stdout
    assert 'non-zero exit_code' in result.stdout
    assert 'input_hash mismatch' in result.stdout


def test_scientific_bootstrap_manifest_rejects_outcome_relevant_analysis_change(tmp_path: Path) -> None:
    # The release trust root intentionally excludes implementation-only code;
    # its mutation is governed by patch records and the later Implementation lock.
    target=ROOT/'analysis/sample_size.py'; original=target.read_text()
    try:
        target.write_text(original+'\n# implementation patch probe\n')
        result=subprocess.run([sys.executable,str(ROOT/'validation/verify_release_manifest.py')],cwd=ROOT,text=True,capture_output=True)
        assert result.returncode!=0
        assert 'bootstrap mismatch' in result.stdout
    finally:
        target.write_text(original)


def test_phase_report_self_hash_is_rejected() -> None:
    from validation.verify_gate import verify_report_artifact_constraints
    plan=resource_plan(False); report=phase_report(None,plan,status='WAITING_APPROVAL')
    report['artifacts'].append({'path':'reports/phase-P01.json','sha256':'sha256:'+'a'*64,'schema_id':None})
    errors=verify_report_artifact_constraints(report)
    assert 'phase report must not contain a self-referential artifact hash' in errors

def test_required_output_directory_requires_every_file_hashed():
    from validation.verify_gate import verify_required_outputs, digest
    import shutil
    reg=yaml.safe_load((ROOT/'docs/operator/phase_registry_v1.yaml').read_text())
    phase=next(x for x in reg['phases'] if x['phase_id']=='P03')
    target=ROOT/'src/contracts'
    shutil.rmtree(target,ignore_errors=True); target.mkdir(parents=True)
    try:
        a=target/'a.py'; b=target/'b.py'; a.write_text('a=1\n'); b.write_text('b=2\n')
        report={'artifacts':[{'path':'src/contracts/a.py','sha256':digest(a),'schema_id':None}]}
        errors=verify_required_outputs(phase,report,ROOT/'reports/phase-P03.json',include_post=False)
        assert any('required output file absent from phase artifacts: src/contracts/b.py' in x for x in errors)
    finally:
        shutil.rmtree(target,ignore_errors=True)
