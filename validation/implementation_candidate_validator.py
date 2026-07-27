from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='work-planner-hash/1.0'


def file_digest(path: Path)->str:
    return 'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(obj: dict)->str:
    payload=dict(obj); payload.pop('candidate_hash',None)
    wrapped={'schema':SCHEMA,'kind':'implementation_lock_candidate','value':payload}
    raw=json.dumps(wrapped,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return 'sha256:'+hashlib.sha256(raw).hexdigest()


def commit_exists(commit: str)->bool:
    return subprocess.run(['git','cat-file','-e',f'{commit}^{{commit}}'],cwd=ROOT,capture_output=True).returncode==0


def commit_files(commit: str, patterns: list[str])->list[str]:
    cp=subprocess.run(['git','ls-tree','-r','--name-only',commit],cwd=ROOT,text=True,capture_output=True)
    if cp.returncode: raise ValueError('cannot list reviewed commit')
    return sorted(rel for rel in cp.stdout.splitlines() if any(fnmatch.fnmatch(rel,p) for p in patterns))


def commit_blob_digest(commit: str, rel: str)->str:
    cp=subprocess.run(['git','show',f'{commit}:{rel}'],cwd=ROOT,capture_output=True)
    if cp.returncode: raise ValueError(f'cannot read reviewed commit blob: {rel}')
    return 'sha256:'+hashlib.sha256(cp.stdout).hexdigest()


def _validate_bound_audit(obj: dict, *, path: Path, hash_field: str, label: str, commit: str)->list[str]:
    errors=[]
    if not path.is_file():
        return [f'{label} file missing']
    if obj.get(hash_field)!=file_digest(path): errors.append(f'{label} file hash mismatch')
    try:
        audit=json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        return errors+[f'invalid {label}: {exc}']
    if audit.get('decision')!='APPROVE': errors.append(f'{label} decision is not APPROVE')
    if audit.get('reviewed_commit')!=commit: errors.append(f'{label} reviewed_commit differs from implementation candidate')
    if audit.get('run_id')!=obj.get('run_id'): errors.append(f'{label} run_id differs from implementation candidate')
    return errors


def validate_candidate(obj: dict)->list[str]:
    errors=[]
    if obj.get('candidate_hash')!=canonical_hash(obj): errors.append('implementation candidate self-hash mismatch')
    commit=obj.get('reviewed_commit','')
    if not commit_exists(commit): return errors+['reviewed_commit is not available']
    policy=ROOT/'docs/operator/implementation_lock_v1.yaml'
    if obj.get('implementation_policy_sha256')!=file_digest(policy): errors.append('implementation policy hash mismatch')
    trust=ROOT/'locks/trust-topology.lock.json'
    if not trust.is_file() or obj.get('trust_topology_lock_sha256')!=file_digest(trust): errors.append('trust topology lock file hash mismatch')
    scientific=ROOT/'locks/scientific.lock.json'
    if not scientific.is_file() or obj.get('scientific_lock_sha256')!=file_digest(scientific): errors.append('scientific lock file hash mismatch')
    preflight=ROOT/'reports/preflight-final.json'
    if not preflight.is_file() or obj.get('preflight_report_sha256')!=file_digest(preflight): errors.append('preflight report hash mismatch')
    else:
        try:
            if json.loads(preflight.read_text()).get('status')!='PASS': errors.append('preflight-final status is not PASS')
        except Exception as exc: errors.append(f'invalid preflight-final report: {exc}')
    errors.extend(_validate_bound_audit(
        obj,path=ROOT/'reports/statistical-implementation-audit.json',
        hash_field='statistical_audit_sha256',label='statistical audit',commit=commit,
    ))
    errors.extend(_validate_bound_audit(
        obj,path=ROOT/'reports/independent-implementation-audit.json',
        hash_field='implementation_audit_sha256',label='implementation audit',commit=commit,
    ))
    patterns=yaml.safe_load(policy.read_text(encoding='utf-8'))['protected_paths']
    try: expected_paths=commit_files(commit,patterns)
    except Exception as exc: return errors+[str(exc)]
    rows=obj.get('protected_files',[]); row_paths=[r.get('path') for r in rows]
    if len(row_paths)!=len(set(row_paths)): errors.append('implementation candidate protected_files contains duplicates')
    if sorted(row_paths)!=expected_paths:
        errors.append(f'implementation candidate path set mismatch: missing={sorted(set(expected_paths)-set(row_paths))}, extra={sorted(set(row_paths)-set(expected_paths))}')
    for row in rows:
        rel=row.get('path')
        if rel in expected_paths:
            try:
                if row.get('sha256')!=commit_blob_digest(commit,rel): errors.append(f'implementation candidate blob hash mismatch: {rel}')
            except Exception as exc: errors.append(str(exc))
    return errors
