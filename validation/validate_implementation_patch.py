from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from validation.verify_lock import DEFAULT_LOCKS, verify as verify_lock
from validation.hashing import hash_json


def sha_bytes(data: bytes)->str:
    return 'sha256:'+hashlib.sha256(data).hexdigest()


def schema_bundle():
    reg=Registry(); schemas={}
    for p in sorted((ROOT/'docs/schemas').glob('*.json')):
        o=json.loads(p.read_text()); schemas[p.name]=o; reg=reg.with_resource(o['$id'],Resource.from_contents(o))
    return schemas,reg


def validate_patch(obj: dict)->list[str]:
    schemas,reg=schema_bundle(); errors=[e.message for e in Draft202012Validator(schemas['implementation_patch.schema.json'],registry=reg,format_checker=FormatChecker()).iter_errors(obj)]
    if errors: return errors
    errors.extend(verify_lock('scientific',DEFAULT_LOCKS['scientific']))
    science=yaml.safe_load((ROOT/'docs/operator/scientific_lock_v1.yaml').read_text())
    implementation=yaml.safe_load((ROOT/'docs/operator/implementation_lock_v1.yaml').read_text())
    patch_policy=implementation['pre_lock_patch_window']
    allowed_patch_globs=patch_policy.get('allowed_path_globs',[])
    for rel in obj['changed_paths']:
        if any(fnmatch.fnmatch(rel,p) for p in science['protected_paths']):
            errors.append(f'scientific-lock path changed by implementation patch: {rel}')
        if not any(fnmatch.fnmatch(rel,p) for p in allowed_patch_globs):
            errors.append(f'changed path is outside the locked implementation-only allowlist: {rel}')
    clone=dict(obj); clone.pop('record_hash',None)
    if obj.get('record_hash')!=hash_json(clone): errors.append('implementation patch record_hash mismatch')
    for field in ('base_commit','patched_commit'):
        cp_commit=subprocess.run(['git','cat-file','-e',f"{obj[field]}^{{commit}}"],cwd=ROOT,capture_output=True)
        if cp_commit.returncode: errors.append(f'{field} is not a repository commit')
    if not errors:
        cp_ancestor=subprocess.run(['git','merge-base','--is-ancestor',obj['base_commit'],obj['patched_commit']],cwd=ROOT,capture_output=True)
        if cp_ancestor.returncode: errors.append('patched_commit does not descend from base_commit')
    preflight=ROOT/obj['preflight_report_path']
    if not preflight.is_file() or 'sha256:'+hashlib.sha256(preflight.read_bytes()).hexdigest()!=obj['preflight_report_sha256']:
        errors.append('preflight report path/hash mismatch')
    else:
        try:
            preflight_obj=json.loads(preflight.read_text())
            if preflight_obj.get('status')!='PASS': errors.append('implementation patch requires PASS preflight')
        except Exception as exc: errors.append(f'invalid preflight report: {exc}')
    cp=subprocess.run(['git','diff','--binary',obj['base_commit'],obj['patched_commit'],'--'],cwd=ROOT,capture_output=True)
    if cp.returncode: errors.append('cannot compute declared git diff')
    else:
        if sha_bytes(cp.stdout)!=obj['diff_sha256']: errors.append('diff_sha256 mismatch')
        diff_text=cp.stdout.decode('utf-8',errors='replace')
        forbidden_markers=(
            'threshold:', 'stage_gates:', 'loss_weights:', 'd_model:', 'latent_dim:',
            'intent_catalog', 'semantic_target', 'blocks_world', 'planner_architecture',
            'dataset_split', 'final_seeds:', 'prompt candidate', 'comparison:', 'rule:'
        )
        if any(marker in diff_text for marker in forbidden_markers):
            errors.append('diff contains outcome-relevant markers forbidden for implementation-only patches')
        names=subprocess.check_output(['git','diff','--name-only',obj['base_commit'],obj['patched_commit'],'--'],cwd=ROOT,text=True).splitlines()
        if sorted(names)!=sorted(obj['changed_paths']): errors.append('changed_paths differ from git diff')
    forbidden_outcome_paths=[
        ROOT/'reports/planner-pilot.json',ROOT/'reports/stage1a-pilot.json',ROOT/'reports/stage1b-pilot.json',
        ROOT/'results/planner-confirmatory',ROOT/'results/stage1a-confirmatory',ROOT/'results/stage1b-confirmatory',
    ]
    forbidden_outcome_paths.extend((ROOT/'sealed').glob('**/evaluator-result-manifest.json') if (ROOT/'sealed').exists() else [])
    materialized=[]
    for path in forbidden_outcome_paths:
        if path.is_file(): materialized.append(path)
        elif path.is_dir() and any(x.is_file() and x.name!='.gitkeep' for x in path.rglob('*')): materialized.append(path)
    if materialized:
        errors.append('implementation patch window closed after pilot/confirmatory outcomes exist: '+', '.join(str(x.relative_to(ROOT)) for x in materialized))
    return errors


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('patch',type=Path); args=ap.parse_args()
    try: obj=json.loads(args.patch.read_text())
    except Exception as exc: print(exc); return 2
    errors=validate_patch(obj)
    if errors:
        print('\n'.join(errors)); return 2
    print('IMPLEMENTATION_PATCH_VALID'); return 0

if __name__=='__main__': raise SystemExit(main())
