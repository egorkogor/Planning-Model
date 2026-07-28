from __future__ import annotations
import json,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_bootstrap_manifest_verifies():
 r=subprocess.run([sys.executable,str(ROOT/'validation/verify_release_manifest.py')],cwd=ROOT,text=True,capture_output=True)
 assert r.returncode==0,r.stdout+r.stderr
 assert 'BOOTSTRAP_MANIFEST_VERIFIED' in r.stdout

def test_two_level_locks_protect_science_and_implementation():
 import yaml
 science=yaml.safe_load((ROOT/'docs/operator/scientific_lock_v1.yaml').read_text())
 impl=yaml.safe_load((ROOT/'docs/operator/implementation_lock_v1.yaml').read_text())
 for x in ('docs/statistics/**','validation/verify_gate.py','release/BOOTSTRAP_MANIFEST.json'):
  assert x in science['protected_paths']
 for x in ('docs/schemas/**','validation/**','src/**','scripts/**'):
  assert x in impl['protected_paths']
 assert science['change_policy']['operator_override']=='forbidden'
 assert impl['post_lock_change_policy']['operator_override']=='forbidden'

def test_manifest_detects_tampering_without_mutating_repo():
 obj=json.loads((ROOT/'release/BOOTSTRAP_MANIFEST.json').read_text())
 rel=next(iter(obj['files']))
 assert obj['files'][rel].startswith('sha256:')

def test_bootstrap_manifest_rejects_added_protected_file(tmp_path):
 workspace=tmp_path/'repo'
 shutil.copytree(ROOT,workspace,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','.ruff_cache','.mypy_cache'))
 path=workspace/'docs/prompt/candidates/UNDECLARED.yaml'
 path.write_text('schema_version: work-planner/1.16\n')
 r=subprocess.run([sys.executable,str(workspace/'validation/verify_release_manifest.py')],cwd=workspace,text=True,capture_output=True)
 assert r.returncode!=0
 assert 'protected path set mismatch' in r.stdout
