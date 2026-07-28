from __future__ import annotations
import fnmatch,hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'release/BOOTSTRAP_MANIFEST.json'
EXCLUDED=['release/BOOTSTRAP_MANIFEST.json','SHA256SUMS.txt','.git/**','RUN_STATUS.md','RUN_STATUS.json','reports/**','results/**','artifacts/**','locks/trust-topology.lock.json','locks/infrastructure-plan.json','locks/public-keys.json','locks/environment.lock.json','locks/llm_model_lock.json','locks/semantic_target_model_lock.json','locks/scientific.lock.json','locks/implementation.lock.json','**/__pycache__/**','**/*.pyc']


def digest(p): return 'sha256:'+hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(obj):
 c=dict(obj); c.pop('manifest_hash',None)
 return 'sha256:'+hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def excluded(rel): return any(fnmatch.fnmatch(rel,p) for p in EXCLUDED)
def expand(pattern):
 if not any(c in pattern for c in '*?['):
  p=ROOT/pattern; return [p] if p.is_file() else []
 prefix=pattern.split('*',1)[0].split('?',1)[0].rstrip('/')
 base=ROOT/prefix if prefix else ROOT
 if base.is_file(): return [base]
 if not base.exists(): base=ROOT
 return [p for p in base.rglob('*') if p.is_file() and fnmatch.fnmatch(p.relative_to(ROOT).as_posix(),pattern)]

def files():
 lock=yaml.safe_load((ROOT/'docs/operator/scientific_lock_v1.yaml').read_text(encoding='utf-8'))
 patterns=list(lock['protected_paths'])
 # The verifier and manifest builder are part of the pre-P00 trust root even if
 # the scientific lock definition is later refactored in a future release.
 patterns += ['docs/operator/scientific_lock_v1.yaml','validation/verify_release_manifest.py','scripts/build_release_manifest.py']
 rows=[]
 for pattern in patterns: rows.extend(expand(pattern))
 return sorted({p for p in rows if not excluded(p.relative_to(ROOT).as_posix())},key=lambda p:p.relative_to(ROOT).as_posix())

def main():
 epoch=int(os.environ.get('SOURCE_DATE_EPOCH','1785000000'))
 generated=datetime.fromtimestamp(epoch,tz=timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
 obj={'schema_version':'work-planner-release/1.1','release_id':'work-planner-v1.18-v2.18','implementation_spec':'1.18','stage1_runbook':'2.18','generated_at':generated,'scope':'SCIENTIFIC_TRUST_ROOT_ONLY','files':{p.relative_to(ROOT).as_posix():digest(p) for p in files()},'excluded_paths':EXCLUDED,'manifest_hash':''}
 obj['manifest_hash']=canonical(obj); OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
 print(f'WROTE {OUT} files={len(obj["files"])}')
if __name__=='__main__': main()
