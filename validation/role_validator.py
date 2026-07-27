from __future__ import annotations

import base64
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


def canonical_identity_hash(identity: dict) -> str:
    data=json.dumps(identity,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return 'sha256:'+hashlib.sha256(data).hexdigest()


def canonical_registry_hash(registry: dict) -> str:
    payload=dict(registry); payload.pop('registry_hash',None)
    data=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return 'sha256:'+hashlib.sha256(data).hexdigest()



def _sha_text(value: str) -> str:
    return 'sha256:'+hashlib.sha256(value.encode('utf-8')).hexdigest()


def role_challenge_bytes(run_id: str, row: dict, identity: dict) -> bytes:
    payload={
      'domain':'work-planner-role-attestation-v1',
      'run_id':run_id,
      'role':row['role'],
      'key_id':row['key_id'],
      'identity_sha256':canonical_identity_hash(identity),
      'challenge_nonce_b64':row['challenge_nonce_b64'],
      'environment_identity_sha256':_sha_text(identity['environment_identity']),
      'credential_principal_sha256':_sha_text(identity['credential_principal']),
    }
    return json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')

def validate_role_independence(resource_plan: dict, key_registry: dict | None=None) -> list[str]:
    errors: list[str] = []
    roles = {k: resource_plan[k] for k in ('builder','data_sealer','evaluator','auditor','statistical_reviewer')}
    principals = [r['credential_principal'] for r in roles.values()]
    if len(principals) != len(set(principals)):
        errors.append('credential_principal must be unique across execution roles')
    envs = {k:v['environment_identity'] for k,v in roles.items()}
    if len(set(envs.values())) != len(envs):
        errors.append('environment_identity must be unique across all execution roles')
    if roles['builder']['reviewer_type'] != 'MODEL':
        errors.append('builder must be a MODEL identity')
    for key in ('data_sealer', 'evaluator'):
        if roles[key]['reviewer_type'] != 'SERVICE_PROCESS':
            errors.append(f'{key} must be a deterministic SERVICE_PROCESS, not an LLM judgment role')
    for key in ('auditor', 'statistical_reviewer'):
        if roles[key]['reviewer_type'] == 'SERVICE_PROCESS':
            errors.append(f'{key} requires independent judgment and cannot be SERVICE_PROCESS')
    builder_family = roles['builder']['model_family']
    for key in ('auditor','statistical_reviewer'):
        r=roles[key]
        independent = r['reviewer_type']=='HUMAN_STATISTICIAN' or r['model_family'] != builder_family
        if not independent:
            errors.append(f'{key} must be HUMAN_STATISTICIAN or use a different model_family from builder')
    if key_registry is not None:
        if key_registry.get('registry_hash')!=canonical_registry_hash(key_registry):
            errors.append('public key registry_hash mismatch')
        rows=key_registry.get('keys',[])
        by_id={row.get('key_id'):row for row in rows}
        if len(by_id)!=len(rows): errors.append('public key IDs must be unique')
        mapping={'data_sealer':'DATA_SEALER','evaluator':'EVALUATION_RUNNER','auditor':'AUDITOR','statistical_reviewer':'STATISTICAL_REVIEWER'}
        for key,expected_role in mapping.items():
            identity=roles[key]; key_id=identity['public_signing_key_id']; row=by_id.get(key_id)
            if row is None:
                errors.append(f'{key}: public_signing_key_id absent from registry'); continue
            if row.get('role')!=expected_role: errors.append(f'{key}: public key role mismatch')
            if row.get('identity_sha256')!=canonical_identity_hash(identity): errors.append(f'{key}: public key identity hash mismatch')
            if row.get('environment_identity_sha256')!=_sha_text(identity['environment_identity']): errors.append(f'{key}: environment identity attestation mismatch')
            if row.get('credential_principal_sha256')!=_sha_text(identity['credential_principal']): errors.append(f'{key}: credential principal attestation mismatch')
            try:
                pub=Ed25519PublicKey.from_public_bytes(base64.b64decode(row['public_key_b64']))
                pub.verify(base64.b64decode(row['challenge_signature_b64']),role_challenge_bytes(key_registry['run_id'],row,identity))
            except Exception as exc:
                errors.append(f'{key}: role challenge signature invalid: {exc}')
    return errors
