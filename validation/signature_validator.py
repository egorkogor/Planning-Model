from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT=Path(__file__).resolve().parents[1]


def canonical_unsigned_bytes(obj: dict) -> bytes:
    payload=dict(obj)
    payload.pop('signature',None)
    payload.pop('manifest_hash',None)
    payload.pop('report_hash',None)
    return json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')


def payload_hash(obj: dict) -> str:
    return 'sha256:'+hashlib.sha256(canonical_unsigned_bytes(obj)).hexdigest()


def load_registry(path: Path | None=None) -> dict:
    path=path or ROOT/'locks/public-keys.json'
    return json.loads(path.read_text(encoding='utf-8'))


def verify_signed_manifest(obj: dict, expected_role: str, registry: dict | None=None, *, hash_field: str='manifest_hash') -> list[str]:
    errors=[]
    if obj.get('signature_algorithm')!='ed25519': errors.append('signature_algorithm must be ed25519')
    if obj.get(hash_field)!=payload_hash(obj): errors.append(f'signed {hash_field} mismatch')
    registry=registry or load_registry()
    rows=[x for x in registry['keys'] if x['key_id']==obj.get('public_key_id')]
    if len(rows)!=1: return errors+['public_key_id not uniquely registered']
    row=rows[0]
    if row['role']!=expected_role: errors.append(f'public key role {row["role"]} != {expected_role}')
    try:
        key=Ed25519PublicKey.from_public_bytes(base64.b64decode(row['public_key_b64']))
        key.verify(base64.b64decode(obj['signature']),canonical_unsigned_bytes(obj))
    except Exception as exc: errors.append(f'Ed25519 signature verification failed: {exc}')
    return errors
