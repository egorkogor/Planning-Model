from __future__ import annotations
import base64
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding,PublicFormat
from validation.signature_validator import canonical_unsigned_bytes,payload_hash,verify_signed_manifest

def test_ed25519_manifest_signature_is_machine_verified():
    private=Ed25519PrivateKey.generate(); public=private.public_key().public_bytes(Encoding.Raw,PublicFormat.Raw)
    registry={'keys':[{'key_id':'sealer-key','role':'DATA_SEALER','algorithm':'ed25519','public_key_b64':base64.b64encode(public).decode(),'identity_sha256':'sha256:'+'1'*64}]}
    obj={'signature_algorithm':'ed25519','public_key_id':'sealer-key','dataset_id':'d','signature':'','manifest_hash':''}
    obj['manifest_hash']=payload_hash(obj)
    obj['signature']=base64.b64encode(private.sign(canonical_unsigned_bytes(obj))).decode()
    assert not verify_signed_manifest(obj,'DATA_SEALER',registry)
    obj['dataset_id']='tampered'
    assert verify_signed_manifest(obj,'DATA_SEALER',registry)
