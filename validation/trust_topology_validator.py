from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs/operator/trust_topology_lock_v1.yaml"
DEFAULT_LOCK = ROOT / "locks/trust-topology.lock.json"


def _require_external_key_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return resolved
    raise ValueError(f"{label} must be outside the repository trust boundary: {resolved}")


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(obj: dict, *, include_manifest_hash: bool) -> bytes:
    payload = dict(obj)
    payload.pop("signature", None)
    if not include_manifest_hash:
        payload.pop("manifest_hash", None)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_hash(obj: dict) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(obj, include_manifest_hash=False)).hexdigest()


def _load_private(path: Path) -> Ed25519PrivateKey:
    raw = path.read_bytes().strip()
    if raw.startswith(b"-----BEGIN"):
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("operator private key is not Ed25519")
        return key
    decoded = base64.b64decode(raw, validate=True)
    if len(decoded) != 32:
        raise ValueError("raw Ed25519 private key must be 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(decoded)


def _load_public(path: Path) -> tuple[Ed25519PublicKey, bytes]:
    raw = path.read_bytes().strip()
    if raw.startswith(b"-----BEGIN"):
        key = serialization.load_pem_public_key(raw)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("operator public key is not Ed25519")
        raw_public = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return key, raw_public
    decoded = base64.b64decode(raw, validate=True)
    if len(decoded) != 32:
        raise ValueError("raw Ed25519 public key must be 32 bytes")
    return Ed25519PublicKey.from_public_bytes(decoded), decoded


def _schema_errors(obj: dict) -> list[str]:
    reg = Registry()
    schemas = {}
    for path in sorted((ROOT / "docs/schemas").glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        schemas[path.name] = schema
        reg = reg.with_resource(schema["$id"], Resource.from_contents(schema))
    return [e.message for e in Draft202012Validator(schemas["trust_topology_lock.schema.json"], registry=reg, format_checker=FormatChecker()).iter_errors(obj)]


def protected_paths() -> list[str]:
    policy = yaml.safe_load(POLICY.read_text(encoding="utf-8"))
    return list(policy["protected_artifacts"])


def create(
    run_id: str,
    operator_key_id: str,
    private_key_path: Path,
    public_key_path: Path,
    lock_path: Path | None = None,
) -> dict:
    lock_path = (lock_path or DEFAULT_LOCK).resolve()
    if lock_path != DEFAULT_LOCK.resolve():
        raise ValueError("Trust Topology lock must be written to locks/trust-topology.lock.json")
    if lock_path.exists():
        raise ValueError("Trust Topology lock already exists and is immutable within a run")
    private_key_path = _require_external_key_path(private_key_path, "operator private key")
    public_key_path = _require_external_key_path(public_key_path, "operator public key")
    private_key = _load_private(private_key_path)
    public_key, raw_public = _load_public(public_key_path)
    expected_public = private_key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if expected_public != raw_public:
        raise ValueError("operator private/public key mismatch")
    rows = []
    for rel in protected_paths():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"missing trust artifact: {rel}")
        rows.append({"path": rel, "sha256": digest(path)})
    obj = {
        "schema_version": "work-planner-trust-lock/1.0",
        "run_id": run_id,
        "operator_key_id": operator_key_id,
        "operator_public_key_sha256": "sha256:" + hashlib.sha256(raw_public).hexdigest(),
        "operator_attestation": {
            "out_of_band_identity_verification": True,
            "all_environment_identities_distinct": True,
            "all_credential_principals_distinct": True,
            "builder_has_no_operator_private_key": True,
        },
        "protected_artifacts": rows,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "manifest_hash": "",
        "signature_algorithm": "ed25519",
        "signature": "",
    }
    obj["manifest_hash"] = manifest_hash(obj)
    obj["signature"] = base64.b64encode(private_key.sign(canonical_bytes(obj, include_manifest_hash=True))).decode("ascii")
    errors = _schema_errors(obj)
    if errors:
        raise ValueError("invalid trust topology lock: " + "; ".join(errors))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return obj


def verify(lock_path: Path | None = None, public_key_path: Path | None = None, expected_run_id: str | None = None) -> list[str]:
    lock_path = lock_path or DEFAULT_LOCK
    errors: list[str] = []
    if not lock_path.is_file():
        return [f"missing trust topology lock: {lock_path}"]
    try:
        obj = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid trust topology lock JSON: {exc}"]
    errors.extend(_schema_errors(obj))
    if expected_run_id is not None and obj.get("run_id") != expected_run_id:
        errors.append(f"trust topology run_id mismatch: {obj.get('run_id')} != {expected_run_id}")
    if obj.get("manifest_hash") != manifest_hash(obj):
        errors.append("trust topology manifest_hash mismatch")
    expected_paths = protected_paths()
    rows = obj.get("protected_artifacts", [])
    row_paths = [row.get("path") for row in rows]
    if row_paths != expected_paths:
        errors.append(f"trust topology protected path set/order mismatch: {row_paths} != {expected_paths}")
    for row in rows:
        rel = row.get("path")
        path = ROOT / str(rel)
        if not path.is_file():
            errors.append(f"trust artifact missing: {rel}")
        elif row.get("sha256") != digest(path):
            errors.append(f"trust artifact hash mismatch: {rel}")
    if public_key_path is None:
        env_path = os.environ.get("PLANNER_OPERATOR_TRUST_PUBLIC_KEY")
        if env_path:
            public_key_path = Path(env_path)
    if public_key_path is None:
        errors.append("external operator public key path is required")
        return errors
    try:
        public_key_path = _require_external_key_path(public_key_path, "operator public key")
        public_key, raw_public = _load_public(public_key_path)
        expected_fingerprint = "sha256:" + hashlib.sha256(raw_public).hexdigest()
        if obj.get("operator_public_key_sha256") != expected_fingerprint:
            errors.append("operator public key fingerprint mismatch")
        public_key.verify(base64.b64decode(obj.get("signature", "")), canonical_bytes(obj, include_manifest_hash=True))
    except Exception as exc:
        errors.append(f"operator signature verification failed: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--run-id", default="run-validation")
    parser.add_argument("--operator-key-id", default="operator-root")
    parser.add_argument("--operator-private-key", type=Path)
    parser.add_argument("--operator-public-key", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    if args.create:
        if args.operator_private_key is None or args.operator_public_key is None:
            parser.error("--create requires --operator-private-key and --operator-public-key")
        create(args.run_id, args.operator_key_id, args.operator_private_key, args.operator_public_key, args.lock)
        print(f"CREATED TRUST_TOPOLOGY {args.lock}")
        return 0
    errors = verify(args.lock, args.operator_public_key)
    if errors:
        print("MISMATCH")
        for error in errors:
            print(error)
        return 2
    print("VERIFIED TRUST_TOPOLOGY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
