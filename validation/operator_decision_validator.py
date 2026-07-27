from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from validation.hashing import canonical_json_bytes, decision_record_hash

_DECISION_NAME = re.compile(r"^D-(\d{4})\.json$")
_GATE_TARGETS = {
    "G00_SCOPE": "artifacts/scope.md",
    "G01_TRUST_AND_RESOURCES": "locks/trust-topology.lock.json",
    "G06_STATISTICAL_IMPLEMENTATION_AUDIT": "freezes/implementation-lock.candidate.json",
    "G07_PLANNER_CONFIRMATORY_FREEZE": "freezes/planner-confirmatory.candidate.json",
    "G12_STAGE1A_CONFIRMATORY_FREEZE": "freezes/stage1a-confirmatory.candidate.json",
    "G16_STAGE1B_CONFIRMATORY_FREEZE": "freezes/stage1b-confirmatory.candidate.json",
    "G20_FINAL_ACCEPTANCE": "reports/final-audit.json",
}

_APPROVED_OUTCOMES = {
    "G00_SCOPE": "APPROVE_SCOPE",
    "G01_TRUST_AND_RESOURCES": "APPROVE_TRUST_AND_RESOURCES",
    "G06_STATISTICAL_IMPLEMENTATION_AUDIT": "APPROVE_G06_AUDITS",
    "G07_PLANNER_CONFIRMATORY_FREEZE": "APPROVE_FREEZE",
    "G12_STAGE1A_CONFIRMATORY_FREEZE": "APPROVE_FREEZE",
    "G16_STAGE1B_CONFIRMATORY_FREEZE": "APPROVE_FREEZE",
    "G20_FINAL_ACCEPTANCE": "APPROVE_FINAL",
}


@lru_cache(maxsize=1)
def _decision_schema() -> tuple[dict, Registry]:
    registry = Registry()
    schemas: dict[str, dict] = {}
    for path in sorted((ROOT / "docs/schemas").glob("*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        schemas[path.name] = obj
        registry = registry.with_resource(obj["$id"], Resource.from_contents(obj))
    return schemas["decision_record.schema.json"], registry


def decision_schema_errors(record: dict) -> list[str]:
    schema, registry = _decision_schema()
    return [
        error.message
        for error in Draft202012Validator(schema, registry=registry, format_checker=FormatChecker()).iter_errors(record)
    ]


def _external(path: Path, label: str, *, root: Path = ROOT) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return resolved
    raise ValueError(f"{label} must be outside the repository trust boundary: {resolved}")


def _load_public(path: Path) -> tuple[Ed25519PublicKey, bytes]:
    raw = path.read_bytes().strip()
    if raw.startswith(b"-----BEGIN"):
        key = serialization.load_pem_public_key(raw)
        if not isinstance(key, Ed25519PublicKey):
            raise ValueError("operator public key is not Ed25519")
        public = key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        return key, public
    public = base64.b64decode(raw, validate=True)
    if len(public) != 32:
        raise ValueError("raw Ed25519 public key must be 32 bytes")
    return Ed25519PublicKey.from_public_bytes(public), public


def _load_private(path: Path) -> Ed25519PrivateKey:
    raw = path.read_bytes().strip()
    if raw.startswith(b"-----BEGIN"):
        key = serialization.load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("operator private key is not Ed25519")
        return key
    private = base64.b64decode(raw, validate=True)
    if len(private) != 32:
        raise ValueError("raw Ed25519 private key must be 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(private)


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def signing_bytes(record: dict) -> bytes:
    payload = dict(record)
    payload.pop("signature", None)
    return canonical_json_bytes(payload)


def resolve_public_key(
    path: Path | None = None,
    *,
    root: Path = ROOT,
) -> tuple[Ed25519PublicKey, bytes, Path]:
    if path is None:
        env = os.environ.get("PLANNER_OPERATOR_TRUST_PUBLIC_KEY")
        if env:
            path = Path(env)
    if path is None:
        raise ValueError("external operator public key path is required")
    path = _external(path, "operator public key", root=root)
    key, raw = _load_public(path)
    return key, raw, path


def _trust_lock(root: Path) -> dict:
    return json.loads((root / "locks/trust-topology.lock.json").read_text(encoding="utf-8"))


def verify_operator_decision(
    record: dict,
    public_key_path: Path | None = None,
    *,
    require_trust_lock: bool = True,
    expected_run_id: str | None = None,
    root: Path = ROOT,
) -> list[str]:
    errors: list[str] = decision_schema_errors(record)
    if record.get("decision_hash") != decision_record_hash(record):
        errors.append("DecisionRecord decision_hash mismatch")
    if expected_run_id is not None and record.get("run_id") != expected_run_id:
        errors.append("DecisionRecord run_id differs from phase report")
    try:
        _timestamp(record.get("timestamp", ""))
    except Exception as exc:
        errors.append(f"DecisionRecord timestamp is invalid: {exc}")
    try:
        key, raw_public, _ = resolve_public_key(public_key_path, root=root)
        fingerprint = "sha256:" + hashlib.sha256(raw_public).hexdigest()
        if record.get("operator_public_key_sha256") != fingerprint:
            errors.append("DecisionRecord operator public key fingerprint mismatch")
        if record.get("signature_algorithm") != "ed25519":
            errors.append("DecisionRecord signature_algorithm must be ed25519")
        key.verify(base64.b64decode(record.get("signature", ""), validate=True), signing_bytes(record))
    except Exception as exc:
        errors.append(f"DecisionRecord operator signature verification failed: {exc}")
        fingerprint = None
    if require_trust_lock:
        try:
            trust = _trust_lock(root)
            if trust.get("run_id") != record.get("run_id"):
                errors.append("DecisionRecord run_id differs from Trust Topology lock")
            if trust.get("operator_key_id") != record.get("operator_key_id"):
                errors.append("DecisionRecord operator_key_id differs from Trust Topology lock")
            if fingerprint is not None and trust.get("operator_public_key_sha256") != fingerprint:
                errors.append("DecisionRecord operator key differs from Trust Topology lock")
        except Exception as exc:
            errors.append(f"cannot bind DecisionRecord to Trust Topology lock: {exc}")
    return errors


def load_decision_history(root: Path, run_id: str) -> tuple[list[tuple[Path, dict]], list[str]]:
    rows: list[tuple[Path, dict]] = []
    errors: list[str] = []
    directory = root / "decisions"
    if not directory.exists():
        return rows, errors
    seen_ids: set[str] = set()
    for path in sorted(directory.glob("D-????.json")):
        match = _DECISION_NAME.fullmatch(path.name)
        if match is None:
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid DecisionRecord JSON {path.name}: {exc}")
            continue
        if obj.get("decision_id") != path.stem:
            errors.append(f"DecisionRecord id/path mismatch: {path.name}")
        if obj.get("decision_id") in seen_ids:
            errors.append(f"duplicate DecisionRecord id: {obj.get('decision_id')}")
        seen_ids.add(obj.get("decision_id"))
        if obj.get("run_id") == run_id:
            rows.append((path, obj))
    rows.sort(key=lambda item: int(item[1]["decision_id"].split("-")[1]))
    return rows, errors


def verify_decision_history(
    current: dict,
    public_key_path: Path | None = None,
    *,
    require_trust_lock: bool,
    root: Path = ROOT,
) -> list[str]:
    """Verify the signed, append-only decision chain for one run.

    The current record must be present as the last signed record. Each record
    commits to the previous DecisionRecord hash, so deleting a rejection or
    replaying an older approval breaks the chain even if JSONL ledgers are edited.
    """
    errors: list[str] = []
    rows, load_errors = load_decision_history(root, current.get("run_id", ""))
    errors.extend(load_errors)
    if not rows:
        return errors + ["DecisionRecord history is empty"]
    current_rows = [obj for _, obj in rows if obj.get("decision_id") == current.get("decision_id")]
    if len(current_rows) != 1 or current_rows[0] != current:
        errors.append("current DecisionRecord is absent from signed history or differs from file")
    previous_hash: str | None = None
    previous_time: datetime | None = None
    previous_number: int | None = None
    operator_fingerprint = current.get("operator_public_key_sha256")
    operator_key_id = current.get("operator_key_id")
    for path, obj in rows:
        errors.extend(
            f"{path.name}: {error}"
            for error in verify_operator_decision(
                obj,
                public_key_path,
                require_trust_lock=require_trust_lock,
                expected_run_id=current.get("run_id"),
                root=root,
            )
        )
        number = int(obj["decision_id"].split("-")[1])
        if previous_number is not None and number <= previous_number:
            errors.append("DecisionRecord identifiers are not strictly increasing")
        previous_number = number
        if obj.get("previous_decision_hash") != previous_hash:
            errors.append(f"{path.name}: previous_decision_hash breaks signed decision chain")
        previous_hash = obj.get("decision_hash")
        try:
            timestamp = _timestamp(obj["timestamp"])
            if previous_time is not None and timestamp <= previous_time:
                errors.append(f"{path.name}: DecisionRecord timestamps are not strictly increasing")
            previous_time = timestamp
        except Exception:
            pass
        if obj.get("operator_public_key_sha256") != operator_fingerprint or obj.get("operator_key_id") != operator_key_id:
            errors.append(f"{path.name}: operator identity changed inside one run")
    same_gate_rows = [obj for _, obj in rows if obj.get("gate_id") == current.get("gate_id")]
    if not same_gate_rows or same_gate_rows[-1].get("decision_id") != current.get("decision_id"):
        errors.append("DecisionRecord is not the latest signed decision for this run/gate")
    prior = [obj for _, obj in rows if obj.get("gate_id") == current.get("gate_id") and obj.get("decision_id") != current.get("decision_id")]
    expected_resubmission = sum(obj.get("decision") == "REJECT" for obj in prior)
    if current.get("resubmission_index") != expected_resubmission:
        errors.append(
            f"DecisionRecord resubmission_index {current.get('resubmission_index')} "
            f"!= signed prior reject count {expected_resubmission}"
        )
    return errors



def verify_run_decision_history(
    run_id: str,
    public_key_path: Path | None = None,
    *,
    require_trust_lock: bool,
    expected_approved_gates: set[str] | None = None,
    root: Path = ROOT,
) -> list[str]:
    """Verify the complete signed chain and the current bytes of approved targets."""
    rows, errors = load_decision_history(root, run_id)
    if not rows:
        if expected_approved_gates:
            errors.append("required signed DecisionRecord history is empty")
        return errors
    errors.extend(
        verify_decision_history(
            rows[-1][1],
            public_key_path,
            require_trust_lock=require_trust_lock,
            root=root,
        )
    )
    latest_by_gate: dict[str, dict] = {}
    for _, record in rows:
        latest_by_gate[record.get("gate_id", "")] = record
        target_rel = _GATE_TARGETS.get(record.get("gate_id"))
        if target_rel is None:
            errors.append(f"{record.get('decision_id')}: unknown approval target for gate {record.get('gate_id')}")
            continue
        # Rejected candidates are intentionally replaceable during the one allowed
        # resubmission. Only an approval makes the target immutable in the run.
        if record.get("decision") != "APPROVE":
            continue
        target_path = (root / target_rel).resolve()
        if target_path != root.resolve() and root.resolve() not in target_path.parents:
            errors.append(f"{record.get('decision_id')}: approval target escapes repository")
        elif not target_path.is_file():
            errors.append(f"{record.get('decision_id')}: approved target is missing: {target_rel}")
        else:
            actual = "sha256:" + hashlib.sha256(target_path.read_bytes()).hexdigest()
            if actual != record.get("target_artifact_hash"):
                errors.append(f"{record.get('decision_id')}: approved target changed after decision: {target_rel}")
    for gate_id in sorted(expected_approved_gates or set()):
        record = latest_by_gate.get(gate_id)
        if record is None:
            errors.append(f"missing required signed approval for {gate_id}")
            continue
        if record.get("decision") != "APPROVE" or record.get("phase_outcome") != _APPROVED_OUTCOMES[gate_id]:
            errors.append(f"latest signed decision for {gate_id} is not the required approval")
    return errors

def _all_existing_decision_numbers(root: Path) -> list[int]:
    numbers: list[int] = []
    directory = root / "decisions"
    if directory.exists():
        for path in directory.glob("D-????.json"):
            match = _DECISION_NAME.fullmatch(path.name)
            if match:
                numbers.append(int(match.group(1)))
    return sorted(numbers)


def sign_record(
    input_path: Path,
    output_path: Path,
    private_path: Path,
    public_path: Path,
    *,
    root: Path = ROOT,
) -> dict:
    private_path = _external(private_path, "operator private key", root=root)
    public_path = _external(public_path, "operator public key", root=root)
    private = _load_private(private_path)
    _, raw_public = _load_public(public_path)
    expected = private.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    if expected != raw_public:
        raise ValueError("operator private/public key mismatch")
    record = json.loads(input_path.read_text(encoding="utf-8"))
    output_path = output_path.resolve()
    decisions_dir = (root / "decisions").resolve()
    if output_path.parent != decisions_dir or _DECISION_NAME.fullmatch(output_path.name) is None:
        raise ValueError("signed DecisionRecord output must be decisions/D-NNNN.json inside the repository")
    run_id = record.get("run_id")
    if not run_id:
        raise ValueError("DecisionRecord run_id is required")
    history, history_errors = load_decision_history(root, run_id)
    if history_errors:
        raise ValueError("invalid prior DecisionRecord history: " + "; ".join(history_errors))
    if history:
        prior_errors = verify_decision_history(
            history[-1][1],
            public_path,
            require_trust_lock=(root / "locks/trust-topology.lock.json").is_file(),
            root=root,
        )
        if prior_errors:
            raise ValueError("invalid signed DecisionRecord history: " + "; ".join(prior_errors))
    if output_path.exists():
        raise ValueError("DecisionRecord output already exists; signed decisions are append-only")
    if input_path.resolve() == output_path:
        raise ValueError("unsigned input and signed DecisionRecord output must be different files")
    existing_numbers = _all_existing_decision_numbers(root)
    requested_number = int(record["decision_id"].split("-")[1])
    expected_number = (max(existing_numbers) + 1) if existing_numbers else 1
    if requested_number != expected_number:
        raise ValueError(f"next DecisionRecord id must be D-{expected_number:04d}")
    previous = history[-1][1] if history else None
    if previous is not None and _timestamp(record["timestamp"]) <= _timestamp(previous["timestamp"]):
        raise ValueError("DecisionRecord timestamp must be later than the previous signed decision")
    record["previous_decision_hash"] = previous.get("decision_hash") if previous else None
    prior_same_gate = [obj for _, obj in history if obj.get("gate_id") == record.get("gate_id")]
    record["resubmission_index"] = sum(obj.get("decision") == "REJECT" for obj in prior_same_gate)
    trust_path = root / "locks/trust-topology.lock.json"
    trust = json.loads(trust_path.read_text(encoding="utf-8")) if trust_path.is_file() else None
    if trust is not None and trust.get("run_id") != run_id:
        raise ValueError("DecisionRecord run_id differs from Trust Topology lock")
    record["operator_key_id"] = trust["operator_key_id"] if trust else record.get("operator_key_id", "operator-root")
    record["operator_public_key_sha256"] = "sha256:" + hashlib.sha256(raw_public).hexdigest()
    if trust is not None and trust.get("operator_public_key_sha256") != record["operator_public_key_sha256"]:
        raise ValueError("operator public key differs from Trust Topology lock")
    record["signature_algorithm"] = "ed25519"
    record["signature"] = ""
    record["decision_hash"] = decision_record_hash(record)
    record["signature"] = base64.b64encode(private.sign(signing_bytes(record))).decode("ascii")
    schema_errors = decision_schema_errors(record)
    if schema_errors:
        raise ValueError("signed DecisionRecord fails schema: " + "; ".join(schema_errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--sign", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--operator-private-key", type=Path)
    parser.add_argument("--operator-public-key", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--without-trust-lock", action="store_true")
    parser.add_argument("--verify-history", action="store_true")
    args = parser.parse_args()
    if args.sign:
        if not args.output or not args.operator_private_key or not args.operator_public_key:
            parser.error("--sign requires --output, --operator-private-key and --operator-public-key")
        sign_record(args.record, args.output, args.operator_private_key, args.operator_public_key)
        print(f"SIGNED {args.output}")
        return 0
    obj = json.loads(args.record.read_text(encoding="utf-8"))
    errors = verify_operator_decision(
        obj,
        args.operator_public_key,
        require_trust_lock=not args.without_trust_lock,
        expected_run_id=args.run_id,
    )
    if args.verify_history:
        errors.extend(
            verify_decision_history(
                obj,
                args.operator_public_key,
                require_trust_lock=not args.without_trust_lock,
            )
        )
    if errors:
        print("MISMATCH")
        print("\n".join(errors))
        return 2
    print("VERIFIED OPERATOR_DECISION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
