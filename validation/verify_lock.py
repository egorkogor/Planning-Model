from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from validation.implementation_candidate_validator import validate_candidate
POLICIES = {
    "scientific": ROOT / "docs/operator/scientific_lock_v1.yaml",
    "implementation": ROOT / "docs/operator/implementation_lock_v1.yaml",
}
DEFAULT_LOCKS = {
    "scientific": ROOT / "locks/scientific.lock.json",
    "implementation": ROOT / "locks/implementation.lock.json",
}


def digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(obj: dict) -> str:
    payload = dict(obj)
    payload.pop("manifest_hash", None)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def load_policy(kind: str) -> dict:
    return yaml.safe_load(POLICIES[kind].read_text(encoding="utf-8"))


def protected_files(kind: str) -> list[Path]:
    patterns = load_policy(kind)["protected_paths"]
    excluded = {DEFAULT_LOCKS[kind].relative_to(ROOT).as_posix()}
    rows: set[Path] = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in excluded:
            continue
        if any(fnmatch.fnmatch(rel, pattern) for pattern in patterns):
            rows.add(path)
    return sorted(rows, key=lambda p: p.relative_to(ROOT).as_posix())


def create(kind: str, lock_path: Path, run_id: str, source_candidate: Path | None = None) -> dict:
    policy = load_policy(kind)
    rows = [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in protected_files(kind)]
    obj = {
        "schema_version": "work-planner-lock/1.1",
        "lock_kind": kind.upper(),
        "run_id": run_id,
        "protocol_version": "work-planner/1.13",
        "policy_sha256": digest(POLICIES[kind]),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_candidate_sha256": None,
        "protected_files": rows,
        "manifest_hash": "",
    }
    if kind == "implementation":
        if source_candidate is None or not source_candidate.is_file():
            raise ValueError("implementation lock creation requires --source-candidate")
        candidate=json.loads(source_candidate.read_text(encoding="utf-8"))
        candidate_errors=validate_candidate(candidate)
        if candidate_errors: raise ValueError("invalid implementation candidate: " + "; ".join(candidate_errors))
        candidate_rows=candidate["protected_files"]
        if rows != candidate_rows: raise ValueError("current implementation files differ from approved candidate")
        obj["source_candidate_sha256"] = candidate["candidate_hash"]
    obj["manifest_hash"] = canonical_hash(obj)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return obj


def verify(kind: str, lock_path: Path, expected_run_id: str | None = None) -> list[str]:
    if not lock_path.exists():
        return [f"missing {kind} lock: {lock_path}"]
    try:
        obj = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid lock JSON: {exc}"]
    errors: list[str] = []
    if expected_run_id is not None and obj.get("run_id") != expected_run_id:
        errors.append(f"{kind} lock run_id mismatch: {obj.get('run_id')} != {expected_run_id}")
    if obj.get("lock_kind") != kind.upper():
        errors.append(f"lock_kind mismatch: {obj.get('lock_kind')} != {kind.upper()}")
    if obj.get("protocol_version") != "work-planner/1.13":
        errors.append("protocol_version mismatch")
    if obj.get("policy_sha256") != digest(POLICIES[kind]):
        errors.append("policy hash mismatch")
    if obj.get("manifest_hash") != canonical_hash(obj):
        errors.append("manifest hash mismatch")
    if kind == "scientific":
        if obj.get("source_candidate_sha256") is not None: errors.append("scientific lock cannot reference implementation candidate")
    else:
        candidate_path=ROOT/"freezes/implementation-lock.candidate.json"
        if not candidate_path.is_file(): errors.append("implementation candidate missing")
        else:
            try:
                candidate=json.loads(candidate_path.read_text(encoding="utf-8")); errors.extend(validate_candidate(candidate))
                if obj.get("source_candidate_sha256")!=candidate.get("candidate_hash"): errors.append("implementation lock source candidate hash mismatch")
                if obj.get("protected_files")!=candidate.get("protected_files"): errors.append("implementation lock protected files differ from approved candidate")
            except Exception as exc: errors.append(f"invalid implementation candidate: {exc}")
    expected = {row["path"]: row["sha256"] for row in obj.get("protected_files", [])}
    actual = {p.relative_to(ROOT).as_posix(): p for p in protected_files(kind)}
    if set(expected) != set(actual):
        errors.append(
            "protected path set mismatch: "
            f"missing={sorted(set(expected) - set(actual))}, added={sorted(set(actual) - set(expected))}"
        )
    for rel in sorted(set(expected) & set(actual)):
        if digest(actual[rel]) != expected[rel]:
            errors.append(f"hash mismatch: {rel}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=sorted(POLICIES), required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--run-id", default="run-validation")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--source-candidate", type=Path)
    args = parser.parse_args()
    lock_path = args.lock or DEFAULT_LOCKS[args.kind]
    if args.create:
        create(args.kind, lock_path, args.run_id, args.source_candidate)
        print(f"CREATED {args.kind.upper()} {lock_path}")
        return 0
    errors = verify(args.kind, lock_path)
    if errors:
        print("MISMATCH")
        for error in errors:
            print(error)
        return 2
    print(f"VERIFIED {args.kind.upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
