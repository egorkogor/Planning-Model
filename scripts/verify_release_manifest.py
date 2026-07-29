from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release/BOOTSTRAP_MANIFEST.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
expected_release_files = importlib.import_module("scripts.build_release_manifest").files


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_hash(obj):
    clone = dict(obj)
    clone.pop("manifest_hash", None)
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    )


def main():
    schemas = {}
    reg = Registry()
    for p in sorted((ROOT / "docs/schemas").glob("*.json")):
        o = json.loads(p.read_text())
        schemas[p.name] = o
        reg = reg.with_resource(o["$id"], Resource.from_contents(o))
    obj = json.loads(MANIFEST.read_text())
    errors = list(
        Draft202012Validator(schemas["bootstrap_manifest.schema.json"], registry=reg).iter_errors(
            obj
        )
    )
    if errors:
        for e in errors:
            print(e.message)
        return 2
    if obj["manifest_hash"] != payload_hash(obj):
        print("bootstrap manifest self-hash mismatch")
        return 2
    expected_paths = {p.relative_to(ROOT).as_posix() for p in expected_release_files()}
    manifest_paths = set(obj["files"])
    if manifest_paths != expected_paths:
        missing = sorted(expected_paths - manifest_paths)
        extra = sorted(manifest_paths - expected_paths)
        print(f"bootstrap protected path set mismatch: missing={missing}, extra={extra}")
        return 2
    for rel, expected in obj["files"].items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected:
            print(f"bootstrap mismatch: {rel}")
            return 2
    print(f"BOOTSTRAP_MANIFEST_VERIFIED files={len(obj['files'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
