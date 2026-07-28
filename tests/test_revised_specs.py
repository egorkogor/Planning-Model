from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "docs/schemas"


def _bundle() -> tuple[dict[str, dict], Registry]:
    schemas: dict[str, dict] = {}
    registry = Registry()
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        schemas[path.name] = schema
        registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return schemas, registry


def test_schema_bundle_is_valid_and_complete() -> None:
    schemas, _ = _bundle()
    assert len(schemas) >= 50
    for name in (
        "typed_action.schema.json",
        "planner_step.schema.json",
        "training_example.schema.json",
        "corpus_manifest.schema.json",
        "prompt_artifact.schema.json",
        "control_certification.schema.json",
        "experiment_freeze.schema.json",
        "phase_report.schema.json",
        "decision_record.schema.json",
    ):
        assert name in schemas


def test_action_contracts_reject_bad_signatures() -> None:
    schemas, registry = _bundle()
    validator = Draft202012Validator(schemas["llm_step_response.schema.json"], registry=registry)
    valid_end = {"schema_version": "work-planner/1.18", "action": "END", "args": []}
    assert not list(validator.iter_errors(valid_end))
    assert list(validator.iter_errors({"schema_version": "work-planner/1.18", "action": "END", "args": ["block_0"]}))
    assert list(validator.iter_errors({"schema_version": "work-planner/1.18", "action": "PICK_UP", "args": []}))


def test_validation_bundle() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "validation/validate_bundle.py"), "--skip-nested-pytest"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "schemas valid" in result.stdout
    assert "pytest skipped" in result.stdout
