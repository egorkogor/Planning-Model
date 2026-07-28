from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

from validation.hashing import hash_json
from validation.model_training_report_validator import VARIANTS, SEEDS, validate_final_training_matrix
from validation.planner_initialization_validator import canonical_tensor_seed
from validation.random_codebook_validator import code_hex, resolve_nearest_signature

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_initialization_seed_is_name_derived_and_order_independent() -> None:
    first = canonical_tensor_seed(101, "planner_decoder.layers.0.self_attn.q_proj.weight")
    assert first == canonical_tensor_seed(101, "planner_decoder.layers.0.self_attn.q_proj.weight")
    assert first != canonical_tensor_seed(101, "planner_decoder.layers.0.self_attn.k_proj.weight")
    assert first != canonical_tensor_seed(202, "planner_decoder.layers.0.self_attn.q_proj.weight")
    contract = yaml.safe_load((ROOT / "docs/training/planner_initialization_contract_v1.yaml").read_text())
    assert contract["arm_comparability"]["same_initial_state_dict_copied_to_A1_A2_A2b_A2c_A3_A3r"] is True
    assert contract["determinism"]["torch_use_deterministic_algorithms"] is True


def test_a3r_resolution_uses_cosine_and_lexicographic_tie_break() -> None:
    seed = "00" * 32
    sig_a = "sha256:" + "1" * 64
    sig_b = "sha256:" + "2" * 64
    entries = [
        {"signature_sha256": sig_b, "code_hex": code_hex(seed, sig_b)},
        {"signature_sha256": sig_a, "code_hex": code_hex(seed, sig_a)},
    ]
    # Zero would be invalid; a code vector resolves to its own signature.
    bits = bytes.fromhex(entries[0]["code_hex"])
    query = []
    for byte in bits:
        query.extend(1.0 if byte & (1 << shift) else -1.0 for shift in range(7, -1, -1))
    assert resolve_nearest_signature(entries, query) == sig_b
    architecture = yaml.safe_load((ROOT / "docs/architecture/planner_architecture_v1.yaml").read_text())
    resolution = architecture["A3r_external_resolution"]
    assert resolution["concept_packer_feedback"].startswith("raw normalized predicted z")
    assert resolution["threshold"].startswith("none")
    assert resolution["tie_break"] == "lexicographically smallest signature_sha256"


def test_a1_has_no_undefined_equal_compute_schedule() -> None:
    grammar = yaml.safe_load((ROOT / "docs/architecture/a1_token_grammar_v1.yaml").read_text())
    accounting = grammar["compute_accounting"]
    assert "A1_equal_compute_secondary_schedule_required" not in accounting
    assert accounting["A1_equal_compute_secondary_schedule"] == "not_required_for_MVP"


def test_model_audit_schema_requires_exact_eight_variants() -> None:
    schema = json.loads((ROOT / "docs/schemas/model_audit_report.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    variants = schema["properties"]["variants"]
    assert variants["minItems"] == variants["maxItems"] == 8
    assert variants["uniqueItems"] is True
    expected = {"A1", "A2", "A2b", "A2c", "A3", "A3r", "A4", "A5"}
    assert set(variants["items"]["enum"]) == expected


def _write_final_matrix(root: Path, mismatch_initialization: bool = False) -> None:
    (root / "docs/architecture").mkdir(parents=True)
    (root / "docs/training").mkdir(parents=True)
    for rel in ("docs/architecture/planner_architecture_v1.yaml", "docs/training/planner_initialization_contract_v1.yaml"):
        src = ROOT / rel
        dst = root / rel
        dst.write_bytes(src.read_bytes())
    out = root / "reports/training/final"
    out.mkdir(parents=True)
    inventory = "sha256:" + "a" * 64
    dataset = "sha256:" + "b" * 64
    for seed in SEEDS:
        init_hash = "sha256:" + f"{seed:064x}"[-64:]
        ordered = root / f"evidence/ordered-{seed}.json"
        ordered.parent.mkdir(parents=True, exist_ok=True)
        ordered.write_text(json.dumps({"seed": seed}))
        for variant in VARIANTS:
            gradient = root / f"evidence/grad-{variant}-{seed}.json"
            checkpoint = root / f"checkpoints/{variant}-{seed}.bin"
            gradient.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            gradient.write_text(json.dumps({"variant": variant, "seed": seed, "dormant_grad_none": True}))
            checkpoint.write_bytes(f"{variant}:{seed}:12000".encode())
            obj = {
                "schema_version": "work-planner-training/1.1",
                "run_id": "run-1",
                "variant": variant,
                "config_id": "cfg-1",
                "seed": seed,
                "dataset_manifest_sha256": dataset,
                "environment_lock_sha256": "sha256:" + "c" * 64,
                "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"),
                "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
                "initialization_checkpoint_sha256": ("sha256:" + "f" * 64) if mismatch_initialization and seed == 101 and variant == "A3r" else init_hash,
                "parameter_inventory_sha256": inventory,
                "ordered_training_examples_path": ordered.relative_to(root).as_posix(),
                "ordered_training_examples_sha256": _sha(ordered),
                "dormant_gradient_audit_path": gradient.relative_to(root).as_posix(),
                "dormant_gradient_audit_sha256": _sha(gradient),
                "final_checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "final_checkpoint_sha256": _sha(checkpoint),
                "optimizer_step": 12000,
                "checkpoint_selection": "FINAL_STEP_ONLY",
                "history": [{"optimizer_step": 12000}],
                "resource_usage": {},
            }
            obj["report_hash"] = hash_json(obj)
            (out / f"{variant}-seed-{seed}.json").write_text(json.dumps(obj))


def test_training_evidence_is_exact_six_by_five_matrix(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path)
    assert validate_final_training_matrix(tmp_path) == []
    (tmp_path / "reports/training/final/A3r-seed-505.json").unlink()
    assert any("matrix mismatch" in error for error in validate_final_training_matrix(tmp_path))


def test_training_evidence_rejects_different_initialization_within_seed(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path, mismatch_initialization=True)
    assert any("initialization_checkpoint_sha256 differs across arms for seed 101" in error for error in validate_final_training_matrix(tmp_path))


def test_implementation_spec_includes_a3r_final_training() -> None:
    spec = (ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.18.md").read_text()
    assert "A1/A2/A2b/A2c/A3/A3r обучаются ровно 12 000" in spec
