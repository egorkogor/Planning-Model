from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import yaml
import numpy as np
from jsonschema import Draft202012Validator

from validation.hashing import hash_json
from validation.model_training_report_validator import VARIANTS, SEEDS, validate_final_training_matrix, validate_model_audit_report
from validation.planner_evidence_validator import canonical_initialized_array, canonical_tensor_table, config_id
from validation.planner_initialization_validator import canonical_tensor_seed
from validation.random_codebook_validator import code_hex, resolve_nearest_signature

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_safetensors(
    path: Path, tensors: list[dict], metadata: dict[str, str],
    values: dict[str, np.ndarray] | None = None,
) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": metadata}
    data = bytearray()
    values = values or {}
    for row in tensors:
        shape = tuple(row["shape"])
        arr = values.get(row["name"], np.zeros(shape, dtype="<f4"))
        arr = np.asarray(arr, dtype="<f4").reshape(shape)
        raw_values = arr.tobytes(order="C")
        size = len(raw_values)
        header[row["name"]] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + size]}
        data.extend(raw_values)
        offset += size
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)


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


def test_exact_module_inventory_is_machine_readable_and_complete() -> None:
    inventory = yaml.safe_load((ROOT / "docs/architecture/planner_module_inventory_v1.yaml").read_text())
    assert inventory["inventory_hash"] == hash_json({k: v for k, v in inventory.items() if k != "inventory_hash"})
    tensors = inventory["tensors"]
    assert len(tensors) == 177
    assert len({row["name"] for row in tensors}) == 177
    by_name = {row["name"]: row for row in tensors}
    assert by_name["task_encoder.token_embedding.weight"]["shape"] == [40, 256]
    assert by_name["planner_decoder.layers.0.self_attn.q_proj.weight"]["shape"] == [256, 256]
    assert by_name["heads.signature.remaining_distance_bucket.weight"]["shape"] == [5, 256]
    impl = inventory["implementation"]
    assert impl["attention"]["projection_layout"] == "separate_q_k_v_out"
    assert impl["attention"]["projection_bias"] is False
    assert impl["layer_norm"]["eps"] == 1e-5
    assert impl["ffn"]["gelu_approximate"] == "none"


def test_model_audit_schema_requires_exact_eight_variants_and_evidence_checks() -> None:
    schema = json.loads((ROOT / "docs/schemas/model_audit_report.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    variants = schema["properties"]["variants"]
    assert variants["minItems"] == variants["maxItems"] == 8
    assert variants["uniqueItems"] is True
    expected = {"A1", "A2", "A2b", "A2c", "A3", "A3r", "A4", "A5"}
    assert set(variants["items"]["enum"]) == expected
    checks = schema["properties"]["checks"]
    assert checks["minItems"] == checks["maxItems"] == 5
    assert {"evidence_path", "evidence_sha256", "recomputed_value", "expected_value"} <= set(checks["items"]["required"])


def _prepare_contracts(root: Path) -> tuple[dict, dict]:
    for rel in (
        "docs/architecture/planner_architecture_v1.yaml",
        "docs/training/planner_initialization_contract_v1.yaml",
        "docs/training/hyperparameter_search_v1.yaml",
    ):
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes())
    tensor = {
        "name": "weight",
        "shape": [1],
        "dtype": "F32",
        "parameter_type": "standalone_parameter",
        "active_arms": list(VARIANTS),
    }
    inventory_contract = {
        "schema_version": "work-planner/1.21",
        "contract_id": "test-inventory/1.0",
        "implementation": {},
        "tensors": [tensor],
    }
    inventory_contract["inventory_hash"] = hash_json(inventory_contract)
    inv_contract_path = root / "docs/architecture/planner_module_inventory_v1.yaml"
    inv_contract_path.write_text(yaml.safe_dump(inventory_contract, sort_keys=False))

    inv_manifest = {
        "schema_version": "work-planner-parameter-inventory/1.0",
        "run_id": "run-1",
        "contract_path": "docs/architecture/planner_module_inventory_v1.yaml",
        "contract_sha256": _sha(inv_contract_path),
        "tensors": [tensor],
        "inventory_hash": inventory_contract["inventory_hash"],
    }
    inv_manifest["manifest_hash"] = hash_json(inv_manifest)
    inv_path = root / "reports/model-evidence/parameter-inventory.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps(inv_manifest))

    params = {
        "learning_rate": 0.0001,
        "dropout": 0.0,
        "decoder_layers": 4,
        "encoder_layers": 4,
        "d_model": 256,
        "attention_heads": 8,
        "ffn_dim": 1024,
        "batch_size_examples": 128,
    }
    cfg = {
        "schema_version": "work-planner-selected-config/1.0",
        "run_id": "run-1",
        "search_contract_sha256": _sha(root / "docs/training/hyperparameter_search_v1.yaml"),
        "config_id": config_id(params),
        "parameters": params,
        "selection_frozen_before_final_seeds": True,
    }
    cfg["report_hash"] = hash_json(cfg)
    cfg_path = root / "reports/training/selected-config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg))

    dataset = {
        "schema_version": "work-planner/1.21",
        "corpus_id": "c",
        "contract_sha256": "sha256:" + "1" * 64,
        "generator_manifest_sha256": "sha256:" + "2" * 64,
        "domain_sha256": "sha256:" + "3" * 64,
        "oracle_sha256": "sha256:" + "4" * 64,
        "examples_sha256": "sha256:" + "5" * 64,
        "counts_by_split": {"train": 1},
        "counts_by_source_kind": {"ORACLE_SUFFIX": 1},
        "counts_by_support_signature": {"x": 1},
    }
    dataset["manifest_hash"] = hash_json(dataset)
    dataset_path = root / "data/manifests/training-corpus.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(json.dumps(dataset))

    env_path = root / "locks/environment.lock.json"
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps({"lock": "test"}))
    return inv_manifest, cfg


def _write_final_matrix(root: Path, corrupt_checkpoint: bool = False) -> None:
    inv, cfg = _prepare_contracts(root)
    inventory_hash = inv["inventory_hash"]
    inv_path = root / "reports/model-evidence/parameter-inventory.json"
    dataset_path = root / "data/manifests/training-corpus.json"
    env_path = root / "locks/environment.lock.json"
    config_path = root / "reports/training/selected-config.json"
    out = root / "reports/training/final"
    out.mkdir(parents=True)

    for seed in SEEDS:
        init_model = root / f"checkpoints/init-{seed}.safetensors"
        init_meta = {
            "run_id": "run-1", "checkpoint_kind": "INITIALIZATION", "variant": "COMMON",
            "seed": str(seed), "optimizer_step": "0", "parameter_inventory_sha256": inventory_hash,
        }
        _write_safetensors(init_model, [{"name": "weight", "shape": [1]}], init_meta, {"weight": canonical_initialized_array(inv["tensors"][0], seed)})
        init_cp = {
            "schema_version": "work-planner-checkpoint-manifest/1.0", "run_id": "run-1", "checkpoint_kind": "INITIALIZATION",
            "variant": "COMMON", "seed": seed, "optimizer_step": 0,
            "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"),
            "module_inventory_contract_sha256": _sha(root / "docs/architecture/planner_module_inventory_v1.yaml"),
            "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
            "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(), "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
            "initialization_manifest_path": None, "initialization_manifest_sha256": None, "selected_config_path": None, "selected_config_sha256": None,
            "dataset_manifest_path": None, "dataset_manifest_sha256": None, "ordered_training_examples_path": None, "ordered_training_examples_sha256": None,
            "environment_lock_path": None, "environment_lock_sha256": None,
            "model_file_path": init_model.relative_to(root).as_posix(), "model_file_sha256": _sha(init_model), "model_format": "SAFETENSORS",
            "model_tensor_table_sha256": canonical_tensor_table([{"name": "weight", "shape": [1], "dtype": "F32"}]),
            "optimizer_state_path": None, "optimizer_state_sha256": None, "optimizer_state_format": None, "optimizer_state_tensor_table_sha256": None,
        }
        init_cp["manifest_hash"] = hash_json(init_cp)
        init_cp_path = root / f"reports/training/checkpoints/init-seed-{seed}.json"
        init_cp_path.parent.mkdir(parents=True, exist_ok=True)
        init_cp_path.write_text(json.dumps(init_cp))
        init = {
            "schema_version": "work-planner-initialization-manifest/1.0", "run_id": "run-1", "seed": seed,
            "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
            "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(), "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
            "checkpoint_manifest_path": init_cp_path.relative_to(root).as_posix(), "checkpoint_manifest_sha256": _sha(init_cp_path),
            "tensor_seeds": [{"name": "weight", "tensor_seed": canonical_tensor_seed(seed, "weight")}],
        }
        init["manifest_hash"] = hash_json(init)
        init_path = root / f"reports/training/initialization/seed-{seed}.json"
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(json.dumps(init))

        ordered = {
            "schema_version": "work-planner-ordered-examples/1.0", "run_id": "run-1", "seed": seed,
            "dataset_manifest_sha256": _sha(dataset_path), "ordering_algorithm": "SHA256(seed || example_id), ascending digest then example_id",
            "example_ids": ["ex-1"], "examples_sha256": hash_json(["ex-1"]),
        }
        ordered["manifest_hash"] = hash_json(ordered)
        ordered_path = root / f"reports/training/ordered-examples/seed-{seed}.json"
        ordered_path.parent.mkdir(parents=True, exist_ok=True)
        ordered_path.write_text(json.dumps(ordered))

        for variant in VARIANTS:
            audit = {
                "schema_version": "work-planner-dormant-gradient-audit/1.0", "run_id": "run-1", "variant": variant, "seed": seed,
                "parameter_inventory_sha256": inventory_hash, "batches_audited": 2,
                "entries": [{"name": "weight", "expected_activity": "ACTIVE", "gradient_state": "PRESENT_FINITE", "nonfinite_count": 0, "materialized_zero_count": 0}],
                "status": "PASS",
            }
            audit["audit_hash"] = hash_json(audit)
            audit_path = root / f"reports/training/dormant-gradients/{variant}-seed-{seed}.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit))

            model = root / f"checkpoints/{variant}-{seed}.safetensors"
            opt = root / f"checkpoints/{variant}-{seed}-optimizer.safetensors"
            model_meta = {"run_id": "run-1", "checkpoint_kind": "FINAL_EQUAL_DATA", "variant": variant, "seed": str(seed), "optimizer_step": "12000", "parameter_inventory_sha256": inventory_hash}
            final_weight = canonical_initialized_array(inv["tensors"][0], seed).copy()
            final_weight += np.asarray([0.1], dtype="<f4")
            _write_safetensors(model, [{"name": "weight", "shape": [1]}], model_meta, {"weight": final_weight})
            _write_safetensors(
                opt,
                [{"name": "weight.exp_avg", "shape": [1]}, {"name": "weight.exp_avg_sq", "shape": [1]}],
                {
                    "optimizer_step": "12000", "variant": variant, "optimizer": "AdamW",
                    "beta1": "0.9", "beta2": "0.95", "eps": "1e-08", "weight_decay": "0.01",
                },
                {
                    "weight.exp_avg": np.asarray([0.1], dtype="<f4"),
                    "weight.exp_avg_sq": np.asarray([0.01], dtype="<f4"),
                },
            )
            if corrupt_checkpoint and variant == "A3" and seed == 101:
                model.write_bytes(b"not a checkpoint")
            cp = {
                "schema_version": "work-planner-checkpoint-manifest/1.0", "run_id": "run-1", "checkpoint_kind": "FINAL_EQUAL_DATA", "variant": variant, "seed": seed, "optimizer_step": 12000,
                "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"), "module_inventory_contract_sha256": _sha(root / "docs/architecture/planner_module_inventory_v1.yaml"), "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
                "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(), "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
                "initialization_manifest_path": init_path.relative_to(root).as_posix(), "initialization_manifest_sha256": _sha(init_path),
                "selected_config_path": config_path.relative_to(root).as_posix(), "selected_config_sha256": _sha(config_path),
                "dataset_manifest_path": dataset_path.relative_to(root).as_posix(), "dataset_manifest_sha256": _sha(dataset_path),
                "ordered_training_examples_path": ordered_path.relative_to(root).as_posix(), "ordered_training_examples_sha256": _sha(ordered_path),
                "environment_lock_path": "locks/environment.lock.json", "environment_lock_sha256": _sha(env_path),
                "model_file_path": model.relative_to(root).as_posix(), "model_file_sha256": _sha(model), "model_format": "SAFETENSORS", "model_tensor_table_sha256": canonical_tensor_table([{"name": "weight", "shape": [1], "dtype": "F32"}]),
                "optimizer_state_path": opt.relative_to(root).as_posix(), "optimizer_state_sha256": _sha(opt), "optimizer_state_format": "SAFETENSORS_ADAMW", "optimizer_state_tensor_table_sha256": canonical_tensor_table([{"name": "weight.exp_avg", "shape": [1], "dtype": "F32"}, {"name": "weight.exp_avg_sq", "shape": [1], "dtype": "F32"}]),
            }
            cp["manifest_hash"] = hash_json(cp)
            cp_path = root / f"reports/training/checkpoints/final-{variant}-seed-{seed}.json"
            cp_path.write_text(json.dumps(cp))
            report = {
                "schema_version": "work-planner-training/1.2", "run_id": "run-1", "training_regime": "FINAL_EQUAL_DATA", "variant": variant, "config_id": cfg["config_id"], "seed": seed,
                "selected_config_path": config_path.relative_to(root).as_posix(), "selected_config_sha256": _sha(config_path), "dataset_manifest_path": dataset_path.relative_to(root).as_posix(), "dataset_manifest_sha256": _sha(dataset_path),
                "environment_lock_path": "locks/environment.lock.json", "environment_lock_sha256": _sha(env_path), "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"),
                "module_inventory_contract_sha256": _sha(root / "docs/architecture/planner_module_inventory_v1.yaml"), "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
                "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(), "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
                "initialization_manifest_path": init_path.relative_to(root).as_posix(), "initialization_manifest_sha256": _sha(init_path), "ordered_training_examples_path": ordered_path.relative_to(root).as_posix(), "ordered_training_examples_sha256": _sha(ordered_path),
                "dormant_gradient_audit_path": audit_path.relative_to(root).as_posix(), "dormant_gradient_audit_sha256": _sha(audit_path), "checkpoint_manifest_path": cp_path.relative_to(root).as_posix(), "checkpoint_manifest_sha256": _sha(cp_path),
                "optimizer_step": 12000, "checkpoint_selection": "FINAL_STEP_ONLY", "history": [{"optimizer_step": 12000}], "resource_usage": {},
            }
            report["report_hash"] = hash_json(report)
            (out / f"{variant}-seed-{seed}.json").write_text(json.dumps(report))



def _write_flops_sensitivity_matrix(root: Path) -> None:
    inv_path = root / "reports/model-evidence/parameter-inventory.json"
    inv = json.loads(inv_path.read_text())
    inventory_hash = inv["inventory_hash"]
    config_path = root / "reports/training/selected-config.json"
    cfg = json.loads(config_path.read_text())
    dataset_path = root / "data/manifests/training-corpus.json"
    env_path = root / "locks/environment.lock.json"
    out = root / "reports/training/flops-sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        init_path = root / f"reports/training/initialization/seed-{seed}.json"
        ordered_path = root / f"reports/training/ordered-examples/seed-{seed}.json"
        for variant in ("A2c", "A3"):
            step = 6000 if variant == "A3" else 12000
            audit_path = root / f"reports/training/dormant-gradients/{variant}-seed-{seed}.json"
            model = root / f"checkpoints/flops-{variant}-{seed}.safetensors"
            opt = root / f"checkpoints/flops-{variant}-{seed}-optimizer.safetensors"
            model_meta = {
                "run_id": "run-1", "checkpoint_kind": "FLOPS_SENSITIVITY", "variant": variant,
                "seed": str(seed), "optimizer_step": str(step), "parameter_inventory_sha256": inventory_hash,
            }
            final_weight = canonical_initialized_array(inv["tensors"][0], seed).copy()
            final_weight += np.asarray([0.2], dtype="<f4")
            _write_safetensors(model, [{"name": "weight", "shape": [1]}], model_meta, {"weight": final_weight})
            _write_safetensors(
                opt,
                [{"name": "weight.exp_avg", "shape": [1]}, {"name": "weight.exp_avg_sq", "shape": [1]}],
                {
                    "optimizer_step": str(step), "variant": variant, "optimizer": "AdamW",
                    "beta1": "0.9", "beta2": "0.95", "eps": "1e-08", "weight_decay": "0.01",
                },
                {
                    "weight.exp_avg": np.asarray([0.2], dtype="<f4"),
                    "weight.exp_avg_sq": np.asarray([0.04], dtype="<f4"),
                },
            )
            cp = {
                "schema_version": "work-planner-checkpoint-manifest/1.0", "run_id": "run-1",
                "checkpoint_kind": "FLOPS_SENSITIVITY", "variant": variant, "seed": seed, "optimizer_step": step,
                "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"),
                "module_inventory_contract_sha256": _sha(root / "docs/architecture/planner_module_inventory_v1.yaml"),
                "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
                "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(),
                "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
                "initialization_manifest_path": init_path.relative_to(root).as_posix(), "initialization_manifest_sha256": _sha(init_path),
                "selected_config_path": config_path.relative_to(root).as_posix(), "selected_config_sha256": _sha(config_path),
                "dataset_manifest_path": dataset_path.relative_to(root).as_posix(), "dataset_manifest_sha256": _sha(dataset_path),
                "ordered_training_examples_path": ordered_path.relative_to(root).as_posix(), "ordered_training_examples_sha256": _sha(ordered_path),
                "environment_lock_path": "locks/environment.lock.json", "environment_lock_sha256": _sha(env_path),
                "model_file_path": model.relative_to(root).as_posix(), "model_file_sha256": _sha(model), "model_format": "SAFETENSORS",
                "model_tensor_table_sha256": canonical_tensor_table([{"name": "weight", "shape": [1], "dtype": "F32"}]),
                "optimizer_state_path": opt.relative_to(root).as_posix(), "optimizer_state_sha256": _sha(opt),
                "optimizer_state_format": "SAFETENSORS_ADAMW",
                "optimizer_state_tensor_table_sha256": canonical_tensor_table([
                    {"name": "weight.exp_avg", "shape": [1], "dtype": "F32"},
                    {"name": "weight.exp_avg_sq", "shape": [1], "dtype": "F32"},
                ]),
            }
            cp["manifest_hash"] = hash_json(cp)
            cp_path = root / f"reports/training/checkpoints/flops-{variant}-seed-{seed}.json"
            cp_path.write_text(json.dumps(cp))
            report = {
                "schema_version": "work-planner-training/1.2", "run_id": "run-1",
                "training_regime": "FLOPS_SENSITIVITY", "variant": variant, "config_id": cfg["config_id"], "seed": seed,
                "selected_config_path": config_path.relative_to(root).as_posix(), "selected_config_sha256": _sha(config_path),
                "dataset_manifest_path": dataset_path.relative_to(root).as_posix(), "dataset_manifest_sha256": _sha(dataset_path),
                "environment_lock_path": "locks/environment.lock.json", "environment_lock_sha256": _sha(env_path),
                "architecture_contract_sha256": _sha(root / "docs/architecture/planner_architecture_v1.yaml"),
                "module_inventory_contract_sha256": _sha(root / "docs/architecture/planner_module_inventory_v1.yaml"),
                "initialization_contract_sha256": _sha(root / "docs/training/planner_initialization_contract_v1.yaml"),
                "parameter_inventory_manifest_path": inv_path.relative_to(root).as_posix(),
                "parameter_inventory_manifest_sha256": _sha(inv_path), "parameter_inventory_sha256": inventory_hash,
                "initialization_manifest_path": init_path.relative_to(root).as_posix(), "initialization_manifest_sha256": _sha(init_path),
                "ordered_training_examples_path": ordered_path.relative_to(root).as_posix(), "ordered_training_examples_sha256": _sha(ordered_path),
                "dormant_gradient_audit_path": audit_path.relative_to(root).as_posix(), "dormant_gradient_audit_sha256": _sha(audit_path),
                "checkpoint_manifest_path": cp_path.relative_to(root).as_posix(), "checkpoint_manifest_sha256": _sha(cp_path),
                "optimizer_step": step, "checkpoint_selection": "FLOPS_CAP_STEP",
                "history": [{"optimizer_step": step}], "resource_usage": {},
            }
            report["report_hash"] = hash_json(report)
            (out / f"{variant}-seed-{seed}.json").write_text(json.dumps(report))

def test_training_evidence_is_exact_six_by_five_matrix(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path)
    assert validate_final_training_matrix(tmp_path) == []
    (tmp_path / "reports/training/final/A3r-seed-505.json").unlink()
    assert any("matrix mismatch" in error for error in validate_final_training_matrix(tmp_path))


def test_training_evidence_rejects_opaque_checkpoint(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path, corrupt_checkpoint=True)
    errors = validate_final_training_matrix(tmp_path)
    assert any("not a valid bound safetensors" in error for error in errors)


def test_implementation_spec_includes_a3r_final_training_and_inventory() -> None:
    spec = (ROOT / "docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.21.md").read_text()
    assert "A1/A2/A2b/A2c/A3/A3r обучаются ровно 12 000" in spec
    assert "planner_module_inventory_v1.yaml" in spec



def test_training_evidence_rejects_false_dormant_gradient_claim(tmp_path: Path) -> None:
    _write_final_matrix(tmp_path)
    audit_path = tmp_path / "reports/training/dormant-gradients/A3-seed-101.json"
    audit = json.loads(audit_path.read_text())
    audit["entries"][0]["gradient_state"] = "NONE"
    audit["status"] = "PASS"
    audit["audit_hash"] = hash_json({k: v for k, v in audit.items() if k != "audit_hash"})
    audit_path.write_text(json.dumps(audit))
    report_path = tmp_path / "reports/training/final/A3-seed-101.json"
    report = json.loads(report_path.read_text())
    report["dormant_gradient_audit_sha256"] = _sha(audit_path)
    report["report_hash"] = hash_json({k: v for k, v in report.items() if k != "report_hash"})
    report_path.write_text(json.dumps(report))
    errors = validate_final_training_matrix(tmp_path)
    assert any("dormant gradient" in error.lower() for error in errors)


def test_model_audit_pass_requires_seed17_initialization_and_all_dormant_audits(tmp_path: Path) -> None:
    inventory, _ = _prepare_contracts(tmp_path)
    inv_path = tmp_path / "reports/model-evidence/parameter-inventory.json"
    checks_dir = tmp_path / "reports/model-evidence/checks"
    checks_dir.mkdir(parents=True, exist_ok=True)
    check_ids = [
        "PARAMETER_TOLERANCE",
        "SAME_INFORMATION",
        "RAW_ROLLOUT",
        "DORMANT_GRADIENT",
        "MODEL_ARCHITECTURE_INITIALIZATION_AND_DORMANT_PARAMETERS",
    ]
    rows = []
    for check_id in check_ids:
        bindings = [{
            "path": "reports/model-evidence/parameter-inventory.json",
            "sha256": _sha(inv_path),
        }]
        value = True
        if check_id == "MODEL_ARCHITECTURE_INITIALIZATION_AND_DORMANT_PARAMETERS":
            value = {"inventory_exact": True, "initialization_exact": True, "dormant_gradients_exact": True}
            bindings.append({
                "path": "docs/architecture/planner_module_inventory_v1.yaml",
                "sha256": _sha(tmp_path / "docs/architecture/planner_module_inventory_v1.yaml"),
            })
        evidence = {
            "schema_version": "work-planner-model-audit-evidence/1.0",
            "run_id": "run-1",
            "check_id": check_id,
            "status": "PASS",
            "recomputed_value": value,
            "expected_value": value,
            "bindings": bindings,
        }
        evidence["evidence_hash"] = hash_json(evidence)
        evidence_path = checks_dir / f"{check_id}.json"
        evidence_path.write_text(json.dumps(evidence))
        rows.append({
            "check_id": check_id,
            "status": "PASS",
            "evidence_path": evidence_path.relative_to(tmp_path).as_posix(),
            "evidence_sha256": _sha(evidence_path),
            "recomputed_value": value,
            "expected_value": value,
        })
    report = {
        "schema_version": "work-planner-model-audit/1.2",
        "run_id": "run-1",
        "variants": ["A1", "A2", "A2b", "A2c", "A3", "A3r", "A4", "A5"],
        "architecture_contract_sha256": _sha(tmp_path / "docs/architecture/planner_architecture_v1.yaml"),
        "module_inventory_contract_sha256": _sha(tmp_path / "docs/architecture/planner_module_inventory_v1.yaml"),
        "initialization_contract_sha256": _sha(tmp_path / "docs/training/planner_initialization_contract_v1.yaml"),
        "parameter_inventory_sha256": inventory["inventory_hash"],
        "checks": rows,
        "status": "PASS",
    }
    report["report_hash"] = hash_json(report)
    errors = validate_model_audit_report(tmp_path, report)
    assert any("seed-17 initialization" in error or "six dormant-gradient" in error for error in errors)


def test_flops_sensitivity_is_required_in_p07_and_p08_lineage() -> None:
    registry = yaml.safe_load((ROOT / "docs/operator/phase_registry_v1.yaml").read_text())
    phases = {phase["phase_id"]: phase for phase in registry["phases"]}
    p07 = " ".join(phases["P07"]["actions"] + phases["P07"]["required_outputs"])
    assert "10 FLOPs-sensitivity" in p07 or "flops-sensitivity" in p07
    p08 = " ".join(phases["P08"]["actions"])
    assert "A2C_FLOPS" in p08 and "A3_FLOPS" in p08
    common = json.loads((ROOT / "docs/schemas/common.schema.json").read_text())
    arm_values = set(common["$defs"]["arm"]["enum"])
    assert {"PLANNER_A2C_FLOPS_RAW", "PLANNER_A3_FLOPS_RAW"} <= arm_values
