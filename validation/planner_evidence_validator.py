from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import yaml

from validation.hashing import hash_json
from validation.planner_initialization_validator import canonical_tensor_seed

TRAINABLE_VARIANTS = ("A1", "A2", "A2b", "A2c", "A3", "A3r")
FINAL_SEEDS = (101, 202, 303, 404, 505)


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, rel: Any) -> Path:
    if not isinstance(rel, str) or not rel:
        raise ValueError("artifact path missing")
    root_r = root.resolve()
    path = (root / rel).resolve()
    if path != root_r and root_r not in path.parents:
        raise ValueError(f"artifact outside repository: {rel}")
    if not path.is_file():
        raise ValueError(f"artifact missing: {rel}")
    return path


def without(obj: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if k != key}


def canonical_tensor_table(rows: list[Mapping[str, Any]]) -> str:
    normalized = [
        {"name": str(row["name"]), "shape": [int(x) for x in row["shape"]], "dtype": str(row["dtype"])}
        for row in rows
    ]
    normalized.sort(key=lambda row: row["name"])
    return hash_json(normalized)


def load_module_inventory(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    path = root / "docs/architecture/planner_module_inventory_v1.yaml"
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"module inventory unreadable: {exc}"]
    if obj.get("inventory_hash") != hash_json(without(obj, "inventory_hash")):
        errors.append("module inventory self-hash mismatch")
    tensors = obj.get("tensors", [])
    names = [row.get("name") for row in tensors if isinstance(row, dict)]
    if len(names) != len(set(names)) or not names:
        errors.append("module inventory tensor names missing or duplicated")
    valid_arms = set(TRAINABLE_VARIANTS)
    for i, row in enumerate(tensors):
        if not isinstance(row, dict):
            errors.append(f"module inventory tensors[{i}] is not an object")
            continue
        if not row.get("shape") or any(not isinstance(x, int) or x <= 0 for x in row.get("shape", [])):
            errors.append(f"module inventory tensors[{i}] invalid shape")
        arms = row.get("active_arms", [])
        if not arms or len(arms) != len(set(arms)) or not set(arms) <= valid_arms:
            errors.append(f"module inventory tensors[{i}] invalid active_arms")
    return obj, errors


def validate_parameter_inventory_manifest(root: Path, obj: Mapping[str, Any]) -> list[str]:
    contract, errors = load_module_inventory(root)
    contract_path = root / "docs/architecture/planner_module_inventory_v1.yaml"
    if obj.get("contract_path") != "docs/architecture/planner_module_inventory_v1.yaml":
        errors.append("parameter inventory contract_path mismatch")
    if obj.get("contract_sha256") != file_digest(contract_path):
        errors.append("parameter inventory contract_sha256 mismatch")
    if obj.get("tensors") != contract.get("tensors"):
        errors.append("actual state_dict inventory differs from locked tensor list")
    if obj.get("inventory_hash") != contract.get("inventory_hash"):
        errors.append("parameter inventory hash differs from locked inventory")
    if obj.get("manifest_hash") != hash_json(without(obj, "manifest_hash")):
        errors.append("parameter inventory manifest self-hash mismatch")
    return errors


def config_id(parameters: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return "cfg-" + digest[:16]


def validate_selected_config(root: Path, obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    contract_path = root / "docs/training/hyperparameter_search_v1.yaml"
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if obj.get("search_contract_sha256") != file_digest(contract_path):
        errors.append("selected config does not bind hyperparameter search contract")
    params = obj.get("parameters", {})
    if obj.get("config_id") != config_id(params):
        errors.append("selected config_id is not derived from canonical parameters")
    for name, values in contract.get("grid", {}).items():
        if params.get(name) not in values:
            errors.append(f"selected config parameter outside locked grid: {name}")
    if obj.get("selection_frozen_before_final_seeds") is not True:
        errors.append("selected config was not frozen before final seeds")
    if obj.get("report_hash") != hash_json(without(obj, "report_hash")):
        errors.append("selected config self-hash mismatch")
    return errors


def validate_ordered_examples(obj: Mapping[str, Any], *, expected_seed: int, dataset_sha256: str) -> list[str]:
    errors: list[str] = []
    if obj.get("seed") != expected_seed:
        errors.append("ordered examples seed mismatch")
    if obj.get("dataset_manifest_sha256") != dataset_sha256:
        errors.append("ordered examples dataset hash mismatch")
    ids = obj.get("example_ids", [])
    if not ids or len(ids) != len(set(ids)):
        errors.append("ordered examples IDs empty or duplicated")
    if obj.get("examples_sha256") != hash_json(ids):
        errors.append("ordered examples list hash mismatch")
    if obj.get("manifest_hash") != hash_json(without(obj, "manifest_hash")):
        errors.append("ordered examples manifest self-hash mismatch")
    return errors


def parse_safetensors(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    size = path.stat().st_size
    with path.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError("safetensors file shorter than 8-byte header length")
        header_len = struct.unpack("<Q", raw)[0]
        if header_len <= 1 or header_len > min(size - 8, 64 * 1024 * 1024):
            raise ValueError("invalid safetensors header length")
        header_raw = fh.read(header_len)
    try:
        header = json.loads(header_raw.decode("utf-8").rstrip(" "))
    except Exception as exc:
        raise ValueError(f"invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
        raise ValueError("invalid safetensors metadata")
    tensors: dict[str, Any] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, row in header.items():
        if not isinstance(row, dict) or set(row) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"invalid tensor header: {name}")
        if row["dtype"] != "F32" or not row["shape"] or any(not isinstance(x, int) or x <= 0 for x in row["shape"]):
            raise ValueError(f"unsupported dtype/shape: {name}")
        offsets = row["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(not isinstance(x, int) for x in offsets):
            raise ValueError(f"invalid data offsets: {name}")
        start, end = offsets
        expected_bytes = 4
        for dim in row["shape"]:
            expected_bytes *= dim
        if start < 0 or end <= start or end - start != expected_bytes:
            raise ValueError(f"tensor byte length mismatch: {name}")
        ranges.append((start, end, name))
        tensors[name] = {"name": name, "shape": row["shape"], "dtype": row["dtype"]}
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError(f"overlapping tensor data: {previous[2]} / {current[2]}")
    data_bytes = size - 8 - header_len
    max_end = max((end for _, end, _ in ranges), default=0)
    if max_end != data_bytes:
        raise ValueError("safetensors data length does not match tensor table")
    return tensors, metadata


def _check_bound_file(root: Path, rel: Any, digest: Any, label: str) -> tuple[Path | None, list[str]]:
    try:
        path = safe_path(root, rel)
    except Exception as exc:
        return None, [f"{label}: {exc}"]
    return path, [] if digest == file_digest(path) else [f"{label} sha256 mismatch"]


def validate_checkpoint_manifest(root: Path, obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("manifest_hash") != hash_json(without(obj, "manifest_hash")):
        errors.append("checkpoint manifest self-hash mismatch")
    for rel, field in (
        ("docs/architecture/planner_architecture_v1.yaml", "architecture_contract_sha256"),
        ("docs/architecture/planner_module_inventory_v1.yaml", "module_inventory_contract_sha256"),
        ("docs/training/planner_initialization_contract_v1.yaml", "initialization_contract_sha256"),
    ):
        if obj.get(field) != file_digest(root / rel):
            errors.append(f"checkpoint {field} mismatch")
    inv_path, errs = _check_bound_file(root, obj.get("parameter_inventory_manifest_path"), obj.get("parameter_inventory_manifest_sha256"), "parameter inventory manifest")
    errors += errs
    inventory: dict[str, Any] = {}
    if inv_path:
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
        errors += validate_parameter_inventory_manifest(root, inventory)
        if obj.get("parameter_inventory_sha256") != inventory.get("inventory_hash"):
            errors.append("checkpoint parameter inventory content hash mismatch")
    model_path, errs = _check_bound_file(root, obj.get("model_file_path"), obj.get("model_file_sha256"), "model checkpoint")
    errors += errs
    if model_path:
        try:
            table, metadata = parse_safetensors(model_path)
            expected_rows = inventory.get("tensors", [])
            expected = {row["name"]: {"name": row["name"], "shape": row["shape"], "dtype": row["dtype"]} for row in expected_rows}
            if table != expected:
                errors.append("model safetensors header differs from exact locked inventory")
            if obj.get("model_tensor_table_sha256") != canonical_tensor_table(list(table.values())):
                errors.append("model tensor table hash mismatch")
            expected_meta = {
                "run_id": str(obj.get("run_id")), "checkpoint_kind": str(obj.get("checkpoint_kind")),
                "variant": str(obj.get("variant")), "seed": str(obj.get("seed")),
                "optimizer_step": str(obj.get("optimizer_step")), "parameter_inventory_sha256": str(obj.get("parameter_inventory_sha256")),
            }
            for key, value in expected_meta.items():
                if metadata.get(key) != value:
                    errors.append(f"model safetensors metadata mismatch: {key}")
        except Exception as exc:
            errors.append(f"model checkpoint is not a valid bound safetensors file: {exc}")
    kind = obj.get("checkpoint_kind")
    if kind == "INITIALIZATION":
        if obj.get("variant") != "COMMON" or obj.get("optimizer_step") != 0:
            errors.append("initialization checkpoint must be COMMON at step 0")
        for field in ("initialization_manifest_path", "initialization_manifest_sha256", "selected_config_path", "selected_config_sha256", "dataset_manifest_path", "dataset_manifest_sha256", "ordered_training_examples_path", "ordered_training_examples_sha256", "environment_lock_path", "environment_lock_sha256", "optimizer_state_path", "optimizer_state_sha256", "optimizer_state_format", "optimizer_state_tensor_table_sha256"):
            if obj.get(field) is not None:
                errors.append(f"initialization checkpoint field must be null: {field}")
    else:
        if obj.get("variant") not in TRAINABLE_VARIANTS:
            errors.append("trained checkpoint variant invalid")
        for path_field, hash_field, label in (
            ("initialization_manifest_path", "initialization_manifest_sha256", "initialization manifest"),
            ("selected_config_path", "selected_config_sha256", "selected config"),
            ("dataset_manifest_path", "dataset_manifest_sha256", "dataset manifest"),
            ("ordered_training_examples_path", "ordered_training_examples_sha256", "ordered examples"),
            ("environment_lock_path", "environment_lock_sha256", "environment lock"),
        ):
            _, errs = _check_bound_file(root, obj.get(path_field), obj.get(hash_field), label)
            errors += errs
        opt_path, errs = _check_bound_file(root, obj.get("optimizer_state_path"), obj.get("optimizer_state_sha256"), "optimizer state")
        errors += errs
        if obj.get("optimizer_state_format") != "SAFETENSORS_ADAMW":
            errors.append("optimizer state format must be SAFETENSORS_ADAMW")
        if opt_path and inventory:
            try:
                table, metadata = parse_safetensors(opt_path)
                variant = str(obj.get("variant"))
                expected: dict[str, dict[str, Any]] = {}
                for row in inventory.get("tensors", []):
                    if variant in row.get("active_arms", []):
                        for suffix in ("exp_avg", "exp_avg_sq"):
                            name = f"{row['name']}.{suffix}"
                            expected[name] = {"name": name, "shape": row["shape"], "dtype": "F32"}
                if table != expected:
                    errors.append("optimizer safetensors header differs from exact active parameter state set")
                if obj.get("optimizer_state_tensor_table_sha256") != canonical_tensor_table(list(table.values())):
                    errors.append("optimizer tensor table hash mismatch")
                if metadata.get("optimizer_step") != str(obj.get("optimizer_step")) or metadata.get("variant") != variant:
                    errors.append("optimizer safetensors metadata mismatch")
            except Exception as exc:
                errors.append(f"optimizer state is not a valid bound safetensors file: {exc}")
    return errors


def validate_initialization_manifest(root: Path, obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("manifest_hash") != hash_json(without(obj, "manifest_hash")):
        errors.append("initialization manifest self-hash mismatch")
    contract_path = root / "docs/training/planner_initialization_contract_v1.yaml"
    if obj.get("initialization_contract_sha256") != file_digest(contract_path):
        errors.append("initialization contract hash mismatch")
    inv_path, errs = _check_bound_file(root, obj.get("parameter_inventory_manifest_path"), obj.get("parameter_inventory_manifest_sha256"), "parameter inventory manifest")
    errors += errs
    inventory: dict[str, Any] = {}
    if inv_path:
        inventory = json.loads(inv_path.read_text(encoding="utf-8"))
        errors += validate_parameter_inventory_manifest(root, inventory)
        if obj.get("parameter_inventory_sha256") != inventory.get("inventory_hash"):
            errors.append("initialization inventory hash mismatch")
    rows = obj.get("tensor_seeds", [])
    names = [row.get("name") for row in rows if isinstance(row, dict)]
    expected_names = [row["name"] for row in inventory.get("tensors", [])]
    if names != expected_names:
        errors.append("initialization tensor seed list must exactly follow locked inventory order")
    seed = obj.get("seed")
    for row in rows:
        if isinstance(seed, int) and row.get("tensor_seed") != canonical_tensor_seed(seed, str(row.get("name"))):
            errors.append(f"initialization tensor seed mismatch: {row.get('name')}")
    cp_path, errs = _check_bound_file(root, obj.get("checkpoint_manifest_path"), obj.get("checkpoint_manifest_sha256"), "initialization checkpoint manifest")
    errors += errs
    if cp_path:
        cp = json.loads(cp_path.read_text(encoding="utf-8"))
        errors += validate_checkpoint_manifest(root, cp)
        if cp.get("checkpoint_kind") != "INITIALIZATION" or cp.get("seed") != seed or cp.get("run_id") != obj.get("run_id"):
            errors.append("initialization checkpoint manifest identity mismatch")
    return errors


def validate_dormant_gradient_audit(root: Path, obj: Mapping[str, Any], inventory: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("audit_hash") != hash_json(without(obj, "audit_hash")):
        errors.append("dormant gradient audit self-hash mismatch")
    if obj.get("parameter_inventory_sha256") != inventory.get("inventory_hash"):
        errors.append("dormant gradient audit inventory hash mismatch")
    variant = str(obj.get("variant"))
    entries = obj.get("entries", [])
    expected_rows = inventory.get("tensors", [])
    if [row.get("name") for row in entries] != [row.get("name") for row in expected_rows]:
        errors.append("dormant gradient audit tensor list differs from locked inventory")
    invalid = False
    for actual, expected in zip(entries, expected_rows):
        active = variant in expected.get("active_arms", [])
        expected_activity = "ACTIVE" if active else "DORMANT"
        expected_state = "PRESENT_FINITE" if active else "NONE"
        if actual.get("expected_activity") != expected_activity or actual.get("gradient_state") != expected_state:
            errors.append(f"dormant gradient state mismatch: {expected.get('name')}")
            invalid = True
        if actual.get("nonfinite_count") != 0 or actual.get("materialized_zero_count") != 0:
            invalid = True
    if obj.get("status") != ("FAIL" if invalid or errors else "PASS"):
        errors.append("dormant gradient audit status mismatch")
    return errors


def validate_model_audit_check_evidence(root: Path, obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("evidence_hash") != hash_json(without(obj, "evidence_hash")):
        errors.append("model audit check evidence self-hash mismatch")
    for i, binding in enumerate(obj.get("bindings", [])):
        _, errs = _check_bound_file(root, binding.get("path"), binding.get("sha256"), f"model audit binding[{i}]")
        errors += errs
    if obj.get("status") == "PASS" and obj.get("recomputed_value") != obj.get("expected_value"):
        errors.append("model audit check cannot PASS when recomputed value differs")
    return errors
