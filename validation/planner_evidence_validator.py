from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Mapping

import yaml
import numpy as np

from validation.hashing import hash_json
from validation.planner_initialization_validator import canonical_tensor_seed
from docs.domain.intent_labeler_v1 import label_intent

TRAINABLE_VARIANTS = ("A1", "A2", "A2b", "A2c", "A3", "A3r")
FINAL_SEEDS = (101, 202, 303, 404, 505)


PADDING_EMBEDDING_NAMES = {
    "task_encoder.token_embedding.weight",
    "planner_decoder.a1_token_embedding.weight",
}


def _read_safetensors_arrays(path: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, np.ndarray]]:
    size = path.stat().st_size
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError("safetensors file shorter than 8-byte header length")
    header_len = struct.unpack("<Q", raw[:8])[0]
    if header_len <= 1 or header_len > min(size - 8, 64 * 1024 * 1024):
        raise ValueError("invalid safetensors header length")
    header = json.loads(raw[8:8 + header_len].decode("utf-8").rstrip(" "))
    metadata = header.pop("__metadata__", {})
    data = memoryview(raw)[8 + header_len:]
    table: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for name, row in header.items():
        start, end = row["data_offsets"]
        shape = tuple(int(x) for x in row["shape"])
        arr = np.frombuffer(data[start:end], dtype="<f4").reshape(shape).copy()
        table[name] = {"name": name, "shape": list(shape), "dtype": row["dtype"]}
        arrays[name] = arr
    return table, metadata, arrays


def canonical_initialized_array(row: Mapping[str, Any], seed: int) -> np.ndarray:
    name = str(row["name"])
    shape = tuple(int(x) for x in row["shape"])
    parameter_type = str(row["parameter_type"])
    tensor_seed = canonical_tensor_seed(seed, name)
    rng = np.random.Generator(np.random.PCG64(tensor_seed))
    if parameter_type in {"linear_weight", "bilinear_weight"}:
        if len(shape) < 2:
            raise ValueError(f"Xavier tensor must have at least 2 dimensions: {name}")
        fan_out, fan_in = shape[0], shape[1]
        limit = float(np.sqrt(6.0 / float(fan_in + fan_out)))
        arr = rng.uniform(-limit, limit, size=shape).astype("<f4")
    elif parameter_type in {"embedding_weight", "standalone_parameter"}:
        arr = rng.normal(0.0, 0.02, size=shape).astype("<f4")
        if name in PADDING_EMBEDDING_NAMES:
            arr[0] = 0.0
    elif parameter_type in {"linear_bias", "layer_norm_bias"}:
        arr = np.zeros(shape, dtype="<f4")
    elif parameter_type == "layer_norm_weight":
        arr = np.ones(shape, dtype="<f4")
    else:
        raise ValueError(f"unsupported parameter_type for initialization: {parameter_type}")
    return arr


def _validate_initialization_values(path: Path, inventory: Mapping[str, Any], seed: int) -> list[str]:
    errors: list[str] = []
    try:
        _, _, arrays = _read_safetensors_arrays(path)
        for row in inventory.get("tensors", []):
            name = str(row["name"])
            expected = canonical_initialized_array(row, seed)
            actual = arrays.get(name)
            if actual is None or actual.shape != expected.shape or not np.array_equal(actual, expected):
                errors.append(f"initialization tensor values differ from canonical initializer: {name}")
    except Exception as exc:
        errors.append(f"initialization value validation failed: {exc}")
    return errors


def _validate_trained_values(
    model_path: Path, init_path: Path, inventory: Mapping[str, Any], variant: str,
) -> list[str]:
    errors: list[str] = []
    try:
        _, _, model_arrays = _read_safetensors_arrays(model_path)
        _, _, init_arrays = _read_safetensors_arrays(init_path)
        active_changed = False
        for row in inventory.get("tensors", []):
            name = str(row["name"])
            actual = model_arrays[name]
            initial = init_arrays[name]
            if not np.isfinite(actual).all():
                errors.append(f"model tensor contains NaN/Inf: {name}")
            active = variant in row.get("active_arms", [])
            if active:
                if not np.array_equal(actual, initial):
                    active_changed = True
            elif not np.array_equal(actual, initial):
                errors.append(f"dormant tensor changed from initialization: {name}")
        if not active_changed:
            errors.append("trained checkpoint has no changed active tensor relative to initialization")
    except Exception as exc:
        errors.append(f"trained tensor value validation failed: {exc}")
    return errors


def _validate_optimizer_values(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        _, metadata, arrays = _read_safetensors_arrays(path)
        total_l1 = 0.0
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                errors.append(f"optimizer tensor contains NaN/Inf: {name}")
            if name.endswith(".exp_avg_sq") and np.any(arr < 0):
                errors.append(f"optimizer exp_avg_sq contains negative values: {name}")
            total_l1 += float(np.abs(arr).sum(dtype=np.float64))
        if total_l1 == 0.0:
            errors.append("optimizer active-state tensors are all zero")
        required_metadata = {
            "optimizer": "AdamW",
            "beta1": "0.9",
            "beta2": "0.95",
            "eps": "1e-08",
            "weight_decay": "0.01",
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                errors.append(f"optimizer safetensors metadata mismatch: {key}")
    except Exception as exc:
        errors.append(f"optimizer value validation failed: {exc}")
    return errors


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
    if kind == "INITIALIZATION" and model_path and inventory and isinstance(obj.get("seed"), int):
        errors += _validate_initialization_values(model_path, inventory, int(obj["seed"]))
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
                errors += _validate_optimizer_values(opt_path)
            except Exception as exc:
                errors.append(f"optimizer state is not a valid bound safetensors file: {exc}")
        if model_path and inventory:
            try:
                init_manifest_path = safe_path(root, obj.get("initialization_manifest_path"))
                init_manifest = json.loads(init_manifest_path.read_text(encoding="utf-8"))
                init_checkpoint_path = safe_path(root, init_manifest.get("checkpoint_manifest_path"))
                init_checkpoint = json.loads(init_checkpoint_path.read_text(encoding="utf-8"))
                init_model_path = safe_path(root, init_checkpoint.get("model_file_path"))
                errors += _validate_trained_values(model_path, init_model_path, inventory, str(obj.get("variant")))
            except Exception as exc:
                errors.append(f"trained checkpoint initialization lineage unreadable: {exc}")
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


def _binding_map(obj: Mapping[str, Any]) -> dict[str, str]:
    return {str(row.get("path")): str(row.get("sha256")) for row in obj.get("bindings", [])}


def _validate_exact_bindings(root: Path, actual: Mapping[str, str], required: set[str], label: str) -> list[str]:
    errors: list[str] = []
    if set(actual) != required:
        errors.append(f"{label} must bind exact paths: {sorted(required)}")
    for rel in required:
        path = root / rel
        if not path.is_file() or actual.get(rel) != file_digest(path):
            errors.append(f"{label} binding invalid: {rel}")
    return errors


def _parameter_tolerance_result(root: Path) -> tuple[dict[str, Any], list[str]]:
    inventory_path = root / "reports/model-evidence/parameter-inventory.json"
    errors: list[str] = []
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        errors += validate_parameter_inventory_manifest(root, inventory)
        total = 0
        active: dict[str, int] = {variant: 0 for variant in TRAINABLE_VARIANTS}
        for row in inventory.get("tensors", []):
            count = 1
            for dim in row.get("shape", []):
                count *= int(dim)
            total += count
            for variant in row.get("active_arms", []):
                active[str(variant)] += count
        return {
            "inventory_sha256": inventory.get("inventory_hash"),
            "common_superset_parameter_count": total,
            "active_parameter_counts": active,
            "total_parameter_count_tolerance_fraction": 0.0,
            "all_trainable_arms_share_exact_inventory": True,
        }, errors
    except Exception as exc:
        return {}, [f"parameter tolerance evidence unreadable: {exc}"]


def _same_information_result(details: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    cases = details.get("cases", [])
    if not isinstance(cases, list) or len(cases) < 21:
        return {}, ["SAME_INFORMATION requires at least 21 fixed coverage cases"]
    seen: set[str] = set()
    mismatches = 0
    intent_counts = {intent_id: 0 for intent_id in range(7)}
    for i, case in enumerate(cases):
        try:
            case_id = str(case["case_id"])
            if case_id in seen:
                errors.append(f"duplicate SAME_INFORMATION case_id: {case_id}")
            seen.add(case_id)
            expected = label_intent(
                case["state"], case["goal"], case["all_shortest_first_actions"],
                case["selected_action"], int(case["remaining_oracle_length"]),
            )
            intent_counts[int(expected["intent_id"])] += 1
            if case.get("a2c_semantic_signature") != expected["semantic_signature"]:
                mismatches += 1
                errors.append(f"SAME_INFORMATION A2c signature mismatch: {case_id}")
            if case.get("a3_canonical_text") != expected["canonical_text"]:
                mismatches += 1
                errors.append(f"SAME_INFORMATION A3 canonical text mismatch: {case_id}")
        except Exception as exc:
            mismatches += 1
            errors.append(f"SAME_INFORMATION case[{i}] invalid: {exc}")
    missing = [intent_id for intent_id, count in intent_counts.items() if count < 3]
    if missing:
        errors.append(f"SAME_INFORMATION requires at least three cases per intent_id; insufficient: {missing}")
    return {"case_count": len(cases), "mismatch_count": mismatches, "intent_counts": {str(k): v for k, v in intent_counts.items()}}, errors


def _raw_rollout_result(root: Path, details: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    cases = details.get("cases", [])
    variants = ("A1", "A2", "A2b", "A2c", "A3", "A3r")
    if not isinstance(cases, list) or len(cases) != 36:
        return {}, ["RAW_ROLLOUT requires exactly 36 fixed cases (6 variants x 6 intents)"]
    seen: set[str] = set()
    matrix: set[tuple[str, int]] = set()
    violations = 0
    for i, case in enumerate(cases):
        try:
            case_id = str(case["case_id"])
            if case_id in seen:
                violations += 1
                errors.append(f"duplicate RAW_ROLLOUT case_id: {case_id}")
            seen.add(case_id)
            variant = str(case.get("variant"))
            intent_id = int(case.get("intent_id", -1))
            if variant not in variants or intent_id not in range(6):
                violations += 1
                errors.append(f"RAW_ROLLOUT {case_id} invalid variant/intent pair")
            matrix.add((variant, intent_id))
            required = {
                "planner_calls": 1,
                "execution_planner_calls": 0,
                "domain_action_mask_applied": False,
                "grammar_constrained_decoding": False,
                "replanning": False,
            }
            for field, expected in required.items():
                if case.get(field) != expected:
                    violations += 1
                    errors.append(f"RAW_ROLLOUT {case_id} {field} mismatch")
            for path_field, digest_field in (
                ("raw_logits_path", "raw_logits_sha256"),
                ("frozen_plan_path", "frozen_plan_sha256"),
                ("event_log_path", "event_log_sha256"),
            ):
                rel = case.get(path_field)
                submitted = case.get(digest_field)
                _, file_errors = _check_bound_file(root, rel, submitted, f"RAW_ROLLOUT {case_id} {path_field}")
                if file_errors:
                    violations += len(file_errors)
                    errors.extend(file_errors)
            consumed = int(case.get("plan_positions_consumed", -1))
            if consumed < 1 or consumed > 17:
                violations += 1
                errors.append(f"RAW_ROLLOUT {case_id} invalid consumed positions")
        except Exception as exc:
            violations += 1
            errors.append(f"RAW_ROLLOUT case[{i}] invalid: {exc}")
    expected_matrix = {(variant, intent_id) for variant in variants for intent_id in range(6)}
    if matrix != expected_matrix:
        errors.append("RAW_ROLLOUT variant x intent matrix mismatch")
    return {"case_count": len(cases), "violation_count": violations, "matrix_complete": matrix == expected_matrix}, errors

def _dormant_gradient_result(root: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    inventory_path = root / "reports/model-evidence/parameter-inventory.json"
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        errors += validate_parameter_inventory_manifest(root, inventory)
    except Exception as exc:
        return {}, [f"DORMANT_GRADIENT inventory unreadable: {exc}"]
    passed: list[str] = []
    for variant in TRAINABLE_VARIANTS:
        path = root / f"reports/model-evidence/dormant-gradients/{variant}-seed-17.json"
        try:
            audit = json.loads(path.read_text(encoding="utf-8"))
            before = len(errors)
            errors += validate_dormant_gradient_audit(root, audit, inventory)
            if audit.get("variant") != variant or audit.get("seed") != 17 or audit.get("status") != "PASS":
                errors.append(f"DORMANT_GRADIENT identity/status mismatch: {variant}")
            if len(errors) == before:
                passed.append(variant)
        except Exception as exc:
            errors.append(f"DORMANT_GRADIENT audit unreadable ({variant}): {exc}")
    return {"audit_seed": 17, "variants_passed": passed, "required_variant_count": len(TRAINABLE_VARIANTS)}, errors


def validate_model_audit_check_evidence(root: Path, obj: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if obj.get("evidence_hash") != hash_json(without(obj, "evidence_hash")):
        errors.append("model audit check evidence self-hash mismatch")
    bindings = _binding_map(obj)
    for i, binding in enumerate(obj.get("bindings", [])):
        _, errs = _check_bound_file(root, binding.get("path"), binding.get("sha256"), f"model audit binding[{i}]")
        errors += errs

    check_id = obj.get("check_id")
    details = obj.get("details", {})
    canonical: dict[str, Any] = {}
    check_errors: list[str] = []
    if check_id == "PARAMETER_TOLERANCE":
        required = {
            "docs/architecture/planner_module_inventory_v1.yaml",
            "reports/model-evidence/parameter-inventory.json",
        }
        errors += _validate_exact_bindings(root, bindings, required, "PARAMETER_TOLERANCE")
        canonical, check_errors = _parameter_tolerance_result(root)
    elif check_id == "SAME_INFORMATION":
        required = {
            "docs/domain/intent_labeler_v1.py",
            "docs/domain/intent_catalog_v1.yaml",
            "docs/semantic/semantic_target_v1.yaml",
        }
        errors += _validate_exact_bindings(root, bindings, required, "SAME_INFORMATION")
        canonical, check_errors = _same_information_result(details)
    elif check_id == "RAW_ROLLOUT":
        required = {
            "docs/architecture/a1_token_grammar_v1.yaml",
            "docs/training/planner_training_contract_v1.yaml",
        }
        errors += _validate_exact_bindings(root, bindings, required, "RAW_ROLLOUT")
        canonical, check_errors = _raw_rollout_result(root, details)
    elif check_id == "DORMANT_GRADIENT":
        required = {"reports/model-evidence/parameter-inventory.json"} | {
            f"reports/model-evidence/dormant-gradients/{variant}-seed-17.json"
            for variant in TRAINABLE_VARIANTS
        }
        errors += _validate_exact_bindings(root, bindings, required, "DORMANT_GRADIENT")
        canonical, check_errors = _dormant_gradient_result(root)
    elif check_id == "MODEL_ARCHITECTURE_INITIALIZATION_AND_DORMANT_PARAMETERS":
        canonical = {
            "inventory_exact": True,
            "initialization_exact": True,
            "dormant_gradients_exact": True,
        }
    else:
        errors.append(f"unknown model audit check_id: {check_id}")

    errors += check_errors
    if obj.get("recomputed_value") != canonical:
        errors.append(f"{check_id} submitted recomputed_value differs from validator recomputation")
    if obj.get("expected_value") != canonical:
        errors.append(f"{check_id} expected_value differs from canonical requirement")
    expected_status = "PASS" if not check_errors else "FAIL"
    if obj.get("status") != expected_status:
        errors.append(f"{check_id} status differs from validator result")
    return errors
