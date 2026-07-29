from __future__ import annotations

import hashlib
import itertools
import json
import unicodedata
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "work-planner-hash/1.0"


def canonical_json_bytes(value: Any) -> bytes:
    """RFC-8785-like restricted canonical JSON used by this experiment.

    Inputs are restricted to JSON-safe strings, integers, finite floats, booleans,
    null, lists and dictionaries with string keys. Strings are NFC-normalized;
    dict keys are sorted; separators and UTF-8 are fixed.
    """
    def norm(x: Any) -> Any:
        if isinstance(x, str):
            return unicodedata.normalize("NFC", x)
        if x is None or isinstance(x, (bool, int)):
            return x
        if isinstance(x, float):
            if x != x or x in (float("inf"), float("-inf")):
                raise ValueError("non-finite float is forbidden")
            return x
        if isinstance(x, list):
            return [norm(v) for v in x]
        if isinstance(x, tuple):
            return [norm(v) for v in x]
        if isinstance(x, dict):
            if not all(isinstance(k, str) for k in x):
                raise TypeError("JSON object keys must be strings")
            normalized_items: list[tuple[str, Any]] = []
            seen_keys: set[str] = set()
            for key, value in x.items():
                normalized_key = unicodedata.normalize("NFC", key)
                if normalized_key in seen_keys:
                    raise ValueError(f"duplicate JSON key after NFC normalization: {normalized_key!r}")
                seen_keys.add(normalized_key)
                normalized_items.append((normalized_key, norm(value)))
            return {key: value for key, value in sorted(normalized_items, key=lambda item: item[0])}
        raise TypeError(f"unsupported canonical JSON type: {type(x)!r}")

    return json.dumps(norm(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def hash_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def canonical_facts(facts: Iterable[Sequence[str]]) -> list[list[str]]:
    return sorted((list(f) for f in facts), key=lambda f: tuple(f))


def state_hash(facts: Iterable[Sequence[str]]) -> str:
    return hash_json({"schema": SCHEMA, "kind": "state", "domain": "blocks_world_v1", "facts": canonical_facts(facts)})


def goal_hash(goal: Iterable[Sequence[str]]) -> str:
    return hash_json({"schema": SCHEMA, "kind": "goal", "domain": "blocks_world_v1", "facts": canonical_facts(goal)})


def token_vector_hash(values: Sequence[int], kind: str) -> str:
    return hash_json({"schema": SCHEMA, "kind": kind, "values": list(values)})


def prompt_bytes_hash(rendered_utf8: bytes) -> str:
    return sha256_bytes(rendered_utf8)


def _replace_refs(facts: Iterable[Sequence[str]], mapping: Mapping[str, str]) -> list[list[str]]:
    return canonical_facts([[mapping.get(x, x) for x in fact] for fact in facts])


def canonical_task_payload(task: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalizes object names by exhaustive permutation (n <= 8).

    Surface/display names, task IDs, split and generation metadata are excluded.
    The lexicographically smallest initial+goal serialization is chosen.
    """
    refs = sorted(task["ledger"], key=lambda r: int(r[2:]))
    canonical_refs = [f"@B{i}" for i in range(len(refs))]
    best: bytes | None = None
    best_payload: dict[str, Any] | None = None
    for perm in itertools.permutations(canonical_refs):
        mapping = dict(zip(refs, perm))
        payload = {
            "schema": SCHEMA,
            "kind": "canonical_task",
            "domain": task["domain"],
            "block_count": len(refs),
            "initial": _replace_refs(task["initial"], mapping),
            "goal": _replace_refs(task["goal"], mapping),
        }
        encoded = canonical_json_bytes(payload)
        if best is None or encoded < best:
            best = encoded
            best_payload = payload
    if best_payload is None:
        raise ValueError("canonical task must contain at least one ledger reference")
    return best_payload


def canonical_task_hash(task: Mapping[str, Any]) -> str:
    return hash_json(canonical_task_payload(task))


def plan_content_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "kind": "plan_content",
        "task_id": plan["task_id"],
        "canonical_task_hash": plan["canonical_task_hash"],
        "state_hash": plan["state_hash"],
        "planner_checkpoint_sha256": plan["planner_checkpoint_sha256"],
        "planner_config_sha256": plan["planner_config_sha256"],
        "planner_seed": plan["planner_seed"],
        "planner_variant": plan["planner_variant"],
        "representation": plan["representation"],
        "semantic_artifact_manifest_sha256": plan["semantic_artifact_manifest_sha256"],
        "steps": [
            {
                "step_id": s["step_id"],
                "step_index": s["step_index"],
                "planner_variant": s["planner_variant"],
                "representation": s["representation"],
                "typed_action": s["typed_action"],
                "semantic_ref": s["semantic_ref"],
                "intent_id": s["intent_id"],
                "semantic_signature": s["semantic_signature"],
            }
            for s in plan["steps"]
        ],
    }


def plan_content_hash(plan: Mapping[str, Any]) -> str:
    return hash_json(plan_content_payload(plan))


def plan_artifact_hash(plan: Mapping[str, Any]) -> str:
    p = deepcopy(dict(plan))
    p.pop("plan_artifact_hash", None)
    return hash_json({"schema": SCHEMA, "kind": "plan_artifact", "value": p})


def manifest_content_hash(manifest: Mapping[str, Any]) -> str:
    m = deepcopy(dict(manifest))
    m.pop("manifest_content_hash", None)
    m.pop("manifest_hash", None)
    for artifact in m.get("artifacts", []):
        artifact.pop("created_at", None)
    return hash_json({"schema": SCHEMA, "kind": "semantic_manifest_content", "value": m})


def manifest_artifact_hash(manifest: Mapping[str, Any]) -> str:
    m = deepcopy(dict(manifest))
    m.pop("manifest_hash", None)
    return hash_json({"schema": SCHEMA, "kind": "semantic_manifest_artifact", "value": m})


def pair_group_hash(
    *,
    stage: str,
    task_id: str,
    base_task_id: str,
    split: str,
    snapshot_id: str | None,
    trajectory_policy: str,
    experiment_freeze_hash: str,
) -> str:
    return hash_json(
        {
            "schema": SCHEMA,
            "kind": "pair_group",
            "stage": stage,
            "task_id": task_id,
            "base_task_id": base_task_id,
            "split": split,
            "snapshot_id": snapshot_id,
            "trajectory_policy": trajectory_policy,
            "experiment_freeze_hash": experiment_freeze_hash,
        }
    )


def experiment_freeze_hash(freeze: Mapping[str, Any]) -> str:
    value = deepcopy(dict(freeze))
    value.pop("freeze_hash", None)
    return hash_json({"schema": SCHEMA, "kind": "experiment_freeze", "value": value})


def decision_record_hash(decision: Mapping[str, Any]) -> str:
    value = deepcopy(dict(decision))
    value.pop("decision_hash", None)
    value.pop("signature", None)
    return hash_json({"schema": SCHEMA, "kind": "operator_decision", "value": value})


def approved_freeze_pointer_hash(pointer: Mapping[str, Any]) -> str:
    value = deepcopy(dict(pointer))
    value.pop("pointer_hash", None)
    return hash_json({"schema": SCHEMA, "kind": "approved_freeze_pointer", "value": value})


def dispatch_record_hash(dispatch: Mapping[str, Any]) -> str:
    value = deepcopy(dict(dispatch))
    value.pop("dispatch_hash", None)
    return hash_json({"schema": SCHEMA, "kind": "dispatch_record", "value": value})
