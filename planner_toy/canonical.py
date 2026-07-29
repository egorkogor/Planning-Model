"""Canonical JSON and content-addressing helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import unicodedata
from copy import deepcopy
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    def normalize(item):
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, list | tuple):
            return [normalize(value) for value in item]
        if isinstance(item, dict):
            return {unicodedata.normalize("NFC", key): normalize(item[key]) for key in sorted(item)}
        return item

    return json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()


def artifact_hash(value: dict[str, Any], hash_field: str) -> str:
    payload = dict(value)
    payload.pop(hash_field, None)
    return sha256(payload)


HASH_SCHEMA = "work-planner-hash/1.0"


def _facts(facts) -> list[list[str]]:
    return sorted((list(fact) for fact in facts), key=tuple)


def state_hash(facts) -> str:
    return sha256(
        {
            "schema": HASH_SCHEMA,
            "kind": "state",
            "domain": "blocks_world_v1",
            "facts": _facts(facts),
        }
    )


def goal_hash(facts) -> str:
    return sha256(
        {"schema": HASH_SCHEMA, "kind": "goal", "domain": "blocks_world_v1", "facts": _facts(facts)}
    )


def canonical_task_hash(task: dict) -> str:
    refs = list(task.get("ledger", task.get("blocks", [])))
    canonical_refs = [f"@B{i}" for i in range(len(refs))]
    candidates = []
    for permutation in itertools.permutations(canonical_refs):
        mapping = dict(zip(refs, permutation, strict=True))

        def remap(rows, ref_mapping=mapping):
            return _facts([[ref_mapping.get(value, value) for value in row] for row in rows])

        candidates.append(
            {
                "schema": HASH_SCHEMA,
                "kind": "canonical_task",
                "domain": task.get("domain", task.get("domain_id")),
                "block_count": len(refs),
                "initial": remap(task["initial"]),
                "goal": remap(task["goal"]),
            }
        )
    return sha256(min(candidates, key=canonical_bytes))


def plan_content_hash(plan: dict) -> str:
    payload = {"schema": HASH_SCHEMA, "kind": "plan_content"}
    for field in (
        "task_id",
        "canonical_task_hash",
        "state_hash",
        "planner_checkpoint_sha256",
        "planner_config_sha256",
        "planner_seed",
        "planner_variant",
        "representation",
        "semantic_artifact_manifest_sha256",
    ):
        payload[field] = plan[field]
    payload["steps"] = [
        {
            field: step[field]
            for field in (
                "step_id",
                "step_index",
                "planner_variant",
                "representation",
                "typed_action",
                "semantic_ref",
                "intent_id",
                "semantic_signature",
            )
        }
        for step in plan["steps"]
    ]
    return sha256(payload)


def plan_artifact_hash(plan: dict) -> str:
    value = deepcopy(plan)
    value.pop("plan_artifact_hash", None)
    return sha256({"schema": HASH_SCHEMA, "kind": "plan_artifact", "value": value})
