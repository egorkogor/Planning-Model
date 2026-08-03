"""Classify differences between two quality-evaluation directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

from planner_toy.numeric_identity import (
    CANONICAL_NUMERIC_POLICY,
    canonical_float32_sha256,
    canonical_torch_object_sha256,
    exact_torch_object_sha256,
)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def tensor_leaves(value: object, path: str = "$") -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(path, value)]
    if isinstance(value, dict):
        leaves = []
        for key, child in value.items():
            leaves.extend(tensor_leaves(child, f"{path}[{key!r}]"))
        return leaves
    if isinstance(value, list | tuple):
        leaves = []
        for index, child in enumerate(value):
            leaves.extend(tensor_leaves(child, f"{path}[{index}]"))
        return leaves
    return []


def differences(left: object, right: object, path: str = "$") -> list[dict]:
    if type(left) is not type(right):
        return [{"path": path, "left": left, "right": right, "class": "structural"}]
    if isinstance(left, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                out.append({"path": f"{path}.{key}", "left": left.get(key),
                            "right": right.get(key), "class": "structural"})
            else:
                out.extend(differences(left[key], right[key], f"{path}.{key}"))
        return out
    if isinstance(left, list):
        out = []
        for index in range(max(len(left), len(right))):
            if index >= len(left) or index >= len(right):
                out.append({"path": f"{path}[{index}]", "left": left[index:] or None,
                            "right": right[index:] or None, "class": "structural"})
                break
            out.extend(differences(left[index], right[index], f"{path}[{index}]"))
        return out
    if left != right:
        kind = "exact-numeric" if isinstance(left, int | float) else (
            "derived-hash" if "sha256" in path or "hash" in path else "semantic"
        )
        return [{"path": path, "left": left, "right": right, "class": kind}]
    return []


def compare(left: Path, right: Path) -> dict:
    names = [
        "evaluation-config.json", "dataset-manifest.json", "task-results.jsonl",
        "task-results-semantic.json", "per-seed-summary.json", "aggregate-summary.json",
        "paired-comparisons.json", "evaluation-manifest.json",
        "canonical-semantic-payload.json",
        "compact/data/a2_a3_a4_heldout_summary.json",
    ]
    report: dict[str, object] = {
        "canonical_numeric_policy": CANONICAL_NUMERIC_POLICY,
        "json_differences": {}, "text_differences": {}, "numeric_differences": [],
        "raw_serialization_differences": [],
    }
    for name in names:
        def load(root: Path, filename: str = name) -> object:
            text = (root / filename).read_text()
            return ([json.loads(line) for line in text.splitlines()] if filename.endswith(".jsonl")
                    else json.loads(text))
        found = differences(load(left), load(right))
        if found:
            report["json_differences"][name] = found
    structured_artifacts = sorted({
        path.relative_to(left)
        for pattern in (
            "training-runs/*/seed-*/checkpoint-manifest.json",
            "evidence/*/seed-*/*/semantic-trace.json",
            "evidence/*/seed-*/*/evaluation-result.json",
        )
        for path in left.glob(pattern)
        if (right / path.relative_to(left)).is_file()
    })
    for relative in structured_artifacts:
        found = differences(
            json.loads((left / relative).read_bytes()),
            json.loads((right / relative).read_bytes()),
        )
        if found:
            report["json_differences"][str(relative)] = found
    for name in [
        "human-readable-examples.md", "replay-hash.txt",
        "compact/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md",
    ]:
        a, b = (left / name).read_text(), (right / name).read_text()
        if a != b:
            report["text_differences"][name] = {
                "left_first_difference": next((line for line in a.splitlines()
                                                if line not in b.splitlines()), ""),
                "right_first_difference": next((line for line in b.splitlines()
                                                 if line not in a.splitlines()), ""),
            }
    checkpoint_paths = sorted({
        path.relative_to(left)
        for pattern in (
            "training-runs/*/seed-*/initialization.pt",
            "training-runs/*/seed-*/trained.pt",
            "training-runs/*/seed-*/optimizer-state.pt",
        )
        for path in left.glob(pattern)
    })
    for relative in checkpoint_paths:
        left_bytes, right_bytes = (left / relative).read_bytes(), (right / relative).read_bytes()
        a = torch.load(left / relative, map_location="cpu", weights_only=True)
        b = torch.load(right / relative, map_location="cpu", weights_only=True)
        left_exact, right_exact = exact_torch_object_sha256(a), exact_torch_object_sha256(b)
        left_canonical = canonical_torch_object_sha256(a)
        right_canonical = canonical_torch_object_sha256(b)
        if left_bytes != right_bytes and left_exact == right_exact:
            report["raw_serialization_differences"].append({
                "path": str(relative), "left_raw_sha256": _sha256(left_bytes),
                "right_raw_sha256": _sha256(right_bytes), "class": "raw serialization",
            })
        left_leaves = dict(tensor_leaves(a))
        right_leaves = dict(tensor_leaves(b))
        for name in sorted(set(left_leaves) | set(right_leaves)):
            if name not in left_leaves or name not in right_leaves:
                report["numeric_differences"].append({
                    "path": f"{relative}:{name}",
                    "class": "structural",
                })
                continue
            left_tensor, right_tensor = left_leaves[name], right_leaves[name]
            if left_tensor.shape != right_tensor.shape or left_tensor.dtype != right_tensor.dtype:
                report["numeric_differences"].append({
                    "path": f"{relative}:{name}", "left_shape": list(left_tensor.shape),
                    "right_shape": list(right_tensor.shape),
                    "left_dtype": str(left_tensor.dtype), "right_dtype": str(right_tensor.dtype),
                    "class": "structural",
                })
                continue
            x, y = left_tensor.detach().cpu().double(), right_tensor.detach().cpu().double()
            delta = (x - y).abs()
            if delta.numel() and float(delta.max()) != 0.0:
                denominator = torch.maximum(x.abs(), y.abs())
                relative_delta = torch.where(denominator > 0, delta / denominator, delta)
                index = int(delta.argmax())
                report["numeric_differences"].append({
                    "path": f"{relative}:{name}[{index}]",
                    "left": float(x.reshape(-1)[index]), "right": float(y.reshape(-1)[index]),
                    "max_abs": float(delta.max()), "max_rel": float(relative_delta.max()),
                    "left_exact_hash": left_exact, "right_exact_hash": right_exact,
                    "left_canonical_hash": left_canonical,
                    "right_canonical_hash": right_canonical,
                    "class": "canonical numeric" if left_canonical != right_canonical
                    else "exact numeric",
                })
    for relative in sorted(
        path.relative_to(left) for path in left.glob("evidence/*/seed-*/*/*.f32")
    ):
        a, b = (left / relative).read_bytes(), (right / relative).read_bytes()
        if a != b:
            report["raw_serialization_differences"].append({
                "path": str(relative), "left_raw_sha256": _sha256(a),
                "right_raw_sha256": _sha256(b), "class": "raw serialization",
            })
        if a != b and len(a) == len(b) and len(a) % 4 == 0:
            av = [value for (value,) in struct.iter_unpack("<f", a)]
            bv = [value for (value,) in struct.iter_unpack("<f", b)]
            deltas = [abs(x - y) for x, y in zip(av, bv, strict=True)]
            index = max(range(len(deltas)), key=deltas.__getitem__)
            denom = max(abs(av[index]), abs(bv[index]))
            report["numeric_differences"].append({
                "path": f"{relative}[{index}]", "left": av[index], "right": bv[index],
                "max_abs": deltas[index],
                "max_rel": deltas[index] / denom if denom else deltas[index],
                "finite": math.isfinite(av[index]) and math.isfinite(bv[index]),
                "left_exact_hash": _sha256(a), "right_exact_hash": _sha256(b),
                "left_canonical_hash": canonical_float32_sha256(a),
                "right_canonical_hash": canonical_float32_sha256(b),
                "class": "canonical numeric" if canonical_float32_sha256(a)
                != canonical_float32_sha256(b) else "exact numeric",
            })
    numeric = report["numeric_differences"]
    report["observed_max_abs"] = max(
        (row.get("max_abs", 0.0) for row in numeric), default=0.0
    )
    report["observed_max_rel"] = max(
        (row.get("max_rel", 0.0) for row in numeric), default=0.0
    )
    report["first_numeric_difference"] = numeric[0] if numeric else None
    report["first_json_difference"] = next(
        (
            rows[0]
            for rows in report["json_differences"].values()
            if rows
        ),
        None,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(compare(args.left, args.right), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
