from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from validation.hashing import hash_json

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/controls/random_codebook_contract_v1.yaml"
SOURCE = "semantic_bank/signatures/manifest.json"


def file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def unsigned_hash(obj: dict[str, Any], field: str) -> str:
    return hash_json({k: v for k, v in obj.items() if k != field})


def code_hex(seed_hex: str, signature_sha256: str) -> str:
    payload = bytes.fromhex(seed_hex) + signature_sha256.encode("utf-8")
    return hashlib.shake_256(payload).hexdigest(48)


def abs_cosine_bits(left: str, right: str) -> float:
    a = int(left, 16)
    b = int(right, 16)
    dimension = len(left) * 4
    hamming = (a ^ b).bit_count()
    return abs((dimension - 2 * hamming) / float(dimension))


def maximum_abs_pairwise_cosine(codes: list[str]) -> float:
    if len(codes) < 2:
        return 0.0
    dimension = len(codes[0]) * 4
    if any(len(code) * 4 != dimension for code in codes):
        raise ValueError("random codebook entries have inconsistent dimensions")
    raw = b"".join(bytes.fromhex(code) for code in codes)
    matrix = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(len(codes), dimension).astype(np.int16)
    matrix = matrix * 2 - 1
    maximum_dot = 0
    block_size = 256
    for start in range(0, len(codes), block_size):
        block = matrix[start:start + block_size]
        dots = np.abs(block @ matrix.T)
        for local in range(len(block)):
            dots[local, : start + local + 1] = 0
        maximum_dot = max(maximum_dot, int(dots.max(initial=0)))
    return maximum_dot / float(dimension)



def resolve_nearest_signature(entries: list[dict[str, Any]], query: list[float]) -> str:
    """Reference A3r inference resolver: maximum cosine, lexicographic hash tie-break."""
    dimension = 384
    if len(query) != dimension:
        raise ValueError("A3r query dimension must be 384")
    norm = math.sqrt(sum(float(x) * float(x) for x in query))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("A3r query must be finite and nonzero")
    normalized = [float(x) / norm for x in query]
    best: tuple[float, str] | None = None
    for row in entries:
        signature = str(row["signature_sha256"])
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(str(row["code_hex"])), dtype=np.uint8)).astype(np.float64)
        vector = (bits * 2.0 - 1.0) / math.sqrt(dimension)
        score = float(np.dot(np.asarray(normalized, dtype=np.float64), vector))
        candidate = (score, signature)
        if best is None or score > best[0] or (math.isclose(score, best[0], rel_tol=0, abs_tol=1e-15) and signature < best[1]):
            best = candidate
    if best is None:
        raise ValueError("A3r codebook must be non-empty")
    return best[1]


def validate_signature_bank(bank: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if bank.get("bank_hash") != unsigned_hash(bank, "bank_hash"):
        errors.append("semantic signature bank self-hash mismatch")
    rows = bank.get("entries", [])
    hashes = []
    for i, row in enumerate(rows):
        expected = hash_json(row.get("semantic_signature"))
        if row.get("signature_sha256") != expected:
            errors.append(f"semantic signature bank entry {i} hash mismatch")
        hashes.append(row.get("signature_sha256"))
    if len(hashes) != len(set(hashes)):
        errors.append("semantic signature bank contains duplicate signatures")
    if hashes != sorted(hashes):
        errors.append("semantic signature bank entries must be sorted by signature_sha256")
    return errors


def validate_random_codebook(root: Path, obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if obj.get("manifest_hash") != unsigned_hash(obj, "manifest_hash"):
        errors.append("random codebook manifest self-hash mismatch")
    if obj.get("contract_sha256") != file_digest(CONTRACT):
        errors.append("random codebook contract hash mismatch")
    if obj.get("algorithm") != contract["code_generation"]["algorithm"]:
        errors.append("random codebook algorithm differs from contract")
    seed = contract["code_generation"]["seed_hex"]
    if obj.get("seed_hex") != seed or obj.get("dimension") != 384:
        errors.append("random codebook seed/dimension differs from contract")
    source_rel = obj.get("source_signature_bank_path")
    if source_rel != SOURCE:
        return errors + ["random codebook source path mismatch"]
    source = (root / source_rel).resolve()
    if root.resolve() not in source.parents or not source.is_file():
        return errors + ["random codebook source signature bank missing or outside repository"]
    if obj.get("source_signature_bank_sha256") != file_digest(source):
        errors.append("random codebook source signature bank hash mismatch")
    try:
        bank = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        return errors + [f"invalid semantic signature bank: {exc}"]
    errors.extend(validate_signature_bank(bank))
    expected_hashes = [row["signature_sha256"] for row in bank.get("entries", [])]
    rows = obj.get("entries", [])
    actual_hashes = [row.get("signature_sha256") for row in rows]
    if actual_hashes != expected_hashes:
        errors.append("random codebook entry set/order differs from signature bank")
    codes: list[str] = []
    for i, row in enumerate(rows):
        expected = code_hex(seed, str(row.get("signature_sha256")))
        if row.get("code_hex") != expected:
            errors.append(f"random codebook entry {i} is not exact deterministic regeneration")
        codes.append(str(row.get("code_hex")))
    if len(codes) != len(set(codes)):
        errors.append("random codebook contains duplicate codes")
    observed = maximum_abs_pairwise_cosine(codes)
    if not math.isclose(float(obj.get("maximum_abs_pairwise_cosine", -1)), observed, rel_tol=0, abs_tol=1e-12):
        errors.append("random codebook pairwise cosine summary mismatch")
    if observed > float(contract["quality_checks"]["maximum_abs_pairwise_cosine"]):
        errors.append("random codebook exceeds locked maximum pairwise cosine")
    return errors
