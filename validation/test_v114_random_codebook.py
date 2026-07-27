from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from validation.hashing import hash_json
from validation.random_codebook_validator import (
    CONTRACT,
    abs_cosine_bits,
    code_hex,
    file_digest,
    validate_random_codebook,
    validate_signature_bank,
)

SEED = "905098775febd6fe3b64b9ab8ddb4262f569dcb156915e7c805cef54e14314ca"


def _signature(intent: int, relation: str, remaining: str) -> dict:
    return {
        "intent_id": intent,
        "hand_mode": "HAND_EMPTY",
        "goal_relation": relation,
        "moving_clear": "YES",
        "support_clear": "YES",
        "obstruction_depth_bucket": "ZERO",
        "remaining_distance_bucket": remaining,
    }


def _tree(tmp_path: Path):
    signatures = [
        _signature(0, "ON_TABLE", "ONE_TWO"),
        _signature(2, "ON", "THREE_FIVE"),
        _signature(4, "SATISFIED", "ZERO"),
    ]
    entries = sorted(
        ({"signature_sha256": hash_json(sig), "semantic_signature": sig} for sig in signatures),
        key=lambda row: row["signature_sha256"],
    )
    bank = {
        "schema_version": "work-planner-signature-bank/1.0",
        "run_id": "run-codebook",
        "source_split": "train",
        "intent_labeler_contract_sha256": "sha256:" + "1" * 64,
        "entries": entries,
        "bank_hash": "sha256:" + "0" * 64,
    }
    bank["bank_hash"] = hash_json({k: v for k, v in bank.items() if k != "bank_hash"})
    bank_path = tmp_path / "semantic_bank/signatures/manifest.json"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    codes = [{"signature_sha256": row["signature_sha256"], "code_hex": code_hex(SEED, row["signature_sha256"])} for row in entries]
    observed = max(abs_cosine_bits(a["code_hex"], b["code_hex"]) for i, a in enumerate(codes) for b in codes[i + 1:])
    manifest = {
        "schema_version": "work-planner-random-codebook/1.0",
        "run_id": "run-codebook",
        "contract_sha256": file_digest(CONTRACT),
        "source_signature_bank_path": "semantic_bank/signatures/manifest.json",
        "source_signature_bank_sha256": file_digest(bank_path),
        "algorithm": "SHAKE256_BITS_V1",
        "seed_hex": SEED,
        "dimension": 384,
        "entries": codes,
        "maximum_abs_pairwise_cosine": observed,
        "created_at": "2026-07-27T12:00:00Z",
        "manifest_hash": "sha256:" + "0" * 64,
    }
    manifest["manifest_hash"] = hash_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return bank, manifest


def test_random_codebook_exactly_regenerates_from_train_signature_bank(tmp_path: Path):
    bank, manifest = _tree(tmp_path)
    assert validate_signature_bank(bank) == []
    assert validate_random_codebook(tmp_path, manifest) == []


def test_random_codebook_rejects_manual_code_selection(tmp_path: Path):
    bank, manifest = _tree(tmp_path)
    bad = deepcopy(manifest)
    bad["entries"][0]["code_hex"] = "00" * 48
    bad["manifest_hash"] = hash_json({k: v for k, v in bad.items() if k != "manifest_hash"})
    assert any("exact deterministic regeneration" in e for e in validate_random_codebook(tmp_path, bad))


def test_random_codebook_rejects_post_bank_signature_injection(tmp_path: Path):
    bank, manifest = _tree(tmp_path)
    bad = deepcopy(manifest)
    extra_hash = "sha256:" + "f" * 64
    bad["entries"].append({"signature_sha256": extra_hash, "code_hex": code_hex(SEED, extra_hash)})
    bad["manifest_hash"] = hash_json({k: v for k, v in bad.items() if k != "manifest_hash"})
    assert any("entry set/order" in e for e in validate_random_codebook(tmp_path, bad))


def test_a3r_is_a_full_parameter_matched_training_variant():
    import yaml
    training = yaml.safe_load(Path("docs/training/planner_training_contract_v1.yaml").read_text())
    architecture = yaml.safe_load(Path("docs/architecture/planner_scientific_contract_v1.yaml").read_text())
    assert training["random_code_control"]["variant"] == "A3r"
    assert "A3r" in training["hyperparameter_selection_policy"]["arms_ranked"]
    assert architecture["arms"]["A3r"]["representation"] == "RANDOM_CODE_CONTINUOUS_LATENT"
