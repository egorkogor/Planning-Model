import hashlib
import math
from pathlib import Path

import pytest

from research_programs.planner.semantic_feedback_readiness import (
    DonorUnit,
    ReadinessError,
    calibrate_collision_paths,
    code_hex,
    encode_state,
    generate_code,
    load_config,
    preintervention_opportunity,
    readiness_replay,
    reconstruct_codebooks,
    reconstruct_seed,
    select_wrong_semantic_donor,
    validate_checkpoint_binding,
    validate_collision_fixture,
    validate_composition_split,
)


def test_readiness_config_is_bound_to_accepted_authority_text():
    config = load_config()
    protocol = Path(config["protocol"]["path"]).read_text(encoding="utf-8")
    generator = Path(config["generator_contract"]["path"]).read_text(encoding="utf-8")
    for identity in config["codebooks"]:
        assert identity["id"] in protocol
        assert identity["derivation_input"] in protocol
        assert identity["seed_hex"] in protocol
    for signature in config["composition_split"]["train"] + config["composition_split"]["stress"]:
        assert signature in protocol
    assert "algorithm: SHAKE256_BITS_V1" in generator
    assert "payload: seed_bytes || UTF8(signature_sha256)" in generator
    assert "output_bytes: 48" in generator
    assert "dimension: 384" in generator


def test_frozen_codebook_seeds_reconstruct_exactly_and_are_distinct():
    config = load_config()
    seeds = [reconstruct_seed(identity) for identity in config["codebooks"]]
    assert [seed.hex() for seed in seeds] == [
        "c83c21942a85cd899b0ddb764eb007f205d79f9c498bd3ec9dd94cf859d75c47",
        "088d3f30429645884c69975744ca9414c8bbaaed4f1ac7dfea2832979396af3e",
    ]
    assert seeds[0] != seeds[1]


def test_codebook_generation_is_deterministic_and_exact_sha_ke_bits():
    config = load_config()
    signatures = [hashlib.sha256(value.encode()).hexdigest() for value in ("zeta", "alpha")]
    first = reconstruct_codebooks(config, signatures)
    second = reconstruct_codebooks(config, reversed(signatures))
    assert first == second
    assert sorted(first) == ["A3R-CODEBOOK-170029", "A3R-CODEBOOK-290043"]
    assert all(set(book) == set(signatures) for book in first.values())
    assert first["A3R-CODEBOOK-170029"][signatures[0]] != first["A3R-CODEBOOK-290043"][signatures[0]]
    seed = reconstruct_seed(config["codebooks"][0])
    raw = bytes.fromhex(code_hex(seed, signatures[0]))
    vector = generate_code(seed, signatures[0])
    expected_first_bit = 1.0 if raw[0] & 0x80 else -1.0
    assert math.isclose(vector[0], expected_first_bit / math.sqrt(384), rel_tol=0, abs_tol=1e-8)
    assert vector[0] != expected_first_bit / math.sqrt(384) or isinstance(vector[0], float)


def test_seed_tamper_fails_closed():
    config = load_config()
    identity = dict(config["codebooks"][0])
    identity["seed_hex"] = "00" * 32
    with pytest.raises(ReadinessError, match="codebook seed mismatch"):
        reconstruct_seed(identity)


def test_composition_split_reconstructs_frozen_holdout():
    result = validate_composition_split(load_config())
    assert result["id"] == "SEM-COMP-PAIR-HOLDOUT-V1"
    assert result["train_count"] == 6
    assert result["stress_count"] == 2
    assert result["atoms_seen"] == ["FAR", "NEAR", "RIGHT_OF", "UNDER"]


def test_composition_overlap_fails_closed():
    config = load_config()
    config["composition_split"]["stress"] = [config["composition_split"]["train"][0]]
    with pytest.raises(ReadinessError, match="INVALID_GEOMETRY_ESTIMAND_SIGNATURE_OVERLAP"):
        validate_composition_split(config)


def _unit(unit_id, episode_id, semantic_signature, norm, **overrides):
    values = dict(
        unit_id=unit_id,
        episode_id=episode_id,
        semantic_signature=semantic_signature,
        planner_seed=17,
        split_id="dev",
        intervention_position=3,
        remaining_distance_bucket="r2",
        hand_mode="left",
        feedback_norm_bucket="n1",
        feedback_norm_raw=norm,
    )
    values.update(overrides)
    return DonorUnit(**values)


def test_wrong_semantic_donor_exact_filter_and_tie_break():
    target = _unit("target", "e9", "sig-a", 1.0)
    candidates = [
        _unit("u2", "e2", "sig-b", 1.2),
        _unit("u1", "e1", "sig-c", 0.8),
        _unit("wrong-seed", "e0", "sig-d", 1.01, planner_seed=29),
        _unit("same-semantic", "e0", "sig-a", 1.0),
    ]
    selected = select_wrong_semantic_donor(target, candidates)
    assert selected is not None
    assert (selected.episode_id, selected.unit_id) == ("e1", "u1")


def test_wrong_semantic_donor_unavailable_is_explicit_null():
    target = _unit("target", "e9", "sig-a", 1.0)
    assert select_wrong_semantic_donor(target, [_unit("x", "e1", "sig-a", 1.0)]) is None
    assert encode_state(donor_available=False) == "NOT_EVALUATED_DONOR_UNAVAILABLE"


def test_collision_and_inverse_fixture_calibration_paths():
    config = load_config()
    result = validate_collision_fixture(config)
    assert result == {
        "fixture_id": "SEM-METADATA-COLLISION-V1",
        "metadata_only_collision": "INDETERMINATE_MUST_NOT_SEPARATE_TARGETS",
        "semantic_oracle_collision": "MUST_SEPARATE_TARGETS",
        "inverse_metadata_target_invariant": True,
    }
    assert calibrate_collision_paths(config) == {
        "metadata_only_cannot_distinguish_collision": True,
        "semantic_oracle_separates_collision": True,
        "inverse_metadata_target_invariant": True,
    }


def test_collision_metadata_tamper_fails_closed():
    config = load_config()
    config["collision_fixture"]["collision_pair"][1]["position"] = 4
    with pytest.raises(ReadinessError, match="INVALID_SHORTCUT_NOT_EXCLUDED"):
        validate_collision_fixture(config)


def test_checkpoint_guards_require_exact_a3_and_no_retraining():
    exact = {arm: "ckpt-a3" for arm in ("A3", "A4", "A5", "WRONG_SEMANTIC_DONOR")}
    assert validate_checkpoint_binding("ckpt-a3", exact) == "EVALUATED"
    drift = dict(exact)
    drift["A5"] = "other"
    assert validate_checkpoint_binding("ckpt-a3", drift) == "INVALID_CHECKPOINT_OR_RETRAINING"
    assert validate_checkpoint_binding("ckpt-a3", exact, ["A4"]) == "INVALID_CHECKPOINT_OR_RETRAINING"


def test_preintervention_exposure_is_fixed_from_reference_prefix_only():
    result = preintervention_opportunity([
        {"position": 1, "feedback_eligible": False},
        {"position": 3, "feedback_eligible": True},
        {"position": 5, "feedback_eligible": True},
    ])
    assert result == {
        "source": "PRE_INTERVENTION_REFERENCE_PREFIX_ONLY",
        "opportunity_count": 2,
        "first_feedback_position": 3,
        "post_treatment_survivor_conditioning": False,
    }
    empty = preintervention_opportunity([{"position": 1, "feedback_eligible": False}])
    assert empty["opportunity_count"] == 0
    assert empty["first_feedback_position"] is None


def test_validity_state_precedence_and_all_required_branches():
    assert encode_state(checkpoint_valid=False, shortcut_valid=False) == "INVALID_CHECKPOINT_OR_RETRAINING"
    observed = {
        encode_state(),
        encode_state(checkpoint_valid=False),
        encode_state(shortcut_valid=False),
        encode_state(oracle_valid=False),
        encode_state(metadata_invariant=False),
        encode_state(split_valid=False),
        encode_state(both_codebooks_valid=False),
        encode_state(downstream_opportunities=0),
        encode_state(intervention_applicable=False),
        encode_state(donor_available=False),
        encode_state(terminated_before_intervention=True),
        encode_state(endpoint_defined=False),
    }
    assert observed == set(load_config()["null_precedence"])


def test_readiness_replay_is_deterministic_non_outcome_bearing_and_complete():
    config = load_config()
    first = readiness_replay(config)
    second = readiness_replay(config)
    assert first == second
    assert len(first["config_digest"]) == 64
    assert len(first["implementation_digest"]) == 64
    assert len(first["replay_digest"]) == 64
    assert first["scientific_execution"] is False
    assert first["held_out_access"] is False
    assert first["claim_bearing_evidence"] is False
    assert first["go_latent"] == "NOT EVALUATED"
    assert first["preintervention_exposure"]["first_feedback_position"] == 3
    assert first["endpoint_placeholders"] == {
        "validity_state": "EVALUATED",
        "immediate_next_step_effect": None,
        "free_running_total_effect": None,
        "outcome_bearing": False,
    }
    assert set(first["validity_branches"].values()) == set(config["null_precedence"])
