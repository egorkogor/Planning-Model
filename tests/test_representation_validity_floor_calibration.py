from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = ROOT / ".research/calibration/representation_validity_floor_calibration_v1.yaml"
RANDOM_CODEBOOK_CONTRACT_PATH = ROOT / "docs/controls/random_codebook_contract_v1.yaml"
DISCOVERY_STUDY_PATH = ROOT / ".research/studies/semantic_feedback_discovery_v1.yaml"

DIMENSION = 384
OUTPUT_BYTES = 48


def _calibration() -> dict[str, Any]:
    return yaml.safe_load(CALIBRATION_PATH.read_text(encoding="utf-8"))


def _random_codebook_contract() -> dict[str, Any]:
    return yaml.safe_load(RANDOM_CODEBOOK_CONTRACT_PATH.read_text(encoding="utf-8"))


def _discovery_study() -> dict[str, Any]:
    return yaml.safe_load(DISCOVERY_STUDY_PATH.read_text(encoding="utf-8"))


def _seed_lookup(calibration: dict[str, Any]) -> dict[str, str]:
    seed_identities = calibration["generator_binding"]["seed_identities"]
    return {row["id"]: row["seed_hex"] for row in seed_identities}


def _arm_lookup(calibration: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in calibration["fixture_construction"]["arms"]}


# ---------------------------------------------------------------------------
# Structural / static contract checks
# ---------------------------------------------------------------------------


def test_calibration_is_non_outcome_bearing_and_scoped_correctly() -> None:
    calibration = _calibration()
    assert calibration["status"] == "ADVISORY_DISCOVERY_MEASUREMENT_NON_OUTCOME_BEARING"
    assert calibration["evidence_class"] == "DISCOVERY_MEASUREMENT_NON_OUTCOME_BEARING"
    assert calibration["authority"]["issue"] == 65
    assert calibration["authority"]["authoritative_base"] == (
        "4ed5b5c5843d14ede5ef93a459d3a292e018ec1a"
    )
    non_authority = " ".join(calibration["authority"]["explicit_non_authority"])
    assert "does not modify GO_LATENT" in non_authority
    assert "does not touch issue #61 / PR #62" in non_authority
    assert "calibration/negative-check reference" in non_authority
    boundary = calibration["promotion_boundary"]
    assert boundary["model_training"] == "FORBIDDEN"
    assert boundary["scientific_execution"] == "FORBIDDEN"
    assert boundary["held_out_access"] == "FORBIDDEN"
    assert boundary["go_latent"] == "NOT EVALUATED"
    out_of_scope = " ".join(calibration["scope_boundary"]["out_of_scope"])
    assert "real A3/A3r checkpoint" in out_of_scope
    assert ".research/planner_portfolio.yaml" in out_of_scope
    assert "#61/#62-owned" in out_of_scope


def test_generator_binding_inherits_frozen_random_codebook_fields() -> None:
    calibration = _calibration()
    contract = _random_codebook_contract()
    binding = calibration["generator_binding"]
    assert binding["contract"] == "docs/controls/random_codebook_contract_v1.yaml"
    assert contract["code_generation"]["algorithm"] == "SHAKE256_BITS_V1"
    assert contract["code_generation"]["output_bytes"] == OUTPUT_BYTES
    assert contract["code_generation"]["dimension"] == DIMENSION
    assert contract["code_generation"]["rejection_or_manual_selection"] == "forbidden"
    inherited = " ".join(binding["inherited_fields_unchanged"])
    assert "SHAKE256_BITS_V1" in inherited
    assert "output_bytes = 48" in inherited
    assert "dimension = 384" in inherited
    assert binding["arbitrary_one_hot_or_simple_deterministic_encoding"] == "FORBIDDEN"
    assert binding["analyst_override"] == "FORBIDDEN"


def test_seed_identities_reproduce_from_their_derivation_input() -> None:
    calibration = _calibration()
    for row in calibration["generator_binding"]["seed_identities"]:
        expected = hashlib.sha256(row["derivation_input"].encode("utf-8")).hexdigest()
        assert row["seed_hex"] == expected, row["id"]
        assert len(bytes.fromhex(row["seed_hex"])) == 32


def test_taxonomy_binds_mechanically_to_frozen_vocabulary() -> None:
    calibration = _calibration()
    study = _discovery_study()
    vocabulary = study["novel_compositional_signature_stress"]["vocabulary"]
    taxonomy = calibration["taxonomy"]

    assert taxonomy["operator_axis"]["values"] == vocabulary["relation_atoms"]
    assert taxonomy["operator_axis"]["cardinality_K"] == len(vocabulary["relation_atoms"]) == 4
    assert taxonomy["topic_axis"]["values"] == vocabulary["distance_atoms"]
    assert taxonomy["topic_axis"]["cardinality_K"] == len(vocabulary["distance_atoms"]) == 2

    expected_joint = [
        f"{relation}|{distance}"
        for relation in vocabulary["relation_atoms"]
        for distance in vocabulary["distance_atoms"]
    ]
    assert taxonomy["joint_axis"]["values"] == expected_joint
    assert taxonomy["joint_axis"]["cardinality_K"] == len(expected_joint) == 8
    assert "FORBIDDEN" in taxonomy["K_selection_rule"]
    assert "elbow" in taxonomy["K_selection_rule"].lower()


def test_fit_evaluation_split_is_predeclared_and_disjoint_by_construction() -> None:
    calibration = _calibration()
    split = calibration["fit_evaluation_split"]
    assert split["rule"] == "within_class_index parity: even -> FIT, odd -> EVAL"
    assert split["same_examples_for_fit_and_measurement"] == "FORBIDDEN"
    assert split["post_outcome_adaptation"] == "FORBIDDEN"


def test_estimators_require_bias_corrected_mi_alongside_naive() -> None:
    calibration = _calibration()
    estimators = calibration["estimators"]
    assert estimators["mutual_information"]["both_required"] is True
    assert estimators["mutual_information"]["naive_alone_interpretation"] == "FORBIDDEN"
    assert estimators["mutual_information"]["bias_corrected"]["method"] == "Miller-Madow"


def test_permutation_null_is_predeclared_before_outcome() -> None:
    calibration = _calibration()
    null = calibration["label_permutation_null"]
    assert null["permutation_count"] == 1000
    assert null["quantile_decision_rule"] == 0.95
    assert null["predeclared_before_outcome"] is True
    assert "EVAL-split true labels only" in null["unit_permuted"]


def test_full_signature_arm_is_non_gating_with_interpretation_boundary() -> None:
    calibration = _calibration()
    arms = _arm_lookup(calibration)
    full_signature = arms["ARM-4-FULL-SIGNATURE"]
    assert full_signature["gating_axis"] == "none"
    assert full_signature["non_gating"] is True
    boundary = full_signature["interpretation_boundary"]
    assert "MUST NOT be interpreted as evidence" in boundary
    assert "A3a" in boundary

    non_gating_reference = calibration["fail_closed_calibration_criteria"]["non_gating_reference"]
    assert non_gating_reference["arm"] == "ARM-4-FULL-SIGNATURE"
    assert "never gates PASS/STOP_REDESIGN" in non_gating_reference["role"]

    real_z = calibration["real_z_interpretation_boundary"]
    claim_boundary = real_z["emergent_semantic_structure_claim"]
    assert claim_boundary == "FORBIDDEN regardless of any calibration outcome in this file"
    assert real_z["application_to_real_trained_z"] == "NOT_AUTHORIZED_BY_THIS_CONTRACT"


# ---------------------------------------------------------------------------
# Mechanical calibration: reproduce the frozen generator family and run the
# predeclared purity/MI/permutation-null pipeline on synthetic fixtures.
# ---------------------------------------------------------------------------


def shake_code_vector(seed_hex: str, payload_key: str) -> np.ndarray:
    """Reproduce the frozen SHAKE256_BITS_V1 code-generation semantics for one payload key."""
    seed_bytes = bytes.fromhex(seed_hex)
    payload = seed_bytes + payload_key.encode("utf-8")
    digest = hashlib.shake_256(payload).digest(OUTPUT_BYTES)
    bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8)).astype(np.float64)
    return (bits * 2.0 - 1.0) / math.sqrt(DIMENSION)


def build_fixture(calibration: dict[str, Any], arm_id: str) -> dict[str, np.ndarray]:
    seeds = _seed_lookup(calibration)
    if arm_id not in _arm_lookup(calibration):
        raise ValueError(f"unknown arm {arm_id}")
    taxonomy = calibration["taxonomy"]
    joint_values = taxonomy["joint_axis"]["values"]
    n_per_class = calibration["fixture_construction"]["points_per_joint_class"]
    alpha = calibration["fixture_construction"]["class_signal_weight_alpha"]
    beta = calibration["fixture_construction"]["noise_weight_beta"]

    vectors = []
    operator_labels = []
    topic_labels = []
    joint_labels = []
    within_class_index = []

    for class_index, joint_label in enumerate(joint_values):
        relation_atom, distance_atom = joint_label.split("|")
        for local_index in range(n_per_class):
            global_index = class_index * n_per_class + local_index
            noise = shake_code_vector(seeds["NOISE-SEED"], f"NOISE|{global_index}")
            if arm_id == "ARM-1-RANDOM-NULL":
                vector = shake_code_vector(seeds["NULL-ARM-SEED"], f"NULL|{global_index}")
            else:
                if arm_id == "ARM-2-OPERATOR-ONLY":
                    key = f"OPERATOR|{relation_atom}"
                    seed_hex = seeds["OPERATOR-ARM-SEED"]
                elif arm_id == "ARM-3-TOPIC-ONLY":
                    key = f"TOPIC|{distance_atom}"
                    seed_hex = seeds["TOPIC-ARM-SEED"]
                elif arm_id == "ARM-4-FULL-SIGNATURE":
                    key = f"SIGNATURE|{relation_atom}|{distance_atom}"
                    seed_hex = seeds["FULL-SIGNATURE-ARM-SEED"]
                else:
                    raise ValueError(f"unknown arm {arm_id}")
                base = shake_code_vector(seed_hex, key)
                raw = alpha * base + beta * noise
                vector = raw / np.linalg.norm(raw)
            vectors.append(vector)
            operator_labels.append(relation_atom)
            topic_labels.append(distance_atom)
            joint_labels.append(joint_label)
            within_class_index.append(local_index)

    return {
        "vectors": np.stack(vectors),
        "operator_labels": np.array(operator_labels),
        "topic_labels": np.array(topic_labels),
        "joint_labels": np.array(joint_labels),
        "within_class_index": np.array(within_class_index),
    }


def fit_mask_from(within_class_index: np.ndarray) -> np.ndarray:
    return within_class_index % 2 == 0


def kmeans(vectors: np.ndarray, k: int, seed: int, max_iterations: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = vectors.shape[0]
    init_idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[init_idx].copy()
    assignments = np.full(n, -1, dtype=int)
    for iteration in range(max_iterations):
        distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_assignments = distances.argmin(axis=1)
        if iteration > 0 and np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for cluster in range(k):
            members = vectors[assignments == cluster]
            if len(members) > 0:
                centroids[cluster] = members.mean(axis=0)
            else:
                centroids[cluster] = vectors[rng.integers(0, n)]
    return centroids


def assign_nearest(vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    n = len(cluster_labels)
    total = 0
    for cluster in np.unique(cluster_labels):
        members = true_labels[cluster_labels == cluster]
        if len(members) == 0:
            continue
        _, counts = np.unique(members, return_counts=True)
        total += counts.max()
    return total / n


def mutual_information_bits(
    cluster_labels: np.ndarray, true_labels: np.ndarray
) -> tuple[float, float]:
    n = len(cluster_labels)
    clusters = np.unique(cluster_labels)
    labels = np.unique(true_labels)
    mi = 0.0
    for cluster in clusters:
        cluster_mask = cluster_labels == cluster
        p_c = cluster_mask.sum() / n
        for label in labels:
            label_mask = true_labels == label
            n_cl = int((cluster_mask & label_mask).sum())
            if n_cl == 0:
                continue
            p_cl = n_cl / n
            p_l = label_mask.sum() / n
            mi += p_cl * math.log2(p_cl / (p_c * p_l))
    correction = (len(clusters) - 1) * (len(labels) - 1) / (2 * n * math.log(2))
    return mi, mi - correction


def evaluate_axis(
    vectors: np.ndarray,
    axis_labels: np.ndarray,
    within_class_index: np.ndarray,
    k: int,
    cluster_seed: int,
    permutation_seed: int,
    permutation_count: int,
    quantile: float,
) -> dict[str, Any]:
    fit_mask = fit_mask_from(within_class_index)
    eval_mask = ~fit_mask
    assert fit_mask.any() and eval_mask.any()
    assert not np.any(fit_mask & eval_mask)

    centroids = kmeans(vectors[fit_mask], k=k, seed=cluster_seed, max_iterations=100)
    eval_assignments = assign_nearest(vectors[eval_mask], centroids)
    eval_true_labels = axis_labels[eval_mask]
    _, encoded_true = np.unique(eval_true_labels, return_inverse=True)

    observed_purity = purity(eval_assignments, encoded_true)
    _, observed_mi_corrected = mutual_information_bits(eval_assignments, encoded_true)

    rng = np.random.default_rng(permutation_seed)
    null_purities = np.empty(permutation_count)
    null_mis = np.empty(permutation_count)
    for i in range(permutation_count):
        permuted = rng.permutation(encoded_true)
        null_purities[i] = purity(eval_assignments, permuted)
        _, null_mis[i] = mutual_information_bits(eval_assignments, permuted)

    purity_threshold = float(np.quantile(null_purities, quantile))
    mi_threshold = float(np.quantile(null_mis, quantile))
    separates = bool(observed_purity > purity_threshold and observed_mi_corrected > mi_threshold)

    return {
        "observed_purity": float(observed_purity),
        "observed_mi_corrected": float(observed_mi_corrected),
        "purity_threshold": purity_threshold,
        "mi_threshold": mi_threshold,
        "separates_from_null": separates,
        "eval_n": int(eval_mask.sum()),
    }


AXES = {
    "operator_axis": ("operator_labels", 4),
    "topic_axis": ("topic_labels", 2),
    "joint_axis": ("joint_labels", 8),
}


def run_axis(calibration: dict[str, Any], arm_id: str, axis: str) -> dict[str, Any]:
    fixture = build_fixture(calibration, arm_id)
    label_key, k = AXES[axis]
    null_cfg = calibration["label_permutation_null"]
    return evaluate_axis(
        vectors=fixture["vectors"],
        axis_labels=fixture[label_key],
        within_class_index=fixture["within_class_index"],
        k=k,
        cluster_seed=calibration["clustering_procedure"]["init_seed"],
        permutation_seed=null_cfg["seed"],
        permutation_count=null_cfg["permutation_count"],
        quantile=null_cfg["quantile_decision_rule"],
    )


def test_generator_reproduces_frozen_bit_mapping_and_model_target() -> None:
    calibration = _calibration()
    seeds = _seed_lookup(calibration)
    vector = shake_code_vector(seeds["NULL-ARM-SEED"], "NULL|0")
    assert vector.shape == (DIMENSION,)
    assert set(np.round(vector * math.sqrt(DIMENSION), 6)).issubset({1.0, -1.0})
    assert math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=0, abs_tol=1e-9)


def test_fixture_construction_is_exactly_reproducible() -> None:
    calibration = _calibration()
    first = build_fixture(calibration, "ARM-2-OPERATOR-ONLY")
    second = build_fixture(calibration, "ARM-2-OPERATOR-ONLY")
    assert np.array_equal(first["vectors"], second["vectors"])
    total_n = calibration["fixture_construction"]["total_N_per_arm"]
    assert total_n == first["vectors"].shape[0] == 400


def test_gate_1_operator_positive_control_separates_from_null() -> None:
    calibration = _calibration()
    result = run_axis(calibration, "ARM-2-OPERATOR-ONLY", "operator_axis")
    assert result["separates_from_null"] is True, result


def test_gate_2_topic_positive_control_separates_from_null() -> None:
    calibration = _calibration()
    result = run_axis(calibration, "ARM-3-TOPIC-ONLY", "topic_axis")
    assert result["separates_from_null"] is True, result


def test_gate_3_null_control_stays_at_or_below_floor_on_every_axis() -> None:
    calibration = _calibration()
    for axis in ("operator_axis", "topic_axis", "joint_axis"):
        result = run_axis(calibration, "ARM-1-RANDOM-NULL", axis)
        assert result["separates_from_null"] is False, (axis, result)


def test_cross_check_no_topic_leakage_in_operator_only_arm() -> None:
    calibration = _calibration()
    result = run_axis(calibration, "ARM-2-OPERATOR-ONLY", "topic_axis")
    assert result["separates_from_null"] is False, result


def test_cross_check_no_operator_leakage_in_topic_only_arm() -> None:
    calibration = _calibration()
    result = run_axis(calibration, "ARM-3-TOPIC-ONLY", "operator_axis")
    assert result["separates_from_null"] is False, result


def test_full_signature_reference_runs_but_is_never_asserted_as_a_claim() -> None:
    calibration = _calibration()
    for axis in ("operator_axis", "topic_axis", "joint_axis"):
        result = run_axis(calibration, "ARM-4-FULL-SIGNATURE", axis)
        assert isinstance(result["separates_from_null"], bool)
        assert math.isfinite(result["observed_purity"])
        assert math.isfinite(result["observed_mi_corrected"])
