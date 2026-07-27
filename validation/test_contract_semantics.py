from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(rel):
    return yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))


def test_fixed_dataset_and_training_defaults_exist():
    splits = load("docs/data/dataset_split_contract_v1.yaml")
    assert splits["partitions"]["train"]["target_base_tasks"] == 24000
    assert splits["partitions"]["stage1b_confirmatory_reserve"]["target_base_tasks"] == 4000
    training = load("docs/training/planner_training_contract_v1.yaml")
    assert training["optimizer"]["name"] == "AdamW"
    assert training["final_seeds"] == [101, 202, 303]
    grid = load("docs/training/hyperparameter_search_v1.yaml")
    assert grid["max_configs"] == 4
    assert grid["selection"]["no_candidate_meets_floor"] == "BLOCKED_MODEL_DEVELOPMENT"


def test_concept_packer_has_semantic_feedback_for_a2c_and_a3():
    arch = load("docs/architecture/planner_architecture_v1.yaml")
    feedback = arch["concept_packer"]["semantic_feedback"]
    assert feedback["A2c"]["concatenate_then_projection"] == "448_to_256"
    assert "384_to_512" in feedback["A3"]["projection"]
    assert arch["inference_feedback"]["A3"] == "normalized_predicted_z"
    assert arch["same_information_constraint"]["extra_facts_in_A3_text"] == "forbidden"


def test_prompt_is_unpadded_canonical_and_left_padded_only_for_batches():
    prompt = load("docs/prompt/stage1_prompt_v1.yaml")
    aliases = list(prompt["surface_aliases"]["mapping"].values())
    assert len(aliases) == len(set(aliases)) == 8
    assert prompt["guidance_blocks"]["attended_token_budget"] == 32
    assert prompt["chat_rendering"]["right_padding"] == "forbidden"
    assert prompt["chat_rendering"]["generation"]["tokenizer_padding_side_for_batches"] == "left"
    assert prompt["chat_rendering"]["stage1b"]["independent_trajectory_total_lengths_may_differ"] is True
    assert prompt["parser"]["arity_by_action"]["END"] == 0


def test_prompt_development_is_pre_registered():
    contract = load("docs/prompt/prompt_development_contract_v1.yaml")
    assert [c["candidate_id"] for c in contract["candidates"]] == ["C01", "C02", "C03", "C04"]
    assert contract["floors"]["format_success"] == 0.95
    assert contract["freeze"]["later_prompt_change_requires_new_protocol_version"] is True


def test_confirmatory_requires_separate_evaluator_and_auditor():
    seal = load("docs/controls/confirmatory_sealing_contract_v1.yaml")
    assert seal["roles"]["builder"]["may_read_confirmatory_task_bodies"] is False
    assert seal["roles"]["evaluator"]["may_write_source_repository"] is False
    assert seal["fallback"]["single_agent_or_unsealed_execution"] == "INVALID_CONFIRMATORY_BLINDNESS"
    audit = load("docs/audit/independent_audit_contract_v1.yaml")
    assert audit["acceptance"]["clean_reproduction_required"] is True


def test_two_level_lock_is_mandatory_and_not_overrideable():
    science = load("docs/operator/scientific_lock_v1.yaml")
    impl = load("docs/operator/implementation_lock_v1.yaml")
    assert science["change_policy"]["in_run_change"] == "forbidden"
    assert science["change_policy"]["operator_override"] == "forbidden"
    assert impl["pre_lock_patch_window"]["outcome_data_access"] == "forbidden"
    assert impl["post_lock_change_policy"]["operator_override"] == "forbidden"

def test_infrastructure_budget_and_recovery_are_fixed():
    infra = load("docs/infrastructure/provisioning_contract_v1.yaml")
    assert infra["execution_roles"]["evaluator"]["separate_credentials_from_builder"] is True
    assert infra["budget"]["maximum_total"] == 100
    recovery = load("docs/infrastructure/recovery_contract_v1.yaml")
    assert recovery["failures"]["transient_retry_count"] == 2
    assert recovery["resume"]["require_contract_lock_match"] is True


def test_state_space_golden_counts_follow_closed_form():
    import math
    domain = load("docs/domain/blocks_world_v1.yaml")
    counts = domain["state_space_golden_counts"]
    for n in range(1, 9):
        hand_empty = sum(math.factorial(n) * math.comb(n - 1, k - 1) // math.factorial(k) for k in range(1, n + 1))
        holding_base = 1 if n == 1 else sum(math.factorial(n - 1) * math.comb(n - 2, k - 1) // math.factorial(k) for k in range(1, n))
        holding = n * holding_base
        assert counts[str(n)] == {"hand_empty": hand_empty, "holding": holding, "total": hand_empty + holding}


def test_dataset_quotas_are_feasible_by_design_and_sum_exactly():
    splits = load("docs/data/dataset_split_contract_v1.yaml")
    for name, part in splits["partitions"].items():
        quotas = part.get("quota_by_block_count")
        if quotas:
            assert sum(quotas.values()) == part["target_base_tasks"], name
    horizon = splits["partitions"]["planner_pilot_horizon"]
    assert 3 not in horizon["allowed_block_counts"]
    assert splits["feasibility_audit"]["quota_must_not_exceed_available_count"] is True
    assert splits["feasibility_audit"]["protected_contracts_already_locked"] is True


def test_off_policy_expansion_is_bounded_and_does_not_claim_global_exactness():
    corpus = load("docs/domain/training_corpus_contract_v1.yaml")
    off = corpus["off_policy_expansion"]
    assert off["exact_local_expansion_for_block_count_lte"] == 5
    assert off["exact_local_rule"]["breadth_first_depth"] == 2
    assert off["maximum_examples_per_base_task"] == 4096
    assert "exact_enumeration_for_block_count_lte" not in off


def test_runtime_split_names_match_common_schema_and_reserves_are_not_logged():
    import json
    splits = load("docs/data/dataset_split_contract_v1.yaml")
    common = json.loads((ROOT / "docs/schemas/common.schema.json").read_text(encoding="utf-8"))
    enum = set(common["$defs"]["split"]["enum"])
    assert set(splits["partitions"]).issubset(enum)
    assert set(splits["derived_partitions"]).issubset(enum)
    runtime = splits["runtime_split_names"]
    assert runtime["reserve_partitions_may_not_appear_in_attempt_or_episode_logs"] is True
    for value in runtime.values():
        if isinstance(value, str):
            assert value in enum
        elif isinstance(value, list):
            assert set(value).issubset(enum)


def test_task_encoder_position_budget_matches_encoding_contract():
    arch = load("docs/architecture/planner_architecture_v1.yaml")
    enc = load("docs/architecture/task_encoding_v1.yaml")
    assert arch["task_encoder"]["max_input_positions"] == enc["max_length"] == 192


def test_a2c_covers_all_seven_signature_fields_and_a4_zero_is_post_projection():
    architecture=load("docs/architecture/planner_architecture_v1.yaml")
    a2c=architecture["concept_packer"]["semantic_feedback"]["A2c"]
    assert a2c["field_count"]==7
    assert a2c["concatenate_then_projection"]=="448_to_256"
    assert architecture["concept_packer"]["semantic_feedback"]["A4"]=="compute_exact_A3_projection_then_replace_projected_256_with_zero_before_sum_and_layer_norm"
    assert architecture["parameter_matching"]["total_parameter_count_tolerance_fraction"]==0.0
    assert architecture["parameter_matching"]["strategy"]=="common_superset_parameter_inventory"

def test_size_ood_is_not_used_for_training_expansion():
    corpus=load("docs/domain/training_corpus_contract_v1.yaml")
    assert corpus["off_policy_expansion"].get("block_count_7_8") in (None,"forbidden_in_training")

def test_confirmatory_selection_seed_is_hidden_from_builder():
    import json
    splits=load('docs/data/dataset_split_contract_v1.yaml')
    hidden=splits['assignment']['confirmatory_hidden_selection']
    assert hidden['owner']=='DATA_SEALER'
    assert hidden['builder_access']=='seed commitment, counts and encrypted dataset hash only'
    seal=load('docs/controls/confirmatory_sealing_contract_v1.yaml')
    assert seal['roles']['data_sealer']['secret_seed_generation']=='CSPRNG_256_BIT_INSIDE_SEALER_ENVIRONMENT'
    assert seal['roles']['data_sealer']['seed_plaintext_export']=='forbidden'
    schema=json.loads((ROOT/'docs/schemas/sealer_manifest.schema.json').read_text())
    assert 'generator_seed' not in schema['properties']
    assert 'generator_seed_commitment' in schema['required']
    assert schema['properties']['seed_reveal_policy']['const']=='AUDITOR_AFTER_SIGNED_EVALUATOR_RESULT'

def test_a5_mapping_requires_deterministic_perfect_derangement():
    contract=load('docs/controls/planner_latent_ablation_contract_v1.yaml')
    mapping=contract['A5_shuffled']['mapping_algorithm']
    assert 'maximum bipartite matching' in mapping['matching']
    assert mapping['required_result']=='perfect derangement within every stratum'
    assert mapping['reuse_foreign_unit']=='forbidden'
    assert contract['A5_shuffled']['candidate_missing_policy'].startswith('BLOCKED_CONTROL_CONSTRUCTION')
