from __future__ import annotations
import json, sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.sample_size import calculate_components
from analysis.decision_gates import evaluate_gate, percentile_bootstrap_ci

def y(rel): return yaml.safe_load((ROOT/rel).read_text())

def test_g01_is_mandatory_externally_signed_trust_gate():
    p=next(x for x in y('docs/operator/phase_registry_v1.yaml')['phases'] if x['phase_id']=='P01')
    assert p['approval_mode']=='manual'
    assert p['manual_gate_id']=='G01_TRUST_AND_RESOURCES'
    assert p['approval_target_output']=='locks/trust-topology.lock.json'
    assert 'locks/trust-topology.lock.json' in p['required_outputs']
    assert any(x['check_id']=='P01_pre_05' for x in p['pre_gate_checks'])

def test_data_sealer_is_separate_role():
    a=y('docs/operator/agent_execution_contract_v1.yaml')
    assert 'data_sealer' in a['execution_roles']
    s=y('docs/controls/confirmatory_sealing_contract_v1.yaml')
    assert s['roles']['data_sealer']['must_destroy_plaintext_before_manifest_release'] is True
    for pid in ('P07','P12','P16'):
        p=next(x for x in y('docs/operator/phase_registry_v1.yaml')['phases'] if x['phase_id']==pid)
        assert any('sealer-manifest.json' in z for z in p['required_outputs'])

def test_final_training_fixed_updates_and_ablation_same_checkpoint():
    t=y('docs/training/planner_training_contract_v1.yaml')
    assert t['training']['final_confirmatory_models']['optimizer_updates_exact']==12000
    assert t['training']['final_confirmatory_models']['early_stopping']=='forbidden'
    assert t['ablation_training_policy']['weights_sha256_must_equal_A3'] is True

def test_resolver_thresholds_are_pre_registered():
    r=y('docs/semantic/semantic_resolver_v1.yaml')
    assert r['calibration']['partition']=='development'
    assert r['resolution']['change_after_lock']=='forbidden'
    assert r['calibration']['no_pair_meets_constraints']=='BLOCKED_RESOLVER_CALIBRATION'

def test_exact_prompt_candidate_files_exist():
    c=y('docs/prompt/prompt_development_contract_v1.yaml')
    for row in c['candidates']:
        obj=y(row['source_file'])
        assert obj['candidate_id']==row['candidate_id']
        assert obj['system_message_exact']
        assert obj['text_mutation_during_run'] if 'text_mutation_during_run' in obj else True

def test_guidance_artifact_is_exact_32_tokens_contract():
    g=y('docs/prompt/guidance_artifact_contract_v1.yaml')
    assert g['target_attended_tokens_exact']==32
    assert g['freeze_before_prompt_development'] is True
    assert g['suffix_search']['if_no_exact_length']=='BLOCKED_GUIDANCE_TOKENIZATION'

def test_statistics_reference_functions_and_gates():
    c=calculate_components(0.4)
    assert c['selected_n']==max(v for k,v in c.items() if k!='selected_n')
    assert evaluate_gate('estimate_gte_and_ci_low_gt',0.06,0.01,0.11,0.05)
    assert evaluate_gate('estimate_gte',0.65,0.60,0.70,0.65)
    assert not evaluate_gate('estimate_gte',0.64,0.59,0.69,0.65)
    est,lo,hi=percentile_bootstrap_ci([1,0,1,-1,0,1,0,1],resamples=1000,seed=7301)
    assert lo<=est<=hi

def test_report_registry_covers_new_role_artifacts():
    patterns={r['path_pattern'] for r in y('docs/operator/report_registry_v1.yaml')['rules']}
    for p in ('decisions/D-????.json','reports/resource-usage.jsonl','dispatch/*.json','sealed/**/sealer-manifest.json','release/BOOTSTRAP_MANIFEST.json'):
        assert p in patterns

def test_audit_profiles_handle_early_stops():
    a=y('docs/audit/independent_audit_contract_v1.yaml')
    assert set(a['profiles'])=={'PLANNER_STOP','PLANNER_DIAGNOSTIC_ONLY','INTERFACE_STOP','END_TO_END'}
