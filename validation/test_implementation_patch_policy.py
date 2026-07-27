from __future__ import annotations
import fnmatch,yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_scientific_and_implementation_paths_are_separated():
    science=yaml.safe_load((ROOT/'docs/operator/scientific_lock_v1.yaml').read_text())['protected_paths']
    impl=yaml.safe_load((ROOT/'docs/operator/implementation_lock_v1.yaml').read_text())['protected_paths']
    assert 'docs/architecture/planner_scientific_contract_v1.yaml' in science
    assert 'docs/architecture/planner_architecture_v1.yaml' in science
    assert 'docs/architecture/planner_architecture_v1.yaml' in impl
    assert 'analysis/**' not in science
    assert 'analysis/decision_gates.py' in science
    assert 'analysis/sample_size.py' in science
    assert 'analysis/**' in impl
    assert 'validation/verify_gate.py' in science
    assert 'validation/verify_lock.py' in science

def test_patch_window_forbids_outcome_data_and_requires_preflight():
    c=yaml.safe_load((ROOT/'docs/operator/implementation_lock_v1.yaml').read_text())
    assert c['pre_lock_patch_window']['outcome_data_access']=='forbidden'
    assert 'toy preflight repeated from clean checkout' in c['pre_lock_patch_window']['required_checks']

def test_patch_validator_itself_is_in_scientific_trust_root():
    science=yaml.safe_load((ROOT/'docs/operator/scientific_lock_v1.yaml').read_text())['protected_paths']
    for path in ('validation/validate_implementation_patch.py','validation/phase_check_runner.py','docs/operator/phase_check_contract_v1.yaml'):
        assert path in science

def test_patch_window_checks_all_pilot_and_confirmatory_outcome_locations():
    text=(ROOT/'validation/validate_implementation_patch.py').read_text()
    for token in ('reports/planner-pilot.json','results/planner-confirmatory','results/stage1a-confirmatory','results/stage1b-confirmatory','evaluator-result-manifest.json'):
        assert token in text
