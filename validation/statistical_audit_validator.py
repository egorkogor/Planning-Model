from __future__ import annotations

from validation.role_validator import canonical_identity_hash

REQUIRED={
 'HIERARCHICAL_BOOTSTRAP','CLUSTERED_STAGE1A_BOOTSTRAP','PAIRED_STAGE1B_BOOTSTRAP',
 'PAIRED_TOST','SAMPLE_SIZE_POWER','TIE_BREAK_AND_MISSING_PAIRS','DECISION_GATE_RECOMPUTATION'
}


def validate_statistical_audit(obj: dict, resource_plan_or_builder: dict)->list[str]:
    errors=[]
    if 'builder' in resource_plan_or_builder:
        builder=resource_plan_or_builder['builder']
        expected_reviewer=resource_plan_or_builder['statistical_reviewer']
    else:  # Compatibility for isolated unit tests.
        builder=resource_plan_or_builder; expected_reviewer=None
    reviewer=obj['reviewer']
    ids=[x['check_id'] for x in obj['checks']]
    if set(ids)!=REQUIRED or len(ids)!=len(REQUIRED): errors.append('statistical audit check set incomplete or duplicated')
    if any(x['status']!='PASS' for x in obj['checks']) and obj['decision']=='APPROVE': errors.append('cannot APPROVE with failed statistical checks')
    independent=reviewer['reviewer_type']=='HUMAN_STATISTICIAN' or reviewer['model_family']!=builder['model_family']
    if not independent: errors.append('statistical reviewer is not judgment-independent from Builder')
    expected='HUMAN_STATISTICIAN' if reviewer['reviewer_type']=='HUMAN_STATISTICIAN' else 'DIFFERENT_MODEL_FAMILY'
    if obj['reviewer_independence']!=expected: errors.append('reviewer_independence declaration mismatch')
    if obj.get('builder_identity_sha256') not in (None,canonical_identity_hash(builder)):
        errors.append('builder_identity_sha256 mismatch')
    if expected_reviewer is not None and reviewer!=expected_reviewer:
        errors.append('statistical audit reviewer differs from locked resource plan identity')
    if obj.get('public_key_id') not in (None,reviewer.get('public_signing_key_id')):
        errors.append('statistical audit public_key_id differs from reviewer identity')
    return errors
