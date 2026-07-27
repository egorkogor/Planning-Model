from validation.role_validator import validate_role_independence
from validation.statistical_audit_validator import validate_statistical_audit, REQUIRED

def role(name,principal,family='family-a',kind=None):
 if kind is None:
  kind = 'SERVICE_PROCESS' if name in {'DATA_SEALER','EVALUATION_RUNNER'} else 'MODEL'
 return {'role':name,'reviewer_type':kind,'agent_provider':'p','model_family':family,'model_revision':'r','system_prompt_hash':'sha256:'+'1'*64,'environment_identity':principal+'-env','credential_principal':principal,'public_signing_key_id':principal+'-key'}

def test_same_model_family_auditor_is_rejected():
 plan={'builder':role('BUILDER','b'),'data_sealer':role('DATA_SEALER','s'),'evaluator':role('EVALUATION_RUNNER','e'),'auditor':role('AUDITOR','a'),'statistical_reviewer':role('STATISTICAL_REVIEWER','r')}
 errors=validate_role_independence(plan)
 assert any('auditor' in x for x in errors)
 assert any('statistical_reviewer' in x for x in errors)

def test_different_family_or_human_is_accepted():
 plan={'builder':role('BUILDER','b'),'data_sealer':role('DATA_SEALER','s'),'evaluator':role('EVALUATION_RUNNER','e'),'auditor':role('AUDITOR','a','family-b'),'statistical_reviewer':role('STATISTICAL_REVIEWER','r','human','HUMAN_STATISTICIAN')}
 assert not validate_role_independence(plan)

def test_stat_audit_requires_all_checks_and_independence():
 builder=role('BUILDER','b')
 reviewer=role('STATISTICAL_REVIEWER','r','family-b')
 obj={'reviewer':reviewer,'reviewer_independence':'DIFFERENT_MODEL_FAMILY','checks':[{'check_id':x,'status':'PASS','evidence_sha256':'sha256:'+'2'*64} for x in sorted(REQUIRED)],'decision':'APPROVE'}
 assert not validate_statistical_audit(obj,builder)
 obj['reviewer']['model_family']='family-a'
 assert validate_statistical_audit(obj,builder)
