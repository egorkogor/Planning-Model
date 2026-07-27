# JSON Schemas — work-planner/1.14

Основное разделение:

- `typed_action.schema.json` — действие Oracle/LLM Parser/Validator/Executor;
- `planner_step.schema.json` — Planner prediction и semantic metadata;
- `work_plan.schema.json` — до 16 non-END actions плюс final END;
- `task_spec.schema.json`, `training_example.schema.json`, `corpus_manifest.schema.json` — task/corpus boundary;
- `semantic_signature.schema.json`, `semantic_artifact*.schema.json` — discrete/continuous semantics;
- `attempt_log.schema.json`, `episode_log.schema.json`, `prompt_artifact.schema.json` — runtime evidence;
- `model_lock.schema.json`, `control_certification.schema.json`, `experiment_freeze.schema.json` — pre-outcome locks;
- `phase_report.schema.json`, `decision_record.schema.json` — autonomous agent/operator ledger.

JSON Schema проверяет local shape. Domain semantics, hashes, pair completeness, episode aggregation и lineage проверяет `validation/cross_object_validator.py` и phase-specific validators.
