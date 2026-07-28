# Нормативный hash protocol v1.19

Все runtime hashes имеют вид `sha256:<64 lowercase hex>`. `SHA256SUMS.txt` использует стандартный формат `<hex><two spaces><relative POSIX path>`.

## 1. Canonical JSON

- Unicode NFC;
- UTF-8 без BOM;
- sorted object keys;
- separators `(',', ':')`;
- NaN/Infinity запрещены;
- array order сохраняется, кроме явно заданной canonical sorting;
- self-hash field удаляется, а не заменяется `null`.

Reference implementation: `validation/hashing.py`.

## 2. State, goal and task

`state_hash` — hash `{"domain":"blocks_world_v1","facts":CANONICAL_FACTS}`. Facts сортируются по predicate rank and numeric ref rank.

`goal_hash` — hash `{"domain":"blocks_world_v1","goal":CANONICAL_GOAL}`.

`canonical_task_hash` — минимальные canonical bytes среди всех ref bijections n≤8 для initial+goal. Surface aliases, IDs and timestamps excluded.

## 3. Tensor/vector hashes

`tensor_sha256` — exact C-contiguous little-endian float32 bytes.

`decoder_state_hash` hashes header:

```text
dtype=float32;shape=<dims>;byteorder=little\n
```

plus exact bytes.

Semantic URI:

```text
latent+sha256://<raw tensor hex>#S00
```

URI digest обязан совпасть с `tensor_sha256`.

Token/vector hashes use the same header rule with pinned dtype (`int32` for IDs/positions, `uint8` for masks).

## 4. Prompt identity

- `rendered_prompt_hash`: exact UTF-8 bytes after pinned chat template;
- `input_ids_hash`: int32 vector hash;
- `attention_mask_hash`: uint8 vector hash;
- `position_ids_hash`: int32 vector hash;
- `guidance_token_sequence_hash`: attended 32-token subvector.

Whitespace после template не нормализуется.

## 5. WorkPlan

`plan_content_hash` includes task/canonical/state/checkpoint/config/variant/representation and ordered semantic PlannerStep content; excludes IDs, timestamps, file paths and diagnostics.

`plan_artifact_hash` hashes full WorkPlan without `plan_artifact_hash`.

Exactly one END and continuity are validated before hashing.

## 6. Semantic manifest

`manifest_content_hash` includes task/state/checkpoint/config and sorted records `{step_id,semantic_ref,tensor_sha256,decoder_state_hash}`.

`manifest_hash` hashes full manifest without `manifest_hash`. `WorkPlan.semantic_artifact_manifest_sha256` must equal it.

## 7. Dataset and corpus

`training_example_hash` excludes storage path and generation timestamp. `corpus_content_hash` hashes ordered row hashes plus domain/generator/oracle/contract hashes. `corpus_artifact_hash` hashes full manifest without self-hash.

## 8. Pair group

Payload contains stage, split, base task, optional snapshot, trajectory policy and experiment freeze hash. Arm excluded. Stage 1A pair completeness is checked against this hash.

## 9. Controls

`intent_compatibility_hash` includes state, goal, sorted shortest first action hashes, sorted compatible intents, oracle and labeler hashes.

`control_mapping_hash` includes state, goal, selected intent, catalog/control hashes and compatibility hash.

`control_certification_hash` is stage-specific: Stage 1A binds frozen snapshot-control mappings; Stage 1B binds only task/domain/split eligibility and pre-outcome artifact manifests. Stage 1B never hashes Planner/LLM outputs into an inclusion decision.

## 10. Freeze and decisions

`freeze_hash` hashes full `ExperimentFreeze` without `freeze_hash` and approval fields. Approval record separately hashes gate ID, freeze hash, decision and operator timestamp.

`decision_record_hash` hashes decision inputs, estimates, CIs and thresholds without narrative or self-hash.

## 11. Folder hash

- regular files only, symlinks forbidden;
- relative POSIX paths sorted bytewise;
- exclude external self-manifest;
- per file line `path\0size\0raw_hex\n`;
- hash concatenated UTF-8 manifest bytes.

## 12. Verification rule

Hash is computed only after schema and cross-object validation. Writers recalculate after read-back. Mismatch is a fatal contract violation; no runtime row may be silently rehashed after outcome.

## Signed role manifests (v1.19)

For Data Sealer and Evaluation Runner manifests, canonical signed bytes are UTF-8 canonical JSON with sorted keys and compact separators after removing `signature` and `manifest_hash`. `manifest_hash` is SHA-256 of those bytes. `signature` is Ed25519 over exactly the same bytes. The public key must be present in `locks/public-keys.json` and bound to the expected role and role-identity hash.

## 13. Analysis code identity

`analysis_code_sha256` is the folder-protocol hash over all regular `analysis/*.py` files plus `validation/statistics_validator.py`, sorted by relative POSIX path. A scientific decision is invalid when this digest or the exact `docs/statistics/statistics_contract_v1.yaml` digest differs from the locked runtime.

`analysis_input_hash`, `sample_size_input.input_hash`, `sample_size_report.report_hash` and `scientific_decision.decision_hash` use canonical JSON wrapped as `{"schema":"work-planner-hash/1.0","kind":<kind>,"value":<object_without_self_hash>}`.
