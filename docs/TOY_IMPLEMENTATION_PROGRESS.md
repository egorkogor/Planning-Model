# Toy A2 implementation progress (non-normative)

## Domain and toy dataset

- Implemented: canonical BlocksWorld facts/goals, contract actions, preconditions, deterministic application, goal verification, BFS oracle, canonical hashing, and seed-17 train/validation generation for `n=1..3`.
- Passing tests: `python -m pytest -q tests/toy/test_domain_dataset.py` (3 passed).
- Last commit before this block: `1cbe957`.
- Implemented: exact 177-tensor locked A2 inventory, contract initialization seed 17,
  task encoder, A2 ConceptPacker, causal decoder, typed/pointer heads, active/dormant
  policy, real CPU forward/backward/AdamW training, clipping, checkpoints and evidence.
- Implemented: single-call frozen WorkPlan pipeline, fail-closed parser and executor,
  development EpisodePlanManifest, AttemptLog, EvaluationResult, lineage validation,
  mutation tests, deterministic replay and `python -m scripts.run_toy_a2_e2e`.
- Review remediation: task inputs now match `task-encoding/1.6` byte-for-byte,
  including 192 positions, segment/argument-position IDs, padding mask, and pointer
  states selected only from ledger `REF_SLOT_i` positions. The E2E task executes four
  predicted state-changing actions before END; no oracle plan enters Planner decoding.
- Review remediation: removed zero-valued synthetic loss edges. Pointer gradients now
  arise from real UNSTACK/STACK targets, AdamW uses locked betas `(0.9, 0.95)`, and
  replay comparison recursively covers both checkpoints and all nested evidence.
- Lineage artifacts use the normative WorkPlan and EpisodePlanManifest schemas and
  recompute content/artifact hashes across PlannerRequest, task, checkpoint, config,
  WorkPlan, EpisodePlanManifest, AttemptLog, and EvaluationResult.
- Seed conflict resolution: the EpisodePlanManifest schema gained a backwards-compatible
  optional `run_class`; seed 17 is valid only for explicit `DEVELOPMENT_TOY`, while an
  absent or `FINAL_CONFIRMATORY` class retains the five final seeds and prior behavior.
- This is implementation evidence only. No sealed data was accessed, no confirmatory
  experiment was run, and P06/P07 are not claimed complete.
- Remaining before A3: broaden training beyond the deliberately minimal toy task,
  produce protocol-governed A2 pilot/freeze evidence, then implement and audit the
  locked latent head/feedback path without changing scientific contracts.
