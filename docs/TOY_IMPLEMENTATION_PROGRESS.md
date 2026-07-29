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
- Seed conflict resolution: the normative FINAL_CONFIRMATORY schema remains untouched.
  Seed 17 is accepted only by the separate, explicitly non-confirmatory
  `DEVELOPMENT/TOY` lineage validator and artifact schema identifiers.
- This is implementation evidence only. No sealed data was accessed, no confirmatory
  experiment was run, and P06/P07 are not claimed complete.
- Remaining before A3: broaden training beyond the deliberately minimal toy task,
  produce protocol-governed A2 pilot/freeze evidence, then implement and audit the
  locked latent head/feedback path without changing scientific contracts.
