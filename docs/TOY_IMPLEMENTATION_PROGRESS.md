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
- Lineage has an explicit development boundary under `planner_toy/schemas`: emitted
  versions are `toy-planner-request/1.0`, `toy-development-config/1.0`,
  `toy-checkpoint-manifest/1.0`, `toy-work-plan/1.0`,
  `toy-episode-plan-manifest/1.0`, `toy-attempt-log/1.0`, `toy-episode-log/1.0`, and
  `toy-evaluation-result/1.0`, plus `toy-optimizer-evidence/1.0`. It semantically replays every transition and binds real
  persisted checkpoint/config files without claiming v1.21 evidence validity.
- Protected runtime, dependency, and schema contracts are unchanged. Seed 17 remains a
  non-normative development profile constrained in code to `split=development`,
  `stage=PLANNER_ONLY`, `arm=PLANNER_A2_RAW`, and development-only plan paths. Toy
  artifacts validate only against their separate schemas and are not v1.21 evidence.
- Decoder cross-attention masks all PAD keys. CI discovers toy tests recursively, installs
  the official CPU wheel, and prints failed per-file logs. Runtime validation requires
  base 2.12.0, `torch.version.cuda is None`, unavailable CUDA, and working CPU autograd.
- Generation/parsing failures emit a FAILED manifest, empty per-step AttemptLog,
  zero-execution EpisodeLog, EvaluationResult, and typed failure code rather than raising
  before evidence is recorded.
- Executor/precondition failures retain the READY frozen WorkPlan, emit a FAILED AttemptLog
  row at the exact plan position, preserve the actual final state, and never replan.
- Development config is persisted before training; training and optimizer reports bind its
  hash and full provenance. Reuse validates and copies the original chain rather than
  wrapping model bytes in newly asserted provenance.
- This is implementation evidence only. No sealed data was accessed, no confirmatory
  experiment was run, and P06/P07 are not claimed complete.
- Remaining before A3: broaden training beyond the deliberately minimal toy task,
  produce protocol-governed A2 pilot/freeze evidence, then implement and audit the
  locked latent head/feedback path without changing scientific contracts.
