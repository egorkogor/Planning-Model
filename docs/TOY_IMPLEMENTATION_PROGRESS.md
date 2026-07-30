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
- Earlier A2 backlog was to broaden training and add the latent path; the bounded
  development implementation status of that latent path is recorded below without
  claiming protocol-governed confirmatory evidence.

## A3 Latent Thought Feedback — DEVELOPMENT MVP

- Реализован явный development-only выбор `A2`, `A3` и `A4` поверх общего
  177-тензорного parameter inventory. A2 сохраняет нулевой semantic component и
  dormant latent-параметры; A3 вычисляет `Linear(256→384)` с L2-нормализацией и
  передаёт предыдущий predicted latent через
  `Linear(384→512) → GELU → Linear(512→256) → LayerNorm`; A4 вычисляет тот же
  путь, но обнуляет projection непосредственно перед ConceptPacker.
- Training A3/A4 использует **ненормативный toy adapter**: детерминированный
  seed-17 вектор размерности 384 строится только из типизированной сигнатуры
  текущего BlocksWorld action (`action`, `arg1`, `arg2`), нормализуется и
  обучается cosine-distance loss. Adapter не получает будущие шаги, конечное
  состояние или полный план через latent channel и не является предлагаемым
  общим способом supervision человеческих мыслей.
- На inference A3 использует только собственный predicted latent предыдущей
  позиции. Позиция 0 нулевая; один Planner call формирует frozen многошаговый
  WorkPlan, после чего Executor не перепланирует. Отдельные binary latent и JSON
  semantic-trace artifacts связываются с config, checkpoint, request и WorkPlan и
  проверяются fail-closed runtime validator-ом.
- Технические sensitivity tests проверяют изменение hidden/logits A3 при controlled
  latent substitution, инвариантность A2/A4 и отсутствие влияния будущего latent
  на прошлую позицию. Это доказывает наличие вычислительного канала, но **не**
  улучшение reasoning или качества решения.
- Не реализованы Verbalizer, естественный язык, отдельная Answer trajectory,
  closed-loop agent, A5 и A3r. BlocksWorld не доказывает перенос на общий
  reasoning; MVP не является confirmatory experiment и не изменяет scientific
  lock или нормативные v1.21 contracts.
- Backlog вне scope: внешне зафиксированный общий semantic-target artifact,
  полноценные A5/A3r controls, статистические сравнения между seeds и задачами,
  FLOPs measurement и protocol-governed confirmatory design.

### CPU wheel installation smoke path

Изолированная поддерживаемая установка ML extra использует официальный CPU index:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  'planner-llm-mvp-stage1[ml] @ file:///absolute/path/to/wheel.whl'
```

Smoke import/inference выполняется вне repository root без `PYTHONPATH`.

## Development Quality Evaluation v0.1

Реализован runner и завершена development-only held-out диагностика существующих
`A2-structured-baseline`, `A3a-codebook` и `A3a-zero` с seeds 17, 29 и 43.
Фактические результаты и конкретные планы находятся в
[`evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md`](evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md).
Это диагностический результат текущего toy BlocksWorld split, не confirmatory evidence,
не доказательство semantic reasoning и **не прохождение Stage 2A semantic gate**.
`A3a-shuffled`, `A3a-foreign` и семейство `A3s` остаются future work; `A3b` остаётся gated.
