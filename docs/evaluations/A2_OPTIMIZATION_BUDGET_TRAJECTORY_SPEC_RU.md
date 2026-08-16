# A2 optimization-budget trajectory — development-only

## Научный вопрос

Достаточен ли исторический canonical budget `3 epochs / 9 updates` для базового A2
memorization и task discrimination в позиции 0, или наблюдаемый END-collapse совместим с
сильным under-training?

Это train-only diagnostic/microexperiment. Он не меняет frozen historical trainer, model
architecture, objective, task order, optimizer hyperparameters, dataset, decoding, A3/A4 или
formal acceptance gates.

## Frozen scope

- variant: `A2`;
- train tasks: `bw-00000001`, `bw-00000002`, `bw-00000003`;
- held-out `bw-00000004/05`: запрещены;
- seeds: `17`, `29`, `43`;
- order в каждой эпохе: `01 -> 02 -> 03`;
- losses: canonical per-task mean operator / arg1 / arg2 cross-entropy;
- optimizer: AdamW, lr `3e-4`, betas `(0.9, 0.95)`, eps `1e-8`, weight decay `0.01`;
- gradient clipping: `1.0`;
- runtime: canonical dedicated fixed CPU target.

## Trajectory

Для каждого seed создаётся ровно одна непрерывная training trajectory до 100 эпох. Никаких
отдельно reinitialized arms для budget checkpoints нет.

Read-only checkpoints:

- epoch 3 = 9 updates;
- epoch 10 = 30 updates;
- epoch 30 = 90 updates;
- epoch 100 = 300 updates.

Дополнительно после каждой эпохи сохраняется read-only position-0 evidence, чтобы определить
первую точную эпоху, на которой хотя бы один gold-UNSTACK в позиции 0 начинает декодироваться
правильно.

## Exact historical prefix contract

Epoch-3 prefix обязан точно воспроизводить реальный frozen canonical control для каждого seed.
Сравнение без tolerances включает:

- initialization canonical state hash;
- trained canonical state hash после epoch 3;
- optimizer canonical hash после epoch 3;
- первые 9 updates;
- task schedule;
- operator / arg1 / arg2 / total losses;
- target counts;
- pre-clip gradient norm;
- clip norm и clipping flag;
- canonical operator position weight.

Любое несовпадение — fail closed. Это non-scientific equivalence probe, а не отдельный arm.

## Persisted evidence

На каждом checkpoint сохраняются raw teacher-forced и free-running observations плюс
пересчитываемые summaries:

- operator / non-END / END accuracy;
- arg1 / arg2 / joint accuracy;
- position-0 gold-UNSTACK accuracy;
- position-0 gold-operator probability и NLL;
- position-0 END probability отдельно для tasks 01/02/03;
- within-seed task-discrimination spread;
- position-4 END metrics;
- free-running exact-plan / goal / initially-unsatisfied goal / zero-action rates.

Training evidence содержит все update losses, gradient norms и clipping для 300 updates per seed.

## Independent validation

`a2_optimization_budget_trajectory_validator.py` не запускает producer/training повторно. Он
проверяет persisted lower-level evidence и независимо пересчитывает:

- checkpoint summaries;
- position-0 task-discrimination claims;
- first-rescue epoch;
- cross-seed checkpoint aggregates;
- free-running goal claims из persisted plans и task semantics;
- exact prefix-equivalence binding;
- update schedule / target counts / canonical loss weighting / clipping semantics.

Validation дополнительно bind-ится к workflow-requested exact implementation SHA и source
inventory, наследующему accepted learnability source closure.

Tamper tests обязаны падать при изменении checkpoint metric, trajectory/first-rescue claim,
prefix-equivalence и raw position evidence даже после re-sign canonical identity.

## Execution contract

Workflow запускается только через `workflow_dispatch` на runner
`planning-model-canonical-cpu-v1`. Push/PR execution запрещён. Artifact скрытого workspace
персистится с `include-hidden-files: true`.

Implementation PR не запускает scientific workflow. После reviewer merge допускается ровно один
fresh results run на новом authoritative main.

## Interpretation boundary

Статус causal language: `SUPPORTED HYPOTHESIS / NOT PROVEN`.

- Rescue при дополнительных canonical updates поддерживает insufficient optimization budget /
  under-training как major cause.
- Persistent position-0 failure до epoch 100, особенно при слабом task-discrimination spread,
  ослабляет budget-only explanation и мотивирует read-only conditioning/representation diagnostic.
- Ни один исход не является доказательством A3/latent/semantic-target claims.
- `GO_LATENT = NOT EVALUATED`.
