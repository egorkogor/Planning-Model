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

На каждом per-seed checkpoint сохраняются raw teacher-forced и free-running observations плюс
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

### Aggregate-safe task semantics

Cross-seed checkpoint aggregate строится из persisted raw teacher records всех seeds. Повторяющийся
`task_id` не может представлять один последний seed и не может перезаписывать предыдущие seeds.

`per_task` для каждого task является истинным pooled cross-seed summary и включает counts и rates,
в том числе `seed_count`, `task_record_count`, target counts и pooled operator/non-END/END/arg1/arg2/
joint accuracies.

Aggregate `position0_by_task` для каждого task содержит только aggregate-safe поля:

- фиксированный `gold_operator`;
- `seed_count` и `target_count`;
- `operator_accuracy`;
- `mean_gold_operator_probability`;
- `mean_operator_nll`;
- `mean_end_probability`.

В aggregate representation запрещены единичные `predicted_operator` и `operator_correct`, потому
что они ложно представляли бы один seed как весь cross-seed aggregate.

Aggregate `position0_task_discrimination` явно имеет семантику
`contrast_of_cross_seed_means`: сначала для каждого task вычисляется cross-seed mean END
probability, затем из этих means вычисляется task01-vs-nontrivial contrast и range.

Это отдельное понятие от `mean_within_seed_position0_task_discrimination`: последнее сначала
вычисляет contrast внутри каждого seed, а потом усредняет contrasts. Оба сохраняются отдельно и
не подменяют друг друга даже в balanced design, где численно они могут совпасть.

## Independent validation

`a2_optimization_budget_trajectory_validator.py` не запускает producer/training повторно и не
импортирует producer aggregate helpers. Он проверяет persisted lower-level evidence и независимо
пересчитывает:

- per-seed checkpoint summaries;
- per-seed position-0 task-discrimination claims;
- first-rescue epoch;
- aggregate-safe `per_task` напрямую из raw teacher records всех seeds;
- aggregate-safe `position0_by_task` напрямую из raw teacher records всех seeds;
- cross-seed `position0_task_discrimination` как contrast истинных cross-seed task means;
- `mean_within_seed_position0_task_discrimination` отдельно из raw per-seed teacher evidence;
- остальные cross-seed checkpoint aggregates;
- free-running goal claims из persisted plans и task semantics;
- exact prefix-equivalence binding;
- update schedule / target counts / canonical loss weighting / clipping semantics.

Validation дополнительно bind-ится к workflow-requested exact implementation SHA и source
inventory, наследующему accepted learnability source closure.

Tamper/regression coverage использует намеренно разные значения для одного `task_id` между seeds и
обязана отвергать:

- last-seed overwrite вместо pooled `per_task`;
- last-seed overwrite вместо aggregate-safe `position0_by_task`;
- неверный cross-seed task-discrimination;
- изменение raw probability/NLL/END evidence одного seed при сохранённом старом aggregate claim;
- checkpoint metric, trajectory/first-rescue claim, prefix-equivalence и raw position evidence.

Таким образом aggregate claim остаётся связан с каждым persisted per-seed raw record, а не только
с самосогласованной producer projection.

## Execution contract

Workflow запускается только через `workflow_dispatch` на runner
`planning-model-canonical-cpu-v1`. Push/PR execution запрещён. Artifact скрытого workspace
персистится с `include-hidden-files: true`.

Implementation/fix PR не запускает scientific workflow. После reviewer acceptance + merge
допускается ровно один fresh results run на новом authoritative main. Для evidence-accounting fix
это evidence repair/republication того же frozen trajectory contract, а не новый hypothesis search;
старый workflow run не rerun-ится.

## Interpretation boundary

Статус causal language: `SUPPORTED HYPOTHESIS / NOT PROVEN`.

- Rescue при дополнительных canonical updates поддерживает insufficient optimization budget /
  under-training как major cause.
- Persistent position-0 failure до epoch 100, особенно при слабом task-discrimination spread,
  ослабляет budget-only explanation и мотивирует read-only conditioning/representation diagnostic.
- Evidence-accounting correction не меняет model/training science и не ищет новый numerical result.
- Ни один исход не является доказательством A3/latent/semantic-target claims.
- `GO_LATENT = NOT EVALUATED`.
