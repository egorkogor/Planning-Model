# A2 sufficient-budget task-order causal discrimination

## Статус

Development-only, train-only causal microexperiment. Он не изменяет frozen historical trainer,
архитектуру модели, objective, dataset, optimizer, fixed-target contract, formal gates, A3/A4 или
held-out policy.

Интерпретация всегда ограничена формулировкой:

`SUPPORTED HYPOTHESIS / NOT PROVEN`

`GO_LATENT = NOT EVALUATED`.

## Научный вопрос

Принятый A2 optimization-budget trajectory показал, что 9 updates / 3 epochs недостаточны, а
неизменённая canonical training trajectory восстанавливает A2 при большем budget. Предыдущая
task-order интервенция была измерена только в under-trained 9-update режиме.

Этот раунд проверяет только одно: влияет ли положение тривиальной END-only задачи
`bw-00000001` внутри каждой эпохи на timing и устойчивость rescue при достаточном budget.

## Arms

Ровно три arm, независимо инициализируемые одной и той же deterministic per-seed policy:

1. `canonical_order`: `01 -> 02 -> 03`
2. `task01_middle`: `02 -> 01 -> 03`
3. `task01_last`: `02 -> 03 -> 01`

Между arms неизменны:

- задачи и exposure counts;
- 100 epochs / 300 optimizer updates per seed-arm;
- A2 architecture;
- canonical per-task mean operator/arg1/arg2 cross-entropy;
- AdamW `lr=3e-4`, `betas=(0.9,0.95)`, `eps=1e-8`, `weight_decay=0.01`;
- pre-clip gradient norm evidence и clipping at `1.0`;
- seeds `17/29/43`;
- fixed-target runtime;
- train-only dataset `01/02/03`.

Held-out `04/05` не материализуется и не используется.

## Checkpoints и trajectory evidence

Full checkpoints сохраняются на epochs `3/10/30/100`. На каждом full checkpoint сохраняются
raw teacher-forced и true free-running records, включая:

- task-specific position-0 gold operator probability, END probability, NLL и correctness;
- position-4 terminal END correctness/probability;
- teacher-forced operator/non-END/END/arg1/arg2/joint accuracy;
- free-running exact-plan, goal success, initially-unsatisfied success, zero-action rate и plan;
- model/optimizer canonical hashes.

Для каждого из 100 epochs дополнительно сохраняется lightweight claim-bearing evidence:

- position-0 raw record для всех `01/02/03`;
- true free-running plan/result для всех `01/02/03`;
- epoch и cumulative update count.

Update trace сохраняет все 300 updates per seed-arm: task order, total/operator/pointer losses,
target counts, pre-clip gradient norm и clipping.

## Exact canonical prefix

`canonical_order` обязан в first 9 updates точно воспроизводить accepted frozen control для
каждого seed. Producer fail-closed сравнивает:

- deterministic initialization hash;
- epoch-3 trained-state canonical hash;
- epoch-3 optimizer canonical hash;
- все first-nine update records по frozen trace fields.

Никаких tolerances. Noncanonical arms не притворяются frozen-control equivalent.

## Pre-specified rescue events

До результатов определены два разных event:

### `first_position0_operator_rescue`

Самый ранний completed epoch/update, где обе nontrivial train tasks `02` и `03` декодируют
gold `UNSTACK` в position 0.

### `first_full_free_running_rescue`

Самый ранний completed epoch/update, где обе initially-unsatisfied tasks `02` и `03` успешно
решаются настоящим free-running execution.

После first rescue отдельно сохраняется persistence на later full checkpoints `10/30/100`
(только на checkpoints, которые не раньше самого event).

Rescue definition после результатов не меняется.

## Cross-seed и cross-arm claims

Для каждого arm сохраняются:

- rescue update по каждому seed;
- mean rescue update только если rescue есть у всех трёх seeds;
- explicit task order и seed count.

Для `task01_middle` и `task01_last` сохраняются per-seed deltas относительно
`canonical_order` отдельно для:

- first position-0 rescue update;
- first full free-running rescue update.

Если любой из сравниваемых event отсутствует, delta остаётся `null`, а не заменяется
favorable subset statistic.

## Independent validation

`a2_sufficient_budget_task_order_validator.py` не импортирует producer aggregation/contrast
helpers нового experiment и не запускает training повторно.

Он независимо:

- bind-ит workflow-requested implementation SHA;
- строит transitive source inventory поверх accepted budget source closure;
- проверяет exact arm × seed × update × epoch × checkpoint coverage;
- проверяет, что arm-order metadata совпадает с фактическим update trace;
- проверяет target counts, canonical per-task weighting и clipping semantics;
- валидирует raw position-0 probability/NLL/correctness;
- независимо исполняет persisted free-running predicted plans через task/domain semantics;
- пересчитывает full-checkpoint teacher/free summaries;
- заново определяет оба first-rescue event;
- заново определяет persistence;
- заново строит cross-seed summaries и cross-arm rescue deltas;
- bind-ит canonical prefix-equivalence к фактическому persisted first-nine trace + epoch-3 hashes.

Validator не принимает auto-generated favorable label как доказательство.

## Tamper coverage

Tests обязаны отвергать как минимум:

- изменение одного per-epoch position-0 raw record при stale first-rescue claim;
- изменение free-running plan/result при stale free-running rescue claim;
- arm-order metadata, не совпадающий с update trace;
- duplicate/missing arm/seed/task/checkpoint evidence;
- forged canonical prefix-equivalence;
- cross-arm rescue delta, не совпадающий с raw per-arm rescue events;
- stale full-checkpoint summary после raw teacher mutation.

## Interpretation boundary

Если rescue timing и stability похожи во всех трёх orders, это усиливает гипотезу, что
insufficient optimization budget является dominant explanation, а within-epoch order вторичен в
этом toy regime.

Если один order даёт большой воспроизводимый rescue shift или не rescue-ится, sequential
task-order dynamics также являются material contributor.

Ни один исход не является evidence для A3, latent channel или semantic geometry.

## Execution boundary

Workflow `.github/workflows/a2-sufficient-budget-task-order.yml`:

- только `workflow_dispatch`;
- только dedicated runner `planning-model-canonical-cpu-v1`;
- exact protected-main binding;
- fixed-target preflight;
- producer;
- independent persisted-evidence validator;
- hidden artifact upload;
- cleanup.

Implementation PR scientific workflow **не запускает**. До reviewer acceptance + merge разрешён
только natural CI.
