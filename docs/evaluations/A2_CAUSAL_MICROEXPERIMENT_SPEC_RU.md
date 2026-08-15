# A2 causal microexperiment: operator-loss weighting vs task order

Статус: **development-only scientific microexperiment**.

Цель — проверить поддержанную, но не доказанную гипотезу из persisted A2 diagnostic:
`SUPPORTED HYPOTHESIS / NOT PROVEN`.

Эксперимент не меняет frozen quality trainer, accepted A2 diagnostic, A3/A4, formal
fixed-target acceptance или held-out policy. Используются только train tasks
`bw-00000001/02/03` и seeds `17/29/43`. Held-out `04/05` запрещены.

## Причинный вопрос

Frozen control делает один AdamW update на задачу и считает operator CE как mean по
валидным позициям конкретной задачи. Поэтому task 01 (одна позиция, END) получает вес
1.0 на position 0 за epoch, а position-0 UNSTACK из task 02 и 03 — 0.2 + 0.2 = 0.4.

Одновременно task 01 всегда идёт первым update каждого epoch. Эксперимент должен
разделить эффект нормализации operator objective и эффект порядка.

## Arms

1. `canonical_control`
   - вызывает существующий `_train_a2_with_loss_trace -> quality._train`;
   - порядок `01 → 02 → 03`;
   - operator loss: per-task mean CE;
   - это неизменённый control.

2. `equal_position_operator_loss`
   - порядок остаётся `01 → 02 → 03`;
   - меняется только operator loss;
   - operator CE = `sum(position CE) * task_count / total_operator_positions`;
   - для текущего train split coefficient каждой operator position = `3/11`;
   - сумма operator weights при одинаковом loss остаётся 3.0 за epoch, как у control;
   - arg1/arg2 objective, AdamW, clipping, epochs, model и seed не меняются.

3. `task_order_only`
   - operator objective остаётся canonical per-task mean CE;
   - меняется только порядок на `02 → 03 → 01`, то есть короткая END-задача переносится
     с первого update на последний;
   - остальное идентично control.

Combined normalization+order arm намеренно не добавляется в первый causal round:
минимальный дизайн должен отдельно оценить два предложенных механизма.

## Обязательный control-equivalence probe

Новый intervention trainer не считается эквивалентным control по предположению.
Перед принятием causal contrasts для каждого seed выполняется отдельный
**не-научный equivalence probe**: intervention trainer запускается с canonical order
`01 → 02 → 03` и canonical per-task mean operator CE.

Probe обязан **точно**, без tolerance, совпасть с frozen control по:

- initialization canonical state hash;
- trained canonical state hash;
- optimizer canonical hash;
- всем 9 update по task schedule;
- operator/arg1/arg2/total loss;
- operator/arg target counts;
- pre-clip gradient norm, clip norm и clipping flag;
- operator position weight.

Любое расхождение — fail-closed `CONTROL_EQUIVALENCE_*`; causal artifact не создаётся.
Probe не является четвёртым arm и не участвует в научных aggregate/contrasts.

## Required output

Для каждого arm/seed и aggregate:

- position-0 gold `UNSTACK` operator accuracy;
- mean END probability на этих position-0 targets;
- position-4 gold END accuracy;
- full teacher-forced operator/non-END/END, arg1, arg2 и joint accuracy;
- free-running exact-plan, goal-success, initially-unsatisfied goal-success,
  zero-action rate;
- per-update loss/gradient/clipping trace;
- initialization/trained/optimizer canonical hashes.

Также сохраняются exact arm contract, train-only lineage, transitive source identity,
control-equivalence evidence, contrast каждого intervention против canonical control и
`automatic_gate = null`.

Результат не делает автоматического causal verdict. Интерпретация только
`SUPPORTED HYPOTHESIS / NOT PROVEN`. `GO_LATENT = NOT EVALUATED`.

## Независимая validation semantics

`--validate-only` не повторяет producer и не сравнивает artifact с результатом того же
producer-кода. Validator независимо пересчитывает claim-bearing semantics из
persisted lower-level evidence:

- train-only lineage и weighting contract;
- arm/seed matrix и 9-update schedule;
- operator position weights;
- seed-level teacher-forced и free-running summaries из raw observations;
- aggregate summaries;
- intervention-vs-control contrasts;
- exact control-equivalence binding.

Validation дополнительно bind'ится к workflow-requested `implementation_commit` и
пересчитывает source identity именно для этого commit. Source closure строится как
causal-specific files + accepted A2 learnability source inventory, включая как минимум
`dataset.py`, `domain.py`, `semantic.py`, `quality.py`, `training.py`, model/e2e и
train-only materialization.

Mutation tests должны доказывать, что re-signed tampering aggregate, contrast, raw
observation или equivalence evidence отвергается validator'ом.

## Execution

Workflow только manual `workflow_dispatch` на fixed canonical runner
`planning-model-canonical-cpu-v1`, с exact protected-main binding и target preflight.
После producer выполняется independent persisted-evidence validator, затем artifact
upload.

Эксперимент нельзя запускать из PR. После merge reviewer должен сначала подтвердить
implementation и только затем поручить ровно один fresh run на новом authoritative main.
