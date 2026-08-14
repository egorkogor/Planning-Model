# A2 END-only learnability diagnostic v0.2 — observability extension

## Статус и граница выводов

```text
Diagnostic version: development-learnability-diagnostic/0.2
Status: development-only-diagnostic
Read-only: true
Gate decision: null
Learnability thresholds: null
Model changed: false
Training policy changed: false
Held-out accessed: false
```

Эта версия **расширяет observability существующего PR #22 diagnostic v0.1**, а не создаёт новый training algorithm. Базовый `development-learnability-diagnostic/0.1` остаётся отдельным sealed core bundle и по-прежнему вызывает frozen `planner_toy.quality._train` для A2. v0.2 оборачивает реальные вызовы gradient clipping и `AdamW.step` read-only наблюдателем, после каждого уже совершившегося optimizer update строит отдельную диагностическую копию модели из фактических active parameter tensors и измеряет teacher-forced поведение. Наблюдатель не пишет в training tensors, gradients, optimizer state или checkpoint producer.

Запрещены и не реализованы: изменение architecture, optimizer, LR, betas, eps, weight decay, epochs, update count, loss weights, examples, seeds, decoding, END suppression, minimum plan length, scheduled sampling, held-out selection, A3/A4 training и любые learnability thresholds/gates.

## Train-only boundary

Единственный materialized dataset path остаётся `generate_train_only(17)` из v0.1. Он строит только:

```text
bw-00000001
bw-00000003
bw-00000002
```

`bw-00000004` и `bw-00000005` не materialize'ятся в diagnostic execution: их states, goals и oracle plans не создаются, `shortest_plan` для них не вызывается, они не входят в metrics или evaluated-data hash.

Две разные identities сохраняются явно:

```text
dataset_lineage_order:
01, 03, 02

optimizer_execution_task_order:
01, 02, 03
```

`frozen_dataset_lineage_hash` — immutable provenance binding исторического полного dataset. Он не выдаётся за hash прочитанных данных. Фактически прочитанные train rows имеют отдельный `evaluated_train_split_hash`.

## Frozen training semantics

Training producer остаётся `planner_toy.quality._train` без копии или альтернативного loop:

- A2 only;
- seeds 17/29/43;
- 3 epochs;
- 9 updates/seed;
- AdamW `lr=3e-4`, `betas=(0.9, 0.95)`, `eps=1e-8`, `weight_decay=0.01`;
- action CE + arg1 CE на gold non-END targets + arg2 CE на gold binary-pointer targets;
- gradient clipping 1.0;
- final checkpoint only, no held-out selection.

Observer фиксирует уже существующие loss tensors и фактический clip/step path. Он не добавляет loss terms и не изменяет порядок вызовов frozen trainer.

## 1. Basic optimization / no-learning observability

Для каждого real update сохраняются:

- total/action/arg1/arg2 loss из v0.1 observer;
- target counts;
- gradient norm до clipping — exact return существующего `clip_grad_norm_`;
- gradient norm после clipping, посчитанный read-only;
- число gradient tensors и число ненулевых gradient tensors;
- finite-gradient flag;
- clipping occurrence;
- число active parameter tensors, изменившихся именно этим optimizer step;
- суммарный L2 update active parameters.

Для каждого seed сохраняется initialization → final checkpoint comparison:

- L2 delta каждого active parameter tensor;
- fraction active tensors changed;
- total active delta norm;
- dormant tensors changed count, обязанный быть 0;
- finite checkpoint values;
- optimizer state parameter count;
- число ненулевых moment tensors и finite optimizer-state flag;
- action-head weight/bias delta;
- END action-row weight delta и END bias delta;
- arg1/arg2 pointer-head weight deltas.

Это evidence обучения, а не критерий acceptance.

## 2. Operator-head / END collapse observability

После каждого **реального** optimizer update модель диагностируется на всех train rows с gold history, без нового training step. Для каждого update и final checkpoint сохраняются:

- teacher-forced operator accuracy;
- END accuracy и non-END accuracy отдельно;
- predicted END count/rate;
- END probability;
- `END logit - best non-END logit` margin;
- full operator confusion matrix;
- breakdown по gold position;
- structural flags `END is modal prediction` и `all positions predict END`.

`first_end_modal_update_index` и `first_all_positions_end_update_index` — descriptive structural events, не tuneable thresholds и не gates.

## 3. Pointer-head observability

Pointer heads оцениваются независимо от predicted operator:

- arg1 denominator = позиции, где gold arg1 target существует;
- arg2 denominator = позиции, где gold arg2 target существует;
- predicted `END` не обнуляет и не объявляет pointer head ошибочным;
- joint step correctness отдельно требует correct operator и все применимые gold-pointer predictions.

Gold END position не имеет pointer denominator.

## 4. Gold-history projection vs true free-running

Сохраняются три разные сущности:

1. teacher-forced one-step predictions на каждом gold position;
2. projected plan v0.1, построенный из teacher-forced per-position predictions и обрывающийся на первом predicted END;
3. true free-running `A2Planner.plan` с собственным autoregressive history.

v0.2 не заменяет free-running своим decoder. Дополнительный forward hook только читает logits того же вызова Planner и сохраняет END probability/margin.

Для free-running сохраняются:

- predicted END count/rate;
- END probabilities;
- END-vs-best-non-END margins;
- first predicted END position;
- early-END boolean/rate относительно gold END position;
- predicted plan length;
- plan-length distribution;
- zero-action-plan count/rate;
- breakdown initial goal already satisfied / not satisfied.

Ранний END нигде не маскируется.

## Интерпретация

Diagnostic не делает causal claim. Он выбирает только descriptive localization:

- если final teacher-forced operator accuracy не perfect — `TEACHER_FORCED_OPERATOR_ERRORS_PRESENT`;
- иначе если pointer-head accuracy не perfect — `TEACHER_FORCED_POINTER_ERRORS_PRESENT`;
- иначе если teacher-forced heads perfect, но free-running exact-plan не perfect — `FREE_RUNNING_ONLY_ERRORS_PRESENT`;
- иначе — `NO_ERROR_LOCALIZED_ON_EVALUATED_TRAIN_TASKS`.

Это использует только exact correctness boundaries, без произвольных numerical thresholds.

Поддерживаемая гипотеза всегда маркируется:

```text
SUPPORTED HYPOTHESIS / NOT PROVEN
```

Она не является `GO`, не разрешает intervention и не меняет historical `REDESIGN`.

## Runtime / dedicated target distinction

Успешный historical formal fixed-target acceptance не переносится автоматически на этот diagnostic.

Отдельный manual workflow:

```text
.github/workflows/a2-learnability-diagnostic.yml
```

может после reviewer acceptance implementation state запустить diagnostic на `planning-model-canonical-cpu-v1`. Он:

- требует exact protected-main implementation SHA;
- проверяет concrete target contract и live observation через существующий non-formal fixed-target preflight;
- запускает diagnostic и independent validate-only;
- загружает preflight + diagnostic bundle;
- **не** вызывает formal `validate-bundle`/`final-gate` и не создаёт formal acceptance claim.

Workflow не запускается автоматически из PR и не генерирует result artifacts в implementation PR.

## Outputs

v0.2 bundle:

```text
a2-end-collapse-diagnostic-v0.2.json
A2_END_COLLAPSE_DIAGNOSTIC_V0_2.md
core-v0.1/
  a2-end-collapse-diagnostic.json
  A2_END_COLLAPSE_DIAGNOSTIC.md
  training-runs/...
```

Top-level JSON является canonical machine-readable observability record и cryptographically binds core v0.1 canonical identity. Markdown — краткая интерпретация семи обязательных diagnostic questions без gate decision.
