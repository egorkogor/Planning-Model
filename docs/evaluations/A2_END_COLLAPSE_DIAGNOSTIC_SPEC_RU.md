# Спецификация диагностики A2 END-only collapse

## Статус

```text
Diagnostic version:
development-learnability-diagnostic/0.1

Status:
development-only-diagnostic

Gate decision:
null

Learnability thresholds:
null

Training policy changed:
false

Held-out accessed:
false
```

Это diagnostics-only стадия для существующего `A2 structured baseline`. Она добавляет
наблюдаемость, но не меняет модель, optimizer, objective, epochs, update budget, seeds,
decoding, parser или executor, не задаёт threshold и не является
`Development Learnability Gate v0.2`.

## Реальная граница данных

Диагностика не вызывает обычный `generate(17)`. Она использует versioned train-only helper,
который строит только существующие строки:

```text
bw-00000001
bw-00000002
bw-00000003
```

Строки `bw-00000004` и `bw-00000005`, их states, goals и oracle plans не создаются. Для них
не вызывается `shortest_plan`, validation split не материализуется и не участвует в hash
фактически прочитанных данных.

Output различает две идентичности:

- `frozen_dataset_lineage_hash` — immutable hash исторического полного dataset, необходимый
  для совместимости с существующим checkpoint/training lineage;
- `evaluated_train_split_hash` — hash канонических bytes только трёх реально созданных и
  оценённых train rows.

Первый не выдаётся за hash прочитанных данных. `heldout_accessed: false` допустим только при
совпадении train-only task IDs, порядка и обоих versioned bindings.

Обычный `generate(17)` и frozen dataset hash остаются неизменными. Обучение сохраняет
historical конфигурацию quality-v0.1:

- seeds `17`, `29`, `43`;
- `3` epochs;
- train-задачи в существующем порядке;
- `9` updates/run;
- final checkpoint only;
- существующие AdamW, loss weights и gradient clipping.

## Teacher-forced per-head metrics

Для каждой позиции gold-плана модель получает правильный gold prefix. Action и pointer heads
диагностируются раздельно:

- `predicted_operator` берётся из action head;
- `arg1_head_prediction` сохраняется при наличии gold arg1 target;
- `arg2_head_prediction` сохраняется при наличии gold arg2 target;
- `arg1_correct` и `arg2_correct` сравнивают соответствующий head с gold target независимо
  от predicted operator;
- `joint_step_correct` требует правильный operator и все применимые pointer-head predictions.

Например, при gold `STACK`, predicted `END` и двух правильных pointer heads результат равен:

```text
operator_correct = false
arg1_correct = true
arg2_correct = true
joint_step_correct = false
```

Pointer denominator определяется только наличием gold targets. Для gold `END` targets,
head predictions и pointer correctness равны `null`.

`decoded_arg1` и `decoded_arg2` — отдельные поля projected decoding. Они следуют arity
predicted operator и не используются как замена per-head teacher-forced metrics.

## Gold-history one-step и projected-plan metrics

Gold-history диагностика разделяет две сущности:

- `gold-history one-step metrics` содержат все позиции gold plan, даже если action head раньше
  предсказал `END`;
- `gold-history projected plan metrics` интерпретируют predictions как последовательный план
  и заканчиваются на первом predicted `END`.

Поэтому operator/joint denominator равен полному количеству gold positions, а exact-plan,
executable-prefix, goal-success, predicted-action-count и first-error используют укороченный
projected plan. Ранний `END` остаётся `EARLY_END`.

## Free-running mode

Free-running использует неизменённый `A2Planner.plan`:

- один Planner call;
- фактический autoregressive prefix модели;
- отсутствие gold prefix;
- отсутствие replanning и suffix correction;
- отсутствие forced minimum length или `END` suppression.

Фактический decoded plan содержит pointers только согласно arity predicted operator. Logits
наблюдаются read-only forward hook-ом; decoding остаётся в существующем Planner.

## First-error и executable-prefix semantics

Используется фиксированный набор:

```text
NONE
EARLY_END
WRONG_OPERATOR
WRONG_ARG1
WRONG_ARG2
EXTRA_ACTION_AFTER_GOLD_END
PARSE_FAILURE
PRECONDITION_FAILURE
GOAL_NOT_ACHIEVED
```

Executable prefix — максимальный начальный префикс predicted actions, где каждое действие
прошло parser, удовлетворило preconditions в evolving state и было применено canonical
transition function. Пустой план при неудовлетворённой initial goal не является fully
executable.

## Loss breakdown

Диагностика не создаёт альтернативный training loop. Observer сохраняет уже вычисляемые
operator, arg1, arg2 и total losses, target counts, gradient norm и clipping occurrence,
возвращая исходные tensors и результат clipping без изменения gradients или optimizer step.

## Полный source closure

Diagnostic source identity — versioned явный tuple фактических runtime/output dependencies.
Он включает, в частности:

```text
docs/architecture/planner_module_inventory_v1.yaml
docs/architecture/task_encoding_v1.yaml
planner_toy/semantic.py
planner_toy/dataset.py
planner_toy/model.py
planner_toy/quality.py
planner_toy/training.py
```

Также включены canonical/runtime, domain/e2e/numeric-identity, learnability implementation,
его schema, CLI, эта спецификация и quality schemas, реально используемые для строгой
training-lineage validation. Тесты и unrelated schemas в inventory не входят.

Переданный полный `--implementation-commit` проверяется fail-closed: commit должен
существовать, быть доступным, являться ancestor results checkout и содержать
`requirements.lock` и каждый `SOURCE_FILES`. Source hashes и requirements hash читаются из
implementation tree через `git show <commit>:<path>`. Перед generation implementation-tree
identity обязана совпасть с working-tree identity. Validator исторического artifact повторяет
расчёт по `payload["implementation_commit"]`.

## Полная training-runs lineage

Для каждого seed обязательны ровно семь файлов:

```text
initialization.pt
trained.pt
optimizer-state.pt
checkpoint-manifest.json
training-config.json
training-report.json
optimizer-evidence.json
```

Лишние и отсутствующие файлы запрещены. Top-level `training_artifact_hashes` связывает
canonical diagnostic JSON с полной recursive map всех generated training files.

Validation переиспользует strict quality schemas и независимо проверяет:

- file SHA-256 и recursive file coverage;
- exact и canonical initialization/trained state-dict identities;
- exact и canonical optimizer identity и структуру AdamW state;
- config, training-report и optimizer-evidence hashes;
- frozen dataset lineage и evaluated train-task ordering;
- seed, A2 variant, epochs, update count и optimizer config;
- active/dormant parameter policy;
- checkpoint policy и initialization/trained distinction;
- соответствие top-level checkpoint records manifests на диске.

Поверхностное пересчитывание artifact hashes и `canonical_identity` не позволяет принять
изменённый или неполный bundle.

## Canonical output

CLI:

```bash
python -m scripts.run_toy_learnability_diagnostic \
  --output-dir <path> \
  --implementation-commit <sha>
```

Создаются:

```text
a2-end-collapse-diagnostic.json
A2_END_COLLAPSE_DIAGNOSTIC.md
training-runs/A2/seed-*/...
```

Markdown отдельно показывает operator, arg1-head, arg2-head и joint-step metrics, а также
раздельные frozen lineage и evaluated train hashes. Generated learnability JSON/Markdown не
коммитятся в implementation PR.

## Ограничения выводов

Диагностические метрики не доказывают причину collapse автоматически.

- Хороший teacher-forced результат и плохой free-running rollout поддерживают гипотезу
  exposure/rollout failure, но не доказывают её.
- Плохой teacher-forced результат поддерживает гипотезу базовой learnability failure, но не
  определяет конкретное исправление.
- Следующий intervention, threshold и gate decision принимаются отдельным versioned contract
  и decision record.
- Эта диагностика не выдаёт `GO`, не разблокирует A3b и не меняет модель.
