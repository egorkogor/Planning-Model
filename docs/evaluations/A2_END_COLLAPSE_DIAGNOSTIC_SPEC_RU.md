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

Это diagnostics-only стадия. Она добавляет наблюдаемость для существующего `A2 structured
baseline`, но не исправляет поведение модели, не задаёт threshold и не является
`Development Learnability Gate v0.2`.

## Граница данных

Текущий toy dataset содержит только `train` и `validation`. Две строки существующего
`validation` split (`bw-00000004` и `bw-00000005`) уже являются frozen held-out задачами
Development Quality Evaluation v0.1. Поэтому они не используются в этой диагностике.

Диагностические метрики вычисляются только для существующего `train` split. Новые задачи,
новый split и post-hoc переименование validation-задач не создаются. Отсутствие отдельного
non-held-out development validation split является явным ограничением формата 0.1.

Отдельный non-held-out development validation split в текущем frozen dataset отсутствует.

Обучение использует неизменённую historical конфигурацию quality-v0.1:

- seeds `17`, `29`, `43`;
- `3` epochs;
- три существующие train-задачи в порядке `task_id`;
- `9` updates/run;
- final checkpoint only;
- существующие AdamW, loss weights и gradient clipping.

## Teacher-forced mode

Для каждой позиции gold-плана модель получает правильный gold prefix. Gold plan используется
только как diagnostic history и target. Для позиции сохраняются:

- gold и predicted operator;
- operator top-1 correctness;
- probability gold operator;
- probability `END`;
- operator negative log-likelihood;
- наличие pointer targets;
- predicted arg1/arg2;
- pointer correctness;
- joint exact-step correctness.

Pointer denominator включает только позиции с определённым target. Predicted pointer-поля
следуют arity предсказанного operator: для predicted `END` оба pointer-поля равны `null`,
для unary operator заполнен только `arg1`, для binary operator заполнены `arg1` и `arg2`.
Correctness pointer-поля остаются `null`, если соответствующего gold target нет.

## Gold-history one-step и projected-plan metrics

Gold-history диагностика разделяет две разные сущности.

`gold-history one-step metrics` включают все позиции gold plan. На каждой позиции модель
получает правильный gold prefix, а denominator operator/joint metrics равен полному числу
gold positions. Ранний predicted `END` на одной позиции не удаляет predictions последующих
позиций.

`gold-history projected plan metrics` интерпретируют те же one-step predictions как один
последовательный план. Для exact-plan match, executable prefix, goal success, predicted action
count и first-error classification projected plan заканчивается на первом predicted `END`.
Поэтому ранний `END` сохраняет `EARLY_END`, но не сокращает denominator one-step metrics.

## Free-running mode

Free-running использует неизменённый `A2Planner.plan`:

- один Planner call;
- фактический autoregressive prefix модели;
- отсутствие gold prefix;
- отсутствие replanning и suffix correction;
- отсутствие forced minimum length или `END` suppression.

Logits наблюдаются read-only forward hook-ом; decoding остаётся в существующем Planner.

## First-error taxonomy

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

Фиксируется первая категория. Parse и precondition failures не сворачиваются в generic
mismatch. При precondition failure на той же позиции он имеет приоритет над содержательным
mismatch как фактический первый отказ исполнения.

## Executable-prefix semantics

Executable prefix — максимальный начальный префикс predicted actions, где каждое действие:

1. прошло существующий parser;
2. удовлетворило preconditions в evolving state;
3. применено существующей canonical transition function.

Сохраняются длина префикса и доли относительно predicted и gold action counts. Для пустого
predicted plan fraction относительно predicted равен `null`. Пустой план при изначально
неудовлетворённой цели не считается fully executable.

## Loss breakdown

Диагностика не создаёт альтернативный training loop. Она прозрачно наблюдает существующую
quality-v0.1 training function и записывает уже вычисляемые:

- operator loss;
- arg1 pointer loss;
- arg2 pointer loss;
- total loss;
- target counts;
- gradient norm;
- clipping occurrence.

Observer возвращает исходные loss tensors и исходный результат clipping без изменения
objective, gradients или optimizer step. Regression tests сравнивают checkpoints и optimizer
state с запуском без observer.

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

JSON содержит provenance, runtime fingerprint, unchanged training config, checkpoint и
optimizer identities, per-update losses, teacher-forced/free-running diagnostics,
aggregates, first-error distribution и invariance evidence. Markdown является
детерминированным rendering JSON.

Переданный полный `--implementation-commit` проверяется fail-closed: commit должен
существовать, быть доступным и являться ancestor текущего results checkout, а его tree должен
содержать `requirements.lock` и все diagnostic `SOURCE_FILES`. Source hashes, aggregate
`diagnostic_source_sha256` и `requirements_lock_sha256` вычисляются из bytes implementation
tree через `git show <commit>:<path>`, а не из текущего working tree. Validator повторяет
этот расчёт по `payload["implementation_commit"]`, поэтому historical artifact не привязан к
случайному состоянию текущего checkout.

Schema хранится только в `planner_toy/schemas/toy_learnability_diagnostic.schema.json`.
Она не входит в immutable source inventory quality-v0.1: quality subsystem использует явный
historical список принадлежащих ему schemas вместо широкого `toy_*.schema.json` glob.

Generated JSON/Markdown не коммитятся в implementation PR. После code freeze внешний
reviewer передаёт remote-reachable implementation SHA и выполняет отдельную results-фазу.


## Semantic validation derived fields

Validator независимо восстанавливает derived metrics из persisted primitive fields.

Для teacher-forced rows пересчитываются operator, END, arg1, arg2 и joint correctness,
operator arity, target flags и operator NLL. Для free-running rows из
`predicted_history_positions` восстанавливается raw predicted plan и проверяются termination,
model-forward coverage, parser status, generation failure, failure code, counts, exact match,
executable-prefix и first-error semantics. Persisted booleans не используются как источник
истины.

Adversarial tests изменяют derived values, заново считают aggregates и `canonical_identity`,
после чего validator всё равно отклоняет artifact стабильным `ValueError` code.

## Ограничения выводов

Диагностические метрики не доказывают причину collapse автоматически.

- Хороший teacher-forced результат и плохой free-running rollout поддерживают гипотезу
  exposure/rollout failure, но не доказывают её.
- Плохой teacher-forced результат поддерживает гипотезу базовой learnability failure, но не
  определяет конкретное исправление.
- Следующий intervention, threshold и gate decision принимаются отдельным versioned contract
  и decision record.
- Эта диагностика не выдаёт `GO`, не разблокирует A3b и не меняет модель.
