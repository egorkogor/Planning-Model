# Changelog v1.9/v2.9 → v1.10/v2.10

## Исполняемая оркестрация

- Исправлен вызов `verify_lock(kind, lock_path)` в `phase_check_runner.py`.
- `P02_exec_02` теперь запускает schema/static validation и полный `pytest validation`; название check больше не маскирует пропуск тестов.
- Добавлены regression tests, которые проверяют порядок аргументов и реальный запуск pytest.

## Контрактная согласованность

- `numpy==2.3.5` зафиксирован одновременно в generator contract, runtime dependency contract и `requirements.lock`.
- Hyperparameter search использует один exact алгоритм: per-arm validation composite → mean rank across A1/A2/A2b/A2c/A3 → единые floors/tie-breaks.
- Parameter count matching везде exact (`tolerance=0.0`) через общий superset inventory hash.

## Statistics / sample size

- Нормативный метод обновлён до `estimator_matched_paired_binary_simulation_v3`.
- Симулятор генерирует только клетки `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`; differences всегда `-1/0/1`.
- Planner сохраняет seed→task hierarchy, Stage 1A — task→snapshots clusters, Stage 1B — paired tasks.
- JSON Schema sample-size input требует binary `left/right` и discrete difference.
- Stage 1A confirmatory reserve увеличен с 2 000 до 4 000 base tasks.

## Sealed execution

- P15 теперь только public/preflight certification; Builder не материализует hidden confirmatory IDs.
- Data Sealer обязан сертифицировать hidden Stage 1B control availability и Planner support до HMAC ranking, encryption release и outcome access.
- Signed `sealer_manifest` для Stage 1B содержит `control_certification`, coverage `1.0`, counts и hashes coverage/support artifacts.
- Добавлен semantic validator `candidate >= eligible >= selected` и binding contract hashes.

## Prompt contract

- Каждый C01–C04 содержит `message_sequence_template_exact`.
- Few-shot examples C02/C04 представлены отдельными user/assistant messages.
- Prompt hash включает role/content array и rendered unpadded tensors.

## Scientific lock

- Intent Labeler rules, catalog и golden outputs объявлены неизменяемыми после P02.
- Implementation-only patch может исправлять adapter/serialization code только при byte-identical labeler outputs.
