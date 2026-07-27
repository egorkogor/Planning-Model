# Changelog v1.10/v2.10 → v1.11/v2.11

## Protocol capacity

- Stage 1B confirmatory reserve увеличен с 1 000 до 4 000 base tasks.
- Quotas Stage 1B reserve изменены на 200/600/1 200/2 000 для n=3/4/5/6.
- Добавлено fail-closed правило: reserve после eligibility должен покрывать locked `selected_n`, иначе `BLOCKED_PROTOCOL_CAPACITY`.

## Sample size

- Компоненты расчёта N теперь stage-specific.
- Planner использует `primary_ci`, `primary_power`, `current_vs_shuffled_power`; несуществующий Planner TOST удалён.
- Stage 1A и Stage 1B сохраняют TOST для shuffled≈neutral.
- JSON Schema запрещает placeholder `equivalence_TOST_power` на Planner-stage.

## Independent trust boundary

- G06 разделён на два подписанных аудита: статистический и независимый аудит реализации.
- Implementation audit покрывает oracle, generator, runtime checkers, dataset split/leakage, evaluator harness, model/runtime loading, persistence/hashing и clean preflight.
- Builder-side `src.validation.phase_checks` больше не считается независимым доказательством.
- Implementation lock создаётся только после PASS обоих аудитов.

## Role topology

- Builder LLM может только оркестрировать внешние роли и не может их impersonate.
- Data Sealer и Evaluation Runner обязаны быть отдельными deterministic `SERVICE_PROCESS` с уникальными environment identities и principals.
- Auditor и Statistical Reviewer требуют независимого суждения; одна LLM-сессия не может выполнить все роли.

## Runtime reproducibility

- `locks/environment.lock` заменён на schema-valid `locks/environment.lock.json`.
- Lock фиксирует Python executable, Torch/Transformers/VLLM artifacts and revisions, CUDA/driver/cuDNN/kernels, tokenizer/processor revisions, language-model-only mode и offline cache manifest.
- Runtime lock криптографически связывается с обоими model locks.

## Consistency

- Удалён расходящийся дубликат `scripts/phase_check_runner.py`; оставлен thin wrapper на trusted implementation.
- Устаревший термин `contract lock` заменён на `implementation lock` в активных phase descriptions/prompts.
- Исправлены старые 2 000/1 000 reserve значения в основной спецификации и runbook.

## Дополнительное закрытие lineage

- `freezes/implementation-lock.candidate.json` теперь содержит hashes статистического и implementation audit.
- Оба аудита обязаны иметь тот же `run_id` и `reviewed_commit`, что candidate и P06 phase report.
- `P06_pre_02` переведён из Builder runtime checker в bootstrap-protected core check.
