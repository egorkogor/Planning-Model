# Changelog v1.14 / v2.14

## Архитектура эксперимента

- Stage 1B переведён с reactive next-intent на один pre-execution frozen full plan без replanning.
- Добавлены семь E0–E5/P arms, два раздельных replay-contexts и обязательная plan → episode → attempt lineage.
- A3r стал шестым обучаемым parameter-matched arm с frozen random codebook.
- A2c unknown signature fail-closed как `SEMANTIC_UNRESOLVED`; heuristic mapping запрещён.

## Причинная валидность

- Stage 1B selection ограничен task/domain/split metadata.
- Удалён старый reachable-state control-coverage veto.
- Plan/control failures и degeneracy не исключают задачи и остаются paired outcomes.
- E2 shuffle воспроизводится из hash-bound cyclic rotation manifest.

## Статистика и compute

- Confirmatory TOST удалён.
- Design alternative 7,5 п.п. отделена от GO threshold 5 п.п.
- Estimator types и power lower-bound проверяются машинно.
- Capacity включает 64 training workloads и семь Stage 1B arms.
- Compute profile, evidence, code и per-episode cap связаны hashes.

## Trust и аудит

- Scientific/Implementation lock scopes расширены на full-plan, random-codebook и capacity validators.
- Implementation audit содержит единый exact set из 15 checks.
- Phase prompts и release bootstrap manifest синхронизированы с v1.14/v2.14.
