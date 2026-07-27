# Changelog v1.8/v2.8 → v1.9/v2.9

## Scientific lock

- В Scientific lock перенесены точная архитектура Planner, Task Encoding, A1 grammar, BlocksWorld domain, Intent Labeler, semantic target, статистический код и dependency lock.
- Implementation-only patch теперь разрешён только для runtime/I/O/storage/device plumbing и implementation notes/tests. Изменение model wiring, domain, labels, targets, losses, arms, thresholds, prompts, controls или statistics требует новой версии протокола.

## Machine-enforced decisions

- Добавлен обязательный gate сокращения causal positions для `GO_TYPED`.
- Добавлены seed-direction gates: заявляемое улучшение должно иметь правильное направление минимум в 2 из 3 final seeds.
- Для Stage 1A и Stage 1B machine decision теперь включает parse, infrastructure, incomplete-pair, contract, hash и determinism floors.
- Все обязательные условия GO находятся в `statistics_contract_v1.yaml` и пересчитываются validator-ом.

## Sample size

- Flat paired input заменён stage-specific структурами: `seed_groups` для Planner, `task_clusters` для Stage 1A и `pairs` для Stage 1B.
- N рассчитывается estimator-matched pilot resampling: используется тот же hierarchical/clustered/paired estimator и тот же CI/TOST, что в confirmatory analysis.
- В отчёт добавлено число CI-resamples на одну power simulation.

## Reproducibility

- `scipy` и `cryptography` добавлены в runtime dependencies.
- Добавлен `requirements.lock` и CI matrix Python 3.11/3.13.
- CI проверяет bootstrap manifest, schema bundle, полный pytest и отсутствие изменений checkout после тестов.
- Mutating gate/lock tests выполняются в disposable repository copies.

## Role isolation

- Public-key registry теперь содержит challenge-response подпись каждой отдельной execution role, hashes environment/credential identity и timestamp attestation.
- Auditor/Statistical Reviewer по-прежнему обязаны быть человеком либо моделью другой family относительно Builder.

## P_REPLAY

- Добавлен нормативный `p_replay_contract_v1.yaml`: raw receding-horizon A3 rollout, Planner пересчитывается после каждого действия; LLM, mask, retry и repair запрещены.
