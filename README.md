# Planner → LLM MVP Stage 1

Репозиторий исполнения архивного Work Planner / BlocksWorld по спецификации v1.17 и Stage 1 runbook v2.17.

## Запуск агентом

```bash
python -m pip install -r requirements-validation.txt
python validation/verify_release_manifest.py
python validation/validate_bundle.py
```

Затем передать Builder Agent `prompts/00_MASTER_ORCHESTRATOR.md` и следить за `RUN_STATUS.md`/`RUN_STATUS.json`.

Агент выполняет P00–P20 по explicit state machine. Оператор подтверждает scope, внешнюю Trust Topology, оба G06-аудита, три confirmatory freeze и final acceptance.

## Защита эксперимента

- до завершения P01 внешний оператор подписывает Trust Topology lock; Builder не имеет доступа к private key;
- после P02 жёстко фиксируется Scientific lock; implementation-only уточнения разрешены только до G06 через patch record и повтор toy preflight; после G06 фиксируется Implementation lock;
- prompt tuning только на development;
- confirmatory plaintext создаёт отдельный Data Sealer и читает только Evaluation Runner;
- hidden Stage 1B controls сертифицирует Data Sealer до HMAC-selection и до доступа к outcomes;
- P19 обязательно воспроизводит результат на clean checkout отдельным Audit Agent.

## Scope

Builder LLM исполняет плейбук и пишет реализацию. Сам эксперимент проверяет отдельную Planner-микромодель. Sealer/Evaluator — детерминированные сервисные процессы; reviewer/auditor — независимые роли. Одна LLM-сессия не может подменять эти роли.


Это BlocksWorld Work Planner experiment, не основная Cognitive Planner architecture ML Brain.

## Изменения v1.17/v2.17

- единый 85-position decoder inventory: A1 использует все позиции и grammar head, step-level arms — первые 17 позиций через ConceptPacker;
- зафиксирована точная active-loss matrix и action-conditional masking без двойного END-loss;
- contrastive loss имеет детерминированное поведение при отсутствии positive pair в batch;
- `planner_seed` связан с EpisodePlanManifest и EpisodeLog даже при FAILED plan generation;
- n=7–8 запрещены в training/development и остаются только sealed size-OOD evaluation;
- sensitivity run унифицирован как train-FLOPs-matched A3↔A2c; inference FLOPs — только guardrail;
- сохранены launch-инварианты v1.16: exact task/seed/arm matrix, одинаковые Stage 1A snapshots и единое имя replay-метрики.
