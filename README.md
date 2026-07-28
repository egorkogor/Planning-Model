# Planner → LLM MVP Stage 1

Репозиторий исполнения архивного Work Planner / BlocksWorld по спецификации v1.16 и Stage 1 runbook v2.16.

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

## Изменения v1.16/v2.16

- confirmatory task IDs фиксируются до outcomes в `SelectedTaskManifest`;
- signed SealerManifest связывает точный task set, его SHA-256 и task count;
- full-plan lineage обязан покрывать весь selected task set без удаления неудачных задач;
- Stage 1B требует точное произведение selected tasks × семь arms;
- `run_id` и stage связаны между EpisodePlanManifest, WorkPlan, EpisodeLog и AttemptLog;
- evaluator `task_count` пересчитывается по lineage;
- все statistical comparisons используют один signed task set;
- sample-size components привязаны к заранее заданным comparison IDs;
- Planner replay metric унифицирована как `P_REPLAY_GOAL_SUCCESS`.
