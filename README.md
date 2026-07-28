# Planner → LLM MVP Stage 1

Репозиторий исполнения архивного Work Planner / BlocksWorld по спецификации v1.19 и Stage 1 runbook v2.19.

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

## Изменения v1.19/v2.19

- exact PyTorch module inventory фиксирует 177 `state_dict` tensors, их shapes, parameter types и active-arm masks;
- tensor-name-derived initialization делает начальные веса независимыми от порядка создания модулей;
- A3r inference однозначно разделяет raw-latent autoregressive feedback и nearest-codebook external resolution;
- A1 использует общий 85-position decoder без несуществующего equal-compute retraining;
- active-loss matrix, action-conditional masks и contrastive empty-positive behavior определены машинно;
- P06 model audit обязан проверить inventory, seed-17 initialization checkpoint и dormant gradients всех шести trainable variants;
- P07 принимает ровно 30 final и 10 FLOPs-sensitivity training reports с проверяемыми safetensors/checkpoint sidecars;
- P08 требует exact sealed matrix с отдельными `PLANNER_A2C_FLOPS_RAW` и `PLANNER_A3_FLOPS_RAW` arms;
- `planner_seed` связан с lineage даже при FAILED plan generation;
- n=7–8 запрещены в training/development и остаются только sealed size-OOD evaluation.
