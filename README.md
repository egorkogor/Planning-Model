# Planner → LLM MVP Stage 1

Репозиторий исполнения архивного Work Planner / BlocksWorld по спецификации v1.13 и Stage 1 runbook v2.13.

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

## Изменения v1.13/v2.13

- P01 стал обязательным trust gate: роли, среды, credential principals, public-key registry и resource plan связываются `trust-topology.lock.json`, подписанным внешним операторским Ed25519-ключом;
- валидатор запрещает хранить operator private/public trust-root keys внутри репозитория, поэтому Builder не может выпустить собственную доверенную топологию;
- Scientific lock больше не закрывает будущие P03 builders: нормативная статистика фиксируется в P02, а весь outcome-relevant executable code входит в Implementation lock после G06;
- P03 обязан реализовать analysis builders, resolver/prototype builder, control-certification engine, sealer/evaluator boundaries и runtime loading до независимых аудитов;
- P10 и P15 только запускают уже аудированный код; изменения `src/**`, `analysis/**` и validator logic после G06 запрещены;
- Scientific lock защищает Trust Topology lock, resource/infrastructure plans, public keys и все runtime/model manifests;
- G06 implementation candidate связывает Scientific lock, Trust Topology lock, оба независимых аудита и один reviewed commit;
- слабый `scripts/verify_gate.py` удалён как отдельная реализация и оставлен только thin wrapper на canonical verifier;
- phase/run status разделён на Trust Topology, Scientific и Implementation lock; verifier проверяет соответствие статусов реальной фазе;
- добавлены adversarial-тесты на подмену trust/runtime manifests, ключи внутри репозитория, добавление P03 builders после Scientific lock и эквивалентность entrypoints.
