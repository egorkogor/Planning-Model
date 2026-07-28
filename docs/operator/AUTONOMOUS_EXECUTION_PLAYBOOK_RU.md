# Автономный плейбук Work Planner / BlocksWorld

**Версия плейбука:** 1.3
**Протокол:** Implementation Spec v1.20, Stage 1 Runbook v2.20.

## Что получает оператор

Оператор передаёт агенту один master prompt и далее только наблюдает. Агент самостоятельно:

- выбирает и поднимает среду по фиксированному provisioning contract;
- реализует код и тесты;
- строит данные, обучает модели и выбирает конфигурации только по заранее заданным grids;
- готовит freeze packages;
- отправляет confirmatory run отдельному evaluator;
- запускает независимый audit и clean reproduction;
- публикует статус, стоимость, ETA и риски.

Оператор отвечает только на scope, обязательную externally signed trust topology, независимый G06 audit gate, три confirmatory freeze и финальное принятие.

## Запуск

1. Дай Builder Agent shell, git и доступ к локальной машине либо API/credentials уже разрешённого cloud-провайдера.
2. Передай `prompts/00_MASTER_ORCHESTRATOR.md`.
3. Следи за `RUN_STATUS.md` или `RUN_STATUS.json`.
4. На gate отвечай только одной из показанных агентом допустимых команд.

## Почему агент не может тихо изменить эксперимент

В P01 внешний операторский signer создаёт `locks/trust-topology.lock.json`; private key недоступен Builder. В P02 создаётся `locks/scientific.lock.json`: outcome-relevant решения после этого неизменяемы. Между P02 и P06 допускаются только implementation-only patch records без pilot outcomes и с повтором toy preflight. В P06 создаётся `locks/implementation.lock.json`; после этого executable interpretation также неизменяема. Оба verifier входят в bootstrap manifest и защищены Scientific lock.

## Почему confirmatory считается sealed

- Data Sealer генерирует secret selection seed внутри изолированной среды, строит plaintext по hidden HMAC order, немедленно шифрует и удаляет его. Builder видит только seed commitment, encrypted dataset hash, task count и strata commitment; seed раскрывается только Auditor после подписанного evaluator result manifest.
- Plaintext/key доступны только Evaluation Runner на отдельной среде.
- Evaluator запускается только по approved freeze pointer, не имеет права писать source repository и возвращает signed result manifest.
- Audit Agent отдельно пересчитывает metrics и statistics из raw logs.

Single-agent или unsealed confirmatory автоматически получает статус `INVALID_CONFIRMATORY_BLINDNESS`. Одна LLM-сессия может быть только Builder: она не имеет права объявлять себя Sealer, Evaluator, Statistical Reviewer или Audit Agent.

## Фазы

| Фаза | Что происходит | Gate |
|---|---|---|
| P00 | scope и run ID | G00 |
| P01 | ресурсы, роли, внешне подписанная trust topology | обязательный G01 |
| P02 | environment/model pins и Scientific lock | — |
| P03 | contract-level dry-run, runtime schemas, persistence, machine-check dispatcher и validators; без обучения моделей | — |
| P04 | BlocksWorld, Oracle, generator, executable Intent Labeler | — |
| P05 | fixed dataset splits, suffix/off-policy corpus | — |
| P06 | six trained variants plus A4/A5 interventions, clean preflight, independent statistics audit и Implementation lock | G06 |
| P07 | planner development/pilot/N/freeze | G07 |
| P08 | sealed Planner confirmatory | — |
| P09 | Planner scientific decision | — |
| P10 | frozen LLM и resolver | — |
| P11 | prompt development по frozen grid | — |
| P12 | Stage 1A pilot/N/freeze | G12 |
| P13 | sealed Stage 1A confirmatory | — |
| P14 | GO_INTERFACE | — |
| P15 | Stage 1B support/control certification | — |
| P16 | Stage 1B pilot/N/freeze | G16 |
| P17 | sealed Stage 1B confirmatory | — |
| P18 | GO_END_TO_END | — |
| P19 | обязательный independent audit + clean reproduction | — |
| P20 | финальное принятие | G20 |

Переходы не равны «следующему номеру». При научном STOP ненужные фазы получают `SKIPPED_BY_CONTRACT`, после чего выполняется P19 с аудитом отрицательного результата.

## Что проверять на gate

Оператору достаточно проверить пять вещей:

1. статус `WAITING_APPROVAL` и clean tree;
2. pre-gate checks = PASS;
3. target/freeze hash указан и совпадает в evidence;
4. Trust Topology и Scientific lock = VERIFIED; Implementation lock = VERIFIED после G06; confirmatory blindness = SEALED;
5. стоимость/риски понятны и не превышают утверждённый budget.

Отчёт агента должен содержать точную допустимую команду, например `APPROVE G12 <freeze_hash>`.

## Мониторинг

`RUN_STATUS` обязан показывать:

- текущую фазу и процент;
- последнее действие и check;
- отдельные trust-topology/scientific/implementation lock states и blindness state;
- elapsed time, CPU, RAM, GPU/VRAM, disk;
- фактическую и прогнозную стоимость;
- ETA;
- риски и следующий переход.

Хэш сам по себе не доказывает корректность. `verify_gate.py` повторно запускает locked verifier каждого pre/execution check через `phase_check_runner.py`, проверяет DecisionRecord/ledgers/commits, а P19 выполняется отдельным Audit Agent.

## Машинные checks

Каждый `pre_gate_check` и `execution_check` имеет exact verifier в `phase_registry_v1.yaml`. P00–P02/lock checks реализованы в Scientific trust root. С P03 Builder-side проверки исполняет locked `src.validation.phase_checks`, но они не считаются независимым доказательством. До P07 G06 обязан получить два подписанных PASS-отчёта: статистический аудит и отдельный аудит oracle/generator/runtime-checkers/dataset/evaluator/model-loading/persistence. Только после этого создаётся Implementation lock.

## Recovery

Long jobs checkpoint каждые 15 минут/1000 optimizer steps/10 000 data items. После disconnect агент сверяет environment lock, Scientific lock и Implementation lock и input manifests и продолжает только с последнего verified checkpoint. Частичные файлы имеют `.partial`; завершённый artifact появляется только после atomic validation и `.complete.json` marker.

## Bootstrap integrity до первого gate

До P00 агент обязан выполнить `python validation/verify_release_manifest.py`. Манифест фиксирует Scientific trust root и verifier-компоненты до создания runtime locks. Implementation-only code намеренно не входит в bootstrap root: до G06 он контролируется patch records, после G06 — Implementation lock. G00/G01 не могут выполняться при несовпадении.

## Обязательный G01 trust gate

P01 всегда требует DecisionRecord. Внешний signer подписывает exact hashes resource plan, infrastructure plan и public-key registry. `PLANNER_OPERATOR_TRUST_PUBLIC_KEY` указывает на public key вне репозитория; private key в Builder environment запрещён.

## Статистика и resolver

Методы, random seeds, CI, power, non-inferiority/direction rules and thresholds берутся только из `docs/statistics/statistics_contract_v1.yaml` и `docs/semantic/semantic_resolver_v1.yaml`. Подмена метода считается contract mutation.

## Цикл саморевью

Каждая фаза проходит `implement → tests → adversarial tests → clean restore → self-review → fix → repeat`; правила заданы в `docs/operator/self_review_loop_v1.yaml`.

## Точные пути решений и журналов

- решения оператора: `decisions/D-????.json`;
- ресурсный журнал: `reports/resource-usage.jsonl`;
- dispatch между ролями: `dispatch/*.json`;
- sealer manifests: `sealed/<stage>-confirmatory/sealer-manifest.json`;
- self-review каждой фазы: `reports/self-review-PXX.json`.

DecisionRecord обязан содержать `phase_outcome`, совпадающий с ключом перехода state machine.


## Двухуровневый lock и технические правки

Scientific lock создаётся до реализации и защищает всё, что способно изменить вывод эксперимента. Implementation lock создаётся только после toy preflight и реализации P03–P06. До него агент может исправить обнаруженную неоднозначность только через `reports/implementation-patches/IP-????.json`: exact diff, классификация IMPLEMENTATION_ONLY, подтверждение неизменности Scientific lock и полный повтор preflight. После P06 любые изменения обоих уровней требуют нового protocol version/run.

## Независимость суждения

Data isolation не считается независимым review. Перед P07 G06 требует два отчёта: Statistical Reviewer проверяет статистику, Audit Agent отдельно проверяет реализацию. Оба должны быть человеком либо моделью другой family относительно Builder; сервисный процесс без суждения не подходит.
