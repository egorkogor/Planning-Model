# P01 — Provisioning, externally signed trust topology и бюджет

**Gate:** `G01_TRUST_AND_RESOURCES`
**Execution role:** `BUILDER`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/infrastructure/provisioning_contract_v1.yaml`
- `docs/infrastructure/resource_budget_contract_v1.yaml`
- `docs/infrastructure/recovery_contract_v1.yaml`
- `docs/infrastructure/runtime_dependency_contract_v1.yaml`
- `docs/schemas/resource_plan.schema.json`
- `docs/operator/trust_topology_lock_v1.yaml`
- `docs/schemas/trust_topology_lock.schema.json`
- `validation/trust_topology_validator.py`
- `docs/schemas/decision_record.schema.json`

## Действия
1. инвентаризировать локальные ресурсы и оценить стоимость
2. сформировать resource plan для Builder, Data Sealer, Evaluation Runner, Audit Agent и Statistical Reviewer
3. получить role challenge signatures от четырёх небuilder execution identities
4. создать infrastructure plan и public-key registry
5. передать три trust artifacts внешнему операторскому signer; Builder не получает private key
6. создать и проверить locks/trust-topology.lock.json по внешнему public key

## Обязательные результаты фазы
- `reports/resource-plan.json`
- `locks/infrastructure-plan.json`
- `locks/public-keys.json`
- `reports/self-review-P01.json`
- `reports/phase-P01.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`
- `locks/trust-topology.lock.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- `P01_pre_01` — resource plan проходит schema; verifier: `python validation/phase_check_runner.py --phase P01 --check P01_pre_01 --report reports/phase-P01.json`
- `P01_pre_02` — роли разделены либо confirmatory заранее помечен INVALID; verifier: `python validation/phase_check_runner.py --phase P01 --check P01_pre_02 --report reports/phase-P01.json`
- `P01_pre_03` — стоимость не превышает hard budget; verifier: `python validation/phase_check_runner.py --phase P01 --check P01_pre_03 --report reports/phase-P01.json`
- `P01_pre_04` — Ed25519 public-key registry schema valid and role key IDs match resource plan; verifier: `python validation/phase_check_runner.py --phase P01 --check P01_pre_04 --report reports/phase-P01.json`
- `P01_pre_05` — trust topology lock externally signed, artifact hashes exact and operator key is outside repository; verifier: `python validation/phase_check_runner.py --phase P01 --check P01_pre_05 --report reports/phase-P01.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P01_post_01` — externally signed DecisionRecord schema and signature valid
- `P01_post_02` — approved target hash matches sealed evidence
- `P01_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- нет

## Outcomes и переходы
- `APPROVE_TRUST_AND_RESOURCES` → `{'next': 'P02'}`
- `REJECT_TRUST_AND_RESOURCES` → `{'terminal': 'STOPPED_NO_RESOURCES'}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
