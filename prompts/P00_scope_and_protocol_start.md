# P00 — Scope и запуск протокола

**Gate:** `G00_SCOPE`
**Execution role:** `BUILDER`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `release/BOOTSTRAP_MANIFEST.json`
- `docs/operator/bootstrap_integrity_contract_v1.yaml`
- `docs/Planner_MVP_MicroModel_Implementation_Spec_RU_v1.20.md`
- `docs/Planner_LLM_Stage1_Operator_Runbook_v2.20_RU.md`
- `docs/schemas/decision_record.schema.json`

## Действия
1. verify release/BOOTSTRAP_MANIFEST.json against current checkout
2. зафиксировать архивный scope Work Planner / BlocksWorld и run_id
3. проверить отсутствие confirmatory plaintext/outcomes
4. создать первичные RUN_STATUS и scope artifact

## Обязательные результаты фазы
- `artifacts/scope.md`
- `RUN_STATUS.md`
- `RUN_STATUS.json`
- `reports/self-review-P00.json`
- `reports/phase-P00.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- `P00_pre_01` — bootstrap release manifest VERIFIED; verifier: `python validation/phase_check_runner.py --phase P00 --check P00_pre_01 --report reports/phase-P00.json`
- `P00_pre_02` — scope references work-planner/1.20 and runbook 2.20; verifier: `python validation/phase_check_runner.py --phase P00 --check P00_pre_02 --report reports/phase-P00.json`
- `P00_pre_03` — confirmatory outcomes absent; verifier: `python validation/phase_check_runner.py --phase P00 --check P00_pre_03 --report reports/phase-P00.json`
- `P00_pre_04` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P00 --check P00_pre_04 --report reports/phase-P00.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P00_post_01` — externally signed DecisionRecord schema and signature valid
- `P00_post_02` — approved target hash matches sealed evidence
- `P00_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- нет

## Outcomes и переходы
- `APPROVE_SCOPE` → `{'next': 'P01'}`
- `REJECT_SCOPE` → `{'terminal': 'STOPPED_BY_OPERATOR'}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
