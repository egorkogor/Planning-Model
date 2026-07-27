# P20 — Final operator acceptance

**Gate:** `G20_FINAL_ACCEPTANCE`
**Execution role:** `OPERATOR`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/operator/OPERATOR_GATE_CHECKLIST.md`
- `docs/audit/independent_audit_contract_v1.yaml`
- `docs/schemas/decision_record.schema.json`

## Действия
1. prepare one-page operator summary
2. seal final audit and release hashes
3. request accept/reject/stop only

## Обязательные результаты фазы
- `reports/operator-summary.md`
- `reports/self-review-P20.json`
- `reports/phase-P20.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- `reports/final-audit.json`

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- `P20_pre_01` — P19 audit PASS; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_01 --report reports/phase-P20.json`
- `P20_pre_02` — release checksums pass; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_02 --report reports/phase-P20.json`
- `P20_pre_03` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_03 --report reports/phase-P20.json`
- `P20_pre_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_04 --report reports/phase-P20.json`
- `P20_pre_05` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_05 --report reports/phase-P20.json`
- `P20_pre_06` — operator executor identity equals Trust Topology operator key fingerprint; verifier: `python validation/phase_check_runner.py --phase P20 --check P20_pre_06 --report reports/phase-P20.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P20_post_01` — externally signed DecisionRecord schema and signature valid
- `P20_post_02` — approved target hash matches sealed evidence
- `P20_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- нет

## Outcomes и переходы
- `APPROVE_FINAL` → `{'terminal': 'ACCEPTED'}`
- `REJECT_FINAL` → `{'repeat': 'P19', 'max_resubmissions': 1}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
