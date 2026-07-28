# P14 — GO_INTERFACE decision

**Gate:** `нет`
**Execution role:** `AUDITOR`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/statistics/statistics_contract_v1.yaml`
- `analysis/decision_gates.py`
- `docs/schemas/scientific_decision.schema.json`

## Действия
1. build AnalysisInput from the signed Stage 1A selected-task manifest; no comparison-specific filtering is allowed
2. recompute Stage 1A interface decision from sealed raw logs
3. combine interface result with immutable Planner Stage 1B eligibility lineage
4. permit Stage 1B only for GO_INTERFACE_STAGE1B_ELIGIBLE

## Обязательные результаты фазы
- `reports/stage1a-decision.json`
- `reports/stage1b-eligibility.json`
- `reports/self-review-P14.json`
- `reports/phase-P14.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- нет

## Проверки исполнения
- `P14_exec_01` — all numeric gates applied; verifier: `python validation/phase_check_runner.py --phase P14 --check P14_exec_01 --report reports/phase-P14.json`
- `P14_exec_02` — I3 remains diagnostic; verifier: `python validation/phase_check_runner.py --phase P14 --check P14_exec_02 --report reports/phase-P14.json`
- `P14_exec_03` — eligibility machine-readable; verifier: `python validation/phase_check_runner.py --phase P14 --check P14_exec_03 --report reports/phase-P14.json`
- `P14_exec_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P14 --check P14_exec_04 --report reports/phase-P14.json`
- `P14_exec_05` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P14 --check P14_exec_05 --report reports/phase-P14.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `GO_INTERFACE_STAGE1B_ELIGIBLE` → `{'next': 'P15', 'requires_flags': {'stage1b_eligible': True}, 'set_flags': {'interface_go': True}}`
- `GO_INTERFACE_DIAGNOSTIC_ONLY` → `{'mark_skipped': ['P15', 'P16', 'P17', 'P18'], 'next': 'P19', 'set_flags': {'interface_go': True}}`
- `STOP_INTERFACE` → `{'mark_skipped': ['P15', 'P16', 'P17', 'P18'], 'next': 'P19', 'set_flags': {'interface_go': False}}`
- `INVALID_RUN` → `{'terminal': 'INVALID_RUN'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
