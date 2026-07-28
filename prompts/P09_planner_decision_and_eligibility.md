# P09 — Planner decision и Stage 1 eligibility

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
1. build AnalysisInput from the signed Planner selected-task manifest; every comparison must contain the exact same task IDs
2. recompute Planner metrics from sealed raw logs through locked analysis code
3. apply architecture relative gates separately from absolute Stage 1B eligibility floor
4. record one of GO_PLANNER_STAGE1B_ELIGIBLE, GO_PLANNER_DIAGNOSTIC_ONLY or STOP_PLANNER

## Обязательные результаты фазы
- `reports/planner-decision.json`
- `reports/stage1-eligibility.json`
- `checkpoints/stage1/manifest.json`
- `reports/self-review-P09.json`
- `reports/phase-P09.json`
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
- `P09_exec_01` — all gates numeric; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_01 --report reports/phase-P09.json`
- `P09_exec_02` — equal-data and FLOPs sensitivity direction consistent; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_02 --report reports/phase-P09.json`
- `P09_exec_03` — eligibility machine-readable; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_03 --report reports/phase-P09.json`
- `P09_exec_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_04 --report reports/phase-P09.json`
- `P09_exec_05` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_05 --report reports/phase-P09.json`
- `P09_exec_06` — A3 HORIZON point estimate and CI floor checked; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_06 --report reports/phase-P09.json`
- `P09_exec_07` — A3 valid-action-rate and P_REPLAY floors checked; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_07 --report reports/phase-P09.json`
- `P09_exec_08` — Stage 1B eligibility is separate from architecture GO; verifier: `python validation/phase_check_runner.py --phase P09 --check P09_exec_08 --report reports/phase-P09.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `GO_PLANNER_STAGE1B_ELIGIBLE` → `{'next': 'P10', 'set_flags': {'planner_architecture_go': True, 'stage1b_eligible': True}}`
- `GO_PLANNER_DIAGNOSTIC_ONLY` → `{'next': 'P10', 'set_flags': {'planner_architecture_go': True, 'stage1b_eligible': False}}`
- `STOP_PLANNER` → `{'next': 'P10', 'set_flags': {'planner_architecture_go': False, 'stage1b_eligible': False}, 'note': 'Stage 1A diagnostic remains allowed; Stage 1B forbidden'}`
- `INVALID_RUN` → `{'terminal': 'INVALID_RUN'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
