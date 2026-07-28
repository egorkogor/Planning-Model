# P18 — GO_END_TO_END decision

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
1. build AnalysisInput from the signed Stage 1B selected-task manifest; task IDs must equal lineage and evaluator task_count
2. recompute paired task-level estimates
3. apply all end-to-end gates
4. publish scientific decision and limitations

## Обязательные результаты фазы
- `reports/stage1b-decision.json`
- `reports/self-review-P18.json`
- `reports/phase-P18.json`
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
- `P18_exec_01` — all thresholds exact; verifier: `python validation/phase_check_runner.py --phase P18 --check P18_exec_01 --report reports/phase-P18.json`
- `P18_exec_02` — diagnostic arms not substituted for primary; verifier: `python validation/phase_check_runner.py --phase P18 --check P18_exec_02 --report reports/phase-P18.json`
- `P18_exec_03` — raw lineage complete; verifier: `python validation/phase_check_runner.py --phase P18 --check P18_exec_03 --report reports/phase-P18.json`
- `P18_exec_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P18 --check P18_exec_04 --report reports/phase-P18.json`
- `P18_exec_05` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P18 --check P18_exec_05 --report reports/phase-P18.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `GO_END_TO_END` → `{'next': 'P19', 'set_flags': {'end_to_end_go': True}}`
- `STOP_END_TO_END` → `{'next': 'P19', 'set_flags': {'end_to_end_go': False}}`
- `INVALID_RUN` → `{'terminal': 'INVALID_RUN'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
