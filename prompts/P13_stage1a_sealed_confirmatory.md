# P13 — Stage 1A sealed confirmatory execution

**Gate:** `нет`
**Execution role:** `EVALUATION_RUNNER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/controls/confirmatory_sealing_contract_v1.yaml`
- `docs/schemas/evaluator_result_manifest.schema.json`
- `docs/schemas/attempt_log.schema.json`
- `docs/schemas/approved_freeze_pointer.schema.json`

## Действия
1. dispatch approved freeze to evaluator
2. run I0/I1/I2/I3 once per snapshot
3. save raw bytes before parsing
4. collect signed evaluator manifest

## Обязательные результаты фазы
- `results/stage1a-confirmatory/`
- `reports/stage1a-confirmatory-run.json`
- `results/stage1a-confirmatory/evaluator-result-manifest.json`
- `reports/self-review-P13.json`
- `reports/phase-P13.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- `freezes/stage1a-confirmatory.approved.json`
- `dispatch/evaluator-stage1a.json`

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- нет

## Проверки исполнения
- `P13_exec_01` — pair completeness within threshold; verifier: `python validation/phase_check_runner.py --phase P13 --check P13_exec_01 --report reports/phase-P13.json`
- `P13_exec_02` — no retry or domain mask; verifier: `python validation/phase_check_runner.py --phase P13 --check P13_exec_02 --report reports/phase-P13.json`
- `P13_exec_03` — contract/hash violations zero; verifier: `python validation/phase_check_runner.py --phase P13 --check P13_exec_03 --report reports/phase-P13.json`
- `P13_exec_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P13 --check P13_exec_04 --report reports/phase-P13.json`
- `P13_exec_05` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P13 --check P13_exec_05 --report reports/phase-P13.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P14'}`
- `INVALID_CONFIRMATORY` → `{'terminal': 'INVALID_RUN'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
