# P08 — Planner sealed confirmatory execution

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
- `docs/schemas/dispatch_record.schema.json`
- `docs/schemas/evaluator_result_manifest.schema.json`
- `docs/schemas/approved_freeze_pointer.schema.json`

## Действия
1. dispatch approved freeze to separate evaluator
2. run all frozen arms once
3. collect signed raw result manifest
4. validate no source mutation

## Обязательные результаты фазы
- `results/planner-confirmatory/`
- `reports/planner-confirmatory-run.json`
- `results/planner-confirmatory/evaluator-result-manifest.json`
- `reports/self-review-P08.json`
- `reports/phase-P08.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- `freezes/planner-confirmatory.approved.json`
- `dispatch/evaluator-planner.json`

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- нет

## Проверки до gate
- нет

## Проверки исполнения
- `P08_exec_01` — evaluator commit equals freeze; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_01 --report reports/phase-P08.json`
- `P08_exec_02` — signed manifest valid; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_02 --report reports/phase-P08.json`
- `P08_exec_03` — contract/hash violations zero; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_03 --report reports/phase-P08.json`
- `P08_exec_04` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_04 --report reports/phase-P08.json`
- `P08_exec_05` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_05 --report reports/phase-P08.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P09'}`
- `INVALID_CONFIRMATORY` → `{'terminal': 'INVALID_RUN'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
