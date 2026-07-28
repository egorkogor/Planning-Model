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
- `docs/schemas/episode_plan_manifest.schema.json`
- `docs/schemas/full_plan_lineage_index.schema.json`
- `docs/schemas/checkpoint_manifest.schema.json`
- `docs/training/planner_training_contract_v1.yaml`

## Действия
1. dispatch approved freeze to separate evaluator
2. run the exact matrix `selected task × seeds 101/202/303/404/505 × A1/A2/A2b/A2c/A3/A3r/A4/A5/A2C_FLOPS/A3_FLOPS/P_FULL_PLAN_REPLAY_RAW` once per cell; duplicates and omissions are invalid
3. collect signed raw result manifest
4. validate no source mutation
5. verify lineage covers the exact selected-task set, primary and FLOPs-sensitivity seed/arm matrix, and every nested artifact has the same run_id, PLANNER stage and planner_seed
6. bind every Planner result to the canonical P07 training report and canonical checkpoint manifest path; arbitrary checkpoint link JSON is forbidden

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
- `P08_exec_06` — exact task × five-seed × 11-arm matrix is complete; primary A3 WorkPlan is generated once per task/seed and replayed without replanning; A2c/A3 FLOPs arms use their locked sensitivity checkpoints; verifier: `python validation/phase_check_runner.py --phase P08 --check P08_exec_06 --report reports/phase-P08.json`

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
