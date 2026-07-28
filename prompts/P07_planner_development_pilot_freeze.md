# P07 — Planner development, pilot, N и freeze

**Gate:** `G07_PLANNER_CONFIRMATORY_FREEZE`
**Execution role:** `BUILDER`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/training/hyperparameter_search_v1.yaml`
- `docs/training/seed_selection_contract_v1.yaml`
- `docs/training/planner_training_contract_v1.yaml`
- `docs/statistics/statistics_contract_v1.yaml`
- `docs/controls/confirmatory_sealing_contract_v1.yaml`
- `docs/schemas/approved_freeze_pointer.schema.json`
- `docs/schemas/dispatch_record.schema.json`
- `docs/schemas/decision_record.schema.json`
- `docs/schemas/sealer_manifest.schema.json`
- `docs/schemas/experiment_freeze.schema.json`
- `validation/confirmatory_lineage_validator.py`
- `docs/training/planner_initialization_contract_v1.yaml`

## Действия
1. выполнить bounded hyperparameter grid на development
2. train six variants × five locked final seeds from the deterministic common-superset initialization; emit exactly 30 final training reports
3. рассчитать pilot N and freeze analysis
4. dispatch locked contracts and public-exclusion manifest to Data Sealer; Data Sealer generates a hidden 256-bit seed, returns only seed commitment, encrypted blob hash and signed sealer manifest; create candidate freeze
5. after external approval, create ApprovedFreezePointer and immutable evaluator dispatch linked to the signed decision
6. Data Sealer must emit an exact selected-task manifest before outcomes and bind its path, SHA-256 and task_count in the signed sealer manifest

## Обязательные результаты фазы
- `reports/planner-pilot.json`
- `reports/training/final/`
- `reports/sample-size-planner.json`
- `checkpoints/planner-seeds/manifest.json`
- `freezes/planner-confirmatory.candidate.json`
- `dispatch/sealer-planner.json`
- `sealed/planner-confirmatory/selected-task-manifest.json`
- `sealed/planner-confirmatory/sealer-manifest.json`
- `reports/self-review-P07.json`
- `reports/phase-P07.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- нет

## Результаты после approval
- нет

## Условные результаты после approval
- `APPROVE_FREEZE`:
  - `freezes/planner-confirmatory.approved.json`
  - `dispatch/evaluator-planner.json`

## Проверки до gate
- `P07_pre_01` — G06 independent statistical implementation audit APPROVED; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_01 --report reports/phase-P07.json`
- `P07_pre_02` — Data Sealer manifest verifies plaintext deletion; no confirmatory plaintext visible to Builder; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_02 --report reports/phase-P07.json`
- `P07_pre_03` — grid and seed selection reproducible; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_03 --report reports/phase-P07.json`
- `P07_pre_04` — reserve >= selected N; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_04 --report reports/phase-P07.json`
- `P07_pre_05` — candidate freeze valid; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_05 --report reports/phase-P07.json`
- `P07_pre_06` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_06 --report reports/phase-P07.json`
- `P07_pre_07` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_07 --report reports/phase-P07.json`
- `P07_pre_08` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_08 --report reports/phase-P07.json`
- `P07_pre_09` — exact 6 variants × 5 seeds final training evidence; step-12000 checkpoints, common initialization and ordered examples verified; verifier: `python validation/phase_check_runner.py --phase P07 --check P07_pre_09 --report reports/phase-P07.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P07_post_01` — externally signed DecisionRecord schema and signature valid
- `P07_post_02` — approved target hash matches sealed evidence
- `P07_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- `APPROVE_FREEZE`:
  - `P07_post_04` — externally signed DecisionRecord is bound to the Trust Topology operator key
  - `P07_post_05` — ApprovedFreezePointer hash and candidate/decision lineage verified
  - `P07_post_06` — immutable evaluator dispatch binds approved pointer and all active locks

## Outcomes и переходы
- `APPROVE_FREEZE` → `{'next': 'P08'}`
- `REJECT_FREEZE` → `{'repeat': 'P07', 'max_resubmissions': 1}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
