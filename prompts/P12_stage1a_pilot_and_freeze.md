# P12 — Stage 1A pilot, N и freeze

**Gate:** `G12_STAGE1A_CONFIRMATORY_FREEZE`
**Execution role:** `BUILDER`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/statistics/statistics_contract_v1.yaml`
- `docs/controls/confirmatory_sealing_contract_v1.yaml`
- `docs/controls/intent_control_contract_v1.yaml`
- `docs/prompt/stage1_prompt_v1.yaml`
- `docs/schemas/approved_freeze_pointer.schema.json`
- `docs/schemas/dispatch_record.schema.json`
- `docs/schemas/decision_record.schema.json`
- `docs/schemas/sealer_manifest.schema.json`
- `docs/schemas/experiment_freeze.schema.json`
- `validation/confirmatory_lineage_validator.py`

## Действия
1. build oracle snapshots and pre-outcome controls on pilot/development only
2. run paired dry run and pilot
3. calculate clustered N including equivalence requirement
4. dispatch locked Stage 1A contracts and public-exclusion manifest to Data Sealer; hidden seed is generated inside the sealer environment
5. receive only encrypted blob commitment and sealer manifest; create candidate freeze
6. after external approval, create ApprovedFreezePointer and immutable evaluator dispatch linked to the signed decision

## Обязательные результаты фазы
- `controls/stage1a-certification/`
- `reports/stage1a-pilot.json`
- `reports/sample-size-stage1a.json`
- `freezes/stage1a-confirmatory.candidate.json`
- `dispatch/sealer-stage1a.json`
- `sealed/stage1a-confirmatory/sealer-manifest.json`
- `reports/self-review-P12.json`
- `reports/phase-P12.json`
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
  - `freezes/stage1a-confirmatory.approved.json`
  - `dispatch/evaluator-stage1a.json`

## Проверки до gate
- `P12_pre_01` — Data Sealer manifest verifies plaintext deletion; no Stage 1A confirmatory plaintext visible to Builder; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_01 --report reports/phase-P12.json`
- `P12_pre_02` — all four arms pair-complete; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_02 --report reports/phase-P12.json`
- `P12_pre_03` — prompt totals equal naturally without right padding; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_03 --report reports/phase-P12.json`
- `P12_pre_04` — reserve >= selected N; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_04 --report reports/phase-P12.json`
- `P12_pre_05` — candidate freeze valid; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_05 --report reports/phase-P12.json`
- `P12_pre_06` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_06 --report reports/phase-P12.json`
- `P12_pre_07` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_07 --report reports/phase-P12.json`
- `P12_pre_08` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P12 --check P12_pre_08 --report reports/phase-P12.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P12_post_01` — externally signed DecisionRecord schema and signature valid
- `P12_post_02` — approved target hash matches sealed evidence
- `P12_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- `APPROVE_FREEZE`:
  - `P12_post_04` — externally signed DecisionRecord is bound to the Trust Topology operator key
  - `P12_post_05` — ApprovedFreezePointer hash and candidate/decision lineage verified
  - `P12_post_06` — immutable evaluator dispatch binds approved pointer and all active locks

## Outcomes и переходы
- `APPROVE_FREEZE` → `{'next': 'P13'}`
- `REJECT_FREEZE` → `{'repeat': 'P12', 'max_resubmissions': 1}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
