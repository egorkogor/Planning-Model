# P16 — Stage 1B pilot, N и freeze

**Gate:** `G16_STAGE1B_CONFIRMATORY_FREEZE`
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
- `docs/schemas/approved_freeze_pointer.schema.json`
- `docs/schemas/dispatch_record.schema.json`
- `docs/schemas/decision_record.schema.json`
- `docs/schemas/sealer_manifest.schema.json`
- `docs/schemas/experiment_freeze.schema.json`
- `validation/confirmatory_lineage_validator.py`

## Действия
1. run independent trajectories pilot
2. measure unresolved and infrastructure rates
3. calculate final N with all registered components
4. dispatch locked Stage 1B task-only eligibility contracts, pre-outcome artifact hashes and public-exclusion manifest to Data Sealer
5. Data Sealer fixes eligibility from task/domain/split metadata only, then HMAC-ranks and encrypts the selected tasks before any Planner or LLM output exists
6. receive only encrypted blob commitment and signed sealer manifest containing task-only selection certification; create candidate freeze
7. after external approval, create ApprovedFreezePointer and immutable evaluator dispatch linked to the signed decision

## Обязательные результаты фазы
- `reports/stage1b-pilot.json`
- `reports/sample-size-stage1b.json`
- `freezes/stage1b-confirmatory.candidate.json`
- `dispatch/sealer-stage1b.json`
- `sealed/stage1b-confirmatory/sealer-manifest.json`
- `reports/self-review-P16.json`
- `reports/phase-P16.json`
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
  - `freezes/stage1b-confirmatory.approved.json`
  - `dispatch/evaluator-stage1b.json`

## Проверки до gate
- `P16_pre_01` — signed Data Sealer manifest verifies plaintext deletion and task-only Stage1B selection certification; no hidden task body or id visible to Builder; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_01 --report reports/phase-P16.json`
- `P16_pre_02` — no total-token equality claim across trajectories; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_02 --report reports/phase-P16.json`
- `P16_pre_03` — same 32-token guidance budget; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_03 --report reports/phase-P16.json`
- `P16_pre_04` — reserve >= selected N; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_04 --report reports/phase-P16.json`
- `P16_pre_05` — candidate freeze valid; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_05 --report reports/phase-P16.json`
- `P16_pre_06` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_06 --report reports/phase-P16.json`
- `P16_pre_07` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_07 --report reports/phase-P16.json`
- `P16_pre_08` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P16 --check P16_pre_08 --report reports/phase-P16.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P16_post_01` — externally signed DecisionRecord schema and signature valid
- `P16_post_02` — approved target hash matches sealed evidence
- `P16_post_03` — gate and decision ledgers updated

## Условные проверки после approval
- `APPROVE_FREEZE`:
  - `P16_post_04` — externally signed DecisionRecord is bound to the Trust Topology operator key
  - `P16_post_05` — ApprovedFreezePointer hash and candidate/decision lineage verified
  - `P16_post_06` — immutable evaluator dispatch binds approved pointer and all active locks

## Outcomes и переходы
- `APPROVE_FREEZE` → `{'next': 'P17'}`
- `REJECT_FREEZE` → `{'repeat': 'P16', 'max_resubmissions': 1}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
