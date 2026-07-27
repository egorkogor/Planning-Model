# P06 — Planner implementations, independent implementation/statistics audits и Implementation lock

**Gate:** `G06_STATISTICAL_IMPLEMENTATION_AUDIT`
**Execution role:** `BUILDER`
**Approval mode:** `manual`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/architecture/planner_architecture_v1.yaml`
- `docs/architecture/a1_token_grammar_v1.yaml`
- `docs/architecture/task_encoding_v1.yaml`
- `docs/semantic/semantic_target_v1.yaml`
- `docs/training/planner_training_contract_v1.yaml`
- `docs/operator/implementation_lock_v1.yaml`
- `docs/audit/independent_audit_contract_v1.yaml`
- `docs/schemas/statistical_audit.schema.json`
- `docs/schemas/implementation_audit.schema.json`
- `docs/schemas/decision_record.schema.json`

## Действия
1. finish Planner A1–A5 implementations and all outcome-relevant runtime modules required through P17
2. prove P10 prototype-bank/parser path and P15 graph/control-certification path execute on toy/public fixtures
3. run full toy preflight from clean checkout
4. create implementation commit containing every protected executable path
5. obtain signed independent statistical audit and signed independent implementation audit of the same commit
6. create implementation-lock candidate binding both audits, Scientific lock, Trust Topology lock and reviewed commit
7. after G06 approval create and verify active Implementation lock; no later source implementation is permitted

## Обязательные результаты фазы
- `src/models/`
- `semantic_bank/bootstrap/`
- `reports/model-audit.json`
- `reports/compute-profile.json`
- `reports/preflight-final.json`
- `reports/self-review-P06.json`
- `reports/phase-P06.json`
- `RUN_STATUS.md`
- `RUN_STATUS.json`

## Обязательные входы фазы
- нет

## Результаты до ручного gate
- `reports/preflight-final.json`
- `reports/statistical-implementation-audit.json`
- `reports/independent-implementation-audit.json`
- `freezes/implementation-lock.candidate.json`

## Результаты после approval
- `locks/implementation.lock.json`

## Условные результаты после approval
- нет

## Проверки до gate
- `P06_pre_01` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_01 --report reports/phase-P06.json`
- `P06_pre_02` — implementation-lock candidate matches reviewed commit and policy; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_02 --report reports/phase-P06.json`
- `P06_pre_03` — clean-checkout full toy preflight: every arm forward/backward, two-batch overfit, serialization, fake episode and statistics golden cases PASS; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_03 --report reports/phase-P06.json`
- `P06_pre_04` — signed statistical audit is independent from Builder; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_04 --report reports/phase-P06.json`
- `P06_pre_05` — statistical audit covers every locked statistical check; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_05 --report reports/phase-P06.json`
- `P06_pre_06` — signed implementation audit is independent from Builder; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_06 --report reports/phase-P06.json`
- `P06_pre_07` — implementation audit covers oracle, generator, runtime checks, analysis builders, semantic resolver/prototype builder, control certification, data isolation, sealer, evaluator, model loading and persistence; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_07 --report reports/phase-P06.json`
- `P06_pre_08` — pre-gate evidence sealed; verifier: `python validation/phase_check_runner.py --phase P06 --check P06_pre_08 --report reports/phase-P06.json`

## Проверки исполнения
- нет

## Проверки после approval
- `P06_post_01` — externally signed DecisionRecord schema and signature valid
- `P06_post_02` — approved target hash matches reviewed Implementation-lock candidate; both signed audits remain sealed evidence
- `P06_post_03` — gate and decision ledgers updated
- `P06_post_04` — active Implementation lock created from approved candidate and VERIFIED; verifier: `python scripts/verify_implementation_lock.py`

## Условные проверки после approval
- нет

## Outcomes и переходы
- `APPROVE_G06_AUDITS` → `{'next': 'P07'}`
- `REJECT_G06_AUDITS` → `{'repeat': 'P06', 'max_resubmissions': 1}`
- `STOP` → `{'terminal': 'STOPPED_BY_OPERATOR'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
