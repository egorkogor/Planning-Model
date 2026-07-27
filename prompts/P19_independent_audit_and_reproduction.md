# P19 — Independent audit and clean reproduction

**Gate:** `нет`
**Execution role:** `AUDITOR`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/audit/independent_audit_contract_v1.yaml`
- `docs/hash_protocol.md`
- `docs/operator/self_review_loop_v1.yaml`
- `validation/verify_release_manifest.py`
- `docs/schemas/audit_report.schema.json`
- `validation/verify_gate.py`

## Действия
1. derive audit profile from actual phase-state-machine path: PLANNER_STOP, INTERFACE_STOP, or END_TO_END
2. run audit agent on separate clean checkout
3. validate every registered artifact
4. recompute all statistics
5. reproduce one planner and one Stage1 split from manifests

## Обязательные результаты фазы
- `reports/final-audit.json`
- `reports/final-report.md`
- `release/SHA256SUMS.txt`
- `release/reproducibility-bundle.tar.gz`
- `reports/self-review-P19.json`
- `reports/phase-P19.json`
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
- `P19_exec_01` — clean reproduction mandatory and PASS; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_01 --report reports/phase-P19.json`
- `P19_exec_02` — all reported metrics exact; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_02 --report reports/phase-P19.json`
- `P19_exec_03` — raw logs and lineage complete; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_03 --report reports/phase-P19.json`
- `P19_exec_04` — claims within scope; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_04 --report reports/phase-P19.json`
- `P19_exec_05` — audit profile reproduces only stages that actually executed; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_05 --report reports/phase-P19.json`
- `P19_exec_06` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_06 --report reports/phase-P19.json`
- `P19_exec_07` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P19 --check P19_exec_07 --report reports/phase-P19.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P20'}`
- `FAIL_AUDIT` → `{'terminal': 'INVALID_RUN'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
