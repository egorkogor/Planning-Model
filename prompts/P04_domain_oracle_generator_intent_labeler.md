# P04 — BlocksWorld, Oracle, generator и Intent Labeler

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/domain/blocks_world_v1.yaml`
- `docs/domain/generator_contract_v1.yaml`
- `docs/domain/intent_labeler_contract_v1.yaml`
- `docs/domain/intent_labeler_v1.py`
- `docs/domain/intent_catalog_v1.yaml`

## Действия
1. реализовать structured domain registry
2. реализовать deterministic BFS oracle и canonicalizer
3. реализовать generator по split contract
4. проверить исполняемый intent labeler exhaustive n<=5

## Обязательные результаты фазы
- `src/domain/`
- `data/manifests/generator-smoke.json`
- `data/manifests/intent-labeler-exhaustive.json`
- `reports/domain-audit.json`
- `reports/self-review-P04.json`
- `reports/phase-P04.json`
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
- `P04_exec_01` — domain property tests pass; verifier: `python validation/phase_check_runner.py --phase P04 --check P04_exec_01 --report reports/phase-P04.json`
- `P04_exec_02` — intent labeler manifest complete; verifier: `python validation/phase_check_runner.py --phase P04 --check P04_exec_02 --report reports/phase-P04.json`
- `P04_exec_03` — generator byte reproducible; verifier: `python validation/phase_check_runner.py --phase P04 --check P04_exec_03 --report reports/phase-P04.json`
- `P04_exec_04` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P04 --check P04_exec_04 --report reports/phase-P04.json`
- `P04_exec_05` — all implementation-only changes have patch records and repeated toy preflight; verifier: `python validation/phase_check_runner.py --phase P04 --check P04_exec_05 --report reports/phase-P04.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P05'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
