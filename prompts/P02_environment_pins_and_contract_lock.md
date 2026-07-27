# P02 — Environment, immutable pins и scientific lock

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/operator/scientific_lock_v1.yaml`
- `docs/operator/bootstrap_integrity_contract_v1.yaml`
- `docs/infrastructure/runtime_dependency_contract_v1.yaml`
- `docs/schemas/model_lock.schema.json`
- `validation/verify_lock.py`
- `docs/schemas/runtime_stack_lock.schema.json`
- `docs/operator/trust_topology_lock_v1.yaml`
- `validation/trust_topology_validator.py`

## Действия
1. verify externally signed trust topology lock before reading role-bound credentials
2. создать environment/runtime stack lock
3. разрешить immutable model revisions and offline cache manifests
4. создать outcome-relevant Scientific lock after all runtime/model/trust artifacts exist
5. проверить bootstrap manifest и неизменяемые trust/lock verifiers

## Обязательные результаты фазы
- `locks/environment.lock.json`
- `locks/llm_model_lock.json`
- `locks/semantic_target_model_lock.json`
- `locks/scientific.lock.json`
- `reports/self-review-P02.json`
- `reports/phase-P02.json`
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
- `P02_exec_01` — model revisions immutable; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_01 --report reports/phase-P02.json`
- `P02_exec_02` — bundle tests pass; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_02 --report reports/phase-P02.json`
- `P02_exec_03` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_03 --report reports/phase-P02.json`
- `P02_exec_04` — lock verifier files covered by bootstrap manifest; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_04 --report reports/phase-P02.json`
- `P02_exec_05` — runtime stack lock is immutable, offline-capable and bound to model locks; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_05 --report reports/phase-P02.json`
- `P02_exec_06` — externally signed trust topology lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P02 --check P02_exec_06 --report reports/phase-P02.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P03'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
