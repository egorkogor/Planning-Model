# P03 — Runtime contracts, persistence и harness preflight

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
- `docs/operator/implementation_lock_v1.yaml`
- `docs/schemas/implementation_patch.schema.json`
- `docs/operator/report_registry_v1.yaml`
- `docs/validation_contract.md`
- `docs/schemas/`
- `validation/verify_gate.py`
- `docs/operator/phase_check_contract_v1.yaml`

## Действия
1. implement schemas, strict IO and runtime validators
2. implement locked phase runtime checker without trusting phase-report PASS claims
3. implement analysis/build_analysis_input.py and analysis/build_sample_size_input.py with reproducibility modes
4. implement semantic resolver parser, alias handling and train-only prototype-bank builder used later in P10
5. implement reachable-state graph and control-certification engine used later in P15
6. implement deterministic storage, model loading, evaluator and sealer interfaces needed through P17
7. run toy end-to-end preflight and record exact implementation patches only when fixing implementation defects

## Обязательные результаты фазы
- `src/contracts/`
- `tests/contracts/`
- `src/validation/phase_checks.py`
- `tests/runtime/`
- `src/runtime/`
- `src/sealing/`
- `src/evaluation/`
- `src/controls/`
- `src/semantic/`
- `analysis/build_analysis_input.py`
- `analysis/build_sample_size_input.py`
- `reports/preflight-toy.json`
- `reports/implementation-patches/index.json`
- `reports/self-review-P03.json`
- `reports/phase-P03.json`
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
- `P03_exec_01` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_01 --report reports/phase-P03.json`
- `P03_exec_02` — exact required-output and check-set gate adversarial tests pass; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_02 --report reports/phase-P03.json`
- `P03_exec_03` — contract-level toy fixture содержит 20–50 synthetic задач с n=3; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_03 --report reports/phase-P03.json`
- `P03_exec_04` — stub interfaces и все нормативные artifacts проходят serialization round-trip; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_04 --report reports/phase-P03.json`
- `P03_exec_05` — atomic persistence/recovery и полный fake episode проходят без реальных model outcomes; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_05 --report reports/phase-P03.json`
- `P03_exec_06` — statistics and sample-size golden run pass; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_06 --report reports/phase-P03.json`
- `P03_exec_07` — every implementation-only patch has a schema-valid diff record; verifier: `python validation/phase_check_runner.py --phase P03 --check P03_exec_07 --report reports/phase-P03.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P04'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
