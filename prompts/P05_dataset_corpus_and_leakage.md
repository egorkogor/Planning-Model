# P05 — Dataset, suffix/off-policy corpus и leakage

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/data/dataset_split_contract_v1.yaml`
- `docs/domain/training_corpus_contract_v1.yaml`
- `docs/schemas/training_example.schema.json`
- `docs/schemas/corpus_manifest.schema.json`
- `docs/controls/random_codebook_contract_v1.yaml`
- `docs/schemas/semantic_signature_bank.schema.json`
- `docs/schemas/random_codebook_manifest.schema.json`

## Действия
1. назначить base_task partitions до expansion
2. построить oracle suffix и off-policy corpus
3. валидировать support signatures и quotas
4. заморозить corpus manifests

## Обязательные результаты фазы
- `reports/dataset-capacity-audit.json`
- `semantic_bank/random-codebook/manifest.json`
- `semantic_bank/signatures/manifest.json`
- `data/manifests/training-corpus.json`
- `reports/leakage.json`
- `reports/support-coverage.json`
- `reports/self-review-P05.json`
- `reports/phase-P05.json`
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
- `P05_exec_01` — dataset capacity audit proves every quota feasible; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_01 --report reports/phase-P05.json`
- `P05_exec_02` — split intersections zero; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_02 --report reports/phase-P05.json`
- `P05_exec_03` — all quotas met or protocol capacity BLOCKED; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_03 --report reports/phase-P05.json`
- `P05_exec_04` — holding and hand-empty coverage present; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_04 --report reports/phase-P05.json`
- `P05_exec_05` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_05 --report reports/phase-P05.json`
- `P05_exec_06` — all implementation-only changes have patch records and repeated toy preflight; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_06 --report reports/phase-P05.json`
- `P05_exec_07` — train-only semantic signature bank and deterministic A3r random codebook regenerate exactly; verifier: `python validation/phase_check_runner.py --phase P05 --check P05_exec_07 --report reports/phase-P05.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P06'}`
- `BLOCKED_PROTOCOL_CAPACITY` → `{'terminal': 'BLOCKED_PROTOCOL_CAPACITY'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
