# P10 — Frozen LLM и semantic resolver

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/semantic/semantic_resolver_v1.yaml`
- `docs/prompt/guidance_artifact_contract_v1.yaml`
- `docs/prompt/stage1_prompt_v1.yaml`
- `docs/schemas/model_lock.schema.json`

## Действия
1. verify Scientific, Trust Topology and Implementation locks
2. load exact locked Qwen revision using locked model loader
3. run the locked train-only prototype-bank builder; do not add or modify parser/resolver code
4. run the locked strict parser and alias resolver on deterministic golden prompts
5. persist only generated semantic-bank manifests and determinism reports

## Обязательные результаты фазы
- `semantic_bank/manifest.json`
- `reports/llm-determinism.json`
- `reports/self-review-P10.json`
- `reports/phase-P10.json`
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
- `P10_exec_01` — 20/20 deterministic outputs on golden prompts; verifier: `python validation/phase_check_runner.py --phase P10 --check P10_exec_01 --report reports/phase-P10.json`
- `P10_exec_02` — silent repair absent; verifier: `python validation/phase_check_runner.py --phase P10 --check P10_exec_02 --report reports/phase-P10.json`
- `P10_exec_03` — prototype hashes reproducible; verifier: `python validation/phase_check_runner.py --phase P10 --check P10_exec_03 --report reports/phase-P10.json`
- `P10_exec_04` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P10 --check P10_exec_04 --report reports/phase-P10.json`
- `P10_exec_05` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P10 --check P10_exec_05 --report reports/phase-P10.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P11'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
