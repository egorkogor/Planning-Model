# P11 — Prompt development и final prompt freeze

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/prompt/prompt_development_contract_v1.yaml`
- `docs/prompt/candidates/`
- `docs/prompt/stage1_prompt_v1.yaml`
- `docs/prompt/guidance_artifact_contract_v1.yaml`

## Действия
1. run fixed prompt candidate grid on development only
2. measure format/valid/progress floors
3. select by preregistered score
4. freeze rendered template and tokenizer artifacts

## Обязательные результаты фазы
- `reports/prompt-development.json`
- `artifacts/golden-prompts/`
- `locks/stage1-prompt.lock.json`
- `reports/self-review-P11.json`
- `reports/phase-P11.json`
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
- `P11_exec_01` — selected candidate meets all floors; verifier: `python validation/phase_check_runner.py --phase P11 --check P11_exec_01 --report reports/phase-P11.json`
- `P11_exec_02` — no confirmatory data accessed; verifier: `python validation/phase_check_runner.py --phase P11 --check P11_exec_02 --report reports/phase-P11.json`
- `P11_exec_03` — left padding only for batching; unpadded prompt hashes frozen; verifier: `python validation/phase_check_runner.py --phase P11 --check P11_exec_03 --report reports/phase-P11.json`
- `P11_exec_04` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P11 --check P11_exec_04 --report reports/phase-P11.json`
- `P11_exec_05` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P11 --check P11_exec_05 --report reports/phase-P11.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P12'}`
- `BLOCKED_PROMPT_STACK` → `{'mark_skipped': ['P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18'], 'next': 'P19'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
