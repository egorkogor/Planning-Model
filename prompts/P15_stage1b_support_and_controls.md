# P15 — Stage 1B public support certification run

**Gate:** `нет`
**Execution role:** `BUILDER`
**Approval mode:** `auto`

## Источники истины
- `docs/operator/agent_execution_contract_v1.yaml`
- `docs/operator/phase_state_machine_v1.yaml`
- `docs/operator/phase_registry_v1.yaml`
- `docs/operator/report_registry_v1.yaml`
- `docs/operator/self_review_loop_v1.yaml`
- `docs/controls/intent_control_contract_v1.yaml`
- `docs/semantic/semantic_resolver_v1.yaml`
- `docs/data/dataset_split_contract_v1.yaml`

## Действия
1. verify Scientific, Trust Topology and Implementation locks
2. run the locked reachable-state graph and control-certification engine on public/pilot tasks
3. verify post-treatment exclusion is impossible under the locked contract
4. materialize public exclusion/support manifests only; source-code changes are forbidden
5. do not materialize hidden confirmatory task ids or hidden task bodies in Builder environment

## Обязательные результаты фазы
- `controls/stage1b-certification-preflight/`
- `reports/stage1b-support-audit-public.json`
- `data/manifests/stage1b-public-exclusions.json`
- `reports/self-review-P15.json`
- `reports/phase-P15.json`
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
- `P15_exec_01` — post-treatment exclusion impossible; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_01 --report reports/phase-P15.json`
- `P15_exec_02` — all included tasks have total control coverage; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_02 --report reports/phase-P15.json`
- `P15_exec_03` — support threshold passes; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_03 --report reports/phase-P15.json`
- `P15_exec_04` — implementation lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_04 --report reports/phase-P15.json`
- `P15_exec_05` — scientific lock VERIFIED; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_05 --report reports/phase-P15.json`

## Проверки после approval
- нет

## Условные проверки после approval
- нет

## Outcomes и переходы
- `PASS` → `{'next': 'P16'}`
- `BLOCKED_SUPPORT` → `{'mark_skipped': ['P16', 'P17', 'P18'], 'next': 'P19'}`
- `BLOCKED` → `{'terminal': 'BLOCKED'}`

## Исполнительские правила
- Выполнить только действия этой фазы и не читать outcome-данные следующих sealed фаз.
- Не менять Scientific lock. Implementation-only patch разрешён только в окне и по контракту `implementation_lock_v1.yaml`.
- Зафиксировать команды, exit codes, stdout/stderr, hashes, resource usage и self-review в phase report.
- Не объявлять PASS без прохождения machine verifier `validation/verify_gate.py`.
- Максимум внутренних попыток исправления: 2.
