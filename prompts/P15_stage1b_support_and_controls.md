# P15 — Stage 1B task-only selection and pre-outcome artifact certification run

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
- `docs/controls/full_plan_shuffle_contract_v1.yaml`
- `docs/controls/random_codebook_contract_v1.yaml`
- `docs/controls/confirmatory_sealing_contract_v1.yaml`

## Действия
1. verify Scientific, Trust Topology and Implementation locks
2. run the locked task/domain/split eligibility checker on public and pilot tasks
3. verify no Planner, semantic, LLM, shuffle-degeneracy or arm outcome can affect inclusion
4. verify A3r codebook, signature bank, shuffle algorithm and failure-retention policy are frozen before hidden selection
5. materialize public exclusion and task-only support manifests only; source-code changes are forbidden
6. do not materialize hidden confirmatory task ids or hidden task bodies in Builder environment

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
- `P15_exec_01` — selection predicates use task/domain/split metadata only; post-treatment exclusion impossible; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_01 --report reports/phase-P15.json`
- `P15_exec_02` — all public/pilot included tasks satisfy task-only eligibility; no plan/control degeneracy exclusion; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_02 --report reports/phase-P15.json`
- `P15_exec_03` — pre-outcome A3r codebook, signature bank, shuffle algorithm and retention policy are frozen; verifier: `python validation/phase_check_runner.py --phase P15 --check P15_exec_03 --report reports/phase-P15.json`
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
