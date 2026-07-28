# Validation contract v1.19

Validation состоит из восьми обязательных уровней. Отчёт агента не является доказательством сам по себе: каждый машинный check повторно исполняется locked verifier.

## 1. Local schema validation

Каждый JSON/JSONL до записи сопоставляется с `docs/operator/report_registry_v1.yaml`. Незарегистрированный JSON запрещён. JSONL валидируется построчно.

## 2. Semantic contract validation

Проверяются exact enums, thresholds, identities, signatures, hashes и cross-field conditions. Простого YAML/JSON parsing недостаточно.

## 3. Cross-object validation

Проверяются:

- ledger/ref/domain/goal invariants;
- task/split/corpus manifests и leakage;
- PlannerStep/semantic artifact/checkpoint/state hashes;
- prompt aliases, guidance budget и left-padding policy;
- stage/arm/candidate source и terminal flows;
- pair completeness и attempt→episode aggregation;
- freeze→DecisionRecord→ApprovedFreezePointer→evaluator manifest lineage;
- raw results→canonical `AnalysisInput`→scientific decision.

## 4. Machine check rerun

Все `pre_gate_checks` и `execution_checks` из `phase_registry_v1.yaml` имеют неизменяемый verifier:

```bash
python validation/phase_check_runner.py --phase PXX --check <check_id> --report reports/phase-PXX.json
```

P00–P02 и общие lock checks реализованы в trust-root runner. Остальные checks вызывают `src.validation.phase_checks`, который создаётся в P03, входит в Implementation lock и возвращает schema-valid `MachineCheckResult`. Поле `status=PASS` в phase report без успешного rerun не принимается.

## 5. Two-level locks

- До P00 `release/BOOTSTRAP_MANIFEST.json` защищает только Scientific trust root и сами verifier-компоненты.
- В P02 создаётся `locks/scientific.lock.json`. Он фиксирует hypotheses, arms, splits, seeds, thresholds, statistics, gates, role isolation и lock policy.
- Между P02 и G06 разрешены только implementation-only patches: exact Git diff, неизменный Scientific lock, PASS toy preflight и отсутствие pilot outcomes.
- После одобрения G06 создаётся `locks/implementation.lock.json`, фиксирующий schemas, serializers, validators, architecture wiring, prompt rendering и analysis code.

Scientific change в текущем run запрещён. Implementation change после G06 запрещён. Оба требуют новой версии протокола и нового run.

## 6. Manual gate integrity

`verify_gate.py` проверяет:

- точный набор checks и required outputs;
- rerun каждого locked verifier;
- artifact hashes, schemas и command evidence;
- state-machine outcome и `next_phase`;
- execution role identity;
- exact approval target hash;
- DecisionRecord self-hash;
- согласованные `gate-ledger.jsonl` и `decision-log.jsonl`;
- clean tree и commit lineage.

G01 обязателен: переход из P01 всегда требует внешне подписанный DecisionRecord и проверенный Trust Topology lock. Автоматический обход бюджетного решения запрещён.

## 7. Confirmatory blindness и signatures

Validator требует Data Sealer, Evaluation Runner и Audit Agent с разными credentials/environments. Confirmatory plaintext и selection seed не доступны Builder: Data Sealer генерирует seed внутри своей среды, публикует только commitment и зашифрованный для Auditor seed envelope. Sealer и evaluator manifests подписываются Ed25519 ключами, связанными с locked role identities. Sealer обязан приложить hashes seed envelope, workspace scan, process log и destroyed-volume evidence.

Single-agent/unsealed execution получает `INVALID_CONFIRMATORY_BLINDNESS` независимо от metric.

## 8. Statistics и independent audit

До первого confirmatory dispatch G06 требует независимый review статистической реализации человеком-статистиком либо моделью другой family относительно Builder. Canonical `AnalysisInput` хранит pair-level rows; валидатор пересчитывает differences, hierarchical bootstrap, clustered bootstrap, non-inferiority and direction gates, sample size и решения из locked definitions. P19 повторяет tests, statistics и выполненные experiment stages на clean checkout.
