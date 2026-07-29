# Checklist оператора

На каждом gate проверь:

1. `STATUS=WAITING_APPROVAL`, working tree clean.
2. Trust Topology lock = VERIFIED; Scientific/Implementation locks = VERIFIED, когда уже активны; pre-gate checks PASS.
3. Target hash совпадает в phase report, gate ledger и показанном artifact.
4. Для confirmatory: blindness `SEALED`, builder не имеет plaintext/key, evaluator environment указан.
5. Стоимость и риски не превышают утверждённый resource plan.
6. Агент дал ровно допустимую команду и не просит технический выбор.

## G00_SCOPE

Подтверждает только область архивного эксперимента и protocol v1.21/v2.21.

```text
APPROVE G00_SCOPE <scope_sha256>
REJECT G00_SCOPE <reason>
STOP G00_SCOPE
```

## G01_TRUST_AND_RESOURCES

Обязательный gate независимо от стоимости. Проверить, что `locks/trust-topology.lock.json` подписан внешним операторским Ed25519-ключом и связывает exact hashes `resource-plan`, `infrastructure-plan` и `public-keys`. Private key не должен быть доступен Builder.

```text
APPROVE G01_TRUST_AND_RESOURCES <trust_topology_lock_sha256>
REJECT G01_TRUST_AND_RESOURCES <reason>
STOP G01_TRUST_AND_RESOURCES
```


## G06_STATISTICAL_IMPLEMENTATION_AUDIT

Проверить два подписанных независимых APPROVE-отчёта по одному implementation commit. Implementation audit обязан покрывать analysis builders, resolver/prototype builder, control certification, sealer/evaluator, model loading, persistence, oracle и generator. Candidate должен связывать оба аудита, Scientific lock и Trust Topology lock.

```text
APPROVE G06_STATISTICAL_IMPLEMENTATION_AUDIT <implementation_candidate_sha256>
REJECT G06_STATISTICAL_IMPLEMENTATION_AUDIT <reason>
STOP G06_STATISTICAL_IMPLEMENTATION_AUDIT
```

## G07_PLANNER_CONFIRMATORY_FREEZE

Проверить sealed dataset commitment, N≤reserve, exact config/seeds/analysis hashes и evaluator boundary.

```text
APPROVE G07_PLANNER_CONFIRMATORY_FREEZE <freeze_sha256>
REJECT G07_PLANNER_CONFIRMATORY_FREEZE <reason>
STOP G07_PLANNER_CONFIRMATORY_FREEZE
```

## G12_STAGE1A_CONFIRMATORY_FREEZE

Проверить prompt lock, snapshot/control manifest, clustered N and diagnostic/core gate separation и sealed evaluator dispatch.

```text
APPROVE G12_STAGE1A_CONFIRMATORY_FREEZE <freeze_sha256>
REJECT G12_STAGE1A_CONFIRMATORY_FREEZE <reason>
STOP G12_STAGE1A_CONFIRMATORY_FREEZE
```

## G16_STAGE1B_CONFIRMATORY_FREEZE

Проверить task-only eligibility manifest, нулевое использование Planner/LLM outputs при selection, сохранение degeneracy/failures как paired outcomes и reserve capacity.

```text
APPROVE G16_STAGE1B_CONFIRMATORY_FREEZE <freeze_sha256>
REJECT G16_STAGE1B_CONFIRMATORY_FREEZE <reason>
STOP G16_STAGE1B_CONFIRMATORY_FREEZE
```

## G20_FINAL_ACCEPTANCE

Проверить P19 audit PASS, обязательный clean reproduction, checksums, raw-log completeness и scope-safe claims.

```text
APPROVE G20_FINAL_ACCEPTANCE <release_bundle_sha256>
REJECT G20_FINAL_ACCEPTANCE <reason>
STOP G20_FINAL_ACCEPTANCE
```

REJECT допускает одну resubmission. Изменять protected contracts в resubmission нельзя.
