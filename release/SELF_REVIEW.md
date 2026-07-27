# Release self-review v1.13 / v2.13

## Scope

Проверены документы, YAML/JSON contracts, validators, state machine, prompts, lock policies и packaging. Обучение A1–A5, Qwen inference, pilot и confirmatory не выполнялись.

## Trust boundary

- P01 всегда создаёт обязательный Trust Topology lock.
- Lock подписывается операторским Ed25519-ключом вне репозитория и Builder environment.
- Валидатор отклоняет private или public trust-root key, размещённый внутри repository root.
- Operator attestation фиксирует out-of-band identity verification, разные environment identities и credential principals, а также отсутствие private key у Builder.
- Scientific lock хеширует Trust Topology lock, resource/infrastructure plans, public-key registry и runtime/model manifests.

## Lock ordering

- P02 фиксирует scientific contracts и доверенную runtime topology.
- P03 реализует весь outcome-relevant executable code.
- G06 получает два независимых подписанных аудита одного implementation commit.
- После APPROVE G06 создаётся Implementation lock на `src/**`, `analysis/**`, validators, scripts и tests.
- P10/P15 только запускают заблокированный код и не могут добавлять parser, prototype builder или control-certification logic.

## Independent review

Implementation audit содержит 12 обязательных checks, включая:

- oracle/generator/runtime checkers;
- dataset split/leakage;
- analysis input builders;
- semantic resolver и prototype builder;
- control-certification engine;
- sealer/evaluator boundary;
- model loading/runtime pins;
- persistence/hashing и clean reproduction.

Implementation-lock candidate связывает оба аудита, Trust Topology lock, Scientific lock и один reviewed commit.

## Entry points и status

- `validation/verify_gate.py` — единственная canonical реализация gate verification.
- `scripts/verify_gate.py`, `scripts/verify_trust_topology.py` и `scripts/phase_check_runner.py` — thin wrappers.
- Trust Topology, Scientific и Implementation statuses разделены и семантически сверяются с фазой.

## Adversarial coverage

Проверяется, что:

- изменение resource plan, public keys или runtime/model manifest ломает lock;
- P03 analysis builders можно создать после Scientific lock;
- operator key внутри репозитория отклоняется;
- поздние фазы не реализуют новый outcome-relevant code;
- compatibility entrypoints загружают canonical modules.

## Ограничения

Release делает плейбук исполнимым, но не доказывает корректность ещё не созданной реализации Planner и не является результатом эксперимента. Организационная независимость ролей требует фактически отдельных principals/environments и внешнего оператора, а не только разных строк в manifest.
