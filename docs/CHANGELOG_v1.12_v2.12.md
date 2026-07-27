# Changelog v1.12 / v2.12

## Исправленные блокеры v1.11

### 1. P02 → P03 больше не противоречат друг другу

Scientific lock защищает только нормативную статистику и доверенные runtime/model manifests. Файлы `analysis/build_analysis_input.py` и `analysis/build_sample_size_input.py` создаются в P03 и затем фиксируются Implementation lock. Adversarial-тест подтверждает, что их добавление после P02 не ломает Scientific lock.

### 2. Добавлен внешний Trust Topology lock

P01 всегда требует `locks/trust-topology.lock.json`, подписанный внешним Ed25519-ключом оператора. Lock связывает exact hashes `resource-plan`, `infrastructure-plan` и `public-keys`. Public/private operator trust-root keys запрещено хранить внутри репозитория.

Scientific lock дополнительно защищает Trust Topology lock, resource/infrastructure plans, public-key registry, environment lock и оба model lock.

### 3. Весь outcome-relevant код перенесён до G06

P03 обязан реализовать analysis builders, semantic resolver/parser/aliases, prototype-bank builder, control-certification engine, deterministic model loading, storage, sealer и evaluator interfaces. G06 проверяет этот код двумя независимыми аудитами по одному commit. P10 и P15 только исполняют уже заблокированную реализацию.

### 4. Удалён слабый verifier

`scripts/verify_gate.py` теперь является thin wrapper на `validation.verify_gate`. Отдельной более слабой логики проверки больше нет.

### 5. Разделены lock statuses

Phase report и run status содержат отдельные поля:

- `trust_topology_lock_status`;
- `scientific_lock_status`;
- `implementation_lock_status`.

Canonical verifier проверяет их соответствие фактически активным lock на каждой фазе.

## Дополнительные изменения

- implementation audit расширен до 12 обязательных проверок;
- Implementation-lock candidate хеширует Trust Topology lock и оба audit reports;
- G01 стал обязательным независимо от стоимости запуска;
- bootstrap manifest исключает generated runtime locks из статического release set, но защищает код и policy их создания/проверки;
- добавлены adversarial regression tests для trust boundary и фазовой исполнимости;
- версии документов, schemas, prompts и operator checklist синхронизированы до v1.12/v2.12.
