# Release self-review v1.15 / v2.15

## Scope

Проверены документы, 71 JSON Schema, machine-readable YAML contracts, validators, state machine, generated phase prompts, lock policies и packaging. Реальное обучение, Qwen inference, pilot и confirmatory run не выполнялись.

## Целевая гипотеза

Stage 1B проверяет **один полный frozen plan**, созданный до исполнения. Planner не вызывается повторно по текущему state. Каждая позиция плана, WorkPlan hash, semantic source и attempt lineage проверяются машинно. Plan-generation, resolution, shuffle-degeneracy и compute-cap failures остаются paired zero-success outcomes и не дают исключать задачу.

## Экспериментальные arms

- шесть обучаемых parameter-matched вариантов: A1, A2, A2b, A2c, A3 и A3r;
- A3r использует frozen train-only random codebook, а не несуществующий бесплатный control;
- Stage 1B требует ровно семь arms E0–E5/P для каждой выбранной задачи;
- E2 использует детерминированную перестановку позиций frozen E1 plan;
- P08 и P17 имеют разные precomputed replay-contexts и не могут подменять друг друга.

## Причинная защита selection

Stage 1B hidden selection использует только locked task/domain/split metadata. Reachable-state Planner support, plan validity, semantic resolution, shuffled-plan degeneracy, LLM output и arm outcome запрещены как eligibility inputs. Data Sealer подписывает task-only selection и pre-outcome artifact manifests до HMAC ranking и outcome access.

## Статистика

- design alternative 7,5 п.п. отделена от GO boundary 5 п.п.;
- acceptance мощности требует нижнюю exact-binomial bound ≥0,90 на двух соседних N;
- estimator structure фиксирована по stage: hierarchical Planner, clustered-by-task Stage 1A, paired task-level Stage 1B;
- confirmatory TOST удалён из active rules и sample-size components; legacy helper сохранён только для исторического golden-case.

## Compute и capacity

Preflight пересчитывает 24 development, 30 primary-final и 10 A3/A2c FLOPs-sensitivity workloads — всего 64 training workloads — плюс семь Stage 1B inference arms. Primary final training фиксирован на 12 000 updates × batch 128. Compute profile связан с raw measurement evidence, measurement code, environment и confirmatory freeze. Per-episode FLOPs пересчитываются из locked coefficients; превышение cap — typed paired failure.

## Trust boundary и lock ordering

- P01 создаёт внешний Ed25519-signed Trust Topology lock; operator key внутри репозитория запрещён.
- P02 фиксирует Scientific lock.
- P03 реализует весь outcome-relevant executable code.
- G06 требует statistical audit и implementation audit одного commit.
- Implementation audit содержит ровно 15 обязательных checks, включая full-plan lineage, A3r, task-only selection и FLOPs accounting.
- После G06 Implementation lock запрещает позднее добавление outcome-relevant кода.


## Минимальные launch-инварианты v1.15

- selected task list фиксируется до outcomes и подписанно связывается через SealerManifest;
- lineage exact-cover запрещает удаление целых задач после исполнения;
- evaluator task count пересчитывается по фактическому lineage;
- AnalysisInput связан с тем же SelectedTaskManifest и не допускает разные task subsets между comparisons;
- sample-size requirements используют только locked comparison mapping;
- execution JSON/JSONL artifacts зарегистрированы и валидируются по схемам.

## Известные границы

Release является исполняемым протоколом, но не результатом эксперимента. Организационная независимость Sealer, Evaluator, reviewers и оператора должна обеспечиваться реальными отдельными principals/environments. Корректность будущей реализации Planner будет доказана только P03–P09 checks и sealed runs.
