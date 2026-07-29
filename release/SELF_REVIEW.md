# Release self-review v1.21 / v2.21

## Scope

Проверены документы, 78 JSON Schema, 48 machine-readable YAML contracts, validators, state machine, generated phase prompts, lock policies и packaging. Полный regression suite: 224 tests PASS в 33 изолированных test-файлах; skipped=0. Реальное обучение, Qwen inference, pilot и confirmatory run не выполнялись.

## Целевая гипотеза

Stage 1B проверяет **один полный frozen plan**, созданный до исполнения. Planner не вызывается повторно по текущему state. Каждая позиция плана, WorkPlan hash, semantic source и attempt lineage проверяются машинно. Plan-generation, resolution, shuffle-degeneracy и compute-cap failures остаются paired zero-success outcomes и не дают исключать задачу.

## Экспериментальные arms

- шесть обучаемых вариантов с общим полным parameter inventory: A1, A2, A2b, A2c, A3 и A3r; active capacity различается и не интерпретируется как чистый representation effect;
- A3r использует frozen train-only random codebook, а не несуществующий бесплатный control;
- Stage 1B требует ровно семь arms E0–E5/P для каждой выбранной задачи; Planner confirmatory P08 отдельно требует exact 11-arm matrix, включая A2c/A3 FLOPs-sensitivity и replay;
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
- Implementation audit содержит ровно 16 обязательных checks, включая full-plan lineage, A3r, task-only selection и FLOPs accounting.
- После G06 Implementation lock запрещает позднее добавление outcome-relevant кода.


## Минимальные launch-инварианты v1.21

- selected task list фиксируется до outcomes и подписанно связывается через SealerManifest;
- lineage exact-cover запрещает удаление целых задач после исполнения;
- evaluator task count пересчитывается по фактическому lineage;
- AnalysisInput связан с тем же SelectedTaskManifest и не допускает разные task subsets между comparisons;
- sample-size requirements используют только locked comparison mapping;
- execution JSON/JSONL artifacts зарегистрированы и валидируются по схемам.

## Известные границы

Release является исполняемым протоколом, но не результатом эксперимента. Организационная независимость Sealer, Evaluator, reviewers и оператора должна обеспечиваться реальными отдельными principals/environments. Корректность будущей реализации Planner будет доказана только P03–P09 checks и sealed runs.

## Launch-fixes v1.21

- exact Planner task × five seeds × all frozen arms matrix;
- duplicate/missing Planner outcomes are fail-closed;
- identical Stage 1A snapshot sets across comparisons;
- canonical Stage 1B replay metric name.

## Architecture freeze fixes v1.21

- A1 and step-level variants share one 85-position decoder parameter inventory; only their active position range and input/head mask differ.
- Training loss masks, reductions, active heads and no-positive contrastive batches are fully specified.
- Planner seed is persisted and checked in plan manifests and episode logs even for generation failure.
- Training/development cannot contain n=7–8 states.
- The only compute-matched retraining sensitivity is measured train FLOPs for A3 versus A2c; inference FLOPs are guardrails.

## Architecture evidence fixes v1.21

- exact machine-readable inventory фиксирует 177 PyTorch `state_dict` tensors, shapes, parameter types и active-arm masks;
- deterministic initialization независима от module construction order и одинакова для trainable arms одного seed;
- P06 PASS требует content-validated seed-17 initialization checkpoint и dormant-gradient audits A1/A2/A2b/A2c/A3/A3r;
- A3r использует raw predicted z для autoregressive feedback и nearest-codebook resolution только для external signature;
- A1 имеет compute reporting guardrails, но не undefined equal-compute retraining schedule;
- P07 training evidence — exact 30 final + 10 FLOPs-sensitivity reports с safetensors headers, optimizer states, config, corpus, ordering и environment binding;
- P08 lineage содержит отдельные A2c/A3 FLOPs-sensitivity arms на том же sealed task set.
- P08 каждый Planner lineage record связывает с точным P07 checkpoint manifest; variant, regime, seed и model SHA пересчитываются.

## Gate evidence hardening v1.21

- P08 canonical P07 binding is enforced and mutation-tested.
- Initialization values are recomputed from the locked PCG64 initializer.
- Unchanged trained weights, changed dormant weights, non-finite tensors, zero optimizer states and wrong AdamW metadata are rejected.
- P06 parameter, same-information, raw-rollout and dormant-gradient checks are independently recomputed.
- Full bundle validation includes both repository test trees.
