# Planner MVP и MicroPlanner — нормативная спецификация

**Версия:** 1.21
**Дата:** 27 июля 2026
**Статус:** исполняемая спецификация архивного эксперимента **Work Planner / BlocksWorld**.
**Stage 1:** `Planner_LLM_Stage1_Operator_Runbook_v2.21_RU.md`.
**Автономное исполнение:** `docs/operator/AUTONOMOUS_EXECUTION_PLAYBOOK_RU.md`.

Этот эксперимент не является основной архитектурой Cognitive Planner проекта ML Brain. Он проверяет узкий тезис: может ли одна causal position представлять один исполнимый шаг, а отдельное semantic representation — улучшать следующие шаги и работу frozen LLM.

Текст объясняет замысел. Исполняемыми источниками истины являются YAML, JSON Schema и Python-контракты.

---

# 0. Что исправлено к v1.21

1. Stage 1B переведён с reactive next-intent на один полный frozen plan, созданный до исполнения.
2. Добавлен `EpisodePlanManifest` и машинная цепочка `manifest → WorkPlan → positions → EpisodeLog → AnalysisInput`.
3. Введены семь Stage 1B arms: neutral, A3 full plan, shuffled A3, random-code A3r, structured A2c, self-plan и raw full-plan replay.
4. Planner call выполняется один раз до исполнения; replanning, plan patch и suffix regeneration запрещены.
5. Ошибка plan generation/resolution остаётся paired failure с нулём executor calls; скрытое исключение задачи запрещено.
6. Shuffled degeneracy только логируется и не меняет eligibility. Data Sealer не использует Planner/LLM outputs для отбора задач.
7. A2c принимает только комбинации из frozen signature bank; неизвестная комбинация fail-closed как `SEMANTIC_UNRESOLVED`.
8. Final training использует пять seeds и единственный допустимый checkpoint на optimizer step 12 000.
9. Sample-size design alternative 7.5 п.п. отделена от GO boundary 5 п.п.; power принимается по нижней exact-binomial bound на двух соседних N.
10. Confirmatory TOST удалён из Stage 1A/1B decisions и sample-size components; diagnostics не имеют veto.
11. FLOPs-matched direction стало core gate, а pre-outcome compute cap входит в hash-bound profile; exhaustion остаётся paired failure.
12. Capacity preflight пересчитывает 24 development + 30 primary final + 10 FLOPs-sensitivity workloads и семь Stage 1B arms по evidence-файлам.
13. Runtime фиксирует `trust_remote_code=false` и exact decoding profiles 64/128/128 tokens.
14. Trust Topology, Scientific и Implementation locks, signed manual decisions и confirmatory lineage остаются обязательными.

---


> **Разделение понятий.** Плейбук исполняется Builder LLM-агентом, но Planner, проверяемый научным экспериментом, остаётся отдельной микромоделью. Builder LLM не считается Data Sealer, Evaluation Runner, Statistical Reviewer или Audit Agent и взаимодействует с ними только через подписанные dispatch/artifact manifests.

# 1. Гипотезы и решения

## 1.1. Arms

| Arm | Одна позиция шага | Semantic supervision | Feedback в следующую позицию |
|---|---|---|---|
| A1 | нет, lossless token grammar | нет | предыдущие grammar tokens |
| A2 | action + ordered pointers | нет | action/refs |
| A2b | A2 + `intent_id` | 7 классов | predicted intent embedding |
| A2c | A2 + categorical signature | все поля semantic signature | predicted field embeddings |
| A3 | A2 + normalized 384-d `z` | frozen semantic encoder | predicted `z` projection |
| A3r | A3-shaped latent | frozen deterministic random code per exact semantic signature | predicted random-code projection |
| A4 | A3, но semantic feedback = 0 | как A3 | zero vector |
| A5 | A3, но semantic feedback shuffled | как A3 | foreign `z` по frozen manifest |

## 1.2. Gates

| Gate | Сравнение | Нормативный критерий |
|---|---|---|
| `GO_TYPED` | A2 − A1, HORIZON | lower 95% CI ≥ −2 п.п. и меньше causal positions |
| `GO_DISCRETE_INTENT` | A2b − A2, HORIZON | estimate ≥2 п.п., lower CI >0 |
| `GO_STRUCTURED_DISCRETE` | A2c − A2b, HORIZON | estimate ≥2 п.п., lower CI >0 |
| `GO_LATENT` | A3 − A2c | HORIZON: estimate ≥2 п.п. и lower CI >0; A3 non-inferior A1; A3>A4/A5. SIZE_OOD: A3 non-inferior A2c и A3>A4/A5. FLOPs-matched sensitivity сохраняет положительное направление. |

Направление эффекта должно совпадать в primary equal-data/equal-updates run и FLOPs-matched sensitivity run. Compute не заменяет качество и публикуется отдельно.

## 1.3. Границы вывода

Результат не доказывает перенос на открытые research/reasoning задачи, качество свободного текста, пользу World Model или универсальность continuous thoughts.

---

# 2. Источники истины

При конфликте:

1. approved freeze конкретного confirmatory run;
2. JSON Schemas;
3. executable Python contracts и validators;
4. YAML contracts;
5. hash/validation protocol;
6. эта спецификация;
7. phase prompt.

Runtime version: `work-planner/1.21`.

Ключевые контракты:

- domain: `blocks_world_v1.yaml`, `generator_contract_v1.yaml`, `intent_labeler_v1.py`;
- data: `dataset_split_contract_v1.yaml`, `training_corpus_contract_v1.yaml`;
- model: `planner_architecture_v1.yaml`, `planner_training_contract_v1.yaml`, `hyperparameter_search_v1.yaml`, `seed_selection_contract_v1.yaml`;
- semantic: `semantic_target_v1.yaml`, `intent_catalog_v1.yaml`;
- execution: `phase_registry_v1.yaml`, `phase_state_machine_v1.yaml`, `agent_execution_contract_v1.yaml`;
- safety: `contract_lock_v1.yaml`, `confirmatory_sealing_contract_v1.yaml`;
- infra: `provisioning_contract_v1.yaml`, `resource_budget_contract_v1.yaml`, `recovery_contract_v1.yaml`.

---

# 3. Формальный BlocksWorld

State содержит положительные facts: `ON`, `ON_TABLE`, `CLEAR`, `HOLDING`, `HAND_EMPTY`.

Инварианты:

1. каждый block имеет ровно один location mode;
2. moving block имеет не более одного support;
3. support несёт не более одного moving block;
4. self-edge и циклы запрещены;
5. удерживается не более одного block;
6. `HAND_EMPTY` iff нет `HOLDING`;
7. `CLEAR(b)` iff над `b` ничего нет и `b` не удерживается;
8. ledger contiguous: `@B0..@B(n−1)`.

Actions и preconditions/effects полностью заданы `blocks_world_v1.yaml`. Goal — conjunction только `ON` и `ON_TABLE`. Oracle — deterministic BFS, максимум 16 non-END actions, fixed tuple tie-break.

Terminal semantics:

```text
≤16 executable actions + ровно один END = ≤17 PlannerStep
```

Goal проверяется до первого действия и после каждого accepted action. Достижение goal на action 16 — success.

---

# 4. Dataset

## 4.1. Fixed partitions

`dataset_split_contract_v1.yaml` задаёт точные размеры:

- train: 24 000 base tasks, n=3..6, с feasible quotas 500/2 500/7 000/14 000;
- development: 4 000, quotas 100/400/1 200/2 300;
- Planner pilot: 800 horizon + 800 size OOD;
- Planner confirmatory reserves: 3 000 + 3 000;
- Stage 1A pilot/reserve: 400 / 4 000;
- Stage 1B pilot/reserve: 200 / 4 000.

Assignment выполняется по `base_task_id` до expansion. Outcome-based ordering и повторное использование pilot в confirmatory запрещены. Если рассчитанный N превышает reserve, run получает `BLOCKED_PROTOCOL_CAPACITY`; агент не уменьшает N и не меняет partition.

## 4.2. Corpus expansion

Для каждой base task:

- oracle suffix examples, включая terminal END;
- exact local breadth-first off-policy expansion depth 2 для n≤5;
- для n=6 — все one-step deviations и 16 deterministic walks;
- n=7,8 полностью запрещены в training/development expansion и появляются только в sealed size-OOD evaluation partitions;
- original base-task split сохраняется;
- leakage по base ID и ref-invariant canonical hash =0.

Runtime support audit использует structural signature, но не дообучается на evaluation states.

---

# 5. Intent Labeler и semantic target

`docs/domain/intent_labeler_v1.py` является нормативным алгоритмом. Он не вызывает модель и не использует randomness. Входы: state, goal, all shortest first actions, selected tie-break action, remaining distance. Выходы:

- `intent_id`;
- `SemanticSignature`;
- exact canonical signature text.

P04 строит exhaustive fixture для всех valid state-goal/action cases n≤5 и сохраняет manifest/hash до Implementation lock P06. Нормативные правила Intent Labeler, каталог меток и их golden outputs входят в Scientific lock с P02 и не могут изменяться implementation-only patch. До P06 разрешено исправлять только исполняемый adapter/serialization-код, если exhaustive fixture остаётся byte-identical; любое изменение labeler output требует новой версии протокола и нового run.

A3 target создаётся только locked `sentence-transformers/all-MiniLM-L6-v2`: exact revision, mean pooling, L2 normalization, float32, dimension 384. A2c получает те же source signature fields; A3 text не может содержать дополнительные факты или object IDs.

---

# 6. Архитектура

Source: `planner_architecture_v1.yaml`.

## 6.1. Task Encoder

- 4 layers, `d_model=256`, 8 heads, FFN 1024;
- pre-LN, GELU, learned positions;
- dedicated `[REF_SLOT_i]`; pointer memory — hidden state этого token;
- structural mask скрывает только nonexistent slots;
- domain-validity mask до raw log запрещён.

Token layout и golden fixtures задаёт `task_encoding_v1.yaml`.

## 6.2. Planner Decoder

- 4 causal layers, cross-attention к Task Encoder;
- единый learned positional inventory на 85 decoder positions во всех trainable arms;
- A1 активирует positions 0..84 и `token_grammar_head: Linear(256,24)`;
- step-level arms A2/A2b/A2c/A3/A3r активируют только positions 0..16 через ConceptPacker; A4/A5 используют тот же A3 decoder path;
- dormant positions/heads не входят в outputs и сохраняют `grad=None`;
- heads: token grammar, action, arg1, arg2, A2b intent, A2c signature fields, A3/A3r 384-d latent.

## 6.3. ConceptPacker

Вход position `t+1`:

```text
LayerNorm(
  step_position
  + previous_action_embedding
  + previous_arg1_ref_projection
  + previous_arg2_ref_projection
  + previous_semantic_projection
)
```

Semantic projection:

- A2: zero;
- A2b: predicted/teacher-forced intent embedding;
- A2c: семь embeddings по 64 измерения → concat 448 → Linear 448→256 → LayerNorm;
- A3: `z 384 → Linear 512 → GELU → Linear 256 → LayerNorm`;
- A4: выполняется точный A3 latent projection и normalization, затем projected 256-вектор заменяется нулями перед суммой ConceptPacker; веса и вычислительный путь совпадают с A3;
- A5: frozen foreign `z` проходит через неизменённый A3 projection. Mapping A5 задаётся `planner_latent_ablation_contract_v1.yaml`: derangement между разными base tasks внутри одинаковых step/distance/hand strata, до outcomes и с полной pair-completeness A3/A4/A5.

Training использует ground-truth previous semantic target; autoregressive inference — только predicted output. Scheduled sampling запрещён.

## 6.4. A1

Lossless grammar, max 85 positions. A1 использует общий 85-position decoder backbone, собственный token embedding `24×256` и `Linear(256,24)` grammar head. Step-level arms используют тот же positional parameter inventory, но только первые 17 positions. Grammar tokens не проходят через ConceptPacker; typed-step arms не используют grammar-token embedding/head. Raw parser не ремонтирует syntax. Все trainable arms используют один и тот же common-superset parameter inventory с идентичным inventory hash и нулевой допустимой разницей по числу параметров. Dormant-модули не входят в вычисление outputs и должны иметь `grad=None`; no-op projections как способ скрытого parameter matching запрещены.

---

# 7. Training

Fixed contract:

- AdamW, betas 0.9/0.95, weight decay 0.01;
- linear warmup 5% + cosine decay;
- batch 128, max 20 epochs, min 5, patience 3;
- grad clip 1.0;
- validation каждые 500 updates, checkpoint 1000;
- final seeds: 101, 202, 303, 404, 505.

Development grid — ровно 4 configs: LR {1e−4, 3e−4} × dropout {0, 0.1}; остальные параметры fixed. Development seed 17.

Selection composite:

```text
0.70 goal_success + 0.20 valid_action_rate + 0.10 exact_plan_match
```

Development floor: valid action ≥0.90, goal success ≥0.50. Stage 1B eligibility оценивается отдельно и строже: A3 HORIZON goal success ≥0.65 with lower 95% CI ≥0.60; A3 SIZE_OOD goal success ≥0.60 with lower CI ≥0.55; valid action ≥0.90; Planner-confirmatory A3 full-plan replay ≥0.70. Tie-break фиксирован. После выбора final config grid не открывается повторно.

Loss weights и contrastive temperature заданы `planner_training_contract_v1.yaml`. `END` входит только в `action_ce`; отдельного `end_ce` нет. `arg1` обучается на PICK_UP/PUT_DOWN/UNSTACK/STACK, `arg2` — только на UNSTACK/STACK, semantic losses — только на non-END positions. Каждый head усредняется по собственным valid targets, после чего активные head means суммируются с весами. Contrastive anchor без другого exact-signature positive в batch исключается; если eligible anchors нет, contrastive loss равен нулю без gradient. Active loss matrix фиксирована: A1 использует только grammar CE; A2 — action/arg1/arg2; A2b добавляет intent; A2c добавляет signature fields; A3/A3r добавляют cosine и supervised contrastive. Все неактивные heads исключены из total loss и сохраняют `grad=None`.

---

# 7.1. Stage 1B full-plan transfer

Stage 1B не вызывает Planner на каждом state. Для каждого plan arm сначала создаётся или привязывается один immutable `WorkPlan`; затем независимая arm trajectory потребляет позиции 0..n без replanning. E2 и P могут переиспользовать exact E1 plan artifact, но не E1 state/actions/outcomes. E5 генерирует self-plan отдельным frozen LLM call; его plan tokens учитываются отдельно от executor tokens. Любая plan-generation failure остаётся в paired analysis как `goal_success=false`.

Семь обязательных arms и их machine lineage заданы `common.schema.json`, `episode_plan_manifest.schema.json`, `full_plan_lineage_index.schema.json` и `validation/full_plan_lineage_validator.py`.

# 8. Raw rollout и statistics

Raw:

```text
unmasked logits → persist top-1 → Validator → success/failure
```

Operational mask/retry — отдельный exploratory dataset и не влияет на confirmatory.

Planner decision:

- per-seed metrics;
- hierarchical bootstrap: seed, затем base task;
- 10 000 resamples;
- минимум 3/5 seeds с одинаковым направлением;
- equal-data primary и train-FLOPs-matched A3↔A2c sensitivity должны иметь одинаковое направление эффекта; inference FLOPs публикуются только как guardrail и не меняют training schedule.

---

# 9. Автономное исполнение

P00–P20 заданы registry и explicit state machine. Важные правила:

- manual phase сначала проходит pre-gate checks и останавливается в `WAITING_APPROVAL`; DecisionRecord проверяется только после ответа;
- переход берётся только по declared outcome;
- scientific STOP помечает downstream фазы `SKIPPED_BY_CONTRACT` и идёт в обязательный audit;
- Scientific lock проверяется до/после каждой фазы с P02; Implementation lock — с P06;
- любое изменение Scientific-lock path блокирует run и требует v1.21/new run;
- Builder не видит confirmatory plaintext;
- Evaluation Runner запускает confirmatory на отдельной среде;
- Audit Agent обязательно воспроизводит run на clean checkout.

Ручные gates:

- G00 scope;
- G01 cloud budget, только если spend>0;
- G06 independent statistics/implementation audits и Implementation lock;
- G07 Planner freeze;
- G12 Stage 1A freeze;
- G16 Stage 1B freeze;
- G20 final acceptance.

---

# 10. Acceptance

До Planner confirmatory:

- domain/generator/labeler tests PASS;
- fixed split quotas PASS;
- Scientific lock VERIFIED; Implementation lock VERIFIED перед любыми pilot/confirmatory данными;
- all model arms smoke PASS;
- development grid and final-seed selection reproducible;
- confirmatory reserve ≥ N;
- sealed dataset and evaluator boundary PASS.

До Stage 1:

- required Planner GO;
- immutable checkpoint/model/prompt locks;
- prompt candidate meets format/valid/progress floors;
- deterministic LLM 20/20;
- controls certified before outcomes.

Final acceptance требует independent audit PASS и обязательное clean reproduction. Ограничение нельзя заменить формулировкой «по возможности».

# 18. Исторические нормативные уточнения v1.9

## 18.1 Финальное обучение
Development использует early stopping. Финальные checkpoints A1/A2/A2b/A2c/A3/A3r обучаются ровно 12 000 optimizer updates на одинаковом порядке примеров для каждого seed. A4/A5 не обучаются: это interventions над тем же A3 checkpoint, и hash весов обязан совпадать.
 Единственный допустимый final checkpoint — step 12 000; выбор промежуточного best checkpoint запрещён. P06 capacity preflight пересчитывает полный development grid (24 workloads), шесть primary variants × пять final seeds (30 workloads), десять A3/A2c FLOPs-sensitivity workloads и семь Stage 1B inference arms.

## 18.2 Statistics source of truth
Все CI, power, non-inferiority/direction gates и решения вычисляются только `analysis/sample_size.py` и `analysis/decision_gates.py` по `docs/statistics/statistics_contract_v1.yaml`. Отчёт обязан ссылаться на hash исходного paired-input artifact.

## 18.3 Resolver
Thresholds similarity/margin выбираются только на development по `docs/semantic/semantic_resolver_v1.yaml`, фиксируются до Stage 1 pilot и не меняются.

## 18.4 Sealed data
Builder не материализует confirmatory plaintext и не знает selection seed. Отдельный Data Sealer получает locked contracts и public-exclusion manifest, внутри изолированной среды генерирует 256-bit secret seed, ранжирует кандидатов по HMAC-SHA256, создаёт encrypted blob, уничтожает plaintext и возвращает только seed commitment, encrypted blob hash, counts и подписанный sealer manifest. Seed reveal доступен Auditor только после подписания evaluator result manifest.


# 19. Исторические нормативные уточнения v1.9

## 19.1 Двухуровневая заморозка

- **Scientific lock (P02):** гипотезы, arms, endpoints, split/seeds, статистика, thresholds, N method, control logic и state machine. Изменение всегда означает новый protocol/run.
- **Implementation lock (P06):** schemas, serializers, validator/runtime code, точная model wiring и prompt renderer. До P06 разрешена только техническая правка через `implementation_patch.schema.json`, без доступа к pilot outcomes и с полным повтором toy preflight.
- Проверяющие lock-скрипты входят в bootstrap manifest и Scientific lock; сам verifier нельзя заменить незаметно.

## 19.2 Toy preflight до Implementation lock

Preflight разделён на два уровня. P03 выполняет contract-level dry-run на n=3 и 20–50 synthetic development-only задачах: stub interfaces, serialization round-trip, atomic persistence/recovery, fake episode, golden statistics/sample-size и adversarial gate-verifier — без обучения Planner. После реализации шести обучаемых вариантов A1/A2/A2b/A2c/A3/A3r и двух A3-интервенций A4/A5 P06 выполняет clean-checkout full toy preflight: forward/backward всех arms, two-batch overfit, serialization, полный fake episode и те же golden statistics. Ни один pilot outcome до утверждения G06 и активации Implementation lock недоступен.

## 19.3 Разделение архитектурного GO и Stage 1B eligibility

`GO_PLANNER_ARCHITECTURE` означает, что A3 прошла относительные causal gates. Stage 1B разрешён только при абсолютных floors: A3 HORIZON point ≥0.65, lower CI ≥0.60, valid-action rate ≥0.90, Planner-confirmatory A3 full-plan replay ≥0.70. Иначе результат фиксируется как `GO_PLANNER_DIAGNOSTIC_ONLY`; Stage 1A остаётся допустимым, Stage 1B запрещён.

## 19.4 Независимый статистический аудит

До P07 обязательна ручная точка G06. Reviewer должен быть человеком-статистиком или моделью другой family, чем Builder, и проверить hierarchical bootstrap, clustered Stage 1A bootstrap, non-inferiority/direction gates, sample-size/power, tie-break/missing-pairs и recomputation decision gates.


### Нормативный вход статистики

`AnalysisInput` не хранит только готовые оценки. Каждая paired comparison содержит строки `pair_id, left, right, difference`, причём валидатор пересчитывает `difference = left − right`, запрещает дубли и требует одинаковый набор `base_task_id` во всех Planner seeds. Stage 1A хранит snapshot-pairs внутри task cluster; scalar rates содержат `unit_id, value`. Auditor обязан восстановить эти строки из raw result manifest; готовые агрегаты без pair rows недопустимы.

## Приложение D. Нормативный full-plan replay

`P_FULL_PLAN_REPLAY_RAW` определяется только `docs/controls/p_replay_contract_v1.yaml`. Planner вызывается один раз до исполнения source arm; replay повторно Planner не вызывает и последовательно исполняет exact frozen TypedActions. Контекст P08 использует precomputed Planner-confirmatory A3 WorkPlan, контекст P17 — precomputed Stage 1B E1 WorkPlan. Hash source manifest, WorkPlan и каждой позиции обязателен; подмена контекстов запрещена.

## Launch-инварианты v1.21

- Planner confirmatory output is an exact Cartesian matrix: every selected task × seeds `101,202,303,404,505` × arms `A1,A2,A2b,A2c,A3,A3r,A4,A5,P_FULL_PLAN_REPLAY_RAW`.
- `planner_seed` is explicit in lineage, EpisodePlanManifest, EpisodeLog, WorkPlan and AttemptLog and must match even for FAILED plan generation; duplicates and omissions are `INVALID_RUN`.
- Stage 1A uses the same snapshot `pair_id` set per task in all comparisons.
- Stage 1B replay diagnostic metric is `STAGE1B_E1_FULL_PLAN_REPLAY_GOAL_SUCCESS`.


## Нормативный inventory и training evidence v1.21

Точные PyTorch state_dict keys, shapes, bias policy, LayerNorm eps, GELU и attention layout заданы в `docs/architecture/planner_module_inventory_v1.yaml`. ИИ-реализация не вправе переименовывать tensors или выбирать fused QKV. P07 принимает 30 final и 10 FLOPs-sensitivity отчётов только через структурированные sidecars: parameter inventory, initialization, ordered examples, dormant-gradient audit и checkpoint manifest. Модельные и optimizer checkpoints имеют формат safetensors; валидатор сверяет header с locked inventory. P08 каждый Planner lineage record связывает с конкретным checkpoint manifest: primary arms — `FINAL_EQUAL_DATA`, `PLANNER_A2C_FLOPS_RAW`/`PLANNER_A3_FLOPS_RAW` — `FLOPS_SENSITIVITY`, A4/A5/replay — соответствующий A3 final checkpoint. Variant, seed, regime и model SHA пересчитываются машинно.

## Evidence hardening v1.21

1. Initialization checkpoint tensors are recomputed byte-for-byte from the locked name-derived NumPy PCG64 initializer.
2. A trained checkpoint is invalid when no active tensor differs from initialization, any dormant tensor differs, or any tensor contains NaN/Inf.
3. AdamW sidecars must contain the exact active state set, finite non-zero aggregate state, non-negative second moments and locked optimizer metadata.
4. P06 `recomputed_value` is generated by validator code for parameter counts, same-information cases, raw-rollout invariants and dormant gradients.
5. P08 accepts only canonical P07 checkpoint manifests and their canonical training reports; ad-hoc link JSON is forbidden.
