# Planner → frozen LLM: Stage 1 runbook

**Версия:** 2.16
**Дата:** 27 июля 2026
**Implementation Spec:** `Planner_MVP_MicroModel_Implementation_Spec_RU_v1.20.md`.

Документ задаёт исполняемый контур Stage 1A/1B. Нормативные числа, схемы и правила находятся в YAML/JSON-контрактах; этот runbook не переопределяет их.

---

# 1. Проверяемые гипотезы

## Stage 1A — полезность интерфейса guidance

На одном frozen oracle snapshot правильный текущий intent должен повышать вероятность действия, которое:

```text
parsed AND valid AND distance_after < distance_before
```

Primary comparison:

```text
I1_ORACLE_CURRENT_RAW − I0_EQUAL_TOKENS_RAW
```

Stage 1A проверяет интерфейс `guidance → frozen LLM`, а не качество Planner.

## Stage 1B — полезность одного полного плана

Planner получает исходную задачу **один раз до исполнения**, создаёт полный frozen `WorkPlan`, после чего LLM-исполнитель последовательно потребляет его позиции без replanning.

Primary comparison:

```text
E1_A3_FULL_PLAN_RAW − E0_EQUAL_TOKENS_RAW
```

Stage 1B запрещён без `GO_INTERFACE_STAGE1B_ELIGIBLE` и Planner-решения с `stage1b_eligible=true`.

---

# 2. Роли и blindness

- **Builder Agent:** код, development, pilot, freeze candidate; confirmatory plaintext запрещён.
- **Data Sealer:** материализация hidden confirmatory tasks и pre-outcome eligibility; отдельные credentials и environment.
- **Evaluation Runner:** approved freeze, sealed key, read-only source commit, write-only results.
- **Statistical Reviewer:** независимая проверка estimators, power и gates до G06.
- **Audit Agent:** отдельный clean checkout, read-only воспроизведение.
- **Operator:** подписывает manual DecisionRecord внешним ключом.

Single-agent или unsealed confirmatory = `INVALID_CONFIRMATORY_BLINDNESS`, независимо от результата.

---

# 3. Frozen LLM и runtime

Model: exact immutable revision `Qwen/Qwen3.5-0.8B`; model, tokenizer и processor фиксируются commit SHA и локальными file hashes.

Обязательно:

- greedy decoding, `do_sample=false`, `num_beams=1`;
- `trust_remote_code=false`;
- thinking/tools off;
- Stage 1A executor: `max_new_tokens=64`;
- Stage 1B executor: `max_new_tokens=128`;
- self-plan generator: `max_new_tokens=128`;
- exact tokenizer/chat-template hashes;
- exact Python/Torch/Transformers/vLLM revisions;
- CUDA, driver, cuDNN и kernels;
- `language_model_only=true`, `local_files_only=true`;
- hash-bound offline cache;
- no sampling or parser-repair fallback.

Determinism: 8 golden prompts × 20 calls. Raw bytes, parsed action и unpadded tensors совпадают 20/20.

---

# 4. Prompt development

P11 использует только development partition и exact UTF-8 candidates C01–C04.

Floors:

- format success ≥0.95;
- valid action rate ≥0.85;
- oracle-intent progress ≥0.55.

Score:

```text
0.20 format + 0.30 valid + 0.50 oracle progress
```

Если candidates не проходят floors: `BLOCKED_PROMPT_STACK`. Prompt tuning на pilot или confirmatory запрещён.

---

# 5. Prompt/parser contract

Aliases: `@B0→block_0 ... @B7→block_7`. Display names не участвуют в semantics.

User sections:

1. TASK;
2. AVAILABLE_BLOCKS;
3. CURRENT_STATE;
4. GOAL;
5. GUIDANCE_BLOCK;
6. ACTION_SIGNATURES;
7. OUTPUT_FORMAT.

Parser принимает ровно один JSON object с keys `schema_version`, `action`, `args`. Extra text, repair и guessed action запрещены.

Canonical prompt artifact — **unpadded** roles, UTF-8 content, input IDs, attention mask и position IDs.

- max unpadded length 512;
- truncation forbidden;
- guidance ровно 32 attended tokens;
- batching только с left padding;
- right padding forbidden;
- generation берёт logits последнего attended token.

---

# 6. Stage 1A

## 6.1 Unit и estimator

```text
(base_task_id, oracle_step_index, canonical_snapshot_hash)
```

Сначала усреднение snapshot outcomes внутри `base_task_id`, затем paired task bootstrap. Коррелированные snapshots не считаются независимыми строками.

## 6.2 Arms

| Arm | Guidance |
|---|---|
| I0_EQUAL_TOKENS_RAW | neutral 32-token block |
| I1_ORACLE_CURRENT_RAW | executable labeler intent |
| I2_SHUFFLED_RAW | pre-certified incompatible intent |
| I3_PLANNER_CURRENT_RAW | Planner/resolver intent |

Snapshot eligibility фиксируется до LLM calls. Post-outcome exclusion запрещён.

## 6.3 One-step execution

```text
sealed snapshot
→ render exact unpadded prompt
→ persist PromptArtifact
→ greedy LLM call
→ persist raw bytes
→ strict parser
→ Validator
→ optional one-action Executor
→ recompute exact distance
→ AttemptLog
```

Retry, mask и repair отсутствуют.

## 6.4 GO_INTERFACE

Core gates:

1. `I1−I0` estimate ≥5 п.п.;
2. `I1−I0` lower 95% CI >0;
3. `I1_PROGRESS_RATE ≥0.60`;
4. `I1−I2` lower 95% CI >0;
5. parse failure ≤5%;
6. infrastructure failure ≤1%;
7. incomplete pairs ≤1%;
8. contract/hash/determinism violations =0.

`I2−I0` — diagnostic, не veto и не sample-size component. Confirmatory TOST в v1.20 запрещён.

N = maximum из `primary_ci`, `primary_power`, `current_vs_shuffled_power` и minimum N. Acceptance мощности использует нижнюю одностороннюю exact-binomial bound ≥0.90 на двух соседних candidate N. Design alternative 7.5 п.п. отделена от decision boundary 5 п.п.

---

# 7. Stage 1B pre-outcome certification

Data Sealer до outcomes:

1. генерирует hidden candidates по locked generator;
2. проверяет только domain validity, oracle-length stratum, split membership и заранее зафиксированные task metadata;
3. применяет HMAC ordering только после eligibility;
4. не использует Planner, LLM, shuffled-plan или self-plan outcomes для inclusion;
5. фиксирует полный inclusion manifest до dispatch.

Задача **не может** исключаться из-за:

- ошибки генерации плана;
- `SEMANTIC_UNRESOLVED`;
- degenerate shuffle;
- budget exhaustion;
- результата любого arm.

Такие события остаются paired failures/diagnostics. Post-treatment selection запрещён.

---

# 8. Stage 1B arms

| Arm | Pre-execution plan source | Execution |
|---|---|---|
| E0_EQUAL_TOKENS_RAW | no plan | neutral guidance → frozen LLM |
| E1_A3_FULL_PLAN_RAW | one A3 Planner call | sequential A3 plan positions → frozen LLM |
| E2_SHUFFLED_A3_FULL_PLAN_RAW | exact E1 WorkPlan reused, no Planner call | hash-bound nonzero cyclic permutation of guidance positions → frozen LLM |
| E3_A3R_RANDOM_CODE_FULL_PLAN_RAW | one separately trained parameter-matched A3r Planner call | sequential positions resolved through the frozen deterministic random codebook → frozen LLM |
| E4_A2C_STRUCTURED_FULL_PLAN_RAW | one A2c Planner call | exact known structured signatures → frozen LLM |
| E5_SELF_PLAN_RAW | one frozen LLM self-plan call | sequential self-plan positions → frozen LLM |
| P_FULL_PLAN_REPLAY_RAW | exact E1 WorkPlan reused, no Planner/LLM call | execute frozen TypedActions directly |

Каждый arm имеет независимый state trajectory. E2 и P переиспользуют только immutable E1 plan artifact и его attributed planning cost; они не переиспользуют state, accepted action или outcome E1.

## 8.1 EpisodePlanManifest

До первого executor attempt каждый plan arm сохраняет:

- generator type и exact call count;
- generation start/completion timestamps;
- initial state/goal hashes;
- `WorkPlan` content/artifact hashes;
- source manifest hash для E2/P;
- control-artifact path/hash для E2 shuffle mapping и E3 random codebook;
- actual и attributed planning tokens/latency;
- signature-bank hash для A2c;
- READY либо FAILED status.

FAILED generation:

```text
zero executor calls
+ goal_success=false
+ terminal failure code
+ task remains in all paired analysis
```

## 8.2 A2c fail-closed resolution

A2c предсказывает семь signature fields. Комбинация должна существовать в frozen normative signature bank. Незнакомая комбинация не маппится эвристически: `FAILED / SEMANTIC_UNRESOLVED`.

## 8.3 No replanning invariant

Для каждого plan arm:

```text
one plan-generation call before execution
→ positions 0,1,2,...
→ same manifest hash
→ same WorkPlan hashes
→ replanning_observed=false
```

Patch, suffix regeneration, повтор Planner call или пропуск позиции = `INVALID_RUN`.

---

# 9. Stage 1B execution

Псевдокод:

```python
plan_manifest = generate_or_bind_plan_once(task, arm)
persist(plan_manifest)

if plan_manifest.status == "FAILED":
    return paired_failure_without_executor_calls(plan_manifest.failure_code)

state = task.initial
for position, plan_step in enumerate(plan_manifest.frozen_work_plan.non_end_steps[:16]):
    if goal_pass(state):
        return SUCCESS

    if arm == P_FULL_PLAN_REPLAY_RAW:
        candidate = plan_step.typed_action
    else:
        guidance = render_guidance_from_frozen_position(plan_step, arm)
        prompt = render_exact_prompt(task, state, guidance)
        persist_prompt_before_call(prompt)
        raw = frozen_llm.greedy(prompt)
        persist_raw_before_parse(raw)
        candidate = strict_parser(raw)

    result = validator.check(candidate, state)
    persist_attempt_with_plan_position(position, plan_manifest, result)
    if not result.valid:
        return FAILURE
    state = executor.apply(result.normalized_action, state)

return SUCCESS if goal_pass(state) else HORIZON_EXCEEDED
```

Goal after action 16 has priority over horizon failure.

---

# 10. Compute accounting и FLOPs sensitivity

P06 `reports/compute-profile.json` до confirmatory outcomes фиксирует:

- measured train FLOPs/time per optimizer step for A1/A2/A2b/A2c/A3;
- equal-data schedule: 12 000 updates × batch 128;
- FLOPs-matched update count per variant;
- Stage 1B per-task FLOPs estimate for all seven arms;
- pre-outcome `flops_cap_per_task`;
- policy `PAIRED_FAILURE_NO_TASK_EXCLUSION` при исчерпании cap.

E2/P имеют zero **actual** new plan cost, но получают E1 plan cost в **attributed** comparison. E5 plan-generation tokens отделены от executor tokens.

`FLOPS_MATCHED_DIRECTION` — core gate. Формальный GO при противоположном направлении под locked FLOPs-cap запрещён.

---

# 11. Logging и lineage

Every JSON follows `report_registry_v1.yaml`.

Обязательная цепочка:

```text
EpisodePlanManifest
→ WorkPlan
→ AttemptLog[position 0..n]
→ EpisodeLog
→ evaluator-result-manifest
→ AnalysisInput
→ ScientificDecision
```

`lineage-index.json` содержит ровно семь Stage 1B arms на каждый task. Удаление arm или task после plan generation считается post-treatment exclusion.

AttemptLog хранит:

- task/stage/arm/pair hashes;
- state before/after и goal hash;
- plan manifest/hash/position/step id;
- exact guidance source position/step/semantic ref; for E2 also frozen shuffle mapping hash;
- `replanning_observed=false`;
- prompt/tensor hashes и tokens;
- raw output before parser;
- parsed и normalized actions;
- issue, latency, resource и terminal fields.

---

# 12. GO_END_TO_END

Core gates:

1. `E1−E0` estimate ≥5 п.п. и lower CI >0;
2. `E1−E2` lower CI >0;
3. `E1−E3` lower CI >0;
4. `E1−E4` lower CI ≥−2 п.п.;
5. `E1−E5` lower CI >0;
6. FLOPs-matched estimate ≥0 и lower CI ≥−2 п.п.;
7. plan-generation failure rate ≤10%;
8. unseen support signature rate ≤5%;
9. parse failure ≤5%; infrastructure/incomplete pairs ≤1%;
10. contract/hash/determinism violations =0.

Diagnostics, не veto:

- E2−E0;
- P full-plan replay goal success;
- E1−E4 point direction.

Sample-size components:

```text
primary_ci
primary_power
current_vs_shuffled_power
random_code_power
structured_noninferiority_power
self_plan_power
flops_direction_power
```

Confirmatory TOST не входит ни в decision, ни в N.

---

# 13. Replay contexts

`docs/controls/p_replay_contract_v1.yaml` задаёт две разные lineage:

1. `PLANNER_CONFIRMATORY_A3` в P08 — replay заранее построенного A3 WorkPlan; доступен до P09 и используется как Planner eligibility diagnostic.
2. `STAGE1B_E1` в P17 — replay exact E1 WorkPlan; только end-to-end diagnostic.

Один контекст нельзя подменять другим. Оба запрещают replanning и LLM calls.

---

# 14. Capacity preflight

До G06 machine validator пересчитывает:

- 4 development configs × 6 trained variants × seed 17 = **24 workloads**;
- 6 trained variants × 5 final seeds = **30 workloads**;
- 2 FLOPs-sensitivity variants A3/A2c × 5 final seeds = **10 workloads**;
- итого **64 training workloads**;
- 12 000 × 128 = **1 536 000 processed examples** на каждый final workload;
- **46 080 000** final processed examples суммарно;
- семь Stage 1B arms на reserve capacity 4000 tasks.

Report обязан ссылаться хешами на training, hyperparameter, dataset, compute, resource и corpus manifests. Self-declared multipliers не принимаются. Недостаток GPU time/storage/cost → `BLOCKED_PROTOCOL_CAPACITY` до G06.

Primary final checkpoint — только optimizer step 12 000. FLOPs-sensitivity создаёт отдельные A3/A2c checkpoints с количеством updates из locked compute profile. Выбор промежуточного «best validation checkpoint» запрещён; после завершения сохраняется ровно один checkpoint на workload.

---

# 15. Phases и stop transitions

Stage 1 соответствует P10–P18:

- P10 locked LLM/resolver;
- P11 prompt development/freeze;
- P12 Stage 1A pilot/N/freeze, G12;
- P13 sealed Stage 1A confirmatory;
- P14 GO_INTERFACE;
- P15 public/preflight Stage 1B certification;
- P16 Stage 1B pilot/N/freeze, G16;
- P17 sealed Stage 1B confirmatory;
- P18 GO_END_TO_END.

При `STOP_INTERFACE` P15–P18 получают `SKIPPED_BY_CONTRACT`; P19 всё равно аудирует отрицательный результат. Lock mismatch, unsealed confirmatory, nondeterminism, lineage/hash violation — `BLOCKED` или `INVALID_RUN`, а не limitation.

---

# 16. Normative statistics input

`AnalysisInput` хранит unit-level rows, а не только готовые estimates:

- Planner: exactly five seed groups с одинаковыми paired task IDs;
- Stage 1A: snapshots внутри task clusters;
- Stage 1B: one paired row per base task;
- scalar rates: unit-level binary rows.

Validator пересчитывает differences, estimators, CI, sample size, core/diagnostic gates и final decision из locked contracts. Готовые агрегаты без raw lineage не являются источником истины.

## Обязательная матрица P08/P14

P08 выполняет exact task × five final seeds × all frozen Planner arms matrix. P14 принимает Stage 1A AnalysisInput only when every comparison contains the same snapshot IDs per task. Any missing, duplicate or substituted unit is `INVALID_CONFIRMATORY`. Для каждой P08 записи обязателен hash-bound P07 checkpoint manifest. FLOPs-arms обязаны ссылаться именно на отдельные `FLOPS_SENSITIVITY` checkpoints matching variant/seed; подмена primary checkpoint или обратная подмена является `INVALID_CONFIRMATORY`.

## Обязательные проверки v2.20

- P06: не принимать декларативный PASS. Проверить check-specific evidence и validator-recomputed results.
- P07: проверить реальные значения initialization/final/optimizer safetensors, а не только headers и hashes.
- P08: каждый Planner record обязан ссылаться на canonical P07 training report/checkpoint manifest для точных `variant × seed × regime`.
- Release: `validation/validate_bundle.py` обязан прогонять `validation/` и `tests/`.
