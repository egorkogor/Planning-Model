# Planner → frozen LLM: Stage 1 runbook

**Версия:** 2.13
**Дата:** 26 июля 2026
**Implementation Spec:** `Planner_MVP_MicroModel_Implementation_Spec_RU_v1.13.md`.

Документ задаёт Stage 1A/1B и действия агента. Технические defaults уже вынесены в контракты; оператор их не выбирает.

---

# 1. Hypotheses

## Stage 1A — interface efficacy

На одном frozen oracle state правильный oracle intent должен повышать вероятность valid action, уменьшающего shortest distance.

```text
I1_ORACLE_CURRENT_RAW − I0_EQUAL_TOKENS_RAW
```

Primary metric: `parsed AND valid AND distance_after < distance_before`.

## Stage 1B — end-to-end efficacy

На independent receding-horizon trajectories predicted current intent frozen Planner должен повышать task goal success.

```text
E1_PLANNER_CURRENT_RAW − E0_EQUAL_TOKENS_RAW
```

Stage 1B запрещён без `GO_INTERFACE`.

---

# 2. Roles и blindness

- Builder Agent: code, development, pilot, freeze; confirmatory plaintext запрещён.
- Evaluation Runner: отдельная среда, approved freeze, sealed key, read-only source commit, write-only results.
- Audit Agent: separate clean checkout, read-only verification.

Single-agent/unsealed confirmatory = `INVALID_CONFIRMATORY_BLINDNESS`, независимо от metrics.

---

# 3. Frozen LLM

Model: exact immutable revision `Qwen/Qwen3.5-0.8B`; model, tokenizer и processor revisions фиксируются 40-символьными commit SHA и локальными file hashes.

Runtime:

- greedy, temperature=0, top_p=1;
- thinking/tools off;
- fixed max output;
- exact tokenizer/chat template hashes;
- exact Python/Torch/Transformers revision; если используется vLLM — exact vLLM revision;
- CUDA runtime, driver, cuDNN и kernel versions;
- `language_model_only=true`, `local_files_only=true` и hash offline-cache manifest;
- no sampling fallback.

Determinism: 8 golden prompts × 20 calls. Raw bytes, parsed action и unpadded token tensors должны совпасть 20/20.

---

# 4. Prompt development

P11 использует только development partition и фиксированные C01–C04 из `prompt_development_contract_v1.yaml`.

Floors:

- format success ≥0.95;
- valid action rate ≥0.85;
- oracle-intent progress ≥0.55.

Score:

```text
0.20 format + 0.30 valid + 0.50 oracle progress
```

Если ни один candidate не проходит floors, Stage 1 получает `BLOCKED_PROMPT_STACK`. Нельзя переносить prompt tuning в pilot.

---

# 5. Prompt/parser contract

Aliases: `@B0→block_0 ... @B7→block_7`. Display names игнорируются.

User sections:

1. TASK;
2. AVAILABLE_BLOCKS;
3. CURRENT_STATE;
4. GOAL;
5. GUIDANCE_BLOCK;
6. ACTION_SIGNATURES;
7. OUTPUT_FORMAT.

Action signatures явно задают arity. Parser принимает ровно один JSON object с keys `schema_version`, `action`, `args`; repair и extra text запрещены.


## Exact chat layout

Selected candidate defines the complete ordered message sequence. C02/C04 encode each demonstration as a separate `user`/`assistant` pair; the final runtime task is the last `user` message. Concatenating examples into system text or runtime user text is a contract violation. Prompt hash covers roles, exact UTF-8 contents and rendered unpadded tensors.

## Token policy

Canonical artifact — **unpadded** rendered prompt, input IDs, attention mask и position IDs.

- max unpadded length 512;
- truncation forbidden;
- guidance всегда ровно 32 attended tokens;
- Stage 1A base prompt одинаков, поэтому total unpadded length naturally equal;
- Stage 1B trajectories могут иметь разную total length; сравнивается только guidance budget;
- при batching разрешён только left padding;
- right padding запрещён;
- generation берёт logits последнего attended token.

---

# 6. Stage 1A

## Unit

```text
(base_task_id, oracle_step_index, canonical_snapshot_hash)
```

Task estimator — mean progress по snapshots внутри task; затем равновзвешенное mean по tasks. Bootstrap resamples base tasks cluster-wise.

## Arms

| Arm | Guidance |
|---|---|
| I0_EQUAL_TOKENS_RAW | neutral 32-token block |
| I1_ORACLE_CURRENT_RAW | executable labeler intent |
| I2_SHUFFLED_RAW | pre-certified incompatible intent |
| I3_PLANNER_CURRENT_RAW | Planner/resolver intent |

Controls сертифицируются до LLM calls. Snapshot без incompatible intent исключается из всех arms до outcomes и фиксируется в inclusion manifest.

## One-step execution

```text
sealed snapshot
→ render unpadded prompt per arm
→ persist PromptArtifact
→ greedy call
→ persist raw bytes
→ strict parser
→ Validator
→ optional one-action Executor
→ recompute exact distance
→ AttemptLog
```

Retry/mask/repair отсутствуют.

## GO_INTERFACE

Все условия обязательны:

1. I1−I0 estimate ≥5 п.п.;
2. lower 95% CI >0;
3. I1 progress ≥0.60;
4. I1−I2 lower CI >0;
5. I2≈I0 по paired TOST, margin ±2 п.п.;
6. prompt/hash/contract violations =0;
7. parse failure ≤5%, infrastructure failure ≤1%, incomplete pairs ≤1%.

N = maximum из primary CI, paired power, I1-vs-I2, TOST и minimum budget.

Для Planner sample-size TOST-компонент не применяется и отсутствует в input/report. Он обязателен только для Stage 1A и Stage 1B, где есть equivalence gate shuffled≈neutral. Если N превышает `stage1a_confirmatory_reserve.target_base_tasks` из locked split contract — `BLOCKED_PROTOCOL_CAPACITY`.

---


## Hidden confirmatory certification

Builder certifies only the public algorithm and exclusion manifest in P15. Data Sealer independently generates hidden Stage 1B candidates, enumerates every reachable non-goal state, verifies incompatible-control availability and Planner support eligibility, excludes failures before HMAC ranking, and signs `control_certification` with `coverage_rate=1.0`. Builder receives hashes/counts only. Post-outcome exclusion is forbidden.

# 7. Stage 1B support certification

До pilot outcomes P15:

1. строит exact valid-state transition graph для n=3..6;
2. для каждой candidate task строит depth-16 reachable set;
3. reverse BFS даёт distance и all shortest first actions;
4. executable labeler вычисляет compatible intent set;
5. selector фиксирует incompatible intent для каждого reachable non-goal pair;
6. task включается только при 100% control coverage;
7. support signature threshold проверяется до outcomes.

Post-trajectory exclusion запрещён. Если control отсутствует хотя бы в одном reachable state, task заранее исключается целиком.

---

# 8. Stage 1B arms

| Arm | Candidate source |
|---|---|
| E0_EQUAL_TOKENS_RAW | LLM + neutral guidance |
| E1_PLANNER_CURRENT_RAW | current state → Planner → resolver → LLM |
| E2_SHUFFLED_RAW | certified incompatible intent → LLM |
| E3_ORACLE_REPLAN_RAW | current state → oracle labeler → LLM |
| P_REPLAY_RAW | current state → raw Planner TypedAction, no LLM |

Trajectories независимы. Нельзя переиспользовать state, plan suffix или action другого arm.

---

# 9. Stage 1B loop

```python
state = task.initial
if goal_pass(state):
    return SUCCESS_ZERO_STEP

for step_index in range(16):
    if arm == E1:
        planner_step, semantic_artifact = planner.predict_current_unmasked(task, state)
        persist_before_validation(planner_step, semantic_artifact)
        resolution = resolver.resolve(semantic_artifact)
        if resolution.unresolved:
            save_terminal_attempt(llm_called=False, issue="SEMANTIC_UNRESOLVED")
            return FAILURE
        guidance = resolution.intent_text
    elif arm == E3:
        guidance = oracle_labeler(state, goal)
    elif arm == E2:
        guidance = frozen_control[state_hash(state), goal_hash(goal)]
    elif arm == E0:
        guidance = NEUTRAL
    elif arm == P_REPLAY:
        action = planner.predict_current_unmasked(task, state).typed_action
        persist_raw_candidate(action)
        if not validator.valid(action, state):
            return FAILURE
        state = executor.apply(action, state)
        if goal_pass(state):
            return SUCCESS
        continue

    prompt = render_unpadded(task, state, guidance)
    assert len(prompt.input_ids) <= 512
    persist_prompt(prompt)
    raw = llm.greedy(prompt, batch_padding_side="left")
    persist_raw_before_parse(raw)
    parsed = strict_parser(raw)
    result = validator.check(parsed, state)
    persist_attempt(...)
    if not result.valid:
        return FAILURE
    state = executor.apply(result.normalized_action, state)
    if goal_pass(state):
        return SUCCESS

return HORIZON_EXCEEDED
```

---

# 10. Terminal flows

`SEMANTIC_UNRESOLVED`: LLM not called; prompt/output/action null; resolver diagnostics mandatory.

`LLM_TIMEOUT`: no rerun in confirmatory; raw partial bytes retained; incomplete rate applies.

`PARSE_FAILED`: raw retained; no guessed action.

`INVALID_ACTION`: parsed action retained; no Executor call.

Goal reached after action 16 has priority over horizon failure.

---

# 11. Logging и validation

Every persisted JSON follows `report_registry_v1.yaml`.

AttemptLog stores:

- task/stage/arm/step/pair hashes;
- state before/after, goal hash;
- unpadded prompt/tensor hashes and actual tokens;
- raw output before parser;
- parsed and normalized actions separately;
- Planner/resolver/control diagnostics;
- issue, latency, resource and terminal fields.

EpisodeLog derives from attempts. Validator recomputes call counts, token totals, actions, latency, final state и terminal status. Stage 1A pair must contain I0–I3. Stage 1B verifies frozen task/control eligibility, not equal trajectory length.

---

# 12. GO_END_TO_END

All mandatory:

1. E1−E0 goal success estimate ≥5 п.п.;
2. lower 95% CI >0;
3. E1−E2 lower CI >0;
4. E2≈E0 by paired TOST ±2 п.п.;
5. E3 goal success ≥0.60;
6. semantic unresolved ≤10%;
7. unseen support signature ≤5%;
8. prompt/hash/contract/determinism violations =0;
9. infrastructure ≤1%, incomplete pairs ≤1%, parse failure ≤5%.

P_REPLAY/E3 diagnostic и не заменяют primary gate.

---

# 13. Phases и stop transitions

Stage 1 соответствует P10–P18:

- P10 locked LLM/resolver;
- P11 prompt development/freeze;
- P12 Stage 1A pilot/N/freeze, gate G12;
- P13 sealed confirmatory;
- P14 GO_INTERFACE;
- P15 Stage 1B support audit;
- P16 pilot/N/freeze, gate G16;
- P17 sealed confirmatory;
- P18 GO_END_TO_END.

При `STOP_INTERFACE` P15–P18 получают `SKIPPED_BY_CONTRACT`, затем P19 всё равно проводит audit отрицательного результата. Missing contract, lock mismatch, unsealed confirmatory, nondeterminism и hash violation не являются limitation — это `BLOCKED`/`INVALID_RUN`.

# 15. Исторические исполняемые уточнения v2.9

1. Prompt candidates C01–C04 загружаются как exact UTF-8 из `docs/prompt/candidates/`; переписывание запрещено.
2. Восемь guidance blocks создаются детерминированно после tokenizer lock, имеют ровно 32 attended tokens и замораживаются в `locks/guidance-token-artifacts.json`. Runtime padding search запрещён.
3. Resolver thresholds выбираются на development и замораживаются до pilot.
4. Sample size и scientific decision пересчитываются нормативными Python-модулями; произвольный метод недопустим.
5. Confirmatory datasets создаёт Data Sealer, а не Builder.


# 16. Исторические исполняемые уточнения v2.9

1. Stage 1A разрешён как диагностический interface test даже при `STOP_PLANNER` или `GO_PLANNER_DIAGNOSTIC_ONLY`; Stage 1B в этих случаях запрещён.
2. `GO_INTERFACE_STAGE1B_ELIGIBLE` возможен только при сохранённом flag `stage1b_eligible=true` из Planner decision. Иначе положительный interface result становится `GO_INTERFACE_DIAGNOSTIC_ONLY` и ведёт сразу к audit.
3. Stage 1A gates включают oracle progress floor I1 ≥0.60. Stage 1B gates включают E3 oracle episode floor ≥0.60 и unseen-support-signature rate ≤0.05.
4. Все CI, TOST и решения пересчитываются из schema-valid `AnalysisInput`; готовые estimates не считаются источником истины.
5. Перед первым pilot/confirmatory dispatch должен существовать approved G06 statistical implementation audit от независимого reviewer.


### Нормативный вход статистики

`AnalysisInput` не хранит только готовые оценки. Каждая paired comparison содержит строки `pair_id, left, right, difference`, причём валидатор пересчитывает `difference = left − right`, запрещает дубли и требует одинаковый набор `base_task_id` во всех Planner seeds. Stage 1A хранит snapshot-pairs внутри task cluster; scalar rates содержат `unit_id, value`. Auditor обязан восстановить эти строки из raw result manifest; готовые агрегаты без pair rows недопустимы.

## Приложение C. Источник P_REPLAY floor

До допуска Stage 1B Evaluation Runner исполняет `docs/controls/p_replay_contract_v1.yaml`. `P_REPLAY_RAW` — не заранее записанный полный план: это `current state → A3 raw action → execute → current state` до goal или terminal failure. Любая маска, повтор, исправление или LLM-вызов делает результат недействительным.
