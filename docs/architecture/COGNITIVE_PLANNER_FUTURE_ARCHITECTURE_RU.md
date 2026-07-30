# Cognitive Planner: будущая архитектура

## Статус документа

Это **ненормативный roadmap с proposed interfaces**. Каждый этап становится
обязательным только через отдельный versioned experiment contract. Документ не
изменяет scientific lock, confirmatory protocol, нормативные контракты v1.21,
код, schemas или текущую реализацию A2/A3/A4. При конфликте приоритет имеют
действующие нормативные контракты.

## Почему архитектура была расширена после начала реализации

Предыдущая архитектура не была бесполезной или полностью неверной. Она была
достаточна для технического MVP A2, построения frozen plan за один Planner call,
начала A3a, проверки работы latent feedback path и создания воспроизводимого
training/evaluation pipeline. Однако она недостаточно строго описывала полную
исследовательскую гипотезу.

Первоначальный вопрос был:

> Можно ли добавить латентный вектор шага и использовать его при вычислении
> следующего шага?

Расширенная архитектура добавляет более строгий вопрос:

> Передаётся ли содержание между смысловыми шагами через ограниченный
> concept-level interface, а не через hidden states, KV-cache, autoregressive
> token history или другие обходные каналы?

Новая архитектура не отменяет A2 и не требует выбрасывать текущий A3. Она
отделяет технический channel prototype от claim-bearing semantic bottleneck
experiment, делает более сильную гипотезу причинно проверяемой и опровержимой и
допускает как `GO`, так и `STOP/REDESIGN`.

## Статус текущего A3

Фактический target текущего A3 строится из typed BlocksWorld signature
(`action`, `arg1`, `arg2`), seed и target configuration как deterministic
SHA-derived code — 384-мерный L2-normalized vector. Это deterministic
development-only codebook и код назначения типизированного шага, а не frozen
semantic encoder, semantic geometry или универсальное представление мысли.

Каноническое описание текущего состояния:

```text
implementation A3
→ experimental arm A3a-codebook
→ Stage 2A channel prototype
```

Реализация подтверждает только, что latent feedback path технически существует,
predicted latent можно передавать следующему шагу, pipeline сохраняет и
проверяет latent artifacts, а controlled substitution может изменять downstream
computation. Само наличие канала и current sensitivity tests не доказывают его
причинное использование. `causal use` требует сравнений с zero, shuffled,
foreign/wrong-task feedback и другими predeclared interventions.

Current sensitivity tests также не доказывают улучшение task quality, полезную
semantic representation, semantic geometry, преимущество concept-level
reasoning или подтверждение исходной гипотезы. A2 уже передаёт previous action и
previous argument references через структурный channel; поэтому A3a-codebook в
значительной степени кодирует информацию, частично уже доступную A2.

## Canonical arm and stage registry

Во всех будущих experiment contracts и artifacts используются отдельные поля
`architecture_stage`, `implementation_variant`, `experimental_arm` и
`target_type`. Одно поле `variant` не должно объединять эти четыре смысла.

### Реализованные варианты

| implementation_variant | experimental_arm | architecture_stage | target_type | feedback_policy | history_policy | status | scientific_interpretation |
|---|---|---|---|---|---|---|---|
| A2 | A2-structured-baseline | STAGE_1 | NONE | NONE | STRUCTURED_AUTOREGRESSIVE | IMPLEMENTED | structured planning baseline |
| A3 | A3a-codebook | STAGE_2A | DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK | PREVIOUS_PREDICTED_LATENT | EXISTING_AUTOREGRESSIVE_HISTORY | IMPLEMENTED | latent channel prototype only |
| A4 | A3a-zero | STAGE_2A | DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK | COMPUTE_THEN_ZERO | SAME_AS_A3A_CODEBOOK | IMPLEMENTED | compute-preserving zero-feedback control |

Implementation variant `A4` не связан с архитектурным Stage 4.

### Запланированные arms

- Stage 2A: `A3a-shuffled`, `A3a-foreign`, `A3s-semantic-target`, `A3s-zero`,
  `A3s-shuffled`, `A3s-foreign`;
- A3b: `A3b-history`, `A3b-recurrent`, `A3b-no-history`,
  `A3b-zero-history`, `A3b-shuffled-history`, `A3b-wrong-task-history`.

Текущий `A3a-codebook` уже является random-codebook arm. Второй такой arm не
создаётся автоматически. Дополнительный codebook control допустим только ради
predeclared цели: другого seed, независимого назначения кодов или
capacity-matched сравнения с `A3s-semantic-target`.

## Proposed LatentConceptStep

Следующий интерфейс является proposed и не реализуется этим roadmap:

```text
LatentConceptStep {
  step_id
  trajectory_kind: REASONING | ANSWER
  trajectory_id
  position_index
  previous_step_hash?
  allowed_context_hash
  step_type: REASON | ACTION | TOOL | ANSWER | END
  z_semantic: float[d]
  grounding_refs: unordered set<entity_id>
  tool_ref?: canonical_tool_id
  source_observation_hash?: sha256
  step_hash
}
```

`trajectory_kind` имеет только значения `REASONING | ANSWER`.
`trajectory_id` назначается до первого шага, не зависит от будущих шагов.
`position_index` начинается с 0, индексы непрерывны. `previous_step_hash = null`
допустим только у первого шага; каждый следующий ссылается на предыдущий
canonical step. `step_hash` покрывает все поля объекта. Поэтому одинаковый
`z_semantic` в другой trajectory или context получает другой `step_hash`.

Для `REASONING` разрешены `REASON`, `ACTION`, `TOOL`, `END`; `ANSWER` запрещён.
Для `ANSWER` разрешены только `ANSWER`, `END`; `REASON`, `ACTION`, `TOOL`
запрещены. Planning `END` и answer `END` различаются через `trajectory_kind`,
`trajectory_id`, `position_index` и `allowed_context_hash`.

### allowed_context_hash

`allowed_context_hash` покрывает canonical manifest всего разрешённого
контекста. Произвольный runtime-object нельзя напрямую использовать как context
hash. Любое изменение разрешённого контекста обязано менять hash.

Manifest reasoning trajectory включает:

- task encoding hash;
- `InitialPublicState` hash;
- выбранный `PlanningState` mode и текущий `PlanningState` hash, если режим это
  допускает;
- разрешённые entity/tool catalogs;
- experiment config hash;
- model/checkpoint identity;
- observation prefix hash для closed-loop;
- разрешённые service/position features.

Manifest answer trajectory включает:

- `VerifiedSolution` hash;
- `CompactReasoningSummary` hash, указанный в `VerifiedSolution`;
- permitted user-context policy;
- answer experiment config и answer position;
- разрешённую answer-concept history;
- model/checkpoint identity.

## Reasoning → Answer boundary

> VerifiedSolution является единственным корневым разрешённым интерфейсом между
> reasoning и answer trajectories.

`CompactReasoningSummary` — отдельный content-addressed artifact с собственным
hash. Этот hash записан внутри `VerifiedSolution`; Answer Planner загружает
summary только транзитивно через `VerifiedSolution`. Summary нельзя отдельно
выбрать или заменить. Произвольный summary с подходящей схемой, но другим hash,
запрещён.

```text
VerifiedSolution
→ referenced CompactReasoningSummary
→ allowed answer context
```

Summary не является независимым параллельным интерфейсом.

## Proposed ClosedLoopTrajectory

Future/non-implemented interface:

```text
ClosedLoopTrajectory {
  trajectory_id
  initial_public_state_hash
  ordered_concept_step_hashes[]
  ordered_resolved_action_hashes[]
  ordered_observation_hashes[]
  transition_policy_hash
  verifier_feedback_policy_hash
  planner_call_count
  model_forward_count
  trajectory_status
  terminal_failure_code?
  trajectory_hash
}
```

Каждый action связан с соответствующим предыдущим ConceptStep, каждая
Observation — с исполненным action, а первый ConceptStep после Observation
содержит соответствующий `source_observation_hash`. Порядок
concept/action/observation событий однозначен. Experiment contract задаёт max
steps и max planner/model calls; unbounded repair loop запрещён.

Verifier feedback типизирован и hash-bound через verifier feedback policy. Он не
передаёт gold next action, oracle answer, hidden solution, privileged state delta
или готовый repair instruction. Полная JSON Schema и реализация
`ClosedLoopTrajectory` остаются future backlog.

## Future answer lineage objects

Оба интерфейса proposed/non-implemented; полные schemas в этом PR не
проектируются.

```text
FrozenAnswerPlan {
  source_verified_solution_hash
  source_summary_hash
  answer_concept_step_hashes[]
  answer_context_hash
  answer_planner_call_count
  freeze_status
  answer_plan_hash
}
```

```text
VerbalizationTrace {
  source_answer_plan_hash
  ordered_answer_step_hashes[]
  ordered_span_hashes[]
  verbalizer_model_hash
  tokenizer_hash
  adapter_hash
  prompt_template_hash
  decoding_config_hash
  final_answer_hash
  trace_hash
}
```

`FrozenAnswerPlan` создаётся после `VerifiedSolution`, а `source_summary_hash`
совпадает с hash внутри него. Verbalizer не создаёт следующий AnswerConceptStep
и не меняет FrozenAnswerPlan. Каждый verbalized span связан ровно с одним
`ANSWER` step; final answer — каноническая конкатенация spans. Contextual
production mode не используется как доказательство использования `z_semantic`.

## ResolvedActionStep и tool identity

В proposed `ResolvedActionStep` identity инструмента не дублируется:

```text
# обычный ACTION
action_or_tool: concrete typed action operator
tool_ref: null

# TOOL
action_or_tool: CALL_TOOL
tool_ref: canonical_tool_id
```

`CALL_TOOL` — generic operator. Конкретный tool ID нельзя одновременно кодировать
в `action_or_tool` и `tool_ref`.

## Grounding policy

Для BlocksWorld и иных контролируемых доменов с однозначным gold contract может
требовать exact-set match и exact-minimal-set и регистрировать missing refs,
extra refs и oracle cardinality.

Для текстовых и интерактивных доменов нельзя автоматически требовать один exact
set. Experiment contract определяет equivalence classes, set-valued gold,
multiple valid groundings, admissibility checker, partial-credit policy и
ambiguity policy.

## Versioned Stage 2A experiment contract

До claim-bearing A3b необходим versioned Stage 2A contract. Он фиксирует:

- primary hypothesis и primary metric;
- implementation variants, experimental arms и target types;
- target generation provenance и seeds;
- held-out split, split-before-target-generation rule и leakage checks;
- training-data matching;
- parameter-count и active-parameter reporting;
- FLOPs/compute/token-budget reporting;
- checkpoint-selection rule;
- feedback interventions и failure policy;
- quantitative `GO/STOP` criteria.

Минимальное сравнение:

```text
A2-structured-baseline
A3a-codebook
A3a-zero
A3a-shuffled
A3a-foreign
A3s-semantic-target
A3s-zero
A3s-shuffled
A3s-foreign
```

Второй random-codebook arm добавляется только при predeclared необходимости.

## Stage 2A semantic gate перед A3b

> Переход к claim-bearing A3b experiment запрещён до прохождения Stage 2A
> semantic gate.

До gate разрешён только ограниченный технический scaffolding: interfaces, type
definitions, isolated validators и non-claiming prototypes. Запрещены
содержательное обучение A3b, claim-bearing evaluation, интерпретация результатов
как concept-level reasoning и автоматический переход к следующему этапу.

Обязательные условия gate:

- held-out evaluation и несколько заранее определённых seeds;
- A2, A3a-codebook и A3a-zero;
- shuffled-feedback и foreign/wrong-task-feedback interventions;
- отдельный настоящий semantic-target arm;
- semantic-target против codebook, zero и shuffled/foreign;
- task success, action validity и plan executability;
- заранее определённая structural generalization;
- parameter count и active parameter count;
- compute/FLOPs/token budget;
- checkpoint-selection rule;
- quantitative `GO/STOP` criteria.

> Если semantic-target arm не отличается от codebook, zero и соответствующих
> interventions ожидаемым образом, результат равен STOP/REDESIGN, а не
> автоматическому GO в A3b.

Техническая работоспособность latent-канала сама по себе недостаточна.

### Human-readable examples

Evaluation report обязан включать qualitative diagnostic с input task, initial
state, goal, predicted plan, gold/reference plan, execution trace,
success/failure и сравнением A2/A3a/A4. Эти примеры не являются самостоятельным
BLOCKER gate, не заменяют held-out metrics или causal interventions и не служат
основным quantitative criterion.

## Граница научной интерпретации A3b

Даже положительный A3b позволяет утверждать только:

> В данном домене, распределении задач и заранее зафиксированном
> экспериментальном режиме ограниченное concept-level representation является
> достаточным и причинно полезным интерфейсом между шагами.

Он не доказывает буквальное человеческое мышление, универсальную структуру мысли,
перенос на произвольные reasoning tasks, фундаментальность выбранной размерности
или ConceptStep boundaries и superiority во всех доменах.

## Неблокирующий backlog до A3b

- полные JSON Schema `ClosedLoopTrajectory`, `FrozenAnswerPlan` и
  `VerbalizationTrace`;
- Answer Planner, Verbalizer и closed-loop execution;
- Boundary Head;
- текстовые и интерактивные домены;
- дополнительные random-code seeds сверх predeclared minimum;
- latent-space visualizations;
- улучшение semantic targets после первого заранее зафиксированного semantic arm.

Эти пункты не блокируют Stage 2A gate и не реализуются данным documentation-only
изменением.
