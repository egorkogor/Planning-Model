# Cognitive Planner — расширенная архитектура исходной гипотезы

**Статус:** non-normative architecture roadmap; интерфейсы зафиксированы для перехода к experiment contracts.  
**Назначение:** описать архитектуру, выходящую за пределы узкого эксперимента Work Planner / BlocksWorld v1.21.  
**Важно:** документ не изменяет scientific lock, confirmatory protocol или контракты v1.21.

## 1. Исходная гипотеза

Модель может рассуждать не только последовательностью текстовых токенов, но и последовательностью более крупных смысловых единиц — `LatentConceptStep`.

Один шаг соответствует одному законченному смысловому фрагменту:

- шагу рассуждения;
- шагу плана;
- действию или вызову инструмента;
- смыслу одного предложения или абзаца ответа.

После построения смысловой траектории frozen LLM используется как Verbalizer: переводит уже сформированный смысл шага в естественный язык, но не ищет решение заново.

Ключевая проверяемая версия гипотезы:

> Если смысл шага проходит через отдельный ограниченный интерфейс, а следующие шаги и downstream-модули не могут обойти этот интерфейс через скрытые состояния Reasoner, то последовательность крупных смысловых шагов может быть полезнее или эффективнее последовательности только текстовых токенов.

### 1.1. Почему архитектура была расширена после начала реализации

Предыдущая архитектура не была бесполезной или полностью неверной: она была достаточна для технического MVP A2 и начала A3 и проверяла, можно ли добавить latent feedback path. Однако она недостаточно строго описывала claim-bearing semantic bottleneck. Расширенная архитектура не отменяет A2 или текущий A3; она отделяет технический channel prototype от semantic experiment и делает более сильную гипотезу причинно проверяемой и опровержимой. Результатом проверки может быть как `GO`, так и `STOP/REDESIGN`.

Исходный технический вопрос:

> Можно ли добавить латентный вектор шага и использовать его при вычислении следующего шага?

Более строгий вопрос:

> Передаётся ли содержание между смысловыми шагами через ограниченный concept-level interface, а не через hidden states, KV-cache, autoregressive token history или другие обходные каналы?

## 2. Архитектурные объекты

Полная схема разделяет:

1. смысл шага;
2. привязку смысла к сущностям и инструменту;
3. исполнимую команду;
4. исходное и предполагаемое состояние планирования;
5. внешний результат исполнения;
6. замороженный план;
7. фактическую execution trace;
8. проверенное решение;
9. план ответа.

### 2.1. LatentConceptStep

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

`z_semantic` — предлагаемое латентное смысловое представление шага в полной архитектуре. Размерность не является фундаментальным ограничением гипотезы; размерность `384` у текущего A3 относится к development-only codebook и сама по себе не придаёт ему семантической геометрии.

`trajectory_id` назначается до первого шага и не зависит от будущих шагов. `position_index` начинается с `0` и непрерывен. `previous_step_hash` равен `null` только у первого шага. `step_hash` покрывает все поля объекта, поэтому одинаковый latent в другой trajectory или в другом разрешённом контексте имеет другой hash.

Для `trajectory_kind: REASONING` разрешены только `REASON`, `ACTION`, `TOOL`, `END`; `ANSWER` запрещён. Для `trajectory_kind: ANSWER` разрешены только `ANSWER`, `END`; `REASON`, `ACTION`, `TOOL` запрещены. Planning `END` и Answer `END` различаются как минимум через `trajectory_kind`, `trajectory_id`, `position_index` и `allowed_context_hash`.

#### allowed_context_hash

`allowed_context_hash` хеширует canonical manifest разрешённого контекста, а не произвольный runtime object. Любое изменение разрешённого контекста обязано менять hash.

Reasoning manifest включает task encoding hash, hash `InitialPublicState`, режим `PlanningState` и его hash, когда применимо, entity/tool catalogs, experiment config hash, model/checkpoint identity, observation prefix для closed-loop, а также service и position features.

Answer manifest включает hash `VerifiedSolution`, hash `CompactReasoningSummary`, указанный внутри этого `VerifiedSolution`, permitted user-context policy, answer experiment config, answer position, разрешённую answer-concept history и model/checkpoint identity.

#### Каноническая сериализация z_semantic

До реализации или запуска experiment contract фиксирует:

- `dtype` и точную `shape`;
- byte order;
- перевод в CPU и contiguous layout перед сериализацией;
- запрет `NaN`, `+Inf` и `-Inf`;
- exact-byte либо заранее определённое quantized representation;
- параметры quantization, если она используется;
- каноническое кодирование tensor metadata;
- domain separator и полный набор полей, покрываемых `step_hash`.

Одинаковая serialization policy применяется ко всем сравниваемым arms. Изменение policy после просмотра результатов запрещено. `step_hash` вычисляется только из канонических байтов объекта.

`grounding_refs` связывает общий смысл с конкретными сущностями. `tool_ref` отдельно выбирает разрешённый инструмент для шага `TOOL`.

#### Ограничения grounding_refs

В BlocksWorld и других контролируемых доменах с однозначным gold `grounding_refs`:

- являются семантически неупорядоченным множеством canonical entity IDs;
- сериализуются в каноническом порядке и дедуплицируются;
- не содержат оператор или тип действия;
- не содержат отношения между объектами;
- не содержат роли аргументов вроде `moving`, `support`, `source`, `target`;
- не содержат произвольный текст, готовый план или команду;
- имеют заранее зафиксированное максимальное количество элементов;
- являются точным минимальным множеством сущностей текущего шага;
- не содержат лишних сущностей и не пропускают обязательные.

В контролируемом домене проверяются:

- exact-set match;
- oracle cardinality match;
- missing-ref rate;
- extra-ref rate;
- predicted refs против oracle refs;
- refs-only performance.

Для текстовых и интерактивных доменов единственный exact set не является универсальным требованием. Experiment contract задаёт equivalence classes, set-valued gold, multiple valid groundings, admissibility checker, partial-credit policy и ambiguity policy.

Оператор, отношения и роли аргументов должны восстанавливаться из `z_semantic` и допустимых структурных контрактов.

#### Ограничения tool_ref

`tool_ref`:

- отсутствует для `REASON`, `ACTION`, `ANSWER` и `END`;
- обязателен и единственен для `TOOL`;
- выбирается только из разрешённого canonical tool catalog;
- не содержит аргументы, текст запроса, роли или готовый tool call;
- проверяется zero/shuffled/wrong-tool interventions;
- имеет одинаковый каталог и одинаковую информационную ёмкость во всех сравниваемых arms.

### 2.2. ResolvedActionStep

```text
ResolvedActionStep {
  source_concept_step_id
  action_or_tool: typed operator
  ordered_args: grounded references[]
  tool_ref?: canonical_tool_id
  resolution_status
  step_hash
}
```

Создаётся Action Decoder / Tool Resolver только для `ACTION` или `TOOL`. Не является частью исходного латентного шага.

Tool identity не дублируется: для обычного `ACTION` используются `action_or_tool: concrete typed action` и `tool_ref: null`; для `TOOL` — `action_or_tool: CALL_TOOL` и `tool_ref: canonical_tool_id`. Конкретный tool ID нельзя одновременно кодировать в `action_or_tool` и `tool_ref`.

### 2.3. InitialPublicState

```text
InitialPublicState {
  canonical_state
  entity_catalog_hash
  tool_catalog_hash
  task_constraints_hash
  initial_state_hash
}
```

Это фактическое публичное состояние в начале эпизода. Оно неизменяемо и является единственным реальным состоянием, доступным при начале open-loop planning.

### 2.4. PlanningState

`PlanningState` используется только в варианте deterministic planning rollout.

```text
PlanningState {
  source_initial_state_hash
  source_resolved_action_hashes[]
  transition_function_hash
  predicted_state
  rollout_status
  planning_state_hash
}
```

Правила:

- строится только детерминированной transition function из `InitialPublicState` и уже созданных `ResolvedActionStep`;
- не содержит реальные `ObservationEvent`;
- не подменяет фактическое состояние исполнения;
- одинаковая transition function и одинаковая policy доступа применяются во всех сравниваемых arms;
- ошибка rollout фиксируется типизированно и не исправляется скрытым replanning.

Experiment contract выбирает ровно один planning-state режим:

1. `initial-state-only`: каждый шаг видит только `InitialPublicState`;
2. `deterministic-planning-rollout`: шаг видит текущий `PlanningState`.

Режим нельзя менять после просмотра результатов или смешивать внутри одного confirmatory comparison.

### 2.5. ObservationEvent

Observation является внешним событием среды, а не мыслью модели.

```text
ObservationEvent {
  source_action_step_id
  success_or_failure
  environment_state_delta
  tool_result_or_error
  observation_hash
}
```

### 2.6. FrozenConceptPlan

Open-loop planning завершается отдельным артефактом:

```text
FrozenConceptPlan {
  concept_step_hashes[]
  resolved_action_step_hashes[]
  initial_public_state_hash
  final_planning_state_hash?
  planning_state_mode
  planning_context_hash
  plan_check_hash
  planner_call_count: 1
  freeze_status
  plan_hash
}
```

Последним `LatentConceptStep` замороженного плана является `END`.

После freeze запрещены:

- изменение шага;
- suffix regeneration;
- повторный вызов Planner;
- замена `LatentConceptStep` или `ResolvedActionStep`;
- использование Observation для изменения frozen suffix.

### 2.7. ExecutionTrace

Open-loop execution создаёт отдельный lineage-артефакт:

```text
ExecutionTrace {
  source_frozen_plan_hash
  ordered_action_step_hashes[]
  ordered_observation_hashes[]
  execution_policy_hash
  initial_state_hash
  final_state_hash
  failure_status
  execution_trace_hash
}
```

Правила:

- `source_frozen_plan_hash` обязан ссылаться на исполняемый `FrozenConceptPlan`;
- порядок action и observation hashes совпадает с фактическим порядком исполнения;
- каждый action принадлежит frozen plan;
- каждый Observation ссылается на соответствующий исполненный action;
- `execution_policy_hash` фиксирует stop/continue policy при ошибке;
- Observation не меняет frozen suffix и не возвращается Planner;
- `execution_trace_hash` охватывает весь объект в канонической сериализации.

### 2.8. CompactReasoningSummary

Summary не является свободным текстовым Chain-of-Thought.

```text
CompactReasoningSummary {
  source_concept_step_hashes[]
  source_action_step_hashes[]
  source_observation_hashes[]
  result_ref
  used_actions[]
  established_facts[]
  unresolved_constraints[]
  summary_hash
}
```

Ограничения:

- строится без LLM;
- свободное поле `summary_payload` запрещено;
- схема и максимальный размер фиксируются до эксперимента;
- не получает `h_t`, KV-cache или скрытую историю Reasoner;
- все элементы выводятся только из сохранённых и проверенных артефактов;
- одинаковые входные артефакты дают byte-identical summary.

### 2.9. VerifiedSolution

```text
VerifiedSolution {
  source_mode: OPEN_LOOP | CLOSED_LOOP
  source_execution_trace_hash?: sha256
  source_trajectory_hash?: sha256
  source_concept_step_hashes[]
  source_action_step_hashes[]
  source_observation_hashes[]
  evaluation_hash
  final_state_or_result
  executed_actions_or_tool_results
  compact_reasoning_summary_hash
  task_constraints_hash
  verification_status
  solution_hash
}
```

Правила lineage:

- в open-loop обязателен `source_execution_trace_hash`;
- в closed-loop обязателен `source_trajectory_hash`;
- один из этих двух источников обязателен, одновременное заполнение запрещено;
- open-loop trace обязана ссылаться на исходный `FrozenConceptPlan`;
- все action/observation hashes должны принадлежать указанной trace или trajectory;
- `evaluation_hash` связывает verifier, критерии и фактическое execution evidence;
- `solution_hash` охватывает весь объект в канонической сериализации.

`VerifiedSolution` является единственным **корневым** разрешённым интерфейсом между reasoning trajectory и answer trajectory. `CompactReasoningSummary` — отдельный content-addressed artifact с собственным hash, который находится внутри `VerifiedSolution`. Answer Planner загружает summary только транзитивно по этому hash; summary нельзя отдельно выбрать или заменить, и он не является параллельным root interface:

```text
VerifiedSolution
→ referenced CompactReasoningSummary
→ allowed answer context
```

## 3. Semantic bottleneck

```text
Reasoner temporary hidden state h_t
          ↓
Semantic Bottleneck
          ↓
step_type_t + z_t + grounding_refs_t + tool_ref_t
          ├→ Action Decoder / Tool Resolver
          └→ Answer Planner / Verbalizer только после VerifiedSolution
```

После создания `LatentConceptStep` downstream-модули не получают `h_t` или другие скрытые состояния Reasoner в обход разрешённых полей.

### 3.1. Output bottleneck

Для A3b downstream-модули получают только:

- `step_type`;
- `z_semantic`;
- ограниченные `grounding_refs`;
- ограниченный `tool_ref` для `TOOL`;
- публичные каталоги;
- минимальные структурные ограничения.

Запрещены прямые пути:

```text
h_t → action
h_t → tool call
h_t → answer text
```

### 3.2. Inter-step bottleneck Reasoner

Внутри одного шага временные hidden states и attention разрешены. На границе `LatentConceptStep` decoder cache и скрытая token-level история сбрасываются или становятся недоступными следующему шагу.

Между шагами разрешены только:

- сохранённые `LatentConceptStep`;
- `InitialPublicState` или `PlanningState` согласно experiment contract;
- последний `ObservationEvent` только в closed-loop;
- неизменяемое task encoding;
- явные позиционные и служебные признаки.

Передача `h_t`, token-level hidden-state history или KV-cache между ConceptStep запрещена.

### 3.3. Варианты concept-level memory

#### A3b-history — основной вариант

```text
[ConceptStep_1, ..., ConceptStep_t]
+ разрешённое состояние
+ разрешённая Observation
→ temporary h_(t+1)
→ ConceptStep_(t+1)
```

#### A3b-recurrent — ablation

Следующий шаг видит только предыдущий ConceptStep.

#### A3b-no-history — обязательный контроль

```text
TaskEncoding
+ разрешённое состояние
+ position/service features
→ ConceptStep_t
```

Контроль получает тот же task context, каталоги и вычислительный бюджет, но не получает concept history.

#### Whole-history interventions

Обязательны:

- `zero-history`;
- `shuffled-history` с сохранением длины, позиций и метаданных;
- `wrong-task-history` той же длины и совместимого формата;
- `truncated-history` с заранее заданным окном;
- `history-only` diagnostic там, где он содержательно допустим.

Польза concept history признаётся только при превосходстве над no-history и ожидаемой деградации в interventions.

### 3.4. Inter-step bottleneck Answer Planner

Для этапа 4 основной Answer Planner также работает через concept-level интерфейс:

```text
VerifiedSolution
→ referenced CompactReasoningSummary
+ [AnswerConceptStep_1, ..., AnswerConceptStep_t]
→ temporary answer hidden state
→ AnswerConceptStep_(t+1)
```

На границе каждого `ANSWER` шага:

- временный hidden state и KV-cache сбрасываются;
- следующий шаг видит только сохранённую answer-concept history, `VerifiedSolution`, summary и разрешённый пользовательский контекст;
- скрытая token-level история между answer steps запрещена.

Обязательный контроль этапа 4:

- `answer-concept-history`: основной bottleneck-вариант;
- `answer-autoregressive-cache`: parameter/compute-matched контроль с обычной hidden/KV history;
- `answer-no-history`: контроль без предыдущих answer ConceptStep;
- shuffled/wrong-order answer-history interventions.

Разбиение ответа на ConceptStep не считается содержательным, если основной вариант не превосходит no-history или не проявляет ожидаемую чувствительность к answer-history interventions.

## 4. Фазовая архитектура

```text
Input + InitialPublicState
        ↓
Task Encoder
        ↓
Reasoner + Semantic Bottleneck
        ↓
REASON / ACTION / TOOL ConceptSteps
        ↓
Action Decoder / Tool Resolver
        ↓
Open-loop: END → PLAN_CHECK → FREEZE_CHECK → FrozenConceptPlan
Closed-loop: execution → ObservationEvent → следующий ConceptStep
        ↓
Open-loop: ExecutionTrace
Closed-loop: typed trajectory
        ↓
SOLUTION_CHECK / verifier
        ↓
VerifiedSolution
        ↓
Answer Planner с answer inter-step bottleneck
        ↓
ANSWER ConceptSteps
        ↓
Frozen LLM Verbalizer
        ↓
Final answer
```

`PLAN_CHECK` и `FREEZE_CHECK` не утверждают, что задача решена. Они проверяют только структурную готовность и неизменяемость плана.

`SOLUTION_CHECK` выполняется только по фактическому execution evidence или по непосредственно проверяемому результату задачи.

## 5. Reasoning и Answer trajectories

### 5.1. Reasoning trajectory

Последовательность `REASON`, `ACTION` и `TOOL` описывает поиск и выполнение решения. Машинно значимые действия остаются типизированными и проверяемыми.

### 5.2. PLAN_CHECK и FREEZE_CHECK

`PLAN_CHECK` запускается только после `END` и проверяет до исполнения:

- валидность state machine planning-фазы;
- разрешимость всех `ACTION/TOOL` в `ResolvedActionStep`;
- допустимость grounding;
- отсутствие запрещённых каналов;
- соответствие planning-state policy;
- наличие ровно одного терминального `END`;
- отсутствие содержательных шагов после `END`.

`PLAN_CHECK` является внешним терминальным валидатором. Он не возвращает управление Planner и не запускает repair-loop.

`FREEZE_CHECK` проверяет:

- полноту хешей;
- единственный Planner call;
- неизменяемость шагов;
- корректность `planning_context_hash`;
- готовность создать `FrozenConceptPlan`.

Они не создают `VerifiedSolution`.

### 5.3. SOLUTION_CHECK

```text
ExecutionTrace или closed-loop trajectory
→ SOLUTION_CHECK
→ VerifiedSolution | FailedEvaluation
```

В closed-loop verifier может вернуть typed failure или разрешённый контрактом переход обратно к `REASON`. В open-loop возврат к Planner запрещён.

### 5.4. Answer trajectory

После `VerifiedSolution` Answer Planner строит отдельную последовательность `ANSWER` шагов.

Он получает:

- `VerifiedSolution`;
- hash-bound `CompactReasoningSummary`;
- task constraints;
- разрешённый пользовательский контекст;
- сохранённую answer-concept history.

Он не получает hidden states Reasoner или скрытую token-level историю предыдущих answer steps.

## 6. Раздельные state machines

### 6.1. Open-loop planning state machine

```text
START → REASON | ACTION | TOOL | END
REASON → REASON | ACTION | TOOL | END
ACTION → REASON | ACTION | TOOL | END
TOOL → REASON | ACTION | TOOL | END
END → PLAN_CHECK
PLAN_CHECK → FREEZE_CHECK | FailedPlanCheck
FREEZE_CHECK → FrozenConceptPlan | FailedPlanCheck
FrozenConceptPlan → terminal planning state
FailedPlanCheck → terminal failure state
```

Правила:

- `END` обязателен и создаётся внутри единственного Planner call;
- после `END` Planner больше не вызывается;
- `PLAN_CHECK` и `FREEZE_CHECK` являются внешними fail-closed gates;
- возврат `PLAN_CHECK → REASON` запрещён;
- самокоррекция допустима только внутри исходной генерации до выдачи `END`;
- `ACTION` и `TOOL` являются элементами ещё не исполненного плана;
- реальный `ObservationEvent` во время planning отсутствует.

### 6.2. Open-loop execution state machine

```text
FrozenConceptPlan
→ execute ResolvedActionStep_1
→ ObservationEvent_1
→ execute ResolvedActionStep_2
→ ObservationEvent_2
→ ...
→ ExecutionTrace
→ SOLUTION_CHECK
→ VerifiedSolution | FailedEvaluation
```

Правила:

- Observation логируется, но не передаётся Planner;
- Observation не меняет frozen suffix;
- Planner call count остаётся равен одному;
- ошибка исполнения фиксируется fail-closed;
- продолжение или остановка после ошибки фиксируется до запуска;
- `ExecutionTrace` создаётся до `SOLUTION_CHECK` и является обязательным open-loop evidence artifact.

### 6.3. Closed-loop cognitive state machine

```text
START → REASON | ACTION | TOOL | SOLUTION_CHECK
REASON → REASON | ACTION | TOOL | SOLUTION_CHECK
ACTION → external ObservationEvent
TOOL → external ObservationEvent
ObservationEvent → REASON | ACTION | TOOL | SOLUTION_CHECK
SOLUTION_CHECK → VerifiedSolution | REASON | FailedEvaluation
VerifiedSolution → ANSWER | END
ANSWER → ANSWER | END
END → terminal
```

### 6.4. Answer planning state machine

```text
VerifiedSolution → ANSWER | END
ANSWER → ANSWER | END
END → terminal
```

Каждый переход `ANSWER → ANSWER` проходит через answer inter-step bottleneck. Verbalizer формулирует текущий span, но не создаёт следующий ConceptStep.

## 7. Type-specific invariants

### REASON

- `z_semantic` обязателен;
- entity refs опциональны и оцениваются по domain-scoped grounding policy;
- `tool_ref` отсутствует;
- Action Decoder не вызывается;
- Verbalizer используется только диагностически;
- готовая команда отсутствует.

### ACTION

- `z_semantic` обязателен;
- entity refs обязательны, кроме операторов без сущностных аргументов;
- `tool_ref` отсутствует;
- refs проходят domain-scoped grounding policy (exact-minimal-set только там, где gold однозначен);
- должен появиться `ResolvedActionStep`.

### TOOL

- `z_semantic` обязателен;
- `tool_ref` обязателен и единственен;
- refs проходят domain-scoped grounding policy;
- текст запроса, роли и параметры не передаются через refs/tool_ref;
- должен появиться `ResolvedActionStep`;
- результат или ошибка фиксируются в Observation при исполнении.

### ANSWER

- допускается только после `VerifiedSolution`;
- `z_semantic` обязателен;
- refs содержат только сущности текущего answer span;
- `tool_ref` отсутствует;
- Action Decoder не вызывается;
- Verbalizer обязателен;
- span связывается с source concept step;
- hidden/KV state предыдущего answer step недоступен следующему.

### END

- `z_semantic` использует заранее зафиксированное terminal representation либо нулевой canonical vector согласно contract;
- `grounding_refs` пусты;
- `tool_ref` отсутствует;
- Action Decoder и Verbalizer не вызываются;
- после END новые planning steps запрещены;
- в open-loop `END` непосредственно запускает внешний `PLAN_CHECK`.

### source_observation_hash

- в open-loop planning отсутствует;
- в open-loop execution относится к execution evidence, а не к frozen planning step;
- в closed-loop обязателен для первого ConceptStep после Observation;
- точно ссылается на Observation, вызвавшую следующий шаг.

## 8. Open-loop и closed-loop режимы

### 8.1. Open-loop experimental mode

```text
один Planner call
→ полный план с terminal END
→ PLAN_CHECK / FREEZE_CHECK
→ FrozenConceptPlan
→ исполнение без изменений
→ ExecutionTrace
```

Используется для сравнения:

```text
A2
vs A3a
vs A3b-history
vs A3b-recurrent
vs A3b-no-history
```

Planning-state mode одинаков для всех arms.

#### Ограничение open-loop TOOL

В open-loop допускаются только:

- детерминированные и полностью симулируемые tool effects;
- либо заранее замороженные tool outcomes, чей hash зафиксирован до planning.

Запрещён open-loop план, если последующий шаг зависит от неизвестного runtime-результата инструмента. Такие задачи выполняются только в closed-loop.

Реальный tool outcome не используется для изменения frozen suffix.

### 8.2. Closed-loop cognitive mode

```text
LatentConceptStep
→ действие или инструмент
→ ObservationEvent
→ обновление State Store
→ следующий LatentConceptStep
```

Replanning является явной частью архитектуры. Open-loop и closed-loop результаты не смешиваются без отдельного протокола.

## 9. Модули

### Task Encoder

Кодирует запрос, `InitialPublicState`, каталоги и ограничения.

### Planning State Builder

В режиме deterministic rollout применяет зафиксированную transition function к уже созданным `ResolvedActionStep`. Не видит реальные Observation.

### Latent Planner / Reasoner

Создаёт временное внутреннее состояние шага.

- A3a сохраняет обычную autoregressive hidden history, а `z_semantic` является дополнительным feedback-каналом.
- A3b использует только concept-level memory между шагами.

### Semantic Bottleneck

Создаёт `step_type`, `z_semantic`, `grounding_refs` и при необходимости `tool_ref`.

### Grounding Head

Выбирает точное минимальное множество сущностей. Предсказание канонизируется и дедуплицируется.

### Tool Head

Для `TOOL` выбирает один `tool_ref`; не создаёт параметры или текст вызова.

### Action Decoder / Tool Resolver

Преобразует `LatentConceptStep` в `ResolvedActionStep` и получает только разрешённые поля bottleneck и структурный контракт.

### Plan Checker / Freeze Checker

Проверяют структурную валидность и создают `FrozenConceptPlan`, но не подтверждают фактическое решение и не возвращают управление Planner.

### Executor и Execution Trace Builder

Исполняют frozen actions, сохраняют Observation и создают `ExecutionTrace` либо типизированный failure.

### State Store и Solution Verifier

В closed-loop сохраняют trajectory; в обоих режимах проверяют execution evidence и создают `VerifiedSolution` либо типизированный failure.

### Answer Planner

Создаёт `ANSWER` ConceptStep после `VerifiedSolution` и подчиняется отдельному inter-step bottleneck.

### Verbalizer Adapter и Frozen LLM Verbalizer

Преобразуют текущий `ANSWER` ConceptStep в текст. Не меняют план и не создают новую trajectory.

## 10. Два режима Verbalizer

### 10.1. Strict reconstruction mode

Вход:

```text
z_semantic
+ ограниченные grounding_refs
+ tool_ref только для TOOL
+ минимальная инструкция по формату
```

Verbalizer не получает полный вопрос, reasoning history, summary, hidden state или полный verified result.

Обязательные контроли:

- correct/zero/shuffled/wrong-task `z_semantic`;
- shuffled/wrong-task refs;
- exact oracle refs против predicted refs;
- refs-only;
- z-only там, где refs обязательны;
- correct/zero/shuffled/wrong-tool `tool_ref`;
- неправильный `step_type`.

### 10.2. Contextual production mode

Используется после strict reconstruction.

```text
z_semantic
+ grounding refs/tool_ref
+ исходный запрос
+ уже сформированный ответ
+ разрешённый фактический контекст
```

Сам по себе этот режим не доказывает использование latent-вектора.

## 11. Type-specific оценка Verbalizer

- `ACTION/TOOL`: оператор, аргументы, роли, применимость, отсутствие новых сущностей, соответствие Observation.
- `REASON`: промежуточный вывод, связь с предыдущими ConceptStep, отсутствие неподтверждённых фактов, чувствительность к interventions.
- `ANSWER`: покрытие части ответа, согласованность с `VerifiedSolution`, отсутствие новых утверждений, порядок answer-step.
- `END`: корректное завершение, отсутствие содержательного текста и последующих шагов.

## 12. Что считается мыслью на разных этапах

### Stage 2: codebook prototype и supervised semantic arm

Текущий BlocksWorld A3 target — 384-мерный deterministic development-only codebook для назначения типизированного шага. Отдельный будущий semantic-target arm должен использовать заранее зафиксированный frozen semantic encoder или иной versioned semantic target. Ни codebook, ни само наличие канала не проверяют причинную полезность semantic feedback без controls и interventions.

### Stage 3–4: learned ConceptStep representation

`z_semantic` обучается на совокупности сигналов:

- правильность действия или результата;
- предсказание следующего состояния;
- реконструкция смысла шага;
- согласованность trajectory;
- информационный bottleneck;
- контрастивные и intervention-контроли.

## 13. Граница одного ConceptStep

Фраза «одно предложение или абзац» является инженерной гипотезой.

Первая реализация использует:

- разметку шагов из oracle/reasoning trajectory;
- явный `step_type`;
- максимальное число ConceptStep;
- отдельный `END`;
- ограниченную размерность `z_semantic`.

Boundary head не входит в первый MVP.

## 14. Пять крупных этапов проверки

### Этап 1 — A2: структурированный Planner

Один Planner call создаёт воспроизводимый многошаговый frozen plan с terminal `END`, который исполняется без replanning.

### Этап 2A — A3a: latent feedback

Сравниваются A2, правильный feedback, zero, shuffled и random-code parameter-matched control.

### Этап 2B — A3b: semantic и inter-step bottleneck

Сравниваются:

- A3a;
- A3b-history;
- A3b-recurrent;
- A3b-no-history;
- whole-history interventions;
- initial-state-only против deterministic rollout;
- refs-only и z-only;
- oracle refs и predicted refs;
- capacity-matched grounding controls.

Критерий: A3b-history проходит experiment contract, превосходит no-history и проявляет ожидаемую causal sensitivity к history, `z_semantic`, refs, tool_ref и `step_type`.

### Этап 3 — Verbalizer MVP

Проверяется декодируемость отдельной мысли в strict reconstruction mode.

### Этап 4 — разделение Reasoning и Answer

Добавляются `VerifiedSolution`, отдельная answer trajectory и answer inter-step bottleneck. Сравниваются concept-history, autoregressive-cache и no-history варианты.

### Этап 5 — полный сравнительный эксперимент

Домены расширяются последовательно:

1. BlocksWorld;
2. один текстовый домен;
3. один интерактивный домен;
4. затем широкое сравнение.

## 15. Experiment contract перед каждым этапом

До реализации или запуска фиксируются:

- primary hypothesis и metric;
- evaluation unit;
- baselines, ablations и interventions;
- threshold или non-inferiority margin;
- seeds и split policy;
- STOP/GO rule;
- checkpoint-selection rule;
- training-data matching;
- parameter-count и active-parameter matching/reporting;
- FLOPs и token-budget matching;
- одинаковый доступ к task context;
- planning-state mode и transition function hash;
- одинаковые entity/tool catalogs;
- grounding capacity и oracle/predicted evaluation;
- canonical tensor serialization policy для `z_semantic` и `step_hash`;
- open-loop tool policy и hashes замороженных outcomes, если применимо;
- execution policy и `execution_policy_hash`;
- answer-history policy;
- lineage requirements;
- допустимая вычислительная стоимость;
- неблокирующий backlog.

Числовые границы не выбираются после просмотра результатов.

## 16. Основные метрики

- правильный результат задачи;
- валидность и исполнимость действий;
- перенос на новые структуры и длины;
- A3a против A2 и controls;
- A3b-history против A3a, recurrent и no-history;
- causal sensitivity к whole-history interventions;
- initial-state-only против deterministic rollout;
- causal sensitivity к `z_semantic`, refs, tool_ref и `step_type`;
- exact/minimal grounding quality;
- refs-only performance;
- качество strict reconstruction;
- answer-concept-history против autoregressive-cache и no-history;
- соответствие текста проверенному решению;
- полнота lineage и ExecutionTrace;
- стоимость и стабильность между seeds.

## 17. Ограничения первой реализации

- Один вектор не объявляется человеческим понятием.
- Размерность 384 и граница шага — проверяемые инженерные решения.
- Текущий A3 target является deterministic development-only codebook, а не semantic embedding или supervised semantic proxy.
- BlocksWorld не доказывает перенос на общий reasoning.
- Grounding является отдельным ограниченным каналом.
- A3b-recurrent — ablation, а не единственная реализация памяти.
- Open-loop и closed-loop — разные экспериментальные режимы.
- PlanningState является предсказанным rollout, а не фактическим состоянием среды.
- PLAN_CHECK не равен SOLUTION_CHECK.
- Open-loop tool plan не может зависеть от неизвестного runtime outcome.

## 18. Правило остановки разработки

Для каждого этапа заранее фиксируются одна гипотеза, обязательные controls, ограниченный набор BLOCKER-критериев, количественный GO/STOP и неблокирующий backlog.

Для ближайшей разработки:

- текущий A2 PR не расширяется до A3;
- A3a реализуется отдельным PR;
- claim-bearing A3b начинается только после Stage 2A semantic gate;
- Verbalizer начинается после проверки причинной роли `z_semantic`;
- closed-loop не смешивается с open-loop;
- после этой фиксации широкое архитектурное ревью прекращается;
- новые идеи переходят в backlog или отдельный versioned experiment contract.

## 19. Статус текущей реализации A3

Фактический target текущего A3 строится из `action`, `arg1`, `arg2`, включает seed и target configuration, использует deterministic SHA-derived code, имеет размерность `384` и L2-normalized. Это deterministic development-only codebook — код назначения типизированного BlocksWorld шага, а не frozen semantic encoder, semantic geometry или универсальное представление мысли.

Каноническое соответствие:

```text
implementation A3
→ experimental arm A3a-codebook
→ architecture stage STAGE_2A
→ technical latent channel prototype
```

Текущая реализация показывает только, что latent path технически существует, predicted latent передаётся следующей позиции, pipeline сохраняет и проверяет latent artifacts, а controlled substitution может изменить downstream computation. Реализация канала сама по себе не доказывает causal use. Для причинного вывода нужны заранее зафиксированные сравнения с zero feedback, shuffled feedback, foreign/wrong-task feedback и другими interventions.

Current tests не доказывают улучшение task quality, осмысленную semantic geometry, преимущество concept-level reasoning или общую гипотезу. Более того, A2 уже передаёт previous action и previous argument references через structured channel, поэтому A3a-codebook частично повторно кодирует уже доступную информацию.

## 20. Canonical arm and stage registry

Во всех будущих experiment contracts и artifacts используются отдельные поля `architecture_stage`, `implementation_variant`, `experimental_arm` и `target_type`; одно поле `variant` не заменяет эти разные смыслы.

| implementation_variant | experimental_arm | architecture_stage | target_type | feedback_policy | history_policy | status |
|---|---|---|---|---|---|---|
| A2 | A2-structured-baseline | STAGE_1 | NONE | NONE | STRUCTURED_AUTOREGRESSIVE | IMPLEMENTED |
| A3 | A3a-codebook | STAGE_2A | DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK | PREVIOUS_PREDICTED_LATENT | EXISTING_AUTOREGRESSIVE_HISTORY | IMPLEMENTED |
| A4 | A3a-zero | STAGE_2A | DETERMINISTIC_ACTION_SIGNATURE_CODEBOOK | COMPUTE_THEN_ZERO | SAME_AS_A3A_CODEBOOK | IMPLEMENTED |

Implementation A4 **не связан** с архитектурным Stage 4.

Запланированные arms: `A3a-shuffled`, `A3a-foreign`, `A3s-semantic-target`, `A3s-zero`, `A3s-shuffled`, `A3s-foreign`, `A3b-history`, `A3b-recurrent`, `A3b-no-history`, `A3b-zero-history`, `A3b-shuffled-history`, `A3b-wrong-task-history`.

Текущий `A3a-codebook` уже является random-codebook arm. Второй random-codebook arm допустим только с заранее определённой целью: другой seed, независимое назначение кодов или capacity-matched comparison с semantic arm.

## 21. Future closed-loop lineage

Следующий proposed/non-implemented interface фиксирует lineage, но его полная schema и реализация остаются future backlog:

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

Каждый action связан с предыдущим ConceptStep, каждый Observation — с выполненным action, а следующий ConceptStep после Observation содержит соответствующий `source_observation_hash`. Порядок событий однозначен. Contract задаёт max steps и max model/planner calls; unbounded repair loop запрещён. Verifier feedback типизирован и hash-bound; verifier не передаёт gold next action, oracle answer, hidden solution, privileged state delta или готовую repair instruction.

## 22. Future Answer lineage

Оба интерфейса proposed и non-implemented; они не являются текущими persisted artifacts.

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

`FrozenAnswerPlan` создаётся после `VerifiedSolution`, а `source_summary_hash` обязан совпадать с hash внутри него. Verbalizer не создаёт следующий AnswerConceptStep и не меняет `FrozenAnswerPlan`. Каждый span связан ровно с одним `ANSWER` step; final answer является канонической конкатенацией spans. Contextual production не используется как доказательство causal use `z_semantic`.

## 23. Versioned Stage 2A experiment contract

До claim-bearing A3b обязателен versioned Stage 2A contract. Он фиксирует: primary hypothesis; primary metric; implementation variants; experimental arms; target types и target provenance; predeclared seeds; held-out split; split-before-target-generation; leakage checks; training-data matching; parameter-count и active-parameter reporting; FLOPs/compute/token-budget reporting; checkpoint-selection rule; feedback interventions; failure policy; quantitative GO/STOP criteria.

Минимальные comparisons: `A2-structured-baseline`, `A3a-codebook`, `A3a-zero`, `A3a-shuffled`, `A3a-foreign`, `A3s-semantic-target`, `A3s-zero`, `A3s-shuffled`, `A3s-foreign`.

### 23.1. Stage 2A semantic gate

**Переход к claim-bearing A3b experiment запрещён до прохождения Stage 2A semantic gate.**

До gate разрешён только ограниченный technical scaffolding: interfaces, type definitions, isolated validators и non-claiming prototypes. До gate запрещены содержательное обучение A3b, claim-bearing A3b evaluation, интерпретация результатов как concept-level reasoning и автоматический переход к следующему этапу.

Обязательные условия gate:

- held-out evaluation и несколько predeclared seeds;
- A2, A3a-codebook, A3a-zero, shuffled feedback и foreign/wrong-task feedback;
- настоящий semantic-target arm;
- semantic target против codebook, zero и shuffled/foreign;
- task success, action validity и plan executability;
- predeclared structural generalization;
- parameter count, active parameter count и FLOPs/compute/token budget;
- checkpoint-selection rule;
- quantitative GO/STOP.

Если semantic-target arm не отличается ожидаемым образом от codebook, zero и interventions, результат равен `STOP/REDESIGN`, а не автоматическому `GO` в A3b.

### 23.2. Human-readable examples

Evaluation report обязательно содержит qualitative diagnostic: input task, initial state, goal, predicted plan, reference plan, execution trace, success/failure и сравнение A2/A3/A4. Эти примеры не являются самостоятельным `BLOCKER`, не заменяют held-out metrics или causal interventions и не являются основным quantitative criterion.

### 23.3. Граница научной интерпретации A3b

Даже положительный A3b позволяет утверждать только:

> В данном домене, распределении задач и заранее зафиксированном экспериментальном режиме ограниченное concept-level representation является достаточным и причинно полезным интерфейсом между шагами.

Он не доказывает человеческое мышление, универсальную структуру мысли, общий перенос, фундаментальность размерности или ConceptStep boundaries либо превосходство во всех доменах.

## 24. Неблокирующий backlog до A3b

До A3b в backlog остаются полные JSON Schemas future lineage objects, реализация Answer Planner и Verbalizer, closed-loop execution, Boundary Head, текстовые и интерактивные домены, дополнительные codebook seeds, latent-space visualizations и дальнейшее улучшение semantic targets после первого predeclared arm. Эти пункты не блокируют Stage 2A gate и не означают, что соответствующие artifacts уже реализованы.
