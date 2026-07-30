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

Языковая модель не обязана заново находить решение. После построения смысловой траектории она используется как Verbalizer: переводит уже сформированный смысл шага в естественный язык.

Ключевая проверяемая версия гипотезы:

> Если смысл шага проходит через отдельный ограниченный интерфейс, а следующие шаги и downstream-модули не могут обойти этот интерфейс через скрытые состояния Reasoner, то последовательность крупных смысловых шагов может быть полезнее или эффективнее последовательности только текстовых токенов.

## 2. Архитектурные объекты

Полная схема разделяет:

1. смысл шага;
2. привязку смысла к конкретным сущностям и инструменту;
3. итоговую исполнимую команду;
4. внешний результат исполнения;
5. замороженный план;
6. проверенное решение;
7. план ответа.

### 2.1. LatentConceptStep

```text
LatentConceptStep {
  step_id
  step_type: REASON | ACTION | TOOL | ANSWER | END
  z_semantic: float[d]
  grounding_refs: unordered set<entity_id>
  tool_ref?: canonical_tool_id
  source_observation_hash?: sha256
}
```

`z_semantic` — латентное смысловое представление шага. Размерность `384` используется в текущей архитектуре A3 как первая проверяемая конфигурация, но не является фундаментальным ограничением гипотезы.

`grounding_refs` — отдельный ограниченный канал, связывающий общий смысл с конкретными сущностями задачи. `tool_ref` отдельно выбирает разрешённый инструмент для шага `TOOL`.

#### Ограничения grounding_refs

В первой версии `grounding_refs`:

- являются семантически неупорядоченным множеством canonical entity IDs;
- сериализуются в каноническом порядке и дедуплицируются;
- не содержат оператор или тип действия;
- не содержат отношения между объектами;
- не содержат роли аргументов вроде `moving`, `support`, `source`, `target`;
- не содержат произвольный текст;
- не содержат готовый план или готовую команду;
- имеют заранее зафиксированное максимальное количество элементов;
- являются точным минимальным множеством сущностей, необходимых текущему шагу;
- не содержат лишних сущностей и не пропускают обязательные сущности.

В контролируемом домене множество сравнивается с oracle entity set:

- exact-set match;
- oracle cardinality match;
- missing-ref rate;
- extra-ref rate;
- predicted refs против oracle refs;
- refs-only performance.

Оператор, отношения и роли аргументов должны восстанавливаться из `z_semantic` и допустимых структурных контрактов.

#### Ограничения tool_ref

`tool_ref`:

- отсутствует для `REASON`, `ACTION`, `ANSWER` и `END`;
- обязателен и единственен для `TOOL`;
- выбирается только из разрешённого canonical tool catalog;
- не содержит аргументы, текст запроса, роли или готовый tool call;
- проверяется отдельными zero/shuffled/wrong-tool interventions;
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

`ResolvedActionStep` создаётся Action Decoder / Tool Resolver только для шагов типа `ACTION` или `TOOL`. Он не является частью исходного латентного шага.

### 2.3. ObservationEvent

Observation является внешним событием среды, а не мыслью модели и не входит в `step_type`.

```text
ObservationEvent {
  source_action_step_id
  success_or_failure
  environment_state_delta
  tool_result_or_error
  observation_hash
}
```

### 2.4. FrozenConceptPlan

Open-loop планирование завершается отдельным артефактом:

```text
FrozenConceptPlan {
  concept_step_hashes[]
  resolved_action_step_hashes[]
  planning_context_hash
  planner_call_count: 1
  freeze_status
  plan_hash
}
```

После freeze запрещены:

- изменение шага;
- suffix regeneration;
- повторный вызов Planner;
- замена `LatentConceptStep` или `ResolvedActionStep`;
- использование Observation для изменения frozen suffix.

### 2.5. CompactReasoningSummary

Summary не является свободным текстовым Chain-of-Thought.

Первая версия представляет собой детерминированный типизированный артефакт:

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

- summary строится без LLM;
- свободное поле `summary_payload` запрещено;
- схема и максимальный размер фиксируются до эксперимента;
- summary не получает `h_t`, KV-cache или скрытую историю Reasoner;
- все элементы выводятся только из сохранённых и проверенных артефактов;
- одинаковый набор входных артефактов должен давать byte-identical summary.

### 2.6. VerifiedSolution

```text
VerifiedSolution {
  final_state_or_result
  executed_actions_or_tool_results
  compact_reasoning_summary_hash
  task_constraints
  verification_status
  solution_hash
}
```

`VerifiedSolution` является единственным разрешённым интерфейсом между reasoning trajectory и answer trajectory.

## 3. Semantic bottleneck

В полной архитектуре `LatentConceptStep` является реальным интерфейсом между Reasoner и последующими модулями.

```text
Reasoner temporary hidden state h_t
          ↓
Semantic Bottleneck
          ↓
step_type_t + z_t + grounding_refs_t + tool_ref_t
          ├→ Action Decoder / Tool Resolver
          └→ Answer Planner / Verbalizer после VerifiedSolution
```

После создания `LatentConceptStep` downstream-модули не должны получать `h_t` или другие скрытые состояния Reasoner в обход `z_semantic` и разрешённых структурированных полей.

Без этого ограничения `z_semantic` может оказаться auxiliary output, который выглядит интерпретируемым, но причинно не участвует в решении.

### 3.1. Output bottleneck

Для A3b downstream-модули получают только:

- `step_type`;
- `z_semantic`;
- ограниченные `grounding_refs`;
- ограниченный `tool_ref` для TOOL;
- публичный каталог сущностей или инструментов;
- минимальные структурные ограничения формата.

Запрещены прямые пути:

```text
h_t → action
h_t → tool call
h_t → answer text
```

### 3.2. Inter-step bottleneck

A3b ограничивает не только выход текущего шага, но и связь между соседними мыслями.

Внутри вычисления одного шага временные hidden states и attention разрешены. На границе `LatentConceptStep` внутренний decoder cache и скрытая token-level история шага сбрасываются или становятся недоступными следующему шагу.

Между шагами разрешено передавать только:

- сохранённые `LatentConceptStep`;
- публичное состояние задачи согласно experiment contract;
- последний `ObservationEvent` только в closed-loop;
- неизменяемое task encoding;
- явные позиционные и служебные признаки.

Передача `h_t`, token-level hidden-state history или KV-cache между ConceptStep запрещена.

### 3.3. Варианты concept-level memory

#### A3b-history — основной вариант

Следующий шаг может attend-ить ко всей сохранённой последовательности ConceptStep:

```text
[ConceptStep_1, ConceptStep_2, ..., ConceptStep_t]
+ PublicState
+ разрешённая Observation
→ temporary h_(t+1)
→ ConceptStep_(t+1)
```

#### A3b-recurrent — ablation

Следующий шаг видит только предыдущий ConceptStep:

```text
ConceptStep_t
+ PublicState
+ разрешённая Observation
→ ConceptStep_(t+1)
```

Отрицательный результат A3b-recurrent не должен интерпретироваться как опровержение concept-level history.

#### A3b-no-history — обязательный контроль

```text
TaskEncoding
+ PublicState
+ position/service features
→ ConceptStep_t
```

Контроль получает тот же task context, public state, каталоги и вычислительный бюджет, но не получает concept history.

#### Whole-history interventions

Обязательны:

- `zero-history`;
- `shuffled-history` с сохранением длины, позиций и служебных метаданных;
- `wrong-task-history` той же длины и совместимого формата;
- `truncated-history` с заранее заданным окном;
- `history-only` diagnostic там, где он содержательно допустим.

Положительный результат A3b-history интерпретируется как польза concept history только при превосходстве над A3b-no-history и ожидаемой деградации в whole-history interventions.

## 4. Фазовая архитектура

Общий поток разделён на reasoning/planning, execution/verification и answer generation.

```text
Input / PublicState
        ↓
Task Encoder
        ↓
Reasoner + Semantic Bottleneck
        ↓
REASON / ACTION / TOOL ConceptSteps
        ↓
Action Decoder / Tool Resolver для ACTION/TOOL
        ↓
Open-loop: FrozenConceptPlan
Closed-loop: исполнение → ObservationEvent → следующий ConceptStep
        ↓
SOLUTION_CHECK / verifier
        ↓
VerifiedSolution
        ↓
Answer Planner
        ↓
ANSWER ConceptSteps
        ↓
Frozen LLM Verbalizer
        ↓
Final answer
```

`ANSWER` ConceptStep не направляется в Verbalizer до появления `VerifiedSolution`.

## 5. Reasoning и Answer trajectories

### 5.1. Reasoning trajectory

Последовательность `REASON`, `ACTION` и `TOOL` шагов описывает поиск и выполнение решения. Машинно значимые действия остаются типизированными и проверяемыми.

### 5.2. Solution Check

Answer trajectory не может начинаться напрямую из `START`, `REASON`, `ACTION`, `TOOL` или `ObservationEvent`.

```text
Reasoning/execution evidence
→ SOLUTION_CHECK
→ VerifiedSolution или fail/возврат к REASON в closed-loop
```

Для тривиальной задачи допускается немедленный `SOLUTION_CHECK`, но интерфейс `VerifiedSolution` всё равно обязателен.

### 5.3. Answer trajectory

После появления `VerifiedSolution` Answer Planner строит отдельную последовательность `ANSWER` шагов.

```text
сначала решить
→ проверить решение
→ спланировать ответ
→ выразить словами
```

Answer Planner получает:

- `VerifiedSolution`;
- hash-bound `CompactReasoningSummary`;
- task constraints;
- разрешённый пользовательский контекст.

Он не получает скрытые состояния Reasoner.

## 6. Раздельные state machines

Единая state machine не используется, потому что open-loop планирование создаёт полный план до появления реальных Observation, а closed-loop создаёт новый шаг после Observation.

### 6.1. Open-loop planning state machine

```text
START → REASON | ACTION | TOOL | SOLUTION_CHECK
REASON → REASON | ACTION | TOOL | SOLUTION_CHECK
ACTION → REASON | ACTION | TOOL | SOLUTION_CHECK
TOOL → REASON | ACTION | TOOL | SOLUTION_CHECK
SOLUTION_CHECK → FrozenConceptPlan | REASON
FrozenConceptPlan → terminal planning state
```

Здесь `ACTION` и `TOOL` являются элементами ещё не исполненного плана. Реальный `ObservationEvent` во время planning отсутствует и не симулируется как внешний факт.

### 6.2. Open-loop execution state machine

```text
FrozenConceptPlan
→ execute ResolvedActionStep_1
→ ObservationEvent_1
→ execute ResolvedActionStep_2
→ ObservationEvent_2
→ ...
→ final verifier
→ VerifiedSolution | FailedEvaluation
```

Правила:

- Observation логируется, но не передаётся Planner;
- Observation не меняет frozen suffix;
- Planner call count остаётся равен одному;
- ошибка исполнения фиксируется fail-closed;
- продолжение или остановка после ошибки задаётся до запуска execution contract.

### 6.3. Closed-loop cognitive state machine

```text
START → REASON | ACTION | TOOL | SOLUTION_CHECK
REASON → REASON | ACTION | TOOL | SOLUTION_CHECK
ACTION → external ObservationEvent
TOOL → external ObservationEvent
ObservationEvent → REASON | ACTION | TOOL | SOLUTION_CHECK
SOLUTION_CHECK → VerifiedSolution | REASON
VerifiedSolution → ANSWER | END
ANSWER → ANSWER | END
END → terminal
```

Правила:

- `ObservationEvent`, `FrozenConceptPlan` и `VerifiedSolution` не являются `LatentConceptStep`;
- `START → ANSWER` и `REASON → ANSWER` запрещены;
- `ANSWER → ACTION/TOOL/REASON` запрещено в первой реализации;
- после `ACTION` или `TOOL` новый closed-loop ConceptStep создаётся только после Observation;
- все переходы логируются и проверяются машинно.

## 7. Type-specific invariants

### REASON

- `z_semantic` обязателен;
- entity refs опциональны, но при наличии проходят exact-minimal-set проверку;
- `tool_ref` отсутствует;
- Action Decoder не вызывается;
- Verbalizer вызывается только в диагностическом reconstruction-режиме;
- готовая команда отсутствует.

### ACTION

- `z_semantic` обязателен;
- entity refs обязательны, кроме операторов без сущностных аргументов;
- `tool_ref` отсутствует;
- refs являются точным минимальным множеством сущностей действия;
- вызывается Action Decoder;
- должен появиться `ResolvedActionStep`.

### TOOL

- `z_semantic` обязателен;
- `tool_ref` обязателен и единственен;
- entity refs содержат только точное минимальное множество сущностей, передаваемых инструменту;
- текст запроса, роли аргументов и параметры не могут передаваться через refs/tool_ref;
- должен появиться `ResolvedActionStep`;
- результат или ошибка фиксируются в Observation.

### ANSWER

- допускается только после `VerifiedSolution`;
- `z_semantic` обязателен;
- `grounding_refs` могут содержать только сущности, необходимые текущему answer span;
- `tool_ref` отсутствует;
- Action Decoder не вызывается;
- Verbalizer обязателен;
- ответный span связывается с source concept step.

### END

- `grounding_refs` пусты;
- `tool_ref` отсутствует;
- Action Decoder и Verbalizer не вызываются;
- после END новые шаги запрещены.

### source_observation_hash

- в open-loop planning отсутствует;
- в open-loop execution относится к execution evidence, а не к frozen planning step;
- в closed-loop обязателен для первого ConceptStep после Observation;
- должен точно ссылаться на Observation, вызвавшую следующий шаг.

## 8. Open-loop и closed-loop режимы

### 8.1. Open-loop experimental mode

```text
один Planner call
→ полный FrozenConceptPlan
→ исполнение без изменений
```

Используется для чистого сравнения:

```text
A2
vs A3a
vs A3b-history
vs A3b-recurrent
vs A3b-no-history
```

Первая проверка A3b проводится именно в open-loop, чтобы не смешивать semantic/inter-step bottleneck с реакцией на среду.

### 8.2. Closed-loop cognitive mode

```text
LatentConceptStep
→ действие или инструмент
→ ObservationEvent
→ обновление State Store
→ следующий LatentConceptStep
```

В этом режиме replanning является явной частью архитектуры. Каждый новый шаг связывается с Observation.

Open-loop и closed-loop результаты нельзя смешивать в одном сравнении без отдельного протокола.

## 9. Модули

### Task Encoder

Кодирует запрос, публичное состояние, доступные сущности, инструменты и ограничения.

### Latent Planner / Reasoner

Авторегрессионно создаёт временное внутреннее состояние шага.

- В A3a обычная autoregressive hidden history сохраняется, а `z_semantic` является дополнительным feedback-каналом.
- В A3b hidden history между ConceptStep недоступна; используется только concept-level memory.

### Semantic Bottleneck

Создаёт `step_type`, `z_semantic`, `grounding_refs` и при необходимости `tool_ref`.

`Step-Type Router`, Grounding Head и Tool Head являются prediction heads внутри Semantic Bottleneck, а не независимыми обходными маршрутизаторами.

### Grounding Head

Выбирает точное минимальное неупорядоченное множество сущностей из разрешённого каталога. Предсказание канонизируется и дедуплицируется.

### Tool Head

Для шага `TOOL` выбирает ровно один `tool_ref` из разрешённого каталога. Не создаёт параметры или текст вызова.

### Action Decoder / Tool Resolver

Преобразует `LatentConceptStep` в `ResolvedActionStep` и получает только разрешённые поля bottleneck и структурный контракт.

### Verbalizer Adapter

Преобразует `z_semantic` и разрешённые grounding-поля в soft/virtual tokens либо другое входное представление для frozen LLM.

### Frozen LLM Verbalizer

Формулирует предложение, абзац или пояснение. Не имеет права менять план, создавать новую reasoning trajectory или обращаться к скрытым состояниям Reasoner.

### Executor, State Store и Verifier

Исполняют frozen или closed-loop действия, сохраняют Observation, проверяют переходы состояния и создают `VerifiedSolution` либо типизированный failure.

## 10. Два режима Verbalizer

### 10.1. Strict reconstruction mode

Вход:

```text
z_semantic
+ ограниченные grounding_refs
+ tool_ref только для TOOL
+ минимальная инструкция по формату
```

Verbalizer не получает полный исходный вопрос, reasoning history, summary, скрытое состояние или полный verified result.

Обязательные контроли:

- correct/zero/shuffled/wrong-task `z_semantic` при правильном grounding;
- correct `z_semantic` с shuffled/wrong-task refs;
- exact oracle refs против predicted refs;
- refs-only без `z_semantic`;
- `z_semantic` без refs там, где refs обязательны;
- correct/zero/shuffled/wrong-tool `tool_ref`;
- неправильный `step_type`.

### 10.2. Contextual production mode

Используется только после проверки reconstruction mode.

```text
z_semantic
+ grounding_refs/tool_ref
+ исходный запрос
+ уже сформированный ответ
+ разрешённый фактический контекст
```

Этот режим оптимизирует качество текста, но сам по себе не доказывает использование latent-вектора.

## 11. Type-specific оценка Verbalizer

### ACTION / TOOL

Проверяются оператор, аргументы и роли, применимость, отсутствие новых сущностей и соответствие Observation.

### REASON

Проверяются промежуточный вывод, причинная связь с предыдущими ConceptStep, отсутствие неподтверждённых фактов и чувствительность к interventions.

### ANSWER

Проверяются покрытие требуемой части ответа, согласованность с `VerifiedSolution`, отсутствие новых утверждений и порядок answer-step.

### END

Проверяются корректное завершение, отсутствие содержательного текста и отсутствие последующих шагов.

## 12. Что считается мыслью на разных этапах

### Stage 2: supervised semantic proxy

В текущем BlocksWorld A3 target является 384-мерным embedding заранее определённой semantic signature.

```text
структурированная semantic signature
→ канонический текст
→ frozen encoder
→ target z
```

Успех доказывает причинную полезность такого semantic feedback в ограниченном домене, но не универсальное представление человеческой мысли.

Точная идентичность сущностей и инструмента передаётся через отдельно проверяемые grounding-каналы.

### Stage 3–4: learned ConceptStep representation

`z_semantic` должен обучаться на совокупности сигналов:

- правильность действия или результата;
- предсказание следующего состояния;
- реконструкция смысла шага;
- согласованность reasoning trajectory;
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

```text
задача → полный frozen plan → исполнение
```

**Критерий завершения:** один Planner call создаёт воспроизводимый многошаговый план, который исполняется без replanning.

### Этап 2A — A3a: latent feedback

Сравниваются A2, правильный feedback, zero, shuffled и random-code parameter-matched control.

**Критерий завершения:** A3a проходит заранее зафиксированный experiment contract, а interventions дают ожидаемое изменение качества.

### Этап 2B — A3b: semantic и inter-step bottleneck

Сравниваются:

- A3a;
- A3b-history;
- A3b-recurrent;
- A3b-no-history;
- A3b-zero/shuffled/wrong-task-history;
- refs-only и z-only там, где допустимо;
- oracle refs и predicted refs;
- capacity-matched grounding controls.

**Критерий завершения:** A3b-history проходит experiment contract, превосходит no-history контроль и демонстрирует ожидаемую causal sensitivity к history, `z_semantic`, refs, tool_ref и `step_type`.

### Этап 3 — Verbalizer MVP

Проверяется декодируемость отдельной мысли в strict reconstruction mode.

### Этап 4 — разделение Reasoning и Answer

Добавляются `SOLUTION_CHECK`, `VerifiedSolution` и отдельная answer trajectory.

### Этап 5 — полный сравнительный эксперимент

Домены расширяются последовательно:

1. BlocksWorld;
2. один текстовый домен;
3. один интерактивный домен;
4. затем широкое сравнение.

## 15. Experiment contract перед каждым этапом

Числовые границы GO/STOP не выбираются после просмотра результатов.

До реализации или запуска фиксируются:

- primary hypothesis и primary metric;
- evaluation unit;
- baselines, ablations и interventions;
- threshold или non-inferiority margin;
- seeds и split policy;
- STOP/GO rule;
- техническая валидность;
- checkpoint-selection rule;
- training-data matching;
- parameter-count и active-parameter matching/reporting;
- FLOPs и token-budget matching;
- одинаковый доступ к task context и PublicState;
- одинаковые entity/tool catalogs;
- правила канонизации и capacity grounding-каналов;
- oracle/predicted grounding evaluation;
- допустимая вычислительная стоимость;
- список неблокирующего backlog.

## 16. Основные метрики

- правильный результат задачи;
- валидность и исполнимость действий;
- перенос на новые структуры и длины;
- A3a против A2 и controls;
- A3b-history против A3a, recurrent и no-history;
- causal sensitivity к whole-history interventions;
- causal sensitivity к `z_semantic`, refs, tool_ref и `step_type`;
- exact/minimal grounding quality;
- refs-only performance;
- качество strict reconstruction;
- соответствие текста проверенному решению;
- стоимость и стабильность между seeds.

## 17. Ограничения первой реализации

- Один вектор не объявляется человеческим понятием.
- Размерность 384 и граница шага являются проверяемыми инженерными выборами.
- Текущий A3 target является supervised semantic proxy.
- BlocksWorld не доказывает перенос на общий reasoning.
- Grounding является отдельным ограниченным каналом.
- A3b-recurrent — ablation, а не единственная реализация памяти.
- Open-loop и closed-loop являются разными экспериментальными режимами.

## 18. Правило остановки разработки

Для каждого этапа заранее фиксируются одна гипотеза, обязательные controls, ограниченный набор BLOCKER-критериев, количественный GO/STOP и неблокирующий backlog.

Для ближайшей разработки:

- текущий A2 PR не расширяется до A3;
- A3a реализуется отдельным PR;
- A3b начинается после технической проверки A3a;
- Verbalizer начинается после проверки причинной роли `z_semantic`;
- closed-loop не смешивается с open-loop;
- после этой фиксации широкое архитектурное ревью прекращается;
- новые идеи переходят в backlog или отдельный versioned experiment contract.
