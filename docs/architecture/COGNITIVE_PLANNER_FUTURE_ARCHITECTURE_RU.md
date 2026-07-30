# Cognitive Planner — расширенная архитектура исходной гипотезы

**Статус:** non-normative roadmap.  
**Назначение:** зафиксировать архитектуру, выходящую за пределы узкого эксперимента Work Planner / BlocksWorld v1.21.  
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
2. привязку смысла к конкретным сущностям;
3. итоговую исполнимую команду;
4. внешнее наблюдение;
5. проверенное решение;
6. план ответа.

### 2.1. LatentConceptStep

```text
LatentConceptStep {
  step_id
  step_type: REASON | ACTION | TOOL | ANSWER | END
  z_semantic: float[d]
  grounding_refs: unordered set<entity_id>
  source_observation_hash?: sha256
}
```

`z_semantic` — латентное смысловое представление шага. Размерность `384` используется в текущей архитектуре A3 как первая проверяемая конфигурация, но не является фундаментальным ограничением гипотезы.

`grounding_refs` — отдельный ограниченный канал, связывающий общий смысл с конкретными объектами, инструментами или сущностями задачи.

В первой версии `grounding_refs`:

- являются неупорядоченным множеством canonical entity IDs;
- не содержат оператор или тип действия;
- не содержат отношения между объектами;
- не содержат роли аргументов вроде `moving`, `support`, `source`, `target`;
- не содержат произвольный текст;
- не содержат готовый план или готовую команду;
- имеют заранее зафиксированное максимальное количество элементов.

Оператор, отношения и роли аргументов должны восстанавливаться из `z_semantic` и допустимых структурных контрактов.

### 2.2. ResolvedActionStep

```text
ResolvedActionStep {
  source_concept_step_id
  action: typed action
  args: ordered grounded references[]
  resolution_status
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

### 2.4. CompactReasoningSummary

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

### 2.5. VerifiedSolution

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
step_type_t + z_t + grounding_refs_t
          ├→ Action Decoder / Tool Resolver
          └→ Answer Verbalizer
```

После создания `LatentConceptStep` downstream-модули не должны получать `h_t` или другие скрытые состояния Reasoner в обход `z_semantic` и разрешённых структурированных полей.

Без этого ограничения `z_semantic` может оказаться auxiliary output, который выглядит интерпретируемым, но причинно не участвует в решении.

### 3.1. Output bottleneck

Для A3b downstream-модули получают только:

- `step_type`;
- `z_semantic`;
- ограниченные `grounding_refs`;
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
- публичное состояние задачи;
- последний `ObservationEvent`, если он существует;
- неизменяемое task encoding;
- явные позиционные и служебные признаки.

Передача `h_t`, token-level hidden-state history или KV-cache между ConceptStep запрещена.

### 3.3. Два варианта concept-level memory

#### A3b-history — основной вариант

Следующий шаг может attend-ить ко всей сохранённой последовательности ConceptStep:

```text
[ConceptStep_1, ConceptStep_2, ..., ConceptStep_t]
+ PublicState
+ Observation
→ temporary h_(t+1)
→ ConceptStep_(t+1)
```

Это основной вариант проверки concept-token гипотезы.

#### A3b-recurrent — ablation

Следующий шаг видит только предыдущий ConceptStep:

```text
ConceptStep_t
+ PublicState
+ Observation
→ ConceptStep_(t+1)
```

Этот режим проверяет достаточность строгой рекуррентной памяти, но не является основным вариантом. Отрицательный результат A3b-recurrent не должен интерпретироваться как опровержение concept-level history.

## 4. Полный поток обработки

```text
Запрос / публичное состояние
          ↓
Task Encoder
          ↓
Latent Planner / Reasoner
          ↓
Semantic Bottleneck
          ↓
LatentConceptStep
      ┌──────────────┴──────────────┐
      ↓                             ↓
Action Decoder / Resolver      Answer Verbalizer
      ↓                             ↓
ResolvedActionStep              Text span
      ↓
Executor
      ↓
ObservationEvent
      ↓
State Store
      └→ следующий LatentConceptStep
```

Архитектура разделяет пять функций:

- **Reasoner** определяет, что нужно понять или сделать дальше;
- **Semantic Bottleneck** создаёт ограниченный `LatentConceptStep`;
- **Grounding Head** выбирает конкретные сущности, но не кодирует оператор и роли;
- **Executor** применяет структурированные действия и возвращает Observation;
- **Verbalizer** переводит смысловые шаги ответа в естественный язык.

## 5. Reasoning и Answer trajectories

### 5.1. Reasoning trajectory

Последовательность `REASON`, `ACTION` и `TOOL` шагов описывает поиск и выполнение решения. Машинно значимые действия остаются типизированными и проверяемыми.

### 5.2. Solution Check

Answer trajectory не может начинаться напрямую из `START`, `REASON`, `ACTION`, `TOOL` или `ObservationEvent`.

Сначала создаётся и проверяется `VerifiedSolution`.

```text
Reasoning trajectory
→ SOLUTION_CHECK
→ VerifiedSolution или возврат к REASON
```

Для тривиальной задачи допускается немедленный `SOLUTION_CHECK`, но интерфейс `VerifiedSolution` всё равно обязателен.

### 5.3. Answer trajectory

После появления `VerifiedSolution` Answer Planner строит отдельную последовательность `ANSWER` шагов. Каждый шаг задаёт смысл одного предложения или абзаца, а frozen LLM Verbalizer преобразует его в текст.

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

## 6. Допустимые переходы

Минимальная state machine:

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

- `ObservationEvent` и `VerifiedSolution` не являются `LatentConceptStep`;
- `START → ANSWER` запрещено;
- `REASON → ANSWER` запрещено;
- `ANSWER → ACTION/TOOL/REASON` запрещено в первой реализации;
- после `ACTION` или `TOOL` следующий смысловой шаг создаётся только после фиксации Observation;
- `END` завершает траекторию;
- все переходы логируются и проверяются машинно.

## 7. Type-specific invariants

### REASON

- `z_semantic` обязателен;
- refs опциональны;
- Action Decoder не вызывается;
- Verbalizer вызывается только в диагностическом reconstruction-режиме;
- готовая команда отсутствует.

### ACTION

- `z_semantic` обязателен;
- refs обязательны, кроме операторов без аргументов;
- вызывается Action Decoder;
- должен появиться `ResolvedActionStep`;
- в closed-loop после исполнения обязателен `ObservationEvent`.

### TOOL

- действуют invariants ACTION;
- дополнительно обязателен разрешённый tool ID;
- результат или ошибка фиксируются в Observation.

### ANSWER

- допускается только после `VerifiedSolution`;
- `z_semantic` обязателен;
- Action Decoder не вызывается;
- Verbalizer обязателен;
- ответный span связывается с source concept step.

### END

- `grounding_refs` пусты;
- Action Decoder и Verbalizer не вызываются;
- после END новые шаги запрещены.

### source_observation_hash

- в open-loop может отсутствовать;
- в closed-loop обязателен для первого ConceptStep после Observation;
- должен точно ссылаться на Observation, вызвавшую следующий шаг.

## 8. Open-loop и closed-loop режимы

### 8.1. Open-loop experimental mode

```text
полный план создаётся заранее
→ замораживается
→ исполняется без изменений
```

Используется для чистого сравнения:

```text
A2 vs A3a vs A3b-history vs A3b-recurrent
```

Replanning и suffix regeneration запрещены.

Первая проверка A3b проводится именно в open-loop, чтобы не смешивать semantic/inter-step bottleneck с реакцией на среду.

### 8.2. Closed-loop cognitive mode

```text
LatentConceptStep
→ действие или инструмент
→ ObservationEvent
→ обновление State Store
→ следующий LatentConceptStep
```

Используется в задачах, где результат действия заранее неизвестен: инструменты, поиск, код, файлы и внешняя среда.

В этом режиме replanning не скрывается и не считается нарушением. Он является явной частью архитектуры, а следующий шаг связывается с Observation.

Open-loop и closed-loop результаты нельзя смешивать в одном сравнении без отдельного протокола.

## 9. Модули

### Task Encoder

Кодирует запрос, публичное состояние, доступные сущности, инструменты и ограничения.

### Latent Planner / Reasoner

Авторегрессионно создаёт временное внутреннее состояние шага.

- В A3a обычная autoregressive hidden history сохраняется, а `z_semantic` является дополнительным feedback-каналом.
- В A3b hidden history между ConceptStep недоступна; используется только concept-level memory.

### Semantic Bottleneck

Создаёт `step_type`, `z_semantic` и `grounding_refs`.

`Step-Type Router` является prediction head внутри Semantic Bottleneck, а не отдельным независимым маршрутизатором.

### Grounding Head

Выбирает неупорядоченное множество сущностей из разрешённого каталога.

Grounding Head не имеет права кодировать:

- оператор;
- роли аргументов;
- отношения;
- произвольный текст;
- готовую команду.

### Action Decoder / Tool Resolver

Для `ACTION` и `TOOL` преобразует `LatentConceptStep` в `ResolvedActionStep`.

Получает только:

- `step_type`;
- `z_semantic`;
- ограниченные refs;
- каталог сущностей/инструментов;
- структурный контракт действий.

### Verbalizer Adapter

Преобразует `z_semantic` и разрешённые refs в soft/virtual tokens либо другое входное представление для frozen LLM.

### Frozen LLM Verbalizer

Формулирует предложение, абзац или пояснение. Verbalizer не имеет права менять план, создавать новую reasoning trajectory или обращаться к скрытым состояниям Reasoner.

### Executor and State Store

Исполняет действия, проверяет переходы состояния, сохраняет Observation и фактическую траекторию выполнения.

## 10. Два режима Verbalizer

### 10.1. Strict reconstruction mode

Используется для проверки, действительно ли смысл содержится в `z_semantic`.

Вход:

```text
z_semantic
+ ограниченные grounding_refs
+ минимальная инструкция по формату
```

Verbalizer не получает:

- полный исходный вопрос;
- reasoning history;
- `CompactReasoningSummary`;
- скрытое состояние Reasoner;
- полный verified result, позволяющий заново решить задачу.

Обязательные контроли:

- правильный `z_semantic` при правильных refs;
- нулевой `z_semantic` при правильных refs;
- shuffled/wrong-task `z_semantic` при правильных refs;
- правильный `z_semantic` с shuffled/wrong-task refs;
- неправильный `step_type`;
- refs-only без `z_semantic`;
- `z_semantic` без refs там, где refs обязательны.

### 10.2. Contextual production mode

Используется только после проверки reconstruction mode.

Вход:

```text
z_semantic
+ grounding_refs
+ исходный запрос
+ уже сформированный ответ
+ разрешённый фактический контекст
```

Этот режим оптимизирует качество текста, но сам по себе не доказывает, что Verbalizer использует latent-вектор.

## 11. Type-specific оценка Verbalizer

### ACTION / TOOL

Проверяются:

- правильный оператор;
- правильные аргументы и их роли;
- применимость;
- отсутствие новых сущностей;
- соответствие Observation после исполнения.

### REASON

Проверяются:

- правильное утверждение или промежуточный вывод;
- причинная связь с предыдущими ConceptStep;
- отсутствие неподтверждённых фактов;
- чувствительность к intervention по `z_semantic`.

### ANSWER

Проверяются:

- покрытие требуемой части ответа;
- фактическая согласованность с `VerifiedSolution`;
- отсутствие новых неподтверждённых утверждений;
- корректный порядок answer-step.

### END

Проверяются:

- корректное завершение;
- отсутствие лишнего содержательного текста;
- отсутствие последующих шагов.

## 12. Что считается мыслью на разных этапах

### Stage 2: supervised semantic proxy

В текущем BlocksWorld A3 target является 384-мерным embedding заранее определённой semantic signature.

```text
структурированная semantic signature
→ канонический текст
→ frozen encoder
→ target z
```

Успех здесь доказывает, что такой semantic feedback причинно полезен в ограниченном домене. Он не доказывает, что найдено универсальное представление человеческой мысли.

Точная идентичность объектов передаётся через отдельно проверяемый grounding-канал.

### Stage 3–4: learned ConceptStep representation

На следующих этапах `z_semantic` должен обучаться не только на заранее заданной метке, но и на совокупности сигналов:

- правильности действия или результата;
- предсказании следующего состояния;
- реконструкции смысла шага;
- согласованности reasoning trajectory;
- информационном bottleneck;
- контрастивных и intervention-контролях.

## 13. Граница одного ConceptStep

Фраза «одно предложение или абзац» является инженерной гипотезой, а не фиксированным свойством.

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

**Критерий завершения:** модель обучается, выдаёт многошаговый план, один раз вызывается, не перепланирует и воспроизводимо достигает цели на toy-задачах.

### Этап 2A — A3a: latent feedback

Каждый шаг дополнительно выдаёт `z_semantic`, который влияет на следующий шаг, но действие и latent могут предсказываться параллельно из одного decoder hidden state, а обычная autoregressive history сохраняется.

Основные сравнения:

- A2;
- A3a с правильным latent feedback;
- zero feedback;
- shuffled feedback;
- random-code parameter-matched control.

**Критерий завершения:** A3a проходит заранее зафиксированный experiment contract, а вмешательства в semantic feedback дают ожидаемое изменение качества.

### Этап 2B — A3b: semantic и inter-step bottleneck

Проверяется более сильная исходная гипотеза:

```text
concept history + PublicState → temporary h_t
h_t → LatentConceptStep_t
LatentConceptStep_t → ResolvedActionStep_t / text_t
```

Сравниваются:

- A3a;
- A3b-history;
- A3b-recurrent;
- A3b refs-only;
- A3b z-only там, где допустимо;
- controls с capacity-matched grounding.

Обязательные независимые interventions:

- zero/shuffled/wrong-task `z_semantic` при сохранённых refs;
- shuffled/wrong-task refs при сохранённом `z_semantic`;
- refs-only без `z_semantic`;
- неправильный `step_type`;
- удаление `source_observation_hash` в closed-loop режиме.

**Критерий завершения:** A3b-history проходит experiment contract, сохраняет установленную долю качества A3a, а intervention каждого канала причинно меняет только ожидаемую часть поведения.

### Этап 3 — Verbalizer MVP

Проверяется декодируемость отдельной мысли в strict reconstruction mode.

**Критерий завершения:** type-specific метрики проходят заранее зафиксированные пороги, а вмешательства по `z`, refs и типу шага дают ожидаемую деградацию.

### Этап 4 — разделение Reasoning и Answer

Добавляются `SOLUTION_CHECK`, `VerifiedSolution` и отдельная answer trajectory.

**Критерий завершения:** разделённая система не уступает по правильности решения и проходит пороги по фактической согласованности, покрытию решения, отсутствию новых утверждений и устойчивости answer-plan.

### Этап 5 — полный сравнительный эксперимент

Эксперименты расширяются последовательно:

1. BlocksWorld — механика и причинность каналов;
2. один текстовый домен — простая символическая математика или логика;
3. один интерактивный домен — инструменты или небольшие изменения кода;
4. затем широкое сравнение.

Сравниваются:

- Direct LLM;
- текстовый Chain-of-Thought;
- текстовый план;
- A2;
- A3a;
- A3b-history;
- A3b-recurrent;
- A3b + Verbalizer;
- A3b + отдельные reasoning и answer trajectories.

## 15. Experiment contract перед каждым этапом

Этот roadmap не задаёт числовые границы GO/STOP. Они не должны выбираться после просмотра результатов.

До начала реализации или запуска каждого этапа создаётся отдельный versioned experiment contract, который фиксирует:

- primary hypothesis;
- primary metric;
- evaluation unit;
- baselines;
- ablations и interventions;
- threshold или non-inferiority margin;
- seeds и split policy;
- STOP/GO rule;
- критерии технической валидности;
- checkpoint-selection rule;
- training-data matching;
- parameter-count matching;
- active-parameter reporting;
- FLOPs и token-budget matching;
- одинаковый доступ к task context;
- одинаковый каталог сущностей и инструментов;
- допустимую вычислительную стоимость;
- список неблокирующего backlog.

Фразы вроде «существенная доля качества», «существенно ухудшается» и «устойчивое улучшение» считаются placeholders до появления experiment contract и не могут использоваться как основание для GO.

## 16. Основные метрики

- достижение правильного результата задачи;
- валидность и исполнимость действий;
- перенос на новые структуры и длины задач;
- польза semantic feedback: A3a против A2 и controls;
- сохранение качества через semantic/inter-step bottleneck: A3b-history против A3a;
- A3b-history против A3b-recurrent;
- causal sensitivity к замене `z_semantic`;
- causal sensitivity к замене `grounding_refs`;
- refs-only performance;
- независимость каналов `z_semantic`, refs и `step_type`;
- восстанавливаемость смысла latent-вектора Verbalizer;
- соответствие текста фактическому решению;
- отсутствие вымышленных фактов;
- стоимость обучения и инференса;
- стабильность между seeds и повторными запусками.

## 17. Ограничения первой реализации

- Не утверждается, что один вектор всегда равен одному человеческому понятию.
- Размерность 384 и граница «одно предложение или абзац» являются проверяемыми инженерными выборами.
- Текст Verbalizer не считается доказательством содержания latent-вектора без strict reconstruction и intervention-контролей.
- Текущий A3 target является supervised semantic proxy, а не доказанной универсальной мыслью.
- BlocksWorld проверяет механику и причинную полезность semantic feedback, но не доказывает перенос на общий reasoning.
- Grounding refs являются отдельным ограниченным каналом и проверяются отдельно от semantics.
- A3b-recurrent является ablation, а не единственной реализацией concept-level memory.
- Переход к каждому следующему этапу выполняется только после выполнения критерия предыдущего этапа.

## 18. Правило остановки разработки

Для каждого этапа заранее фиксируются:

1. одна основная гипотеза;
2. обязательные baselines, ablations и interventions;
3. ограниченный набор BLOCKER-критериев;
4. количественный критерий GO / STOP в отдельном experiment contract;
5. backlog улучшений, которые не блокируют текущий MVP.

Для ближайшей разработки действуют границы:

- текущий A2 PR не расширяется до A3;
- A3a реализуется отдельным PR;
- A3b начинается только после технической проверки A3a;
- Verbalizer начинается только после проверки причинной роли `z_semantic`;
- closed-loop режим не смешивается с open-loop экспериментом;
- после фиксации этого roadmap широкое архитектурное ревью прекращается;
- новые идеи переносятся в backlog или отдельный versioned experiment contract и не возобновляют ревью A2/A3a.
