# Cognitive Planner — расширенная архитектура исходной гипотезы

**Статус:** non-normative roadmap.  
**Назначение:** зафиксировать архитектуру, выходящую за пределы узкого эксперимента Work Planner / BlocksWorld v1.21.  
**Важно:** документ не изменяет scientific lock, confirmatory protocol или контракты v1.21.

## 1. Исходная гипотеза

Модель может рассуждать не только последовательностью текстовых токенов, но и последовательностью более крупных смысловых единиц — `ConceptStep`.

Один `ConceptStep` соответствует одному законченному смысловому фрагменту:

- шагу рассуждения;
- шагу плана;
- действию или вызову инструмента;
- смыслу одного предложения или абзаца ответа.

Языковая модель не обязана заново находить решение. После построения смысловой траектории она используется как verbalizer: переводит уже сформированный смысл шага в исполнимую команду или естественный язык.

Ключевая проверяемая версия гипотезы:

> Если смысл шага проходит через отдельный ограниченный интерфейс, а следующие шаги и downstream-модули не могут обойти этот интерфейс через скрытые состояния Reasoner, то последовательность крупных смысловых шагов может быть полезнее или эффективнее последовательности только текстовых токенов.

## 2. Архитектурные объекты

Полная схема разделяет смысл шага, привязку к конкретным объектам и итоговую исполнимую команду.

### 2.1. LatentConceptStep

```text
LatentConceptStep {
  step_id
  step_type: REASON | ACTION | TOOL | ANSWER | END
  z_semantic: float[d]
  grounding_refs: object/tool references[]
  source_observation_hash?: sha256
}
```

`z_semantic` — латентное смысловое представление шага. Размерность `384` используется в текущей архитектуре A3 как первая проверяемая конфигурация, но не является фундаментальным ограничением гипотезы.

`grounding_refs` — отдельный структурированный канал, связывающий общий смысл с конкретными объектами, инструментами или сущностями задачи. Он не заменяет `z_semantic` и не должен содержать готовое описание всего действия.

### 2.2. ResolvedActionStep

```text
ResolvedActionStep {
  source_concept_step_id
  action: typed action
  args: grounded references[]
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

### 2.4. VerifiedSolution

```text
VerifiedSolution {
  final_state_or_result
  executed_actions_or_tool_results
  compact_reasoning_summary_hash
  task_constraints
}
```

`VerifiedSolution` является проверяемым интерфейсом между reasoning trajectory и answer trajectory.

## 3. Semantic bottleneck

В полной архитектуре `LatentConceptStep` является реальным интерфейсом между Reasoner и последующими модулями:

```text
Reasoner hidden state h_t
          ↓
Semantic Bottleneck
          ↓
step_type_t + z_t + grounding_refs_t
          ├→ Action Decoder / Tool Resolver
          └→ Answer Verbalizer
```

После создания `LatentConceptStep` Action Decoder и Verbalizer не должны получать `h_t` или другие скрытые состояния Reasoner в обход `z_semantic` и разрешённых структурированных полей.

Без этого ограничения `z_semantic` может оказаться auxiliary output, который выглядит интерпретируемым, но причинно не участвует в решении.

### 3.1. Output bottleneck

Для A3b downstream-модули получают только:

- `step_type`;
- `z_semantic`;
- `grounding_refs`;
- ограниченный публичный каталог объектов или инструментов;
- минимальные структурные ограничения формата.

Прямой путь `h_t → action`, `h_t → tool call` или `h_t → answer text` запрещён.

### 3.2. Inter-step bottleneck

A3b должен ограничивать не только выход текущего шага, но и связь между соседними мыслями.

Внутри вычисления одного шага временные hidden states и attention разрешены. На границе `LatentConceptStep` внутренний decoder cache и скрытая история шага сбрасываются или становятся недоступными следующему шагу.

Между соседними шагами разрешено передавать только:

- предыдущий `z_semantic`;
- предыдущий `step_type`;
- предыдущие `grounding_refs`;
- публичное состояние задачи;
- последний `ObservationEvent`, если он существует;
- неизменяемое task encoding;
- явные позиционные и служебные признаки.

```text
внутри шага t:
TaskEncoding + PublicState + PreviousConceptStep
→ temporary h_t
→ LatentConceptStep_t

между t и t+1:
temporary h_t и KV-cache не передаются
LatentConceptStep_t + PublicState + Observation
→ следующий шаг
```

Иначе система проверяет только декодируемость действия из `z`, но продолжает рассуждать скрытой token-level траекторией `h_1 → h_2 → h_3`.

## 4. Полный поток обработки

```text
Запрос / состояние задачи
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
- **Semantic Bottleneck** превращает скрытое состояние шага в ограниченный `LatentConceptStep`;
- **Grounding channel** связывает общий смысл шага с конкретными объектами и инструментами;
- **Executor** применяет структурированные действия и возвращает наблюдаемое состояние;
- **Verbalizer** переводит смысловые шаги ответа в естественный язык.

## 5. Два связанных потока

### 5.1. Reasoning trajectory

Последовательность `REASON`, `ACTION` и `TOOL` шагов описывает поиск и выполнение решения. Она может включать скрытые смысловые шаги, но машинно значимые действия должны оставаться типизированными и проверяемыми.

### 5.2. Answer trajectory

После получения решения модель строит отдельную последовательность `ANSWER` шагов. Каждый шаг задаёт смысл одного предложения или абзаца, а frozen LLM verbalizer преобразует его в текст.

```text
сначала решить → затем спланировать ответ → затем выразить словами
```

### 5.3. Интерфейс между reasoning и answer

Answer Planner не получает скрытые состояния Reasoner напрямую. Его входом является `VerifiedSolution`.

`compact_reasoning_summary` создаётся только из разрешённых, сохранённых и проверяемых данных:

- `LatentConceptStep`;
- `grounding_refs`;
- `ResolvedActionStep`;
- `ObservationEvent`;
- verified final result;
- task constraints.

Запрещено строить summary из:

- `h_t`;
- KV-cache;
- несохранённой hidden-state history;
- отдельной LLM, которой доступен полный скрытый reasoning в обход ConceptStep.

Summary сохраняется как отдельный hash-bound артефакт:

```text
CompactReasoningSummary {
  source_concept_step_hashes[]
  source_action_step_hashes[]
  source_observation_hashes[]
  verified_result_hash
  summary_payload
  summary_hash
}
```

В strict reconstruction mode summary вообще не передаётся Verbalizer.

## 6. Допустимые переходы

Минимальная state machine:

```text
START  → REASON | ACTION | TOOL | ANSWER | END
REASON → REASON | ACTION | TOOL | ANSWER | END
ACTION → external ObservationEvent
TOOL   → external ObservationEvent
ObservationEvent → REASON | ACTION | TOOL | ANSWER | END
ANSWER → ANSWER | END
END    → terminal
```

Правила:

- `ObservationEvent` не является `ConceptStep`;
- `ANSWER → ACTION/TOOL/REASON` запрещено в первой реализации;
- после `ACTION` или `TOOL` следующий смысловой шаг может создаваться только после фиксации Observation;
- `END` завершает траекторию;
- все переходы логируются и проверяются машинно.

## 7. Open-loop и closed-loop режимы

### 7.1. Open-loop experimental mode

```text
полный план создаётся заранее
→ замораживается
→ исполняется без изменений
```

Используется для чистого сравнения A2/A3a и проверки причинной пользы semantic feedback. Replanning и suffix regeneration запрещены.

### 7.2. Closed-loop cognitive mode

```text
LatentConceptStep
→ действие или инструмент
→ ObservationEvent
→ обновление State Store
→ следующий LatentConceptStep
```

Используется в задачах, где результат действия заранее неизвестен: инструменты, поиск, код, файлы и внешняя среда.

В этом режиме replanning не скрывается и не считается нарушением. Он является явной частью архитектуры, а каждый новый шаг связывается с `source_observation_hash`.

Open-loop и closed-loop результаты нельзя смешивать в одном сравнении без отдельного протокола.

## 8. Модули

### Task Encoder

Кодирует запрос, публичное состояние, доступные объекты, инструменты и ограничения.

### Latent Planner / Reasoner

Авторегрессионно создаёт внутреннее состояние текущего шага.

- В A3a предсказанный `z_semantic` возвращается в следующий шаг через semantic feedback, при этом обычная скрытая autoregressive history может сохраняться.
- В A3b действует inter-step bottleneck: скрытый cache между ConceptStep недоступен.

### Semantic Bottleneck

Создаёт `step_type`, `z_semantic` и `grounding_refs`.

`Step-Type Router` является отдельной prediction head внутри Semantic Bottleneck, а не вторым независимым маршрутизатором.

### Grounding Head

Выбирает конкретные ссылки на объекты или инструменты из разрешённого каталога. Он не имеет права кодировать произвольный текст или готовый план.

### Action Decoder / Tool Resolver

Для `ACTION` и `TOOL` преобразует `LatentConceptStep` в `ResolvedActionStep`.

В A3b Action Decoder получает только:

- `step_type`;
- `z_semantic`;
- `grounding_refs`;
- допустимый каталог объектов/инструментов;
- минимальные структурные ограничения.

Ошибка применения фиксируется fail-closed. В open-loop режиме скрытое перепланирование запрещено; в closed-loop режиме следующий шаг создаётся только после явной Observation.

### Verbalizer Adapter

Преобразует `z_semantic` в несколько soft/virtual tokens либо другое входное представление для frozen LLM.

### Frozen LLM Verbalizer

Формулирует предложение, абзац или пояснение действия. На первом этапе Verbalizer не имеет права менять план или возвращать новую reasoning trajectory.

### Executor and State Store

Исполняет действия, проверяет переходы состояния, сохраняет `ObservationEvent` и фактическую траекторию выполнения.

## 9. Два режима Verbalizer

### 9.1. Strict reconstruction mode

Используется для проверки, действительно ли смысл содержится в `z_semantic`.

Вход:

```text
z_semantic
+ grounding_refs или словарь допустимых имён объектов
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
- переставленный `z_semantic` другого шага при правильных refs;
- `z_semantic` другой задачи при правильных refs;
- правильный `z_semantic` с переставленными refs;
- правильный `z_semantic` с refs другой задачи;
- неправильный `step_type`;
- тот же минимальный контекст без `z_semantic`.

### 9.2. Contextual production mode

Используется после проверки reconstruction mode.

Вход:

```text
z_semantic
+ grounding_refs
+ исходный запрос
+ уже сформированный ответ
+ разрешённый фактический контекст
```

Этот режим оптимизирует качество текста, но сам по себе не доказывает, что Verbalizer использует latent-вектор.

## 10. Что считается мыслью на разных этапах

### Stage 2: supervised semantic proxy

В текущем BlocksWorld A3 target является 384-мерным embedding заранее определённой semantic signature.

```text
структурированная semantic signature
→ канонический текст
→ frozen encoder
→ target z
```

Успех здесь доказывает, что такой semantic feedback причинно полезен в ограниченном домене. Он не доказывает, что найдено универсальное представление человеческой мысли.

Точная идентичность объектов не обязана входить в `z_semantic`; она передаётся через отдельно проверяемый `grounding_refs`.

### Stage 3–4: learned ConceptStep representation

На следующих этапах `z_semantic` должен обучаться не только на заранее заданной метке, но и на совокупности сигналов:

- правильности действия или результата;
- предсказании следующего состояния;
- реконструкции смысла шага;
- согласованности reasoning trajectory;
- информационном bottleneck;
- контрастивных и intervention-контролях.

## 11. Граница одного ConceptStep

Фраза «одно предложение или абзац» является инженерной гипотезой, а не фиксированным свойством.

Первая реализация использует:

- разметку шагов из oracle/reasoning trajectory;
- явный `step_type`;
- максимальное число ConceptStep;
- отдельный `END`;
- ограниченную размерность `z_semantic`.

Будущие варианты могут добавить boundary head, но она не входит в первый MVP.

## 12. Пять крупных этапов проверки

### Этап 1 — A2: структурированный Planner

```text
задача → полный frozen plan → исполнение
```

**Критерий завершения:** модель обучается, выдаёт многошаговый план, один раз вызывается, не перепланирует и воспроизводимо достигает цели на toy-задачах.

### Этап 2 — A3: latent thought

Этап делится на две подфазы.

#### Этап 2A — A3a: latent feedback

Каждый шаг дополнительно выдаёт `z_semantic`, который влияет на следующий шаг, но действие и latent могут предсказываться параллельно из одного decoder hidden state, а обычная autoregressive history сохраняется.

Основные сравнения:

- `A2`: semantic feedback отсутствует;
- `A3`: используется правильный предсказанный latent;
- `A4`: вычисление latent сохраняется, но feedback обнуляется;
- `A5`: latent-векторы переставляются между шагами;
- `A3r`: parameter-matched случайный смысловой код.

**Критерий завершения:** A3a улучшает качество на невидимых задачах, а обнуление или перестановка meaningful latent feedback уменьшает улучшение.

#### Этап 2B — A3b: semantic и inter-step bottleneck

Проверяется более сильная исходная гипотеза:

```text
h_t → LatentConceptStep_t
LatentConceptStep_t + PublicState → h_(t+1)
LatentConceptStep_t → ResolvedActionStep_t / text_t
```

Запрещены:

- прямой путь `h_t → action_t`;
- прямой путь `h_t → text_t`;
- передача hidden-state history или KV-cache между ConceptStep.

Action Decoder получает смысл и grounding только через `LatentConceptStep` и разрешённые справочники.

Обязательные независимые interventions:

- zero/shuffled/wrong-task `z_semantic` при сохранённых refs;
- shuffled/wrong-task refs при сохранённом `z_semantic`;
- неправильный `step_type`;
- удаление `source_observation_hash` в closed-loop режиме.

**Критерий завершения:** система проходит заранее зафиксированный experiment contract, сохраняет установленную долю качества A3a, а intervention каждого канала причинно меняет только ожидаемую часть поведения.

### Этап 3 — Verbalizer MVP

Проверяется декодируемость отдельной мысли в strict reconstruction mode:

```text
LatentConceptStep → frozen LLM → описание шага
```

Verbalizer пока не участвует в исполнении.

**Критерий завершения:** на заранее зафиксированном наборе полей текст:

- сохраняет правильный `step_type`;
- называет правильное действие и аргументы;
- не добавляет новые объекты и отношения;
- семантически соответствует target;
- проходит пороги experiment contract;
- ухудшается согласно заранее зафиксированным ожиданиям при intervention по `z`, refs или типу шага.

### Этап 4 — разделение Reasoning и Answer

Добавляются типы `REASON` и `ANSWER`, отдельная answer trajectory и генерация текста только после завершения решения.

**Критерий завершения:** разделённая система не уступает по правильности решения и проходит заранее зафиксированные пороги по:

- фактической согласованности ответа;
- покрытию обязательных частей решения;
- отсутствию новых неподтверждённых утверждений;
- устойчивости к перестановке answer-step;
- качеству относительно единой смешанной траектории.

### Этап 5 — полный сравнительный эксперимент

Эксперименты расширяются последовательно, а не сразу на все домены:

1. BlocksWorld — механика, исполнимость и причинность semantic feedback;
2. один текстовый домен — простая символическая математика или логика;
3. один интерактивный домен — инструменты или небольшие изменения кода;
4. только затем широкое сравнение.

Сравниваются:

- Direct LLM;
- текстовый Chain-of-Thought;
- текстовый план;
- A2 structured planner;
- A3a latent feedback;
- A3b semantic/inter-step bottleneck;
- A3b + Verbalizer;
- A3b + отдельные reasoning и answer trajectories.

**Критерий успеха гипотезы:** латентные крупные смысловые шаги дают устойчивое улучшение качества или эффективности, которое исчезает в корректных intervention/ablation-контролях и не объясняется дополнительными параметрами, данными, контекстом Verbalizer или вычислениями.

## 13. Experiment contract перед каждым этапом

Этот roadmap не задаёт числовые границы GO/STOP. Они не должны выбираться после просмотра результатов.

До начала реализации или запуска каждого этапа создаётся отдельный versioned experiment contract, который фиксирует:

- primary hypothesis;
- primary metric;
- evaluation unit;
- baselines;
- ablations и interventions;
- threshold или non-inferiority margin;
- допустимую вычислительную стоимость;
- seeds и split policy;
- STOP/GO rule;
- критерии технической валидности;
- список неблокирующего backlog.

Фразы вроде «существенная доля качества», «существенно ухудшается» и «устойчивое улучшение» считаются placeholders до появления соответствующего experiment contract и не могут использоваться как основание для GO.

## 14. Основные метрики

- достижение правильного результата задачи;
- валидность и исполнимость действий;
- перенос на новые структуры и длины задач;
- польза semantic feedback: A3a против A2/A4/A5/A3r;
- сохранение качества через semantic/inter-step bottleneck: A3b против A3a;
- causal sensitivity к замене `z_semantic`;
- causal sensitivity к замене `grounding_refs`;
- независимость каналов `z_semantic`, refs и `step_type`;
- восстанавливаемость смысла latent-вектора Verbalizer;
- соответствие текста фактическому решению;
- отсутствие вымышленных фактов;
- стоимость обучения и инференса;
- стабильность между seeds и повторными запусками.

## 15. Ограничения первой реализации

- Не утверждается, что один вектор всегда равен одному человеческому понятию.
- Размерность 384 и граница «одно предложение или абзац» являются проверяемыми инженерными выборами.
- Текст Verbalizer не считается доказательством содержания latent-вектора без strict reconstruction и intervention-контролей.
- Текущий A3 target является supervised semantic proxy, а не доказанной универсальной мыслью.
- BlocksWorld проверяет механику и причинную полезность semantic feedback, но не доказывает перенос на общий reasoning.
- Grounding refs являются отдельным структурированным каналом и должны проверяться отдельно от semantics.
- Переход к каждому следующему этапу выполняется только после выполнения критерия предыдущего этапа.

## 16. Правило остановки разработки

Для каждого этапа заранее фиксируются:

1. одна основная гипотеза;
2. обязательные baselines, ablations и interventions;
3. ограниченный набор BLOCKER-критериев;
4. количественный критерий GO / STOP в отдельном experiment contract;
5. backlog улучшений, которые не блокируют текущий MVP.

Новые требования к будущим confirmatory-стадиям не должны бесконечно расширять критерии приёмки технического prototype.

Для ближайшей разработки действуют границы:

- текущий A2 PR не расширяется до A3;
- A3a реализуется отдельным PR;
- A3b начинается только после технической проверки A3a;
- Verbalizer начинается только после проверки причинной роли `z_semantic`;
- closed-loop режим не смешивается с open-loop экспериментом;
- после закрытия указанных в этом документе архитектурных интерфейсов новые идеи переносятся в backlog и не возобновляют широкое ревью A2/A3a.
