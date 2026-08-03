# Решение по Development Quality Evaluation v0.1

## 1. Статус

```text
Experiment:
Development Quality Evaluation v0.1

Scope:
A2 structured baseline
A3a codebook channel prototype
A4 compute-then-zero control

Status:
COMPLETED

Scientific decision:
REDESIGN

Stage 2A semantic gate:
NOT PASSED

A3b:
REMAINS BLOCKED
```

Решение `REDESIGN` означает следующее:

- инфраструктура development-эксперимента исполняется end-to-end;
- зафиксированный результат воспроизводим;
- v0.1 не создал данных, позволяющих сравнить causal usefulness latent feedback;
- до semantic-target experiment необходимо устранить базовую проблему learnability.

Это решение не является `GO`, confirmatory failure, доказательством superiority A2 или доказательством неработоспособности latent reasoning.

## 2. Зафиксированная конфигурация

Источником числовых значений этого раздела является committed compact summary:
[`data/a2_a3_a4_heldout_summary.json`](data/a2_a3_a4_heldout_summary.json).

| Поле | Зафиксированное значение |
|---|---|
| Evaluator version | `development-quality-evaluation/0.1` |
| Implementation commit | `779172c3bbca3d03552deaed6421e82fcf19a932` |
| Evaluator source SHA-256 | `sha256:9205ad312fc37fa9927505e9c44a599e29fc5e31180db9d2e49ebfcc247b4570` |
| Requirements lock SHA-256 | `sha256:883c8e262c6f3ea917239010be08ac0064ab67de70c1caad7bb98fe9f0b68401` |
| Python | `3.11.15` |
| Torch | `2.12.0+cpu` |
| NumPy | `2.3.5` |
| Execution device | `cpu` |
| Seeds | `17`, `29`, `43` |
| Variants | `A2-structured-baseline`, `A3a-codebook`, `A3a-zero` (`A4`) |
| Train tasks | `3` |
| Held-out tasks | `2` |
| Evaluation units | `18` |
| Training budget | `3` epochs; `9` updates/run |
| Checkpoint policy | `final_epoch_only_no_heldout_selection` |
| Replay hash | `sha256:c676cb4d01798b20349342e91f3a8512dd5710010482f3995256f5430b069b3c` |

Dataset, variants, seeds, decoding behavior and training budget в этом документе только фиксируют historical v0.1 setup и не создают новый экспериментальный контракт.

## 3. Фактический результат

Выполнено `18` evaluation units:

- три варианта: `A2`, `A3`, `A4`;
- три seed: `17`, `29`, `43`;
- две held-out задачи на каждую пару variant × seed.

Во всех evaluation units:

- generation завершилась сразу через `END`;
- predicted plan был пустым;
- исполнимые действия не были получены;
- attempted action count был равен `0`;
- applicable action count был равен `0`;
- task success rate был равен `0.0`;
- nonempty plan rate был равен `0.0`;
- end-only rate был равен `1.0`;
- latent feedback не применялся к downstream action positions;
- число observed feedback application positions было равно `0`;
- applicability comparison для `A3` против `A2` или `A4` не возникло: `action_applicable_rate` остался `null`.

Пустой план при изначально неудовлетворённой цели не считается full-plan executable. Все observed failures имели код `GOAL_NOT_ACHIEVED`.

Полный human-readable результат сохранён в
[`A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md`](A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md).

## 4. Что результат доказывает

1. Development evaluation pipeline исполняется end-to-end.
2. Committed results привязаны к зафиксированному implementation provenance.
3. Raw artifact integrity, semantic validation и replay являются исполняемыми частями pipeline.
4. Два независимых canonical worker воспроизводят одинаковый canonical semantic result для зафиксированного v0.1 setup.
5. Текущий фиксированный v0.1 setup приводит к `END-only` поведению на двух committed held-out задачах для всех трёх variants и всех трёх seeds.
6. При `END-only` поведении невозможно оценить causal usefulness latent feedback на downstream action positions.

## 5. Что результат не доказывает

- Не доказано, что `A3` хуже `A2`.
- Не доказано, что latent feedback бесполезен.
- Не доказано, что semantic bottleneck не работает.
- Не доказано, что полная архитектура не масштабируется.
- Не доказано, что увеличение training budget исправит поведение.
- Не доказано, что проблема вызвана только training budget.
- Не проведён semantic-target experiment.
- Не проведён confirmatory experiment.
- Stage 2A semantic gate не пройден.
- Не получено разрешение на A3b.

## 6. Freeze policy

Development Quality Evaluation v0.1 является неизменяемым historical development baseline.

Для v0.1 действуют правила:

- committed v0.1 results больше не перегенерируются с другим training budget;
- dataset, variants, seeds и decoding policy v0.1 не меняются задним числом;
- post-hoc tuning не относится к v0.1;
- любое изменение training policy создаёт новую versioned evaluation;
- исправление содержательной ошибки v0.1 требует отдельного migration/errata record;
- новый результат не должен молча заменять historical v0.1 artifacts;
- scientific interpretation v0.1 не повышается до confirmatory claim после появления новых development runs.

Следующие generated artifacts являются historical v0.1 artifacts и не изменяются этим decision record:

```text
docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md
docs/evaluations/data/a2_a3_a4_heldout_summary.json
```

## 7. Next required stage: Development Learnability Gate

До semantic-target experiments необходимо показать, что базовая structured модель способна генерировать непустые планы и решать хотя бы заранее определённую долю development-задач.

Следующая стадия начинается с диагностики `END-only` collapse, а не с произвольного увеличения epochs.

Минимальный diagnostic scope:

- teacher-forced action accuracy;
- free-running action accuracy;
- operator accuracy;
- argument/pointer accuracy при gold operator;
- `END` probability по позициям;
- first-error position;
- train/validation breakdown;
- exact-plan rate;
- executable-prefix rate;
- full-plan executable rate;
- goal-success rate;
- loss breakdown по heads;
- сравнение gold-history и predicted-history rollout.

Диагностический PR должен только добавлять наблюдаемость и не должен одновременно менять training policy. Изменение budget, curriculum, loss weights, decoding policy или optimization policy относится к отдельному versioned Development Learnability v0.2 contract.

## 8. Порядок следующих работ

```text
1. Merge and freeze quality v0.1.
2. Diagnose END-only collapse.
3. Define Development Learnability v0.2 contract.
4. Run bounded development search.
5. Require A2 learnability threshold.
6. Define semantic-target experiment contract.
7. Implement A3s and required controls.
8. Evaluate Stage 2A semantic gate.
9. Only after GO consider A3b.
```

Этот порядок не проектирует A3b и не вводит новые нормативные архитектурные интерфейсы. A3b остаётся blocked до отдельного решения `GO` по Stage 2A semantic gate.

## 9. Связанные документы

- [Development quality report](A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md)
- [Committed compact summary](data/a2_a3_a4_heldout_summary.json)
- [Toy implementation progress](../TOY_IMPLEMENTATION_PROGRESS.md)
- [Non-normative Cognitive Planner roadmap](../architecture/COGNITIVE_PLANNER_FUTURE_ARCHITECTURE_RU.md)
