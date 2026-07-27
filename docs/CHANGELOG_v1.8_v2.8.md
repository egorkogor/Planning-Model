# Изменения v1.8 / v2.8

## Lock и жизненный цикл

- Scientific lock отделён от Implementation lock.
- До Implementation lock добавлен обязательный toy preflight: contracts, domain, A1–A5 forward/backward, two-batch overfit, fake episode и golden statistics.
- Implementation-only patch разрешён только до P06, до появления pilot/outcome-данных, с точным diff, hash lineage и полным повтором preflight.
- Изменение scientific contracts всегда требует новой версии протокола и нового run.
- Bootstrap manifest защищает scientific trust root, включая собственные verifier/build scripts, ещё до P00.

## Оркестрация и gates

- P00–P20 заданы fail-closed state machine с условными переходами, флагами и ранними STOP-профилями.
- Conditional G01 не требует DecisionRecord для бесплатного локального пути.
- Gate verifier проверяет exact check set, обязательные outputs, hashes всех файлов в output directories, evidence, команды, transition flags, clean lineage и DecisionRecord.
- Самоссылочные, дублированные, незахэшированные и symlink-артефакты отклоняются.
- `RUN_STATUS.md` и `RUN_STATUS.json` обязательны на каждой фазе.

## Данные и blindness

- Builder, Data Sealer, Evaluation Runner, Auditor и Statistical Reviewer разделены ролями и identity manifests.
- Builder не получает confirmatory selection seed и не видит plaintext.
- Data Sealer генерирует secret seed, выбирает задачи через HMAC, публикует commitment и шифрует seed envelope для Auditor.
- Seed раскрывается Auditor только после подписанного evaluator result.
- Перед первым confirmatory dispatch обязателен независимый аудит статистической реализации человеком или другой model family.

## Модели и causal controls

- Final training использует фиксированные 12 000 optimizer updates; early stopping разрешён только в development.
- A4/A5 используют тот же checkpoint A3 и являются inference interventions.
- A4 обнуляет semantic vector после той же projection/normalization path.
- A5 использует детерминированный perfect derangement внутри strata; невозможность matching блокирует freeze.
- Добавлены абсолютные Stage 1B eligibility floors: A3 goal success, нижняя граница CI, valid-action rate и P_REPLAY.
- `STOP_PLANNER` допускает диагностический Stage 1A, но запрещает Stage 1B.

## Статистика

- Все decisions пересчитываются из canonical `AnalysisInput`, а не доверяют агрегатам отчёта.
- Реализованы hierarchical Planner bootstrap, clustered Stage 1A bootstrap, paired Stage 1B bootstrap и paired TOST.
- Sample size пересчитывается из paired pilot inputs; zero-discordance использует заранее зафиксированный conservative fallback.
- Абсолютные binary rates используют Wilson 95% CI.
- Locked gate definitions нельзя заменить полями отчёта.

## Prompt и resolver

- C01–C04 содержат exact UTF-8 prompts и golden examples.
- Right padding decoder-only LLM запрещён; canonical inference unpadded, batch — только left padding.
- Guidance artifacts заморожены как точные 32-token sequences.
- Semantic resolver thresholds калибруются только на development и замораживаются до confirmatory.

## Проверки релиза

- Добавлены adversarial tests для fabricated decisions, incomplete pairs, zero discordance, contract mutation, hidden seed, output hashing и gate bypass.
- Clean-restore, ZIP, Git bundle, checksum и bootstrap verification обязательны перед публикацией.
