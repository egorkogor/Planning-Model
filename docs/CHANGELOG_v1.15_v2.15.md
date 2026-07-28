# Changelog v1.15 / v2.15

Минимальные исправления, обязательные для запуска MVP без расширения trust-бюрократии.

- Зарегистрированы WorkPlan, EpisodeLog и AttemptLog в report registry.
- Добавлен `SelectedTaskManifest`: точный список confirmatory-задач фиксируется до outcomes.
- Signed SealerManifest напрямую связывает path/hash selected-task manifest и task count.
- Full-plan lineage обязан покрывать точный selected task set; Stage 1B — декартово произведение selected tasks × семь arms.
- `run_id` и stage проверяются между lineage index, EpisodePlanManifest, WorkPlan, EpisodeLog и AttemptLog.
- Evaluator `task_count` пересчитывается по lineage, а не принимается декларативно.
- AnalysisInput содержит единый expected task set; все comparisons обязаны использовать его полностью.
- Sample-size components привязаны к точным comparison IDs по стадиям.
- Planner replay metric унифицирована как `P_REPLAY_GOAL_SUCCESS`.
