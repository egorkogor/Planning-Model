# Changelog v1.16 / v2.16

Минимальный launch-fix поверх v1.15. Новые trust/lock-слои не добавлялись.

## Обязательные исправления

1. Planner confirmatory lineage теперь обязан содержать точную матрицу `selected task × seed × arm`:
   - seeds: `101, 202, 303, 404, 505`;
   - arms: `A1, A2, A2b, A2c, A3, A3r, A4, A5` и deterministic `P_FULL_PLAN_REPLAY_RAW`;
   - дубликаты, пропущенные seeds/arms и подмена `planner_seed` запрещены.
2. Planner AnalysisInput принимает только пять зафиксированных final seed groups с одинаковым task set.
3. Все Stage 1A comparisons обязаны использовать один и тот же exact snapshot `pair_id` set для каждого task.
4. Stage 1B replay diagnostic использует единое каноническое имя `STAGE1B_E1_FULL_PLAN_REPLAY_GOAL_SUCCESS` в replay- и statistics-контрактах.

## Статус

После зелёного CI v1.16 разрешает P00–P02 и начало AI-driven реализации. Дальнейшее расширение защитной архитектуры не требуется до появления практического риска валидности.
