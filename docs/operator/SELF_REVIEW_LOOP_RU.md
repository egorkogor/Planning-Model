# Формальный цикл саморевью

Для каждой фазы агент выполняет:

1. реализует только заявленный scope;
2. запускает locked checks текущей фазы;
3. запускает adversarial tests на обход contracts, lineage, roles и hashes;
4. сравнивает diff с активными locks;
5. повторяет проверку на clean checkout либо в чистой временной директории;
6. пишет `reports/self-review-PXX.json` с P0/P1/P2 findings;
7. исправляет P0/P1 и повторяет цикл;
8. завершает фазу только без открытых P0/P1 либо получает `BLOCKED`.

Максимум — три полных self-review цикла на фазу. Обычные тестовые итерации этим не ограничены.

## Политика изменений

- Scientific paths неизменяемы после P02.
- До G06 implementation-only дефект разрешено исправить только schema-valid patch record с exact diff и повторным toy preflight.
- После G06 Implementation paths также неизменяемы.
- Изменение outcome-relevant параметра в любой момент либо executable interpretation после G06 завершает run и требует новой версии.
