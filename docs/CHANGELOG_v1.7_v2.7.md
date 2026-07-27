# Изменения v1.7 / v2.7

- исправлен conditional gate G01: бесплатный локальный путь не требует DecisionRecord;
- введён bootstrap release manifest, защищающий протокол до P00/P02;
- добавлена отдельная роль Data Sealer и формальные dispatch/sealer manifests;
- report registry покрывает decisions, dispatch, resource JSONL и sealed manifests;
- финальное обучение использует фиксированные 12 000 updates; A4/A5 — inference interventions того же A3 checkpoint;
- добавлен исполняемый statistics contract, sample-size calculator и decision gates;
- добавлен нормативный semantic resolver calibration contract;
- C01–C04 стали exact UTF-8 artifacts, guidance token blocks — frozen 32-token artifacts;
- P19 получил профили раннего STOP;
- добавлен формальный self-review loop и adversarial release checks.
