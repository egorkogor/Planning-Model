# Changelog v1.6 / v2.6

- Заменён линейный порядок P00–P16 на explicit P00–P20 outcome state machine.
- Manual gates разделены на pre-gate и post-approval checks.
- Добавлены fixed split sizes, architecture, optimizer, hyperparameter grid, losses и seed selection.
- Полностью определён semantic feedback `ConceptPacker` для A2b/A2c/A3/A4/A5.
- Intent Labeler перенесён в исполняемый Python-контракт с exhaustive n≤5 fixture requirement.
- Right padding удалён; canonical prompts unpadded, batching только left-padded.
- Добавлена development-only prompt calibration/freeze.
- Добавлены Builder/Evaluator/Auditor roles, sealed datasets и signed evaluator manifest.
- Добавлен immutable protected contract lock.
- Добавлены schemas и report registry для phase/study/decision/audit/resource/ledger artifacts.
- Добавлены provisioning, budget, recovery/checkpoint и mandatory clean reproduction contracts.
- Monitoring расширен ресурсами, стоимостью, ETA, blindness и lock status.
