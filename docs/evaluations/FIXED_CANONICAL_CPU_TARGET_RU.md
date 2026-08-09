# Fixed canonical CPU target — provisioning и acceptance contract

## Статус

```text
TARGET CONTRACT DEFINED
TARGET NOT PROVISIONED
ACCEPTANCE 0/3
RUNTIME/1.1 NOT ACCEPTED

BLOCKED ON FIXED RUNNER PROVISIONING
ACCEPTED RUNS: 0/3
```

Terminal state этого раунда: `FIXED TARGET NOT PROVISIONED / NOT ACCEPTED`.

PR не утверждает, что GitHub-hosted `ubuntu-24.04` является fixed target. Обычные hosted runners остаются только regression/observation environment.

## Почему provisioning заблокирован

Доступный GitHub integration получает `403 Resource not accessible by integration` при чтении repository self-hosted runner inventory. В доступной среде нет отдельного AWS/Azure/GCP/VM provisioner или credentials, которыми можно создать и зарегистрировать dedicated runner. Поэтому hardware identity не наблюдалась и не подменяется фиктивными значениями.

## Что определено

Versioned semantic contract:

```text
toy-quality-fixed-cpu-target/1.0
```

Отдельная observation:

```text
toy-quality-fixed-cpu-target-observation/1.0
```

Candidate runtime:

```text
toy-quality-canonical-cpu-runtime/1.1
```

Acceptance artifact:

```text
toy-quality-fixed-target-acceptance/1.0
```

Runtime/1.1 не имеет самостоятельного accepted status: его acceptance устанавливается только итоговым acceptance artifact после 3/3 exact independent runs. Пока concrete target отсутствует, target/runtime contract hashes намеренно отсутствуют.

## FixedTargetContract и FixedTargetObservation

Contract фиксирует обязательные OS/kernel, architecture, CPU vendor/family/model/stepping/model name, microcode policy, required/forbidden flags policy, logical CPU policy, dedicated self-hosted runner labels/image, CPython build/compiler, PyTorch version/build hash, MKL/OpenMP/MKLDNN availability, ATen dispatch, MKL_CBWR, thread controls, deterministic controls и explicit AdamW execution path `foreach=false`, `fused=false`.

Observation собирается заново перед конкретным run, имеет собственный `observation_sha256` и не содержит hostname, PID, timestamps, workflow/job ID или temp paths. Operational provenance хранится отдельно в acceptance attempt.

Semantic validator сначала проверяет schemas/hashes, затем exact contract↔observation binding. CPU model/stepping, required flags, dispatch, Python/PyTorch build, MKL_CBWR, thread counts, MKLDNN и optimizer execution mode fail-closed. Fully resealed contradictory observation всё равно отклоняется.

## Runtime/1.1

`toy-quality-canonical-cpu-runtime/1.0` остаётся историческим frozen runtime contract и не переопределяется.

Runtime/1.1 candidate содержит обязательную dependency на exact:

```text
fixed_target_contract_version
fixed_target_contract_sha256
```

и закрепляет execution controls, включая:

```text
AdamW
foreach = false
fused = false
MKL_CBWR = COMPATIBLE
threads = 1
mkldnn_enabled = false
deterministic_algorithms = true
deterministic_warn_only = false
```

Это execution migration, а не изменение model/training policy: optimizer class, lr, betas, eps, weight decay, parameter groups, gradient clipping, epochs, updates, dataset/splits/seeds и A2/A3/A4 semantics не должны меняться.

Historical quality-v0.1 evaluator schema жёстко маркирует свой output как runtime/1.0. Поэтому этот PR не запускает historical evaluator и не переименовывает его output в runtime/1.1. Полная runtime/1.1 evaluation будет включена только после provisioning concrete target и должна иметь отдельный versioned lineage/migration record.

## Acceptance semantics

`accepted=true` возможно только при ровно трёх attempts. Все три обязаны иметь один implementation SHA, один target contract/hash, один runtime contract/hash, valid fresh target observation, `TRAINED_IN_RUN`, отсутствие reuse lineage и успешную full evaluation.

Между attempts exact equality требуется минимум для:

- initialization identities;
- training config;
- ordered tasks;
- checkpoint identities;
- optimizer-state identities;
- canonical state-dict identities;
- evaluation task results;
- replay hash;
- probe identity;
- canonical semantic payload;
- derived summaries.

Tolerance не используется. Duplicate workflow/job IDs, reordered attempts и reused checkpoint lineage отклоняются.

## Workflow trust boundary

`.github/workflows/fixed-target-acceptance.yml` имеет только controlled `workflow_dispatch` и labels:

```text
self-hosted
linux
x64
planning-model-canonical-cpu-v1
```

Он принимает exact 40-char implementation SHA, делает clean checkout, не сохраняет checkout credentials, требует committed concrete contract и pinned runner image identity `/etc/planning-model-runner-image-id`, создаёт clean venv и выполняет pre-training target validation.

В blocked state workflow намеренно останавливается после preflight с `FIXED_TARGET_ACCEPTANCE_EXECUTION_NOT_ENABLED`; preflight нельзя засчитать как attempt. Это исключает случайное выполнение historical runtime/1.0 как runtime/1.1 и исключает фиктивное acceptance до provisioning.

Permanent canonical runner не должен выполнять fork PR code. Workflow не использует `pull_request`/`pull_request_target`, secrets ему не требуются. После каждого invocation workspace очищается.

## Точная внешняя provisioning спецификация

Нужно provision одно dedicated execution target:

1. Linux x86_64 bare-metal либо VM, закреплённая за конкретным host/CPU identity без плавающего hosted hardware; live migration на другой CPU identity запрещена.
2. Зарегистрировать GitHub Actions self-hosted runner в `egorkogor/Planning-Model` с labels `self-hosted`, `linux`, `x64`, `planning-model-canonical-cpu-v1`.
3. Создать immutable/versioned VM image identity и записать её в `/etc/planning-model-runner-image-id`.
4. После provisioning снять exact CPU vendor/family/model/stepping/model name; выбрать и зафиксировать microcode policy (`exact` предпочтительно; иной mode требует отдельного обоснования).
5. Зафиксировать exact kernel, logical CPU count policy и required CPU flags; unexpected forbidden flags должны fail-closed.
6. Зафиксировать CPython 3.11 build/compiler и pinned `torch==2.12.0+cpu` build configuration SHA-256.
7. Разрешить repository Actions читать/dispatch этот runner; текущему integration сейчас этого права не хватает.
8. На самом target создать и commit `configs/fixed-cpu-target-1.0.json`, полученный из реальной observation, а не из предполагаемых значений.
9. После external review concrete contract заморозить один final implementation SHA и только затем включить full runtime/1.1 acceptance execution.
10. Выполнить три независимых `workflow_dispatch` full runs, без rerun-as-new-attempt и без reuse checkpoints; собрать 3/3 machine-readable evidence и exact cross-attempt comparison.

До выполнения этих действий concrete target contract/hash, hardware/software identity и runtime/1.1 hash считаются `unavailable`, а не неизвестными значениями, которые можно заполнить предположениями.

## Historical runtime/1.0 и migration

Поддерживаемый вывод PR №18 не меняется:

```text
First observed cross-host divergence occurs during AdamW parameter update.
The underlying instruction-level cause remains unidentified.
```

Будущий runtime/1.1 может дать numerically другой lineage относительно frozen v0.1. Это допустимый migration outcome. В таком случае сохраняется разделение:

```text
runtime/1.0 = historical lineage
runtime/1.1 = new fixed-target execution lineage
migration required
```

Frozen v0.1 файлы не переписываются и их hashes не заменяются.

## Ограничения

Fixed runner не provisioned; accepted attempts `0/3`; cross-attempt comparison `NOT_RUN`; runtime/1.1 candidate не принят. Следующая стадия начинается только после реального provisioning и наблюдения hardware/software target identity.
