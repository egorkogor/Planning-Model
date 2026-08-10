# Fixed canonical CPU target — acceptance foundation

## Статус

```text
TARGET CONTRACT FRAMEWORK DEFINED
TARGET NOT PROVISIONED
ACCEPTANCE 0/3
RUNTIME/1.1 NOT ACCEPTED

BLOCKED ON FIXED RUNNER PROVISIONING
```

```text
PR19 = acceptance foundation
future phase = concrete target + 3/3 acceptance
```

This PR does not perform fixed-target acceptance.
It installs the trusted acceptance foundation on the default branch.

PR №19 не provision runner, не создаёт concrete hardware identity и не принимает
`toy-quality-canonical-cpu-runtime/1.1`. GitHub-hosted runners остаются только
regression/observation environment и не являются fixed-target evidence.

## Lifecycle

`workflow_dispatch` исполняет workflow с default branch, поэтому acceptance не
является pre-merge gate этого PR. Нормативный lifecycle:

```text
PR19
  define contracts
  define schemas
  define fail-closed preflight
  define acceptance evidence format
  define trusted dispatcher
  status BLOCKED / 0/3
→ external review
→ merge to main

THEN

runner provisioning

THEN separate acceptance phase:
  concrete target contract
  runtime/1.1 full execution
  three attempts
  final acceptance artifact
```

Следующий PR или acceptance phase в этом раунде не создаются.

## Contracts

Foundation определяет:

```text
toy-quality-fixed-cpu-target/1.0
toy-quality-fixed-cpu-target-observation/1.0
toy-quality-canonical-cpu-runtime/1.1
toy-quality-fixed-target-acceptance/1.0
toy-quality-fixed-target-source-inventory/1.0
toy-quality-fixed-target-attempt-manifest/1.0
```

Target contract фиксирует CPU/software/runtime policy и
`required_runner_labels`. Scheduler labels являются scheduling requirement, а
не самостоятельно измеренной hardware observation.

Preflight observation измеряет CPU, kernel, immutable runner image identity,
CPython build/compiler, exact `pip` version, PyTorch version/build,
ATen dispatch, MKL/OpenMP/MKLDNN и thread/deterministic controls. Она не
утверждает фактический optimizer path: AdamW на preflight не создаётся.

Runtime/1.1 требует:

```text
optimizer_class = AdamW
optimizer_foreach = false
optimizer_fused = false
MKL_CBWR = COMPATIBLE
threads = 1
mkldnn_enabled = false
deterministic_algorithms = true
deterministic_warn_only = false
```

Фактические `observed_optimizer_foreach` и `observed_optimizer_fused`
появляются только в full attempt evidence и извлекаются из persisted real
optimizer state. `None/None` не эквивалентно `false/false`.

## Record vs bundle validation

`validate_acceptance_record(...)` проверяет schema, exact fields, hashes,
attempt order, duplicate run/job IDs, status transitions и internal binding.
Он намеренно отклоняет `accepted=true` с
`FIXED_TARGET_ACCEPTED_REQUIRES_BUNDLE`.

Final acceptance может подтвердить только:

```text
validate_acceptance_bundle(root)
```

Bundle должен содержать:

```text
acceptance.json
attempt-1/
attempt-2/
attempt-3/
```

Каждый attempt содержит versioned `attempt_manifest.json`, `preflight.json`,
полный `evaluation/` bundle и `probe.json`. Attempt manifest рекурсивно
покрывает фактические файлы и их hashes; symlinks запрещены.

Bundle validator читает реальные artifacts и independently derives:

- initialization identities;
- training config identity;
- ordered task identity;
- trained checkpoint identities;
- optimizer-state identities;
- canonical state-dict identities;
- evaluation task-result identity;
- replay hash;
- probe identity;
- canonical semantic payload identity;
- derived summaries identity.

Значения из `acceptance.json` не являются authority: они сравниваются с
derived values. Checkpoint/optimizer files дополнительно cross-bind к
checkpoint manifests. Missing/extra training-run claim files, stale attempt
manifest, reused lineage и altered evidence fail closed.

## Source closure

Fixed-target source inventory строится не из нового вручную урезанного списка.
База — version-locked `evaluator_source_files` frozen quality-v0.1 artifact с
проверкой его `evaluator_source_sha256`. К ней добавляются fixed-target
workflow/scripts/schemas, `requirements.lock`, `pyproject.toml`, package
initializers и transitive probe validators.

Для каждого final attempt:

1. implementation SHA должен существовать как Git commit;
2. `source_inventory_at_commit(implementation_commit)` регенерируется через
   `git show <commit>:<path>`;
3. inventory и `source_inventory_sha256` должны совпасть exactly с preflight
   evidence;
4. missing source или nonexistent 40-hex SHA отклоняется.

## Trusted dispatcher

Acceptance workflow предназначен для работы после merge foundation в default
branch. Он не исполняет arbitrary same-repository SHA.

Policy:

```text
implementation SHA must be reachable from protected main
```

До checkout requested implementation workflow делает:

```text
git fetch --no-tags origin main
git cat-file -e "${IMPLEMENTATION_SHA}^{commit}"
git merge-base --is-ancestor "$IMPLEMENTATION_SHA" origin/main
```

Только после этого выполняется detached checkout exact SHA.

Dispatch доступен через repository `workflow_dispatch`; выполняться может
только commit из protected-main history. Fork PR code и unmerged branch code
на permanent canonical runner не исполняются. Workflow не использует
`pull_request`/`pull_request_target`, не требует secrets, checkout credentials
не сохраняются, workspace очищается после invocation.

## Reproducible runner image

Canonical acceptance не выполняет mutable package installation. Workflow не
делает `pip install`, не обновляет pip и не запускает dependency resolver.

Будущий immutable/versioned runner image должен уже содержать exact CPython,
pip, PyTorch wheel и native runtime libraries. Workflow только наблюдает и
проверяет их identities против concrete target contract. Image identity
читается из:

```text
/etc/planning-model-runner-image-id
```

## Blocked artifact

`docs/evaluations/data/fixed-target-acceptance.json` остаётся честным blocked
record:

```text
status = BLOCKED_ON_FIXED_RUNNER_PROVISIONING
accepted = false
attempts = []
target_contract = null
runtime_contract = null
cross_attempt_comparison = NOT_RUN
```

Fictitious hardware values не добавляются.

## Full execution disabled

До отдельной post-merge acceptance phase workflow заканчивается fail-closed:

```text
FIXED_TARGET_ACCEPTANCE_EXECUTION_NOT_ENABLED
```

Успешный preflight не является acceptance attempt.

## Historical lineage

Historical runtime остаётся:

```text
runtime/1.0 = historical lineage
runtime/1.1 = future fixed-target execution lineage
```

Поддерживаемый вывод investigation PR №18 не меняется:

```text
First observed cross-host divergence occurs during AdamW parameter update.
The underlying instruction-level cause remains unidentified.
```

Frozen v0.1 artifacts не переписываются.

## Provisioning handoff

После merge PR №19 отдельная фаза должна provision dedicated Linux x86_64
target без плавающей CPU identity/live migration, назначить scheduler labels,
зафиксировать immutable image identity, CPU vendor/family/model/stepping,
microcode policy, kernel, flags, logical CPU policy, CPython/pip/PyTorch/native
runtime identities и создать concrete `configs/fixed-cpu-target-1.0.json` из
реальной observation.

Только затем отдельная acceptance phase включает full runtime/1.1 execution и
создаёт три independent attempts с `TRAINED_IN_RUN`, без checkpoint reuse и
без rerun-as-new-attempt. До этого accepted attempts остаются `0/3`.
