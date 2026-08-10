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
future acceptance phase = runtime/1.1 execution semantics
```

PR №19 не provision runner, не создаёт concrete hardware identity и не выполняет runtime/1.1 evaluation. Final runtime/1.1 execution evidence validation remains disabled until the separate post-provisioning acceptance phase. Historical probe/2.0 is not accepted as runtime/1.1 evidence.

## Lifecycle

```text
PR19
  contracts + schemas
  fail-closed preflight
  source closure
  scientific policy
  execution-evidence interface
  trusted dispatcher
  acceptance status BLOCKED / 0/3
→ external review
→ merge to main
→ runner provisioning
→ separate acceptance phase
  concrete target
  versioned runtime/1.1 evaluator/probe if needed
  three independent full runs
  final acceptance artifact
```

`workflow_dispatch` — post-merge mechanism. PR19 не является pre-merge fixed-target acceptance gate.

## Versioned foundation contracts

```text
toy-quality-fixed-cpu-target/1.0
toy-quality-fixed-cpu-target-observation/1.0
toy-quality-canonical-cpu-runtime/1.1
toy-quality-fixed-target-acceptance/1.0
toy-quality-fixed-target-source-inventory/1.0
toy-quality-fixed-target-attempt-manifest/1.0
toy-quality-fixed-target-scientific-policy/1.0
toy-quality-fixed-target-execution-evidence/1.0
```

Fixed-target schemas deliberately live under `planner_toy/schemas/fixed_target_*.schema.json`. Они не match historical `toy_*.schema.json`, поэтому frozen `planner_toy.quality.SOURCE_FILES` и historical evaluator source identity остаются неизменными:

```text
sha256:9205ad312fc37fa9927505e9c44a599e29fc5e31180db9d2e49ebfcc247b4570
historical implementation = 779172c3bbca3d03552deaed6421e82fcf19a932
```

## Runtime/1.1 boundary

Target/runtime contract по-прежнему требует fixed target и:

```text
AdamW
foreach=false
fused=false
MKL_CBWR=COMPATIBLE
threads=1
mkldnn_enabled=false
deterministic_algorithms=true
deterministic_warn_only=false
```

Preflight не self-attest optimizer execution. Actual foreach/fused должны появиться только в future full execution evidence.

PR19 не имеет code path, способного подтвердить final `accepted=true`. `validate_acceptance_record(...)` и `validate_acceptance_bundle(...)` fail closed с:

```text
FIXED_TARGET_RUNTIME_1_1_EXECUTION_NOT_ENABLED
```

Blocked record и future non-final structural evidence могут валидироваться; claim-bearing final semantics остаются disabled до отдельной acceptance phase.

## Historical probe/2.0

`toy-quality-canonical-training-probe/2.0` относится к historical `toy-quality-canonical-cpu-runtime/1.0`. Он может быть валидирован как retained investigation evidence PR18, но:

- `probe.json` не является обязательной частью fixed-target acceptance bundle;
- `probe_identity` отсутствует в runtime/1.1 acceptance decision contract;
- probe/2.0 не участвует в exact-equality acceptance decision;
- probe/3.0 в PR19 не создаётся.

## Scientific policy

`toy-quality-fixed-target-scientific-policy/1.0` bind’ит migration к frozen quality-v0.1 scientific semantics, а не создаёт новый эксперимент.

Нормативно сохраняются:

```text
variants = [A2, A3, A4]
seeds = [17, 29, 43]
dataset seed = 17
historical dataset identity and frozen train/held-out split
epochs = 3
updates per run = 9
checkpoint policy = final_epoch_only_no_heldout_selection
training execution = TRAINED_IN_RUN

optimizer = AdamW
learning_rate = 3e-4
betas = [0.9, 0.95]
eps = 1e-8
weight_decay = 0.01
gradient clipping = clip_grad_norm_(max_norm=1.0)
```

Model, initialization, dtype, loss, decoding и split semantics не переопределяются PR19: они bind’ятся к immutable historical evaluator source identity и dataset identity. Единственная execution migration:

```text
runtime/1.1
fixed target
foreach=false
fused=false
```

Три одинаковых run неправильного эксперимента поэтому не могут стать acceptance.

## Future execution-evidence interface

PR19 определяет schema/interface, но не генерирует real execution manifest. Future attempt обязан предоставить versioned `execution-evidence.json`, bind’ящий минимум:

```text
execution_evidence_version
implementation_commit
target_contract_sha256
runtime_contract_sha256
target_observation_sha256
source_inventory_sha256
scientific_policy + scientific_policy_sha256

evaluator_version
evaluator_source_sha256
requirements_lock_sha256

dataset_hash
ordered_train_task_ids
ordered_eval_task_ids
variants
seeds
epochs
updates_per_run

optimizer_class
optimizer_hyperparameters
gradient_clipping
observed_optimizer_foreach
observed_optimizer_fused

evaluation_root_identity
```

Future validator должен cross-bind manifest к acceptance, attempt, preflight, evaluation-config, all nine training configs/artifacts и target observation. Foundation helper `validate_execution_binding_contract(...)` уже задаёт exact interface, но final acceptance execution gate остаётся закрытым.

## Commit-bound semantic validation

Claim-bearing semantic revalidation в future phase допускается только кодом implementation commit, evidence которого проверяется. Foundation invariant:

```text
git rev-parse HEAD == acceptance.implementation_commit
tracked working tree clean
```

`require_semantic_validation_checkout(...)` fail closed на другом HEAD и на dirty tracked tree. Альтернатива future phase — isolated clean worktree exact implementation commit. Historical evidence нельзя revalidate произвольным более новым checkout.

## Source closure

Fixed-target source inventory:

```text
historical locked quality evaluator source inventory
+
fixed-target workflow
fixed-target scripts
fixed_target_*.schema.json
requirements.lock
pyproject.toml
planner_toy/__init__.py
scripts/__init__.py
```

Historical source identity проверяется отдельно и не меняется из-за fixed-target schemas. Mutation `pyproject.toml` или fixed-target schema меняет fixed-target source identity.

## Trusted dispatcher

Policy сохраняется:

```text
workflow_dispatch only
runs-on: self-hosted/linux/x64/planning-model-canonical-cpu-v1
implementation SHA must be ancestor of protected main
persist-credentials=false
immutable runner image identity required
no mutable package installation
cleanup always
```

Перед detached checkout:

```text
git fetch --no-tags origin main
git cat-file -e "${IMPLEMENTATION_SHA}^{commit}"
git merge-base --is-ancestor "$IMPLEMENTATION_SHA" origin/main
```

После preflight full execution остаётся disabled:

```text
FIXED_TARGET_ACCEPTANCE_EXECUTION_NOT_ENABLED
```

Это ожидаемое поведение foundation PR.

## Blocked evidence

`docs/evaluations/data/fixed-target-acceptance.json` остаётся без hypothetical values:

```text
status = BLOCKED_ON_FIXED_RUNNER_PROVISIONING
accepted = false
attempts = []
target_contract = null
runtime_contract = null
cross_attempt_comparison = NOT_RUN
```

## Historical lineage

```text
runtime/1.0 = historical lineage
runtime/1.1 = future fixed-target execution lineage
```

Historical compact artifacts и evaluator source identity должны regeneratе byte-identically на implementation commit `779172c3bbca3d03552deaed6421e82fcf19a932`. Frozen quality-v0.1 files не переписываются.

Поддерживаемый вывод PR18 не меняется:

```text
First observed cross-host divergence occurs during AdamW parameter update.
The underlying instruction-level cause remains unidentified.
```
