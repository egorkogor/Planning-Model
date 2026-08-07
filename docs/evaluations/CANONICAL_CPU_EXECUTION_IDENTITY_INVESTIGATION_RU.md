# Investigation: cross-host CPU nondeterminism и требование fixed target

Статус:

```text
INVESTIGATION COMPLETED
COMMON HOSTED PROFILE NOT FOUND
FIXED TARGET REQUIRED
MIGRATION REQUIRED
```

Этот документ фиксирует investigation и retained reproduction tooling. Он не утверждает, что execution identity уже pinned, fixed target выбран или provisioned, либо `runtime/1.1` принят.

## Retained execution profiles

Probe сохраняет три именованных профиля:

```text
historical-default
default-single-tensor
avx2-single-tensor
```

`historical-default` сохраняет исторические AdamW defaults:

```text
foreach=None
fused=None
ATEN_CPU_CAPABILITY absent
MKL_CBWR absent
```

Controlled profiles используют:

```text
foreach=False
fused=False
MKL_CBWR=COMPATIBLE
```

Для `default-single-tensor` требуется `ATEN_CPU_CAPABILITY=default`, для `avx2-single-tensor` — `ATEN_CPU_CAPABILITY=avx2`. Environment проверяется до импорта Torch. Фактический dispatch сохраняется через `torch.backends.cpu.get_cpu_capability()`.

CLI `--optimizer-foreach` и `--optimizer-fused` являются только подтверждением profile contract и не могут переопределить его.

## Execution contract

Contract version:

```text
toy-quality-cpu-execution-contract/1.0
```

Exact fields:

```text
contract_version
profile
profile_kind
ATEN_CPU_CAPABILITY
actual_atten_cpu_capability
MKL_CBWR
foreach
fused
optimizer_class
optimizer_hyperparameters
torch_num_threads
torch_num_interop_threads
mkldnn_enabled
deterministic_algorithms
canonical_runtime_version
python_implementation
python_version
python_compiler
python_build
torch_version
torch_build_configuration_sha256
mkl_available
openmp_available
mkldnn_available
```

Для всех retained profiles canonical runtime invariants фиксированы точно:

```text
torch_num_threads = 1
torch_num_interop_threads = 1
mkldnn_enabled = false
deterministic_algorithms.enabled = true
deterministic_algorithms.warn_only = false
```

Software-runtime identity входит в execution contract. Поэтому два валидных artifacts с разными Python/PyTorch builds, build configuration или canonical runtime version являются несовместимыми:

```text
comparable: false
reason: EXECUTION_CONTRACT_MISMATCH
equal: null
```

CPU vendor/model, CPU flags, Azure region и runner hostname в execution contract не входят. Они остаются observation-only частью hardware fingerprint и могут различаться между hosts.

## Retained probe artifact

Probe version:

```text
toy-quality-canonical-training-probe/2.0
```

Exact top-level fields persisted artifact:

```text
probe_version
variant
seed
epochs
ordered_train_task_ids
parameter_names
initial_parameters
updates
execution_contract
execution_contract_sha256
runtime
hardware_runtime_fingerprint
probe_identity
evidence_identity
```

`validate_probe_artifact(payload)` fail-closed проверяет exact field set и затем:

1. execution contract semantics;
2. `execution_contract_sha256`;
3. probe/2.0 semantics и `probe_identity`;
4. hardware fingerprint integrity;
5. exact runtime fingerprint;
6. согласованность runtime version с execution contract;
7. `evidence_identity`.

`probe_identity` покрывает numerical/probe identity payload и execution contract. Hardware observation не обязана входить в `probe_identity`.

`evidence_identity` определяется независимо:

```text
evidence_identity = sha256(canonical payload without evidence_identity)
```

Поэтому hardware observation и hardware fingerprint входят в evidence identity. Resealed numerical mutation со старым `evidence_identity`, либо resealed hardware mutation со старым `evidence_identity`, отклоняется.

`compare_probes(left, right)` сначала полностью валидирует оба artifacts. Повреждённый artifact выдаёт стабильный `ValueError` до comparable/incomparable classification. `EXECUTION_CONTRACT_MISMATCH` используется только для двух валидных, но разных execution contracts.

## Probe/2.0 semantics

Для `toy-quality-canonical-training-probe/2.0` зафиксировано:

```text
variant = A2
epochs = 3
ordered_train_task_ids =
  bw-00000001
  bw-00000002
  bw-00000003
update count = 9
```

Seed остаётся explicit integer argument.

Каждый update имеет exact field set и проверяется на sequential update index, epoch и canonical task order. Scalar tensor hashes используют только canonical lowercase `sha256:<64 hex>`.

`raw_gradients`, `gradients_after_clipping`, `adamw_exp_avg`, `adamw_exp_avg_sq` являются subsets `parameter_names`. `initial_parameters` и `parameters_after_optimizer_step` покрывают exact `parameter_names`. Forward logits и loss components имеют exact key sets. `parameter_names` уникальны, а их порядок является частью probe identity.

## Historical quality parity

Retained historical parity instrumentирует frozen `planner_toy.quality._train` без изменения `planner_toy/quality.py`.

Parameter capture order-independent:

- `LockedPlanner` wrapper сохраняет model, initial parameters и original forward;
- `_optimizer_named_parameters` сохраняет names/order, если helper вызывается первым;
- AdamW wrapper независимо восстанавливает names по identity фактического parameter list;
- если оба пути capture сработали, они обязаны совпасть;
- `require_captured_parameter_order()` возвращает только полный проверенный порядок и никогда не допускает сырой `KeyError`;
- instrumentation не добавляет искусственный вызов `_optimizer_named_parameters` в real quality path.

Update event capture активируется только реальным `optimizer.zero_grad` training update. Любой backward, который происходит до этого в model/preflight logic, не включается в update trace.

Historical parity требует exact:

```text
quality_trace == probe_trace
```

Сравниваются:

```text
parameter names/order
optimizer defaults
initial parameters
encoded task
forward logits
loss components
raw gradients
gradient norm
gradients after clipping
AdamW exp_avg
AdamW exp_avg_sq
parameters after optimizer step
clip max norm
clip parameter order
event order
```

Controlled single-tensor profiles остаются investigation alternatives и не выдаются за historical quality path.

## Hardware fingerprint integrity

Hardware fingerprint остаётся observation artifact. Его validator проверяет exact top-level/nested field sets, fingerprint version, canonical hash format и пересчитывает `observation_identity_sha256`.

Успех этой проверки не означает acceptance host как canonical fixed target.

## Cross-host investigation evidence

### Triggering canonical failure

```text
run: 30860485483
commit: 33d0ff62c597170d46d1bdb40a05687326aa9369
```

### Controlled AVX2 experiment

```text
run: 30872595002
commit: df126eab5e2da8b47a3d006406f7dbe2d3ed7268
profile: avx2-single-tensor
```

First observed divergence:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.bos_embedding
stage: parameters_after_optimizer_step
```

### Provisional DEFAULT sentinel

```text
run: 30907124345
commit: 2978356a7384164b4388d81cc8e75a5c0e7d03a4
```

Sentinel failed closed и не был принят как runtime contract.

### Controlled DEFAULT observation

```text
run: 30907486933
commit: f3f27ff8428088c29c126fac6cf59bdbae35b2bc
profile: default-single-tensor
```

First observed divergence:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.output_norm.bias
stage: parameters_after_optimizer_step
```

Поддерживаемый вывод остаётся ограниченным:

```text
First observed cross-host divergence occurs during AdamW parameter update.
The underlying instruction-level cause remains unidentified.
```

Investigation не идентифицирует конкретную vector instruction, assembly path или instruction-level root cause.

## Acceptance boundary

Hosted `canonical-compare` — только observation sample конкретной пары workers. Его green или red результат не является fixed-target acceptance evidence.

Fixed-target acceptance:

```text
unavailable
accepted fixed-target runs: 0/3
```

Будущая acceptance phase требует одного provisioned, versioned fixed execution target и трёх независимых полностью green runs на одном final head.

`toy-quality-canonical-cpu-runtime/1.0` не переинтерпретируется. `runtime/1.1` в этом PR не принимается.
