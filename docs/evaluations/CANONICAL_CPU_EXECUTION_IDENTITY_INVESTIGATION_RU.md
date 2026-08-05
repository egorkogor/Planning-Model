# Cross-host CPU nondeterminism: investigation and decision record

## Статус

```text
INVESTIGATION COMPLETED
COMMON HOSTED PROFILE NOT FOUND
FIXED TARGET REQUIRED
MIGRATION REQUIRED
```

Этот документ фиксирует investigation result. Он не объявляет новый canonical
runtime, не утверждает, что execution identity уже pinned, и не выбирает или
provision конкретный fixed target.

`runtime/1.1` не принят.

## Исходный сбой

Triggering canonical run:

```text
workflow run: 30860485483
commit: 33d0ff62c597170d46d1bdb40a05687326aa9369
```

Результат:

```text
checks (3.11): success
checks (3.13): success
canonical-run-a: success
canonical-run-b: success
canonical-compare: failure
```

Первое persisted расхождение было обнаружено в:

```text
training-runs/A2/seed-17/optimizer-state.pt
state[0].exp_avg[8567]
```

После него разошлись trained checkpoints и replay artifacts.

## Retained probe

Самодостаточный retained probe:

```text
scripts/run_canonical_training_probe.py
```

Он сохраняет exact evidence по каждому update и parameter name для:

```text
parameter names and order
initial parameters
encoded task
forward logits
loss components
raw gradients
raw gradient norm
gradients after clipping
AdamW exp_avg
AdamW exp_avg_sq
parameters after optimizer.step
```

Tolerance, rounding и quantization не используются.

Удалённые temporary workflows и monkeypatch больше не являются единственным
способом повторить investigation contracts.

## Versioned execution contract

Каждый probe artifact содержит:

```text
execution_contract
execution_contract_sha256
```

Contract version:

```text
toy-quality-cpu-execution-contract/1.0
```

Поля:

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
```

Validator требует exact field set, корректную contract version, canonical hash,
поддерживаемый named profile, согласованность profile fields и ожидаемые AdamW
hyperparameters. Missing или extra fields не могут быть легализованы простым
пересчётом dependent hashes.

`ATEN_CPU_CAPABILITY` и `MKL_CBWR` принимаются только через environment до
импорта Torch. Probe fail-closed проверяет ожидаемые значения environment и
после импорта Torch проверяет фактический dispatch через:

```python
torch.backends.cpu.get_cpu_capability()
```

Probe не устанавливает `ATEN_CPU_CAPABILITY` после импорта Torch.

## Named profiles

### historical-default

Воспроизводит historical quality path:

```text
ATEN_CPU_CAPABILITY: absent
MKL_CBWR: absent
foreach: None
fused: None
profile_kind: historical
```

### avx2-single-tensor

Controlled investigation alternative:

```text
ATEN_CPU_CAPABILITY=avx2
actual_atten_cpu_capability=AVX2
MKL_CBWR=COMPATIBLE
foreach=False
fused=False
profile_kind=controlled-investigation
```

### default-single-tensor

Controlled investigation alternative:

```text
ATEN_CPU_CAPABILITY=default
actual_atten_cpu_capability=DEFAULT
MKL_CBWR=COMPATIBLE
foreach=False
fused=False
profile_kind=controlled-investigation
```

CLI поддерживает явное подтверждение boolean controls:

```text
--optimizer-foreach true|false
--optimizer-fused true|false
```

Named profile нельзя переопределить противоречащим CLI value. Controlled
single-tensor profile с `true` и historical-default с explicit boolean
отклоняются fail-closed. Поэтому имя profile всегда соответствует фактическому
optimizer contract.

Controlled profiles являются investigation alternatives. Они не выдаются за
historical quality path и не объявляются принятым runtime contract.

## Contract-aware identity и comparison

Полный execution contract и его hash входят в `probe_identity`.

Перед comparison выполняются:

1. строгая validation обоих execution contracts;
2. проверка canonical contract hashes;
3. полное сравнение contracts;
4. проверка пересчитанного `probe_identity`;
5. exact numerical comparison только для compatible artifacts.

Несовместимые contracts возвращают:

```json
{
  "comparable": false,
  "reason": "EXECUTION_CONTRACT_MISMATCH",
  "equal": null
}
```

Numerical equality для них не вычисляется.

Hardware fingerprint может различаться между hosts. Он не делает одинаковый
software execution contract несовместимым.

## Parity с frozen quality training

`quality._train` не изменён.

Parity harness исполняет один реальный update внутри `quality._train` с
transparent instrumentation, а не локальную копию quality update. Временно и
с обязательным восстановлением перехватываются фактические:

```text
LockedPlanner construction and forward
_optimizer_named_parameters
AdamW construction, zero_grad and step
cross_entropy calls
Tensor.backward
clip_grad_norm_
```

Для одного seed/task на одном process actual quality update exact-сравнивается
с probe update под `historical-default` по:

```text
parameter names and order
optimizer defaults
initial parameters
encoded task
forward logits
loss components and total loss
raw gradients
gradient norm
clip threshold and clipped parameter order
gradients after clipping
AdamW exp_avg
AdamW exp_avg_sq
parameters after optimizer.step
update event order
```

Такой parity test ломается при изменении parameter order, optimizer defaults,
loss construction, clipping или update semantics.

## Hardware/runtime fingerprint integrity

Retained function:

```text
validate_hardware_runtime_fingerprint
```

проверяет только observation artifact integrity:

```text
exact outer field set
outer fingerprint version
exact observation field set
nested observation version
canonical lowercase sha256:<64 hex> format
recomputed observation_identity_sha256
rejection of stale hash after observation mutation
```

Passing validation не объявляѵт host принятым canonical target.

Premature fixed-target API отсутствует:

```text
FIXED_TARGET_POLICY_VERSION
validate_fixed_execution_target
validate_supported_cpu_software_path
```

До provisioning реального target частичный acceptance validator не публикуется.

## Investigation evidence

Temporary workflows использовались только для investigation runs ниже. Они не
входят в retained diff.

### Controlled AVX2 experiment

```text
workflow run: 30872595002
commit: df126eab5e2da8b47a3d006406f7dbe2d3ed7268
profile: avx2-single-tensor
```

Первое наблюдаемое cross-host расхождение:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.bos_embedding
stage: parameters_after_optimizer_step
```

До `optimizer.step()` совпали encoded task, forward logits, loss components,
raw gradients, gradient norm и clipped gradients.

### Provisional DEFAULT sentinel experiment

```text
workflow run: 30907124345
commit: 2978356a7384164b4388d81cc8e75a5c0e7d03a4
```

Provisional sentinel failed closed и не был принят как runtime contract.

### Controlled DEFAULT observation

```text
workflow run: 30907486933
commit: f3f27ff8428088c29c126fac6cf59bdbae35b2bc
profile: default-single-tensor
```

Первое наблюдаемое cross-host расхождение:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.output_norm.bias
stage: parameters_after_optimizer_step
```

Forward, loss, raw gradients, clipping и clipped gradients совпали.

## Поддерживаемый вывод

```text
First observed cross-host divergence occurs during AdamW parameter update.
The underlying instruction-level cause remains unidentified.
```

Investigation локализует первую divergence boundary. Она не идентифицирует
конкретную vectorized instruction, assembly path или instruction-level root
cause.

## Decision

Общий byte-exact profile для heterogeneous standard GitHub-hosted x86 runners
не найден ни для controlled AVX2, ни для controlled DEFAULT experiment.

Будущая canonical acceptance требует одного provisioned и versioned fixed
execution target. Конкретный target ещё не выбран и не provisioned.

Будущий policy должен fail-closed закреплять как минимум:

```text
policy version and complete required field set
CPU vendor/family/model/stepping/model name
canonical CPU flags hash
OS, kernel and image
Python build
PyTorch build configuration
ATen dispatch
BLAS, MKL, OpenMP and oneDNN configuration
canonical environment variables
accepted update-level execution sentinel
observation identity recomputation
rejection of missing and extra target fields
```

Обычный hosted runner label или container image сами по себе не фиксируют
physical CPU identity.

## Версионирование и migration

Существующий runtime:

```text
toy-quality-canonical-cpu-runtime/1.0
```

не переинтерпретируется.

Новый runtime contract и новая versioned evaluation возможны только после
provisioning fixed target и отдельной acceptance-фазы.

Historical v0.1 не сохранял достаточный hardware fingerprint, поэтому его
нельзя доказуемо привязать к будущему target.

## Frozen scope

Не изменяются и не перегенерируются:

```text
planner_toy/canonical_runtime.py
.github/workflows/ci.yml
docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md
docs/evaluations/data/a2_a3_a4_heldout_summary.json
docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md
```

## CI и acceptance

Investigation validation и canonical runtime acceptance остаются разными
статусами:

```text
investigation tests: must pass on the current head
canonical runtime acceptance: failed / unavailable
accepted fixed-target runs: 0/3
```

Canonical comparison не ослабляется. Tolerance не добавляется. Cross-host
equality не требуется от retained investigation profiles до provisioning fixed
target.

PR остаётся draft до повторного внешнего review.
