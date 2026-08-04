# Cross-host CPU nondeterminism: investigation and decision record

## Статус

```text
INVESTIGATION COMPLETED
COMMON HOSTED PROFILE NOT FOUND
FIXED TARGET REQUIRED
MIGRATION REQUIRED
```

Документ фиксирует результат расследования, но не объявляет новый canonical
runtime, не выбирает конкретный fixed target и не утверждает, что такой target
уже provisioned.

## Исходный сбой

Implementation head:

```text
33d0ff62c597170d46d1bdb40a05687326aa9369
```

CI run:

```text
30860485483
```

Результат:

```text
checks (3.11): success
checks (3.13): success
canonical-run-a: success
canonical-run-b: success
canonical-compare: failure
```

Первое persisted расхождение было найдено в:

```text
training-runs/A2/seed-17/optimizer-state.pt
state[0].exp_avg[8567]
```

После него разошлись trained checkpoints и replay artifacts.

## Retained update-level probe

Самодостаточный retained probe:

```text
scripts/run_canonical_training_probe.py
```

Он сохраняет exact hashes по каждому update и parameter name для:

```text
parameter names and order
initial parameters
encoded task
forward logits
loss components
raw gradients before clipping
gradient norm
gradients after clipping
AdamW exp_avg
AdamW exp_avg_sq
parameters after optimizer.step
```

Сравнение exact: tolerance, rounding и quantization не применяются.

### Versioned execution contract

Каждый artifact содержит:

```text
execution_contract
execution_contract_sha256
```

Execution contract входит в `probe_identity` и включает:

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

`ATEN_CPU_CAPABILITY` и `MKL_CBWR` принимаются только через environment до
импорта Torch. Probe fail-closed проверяет ожидаемое environment value, а после
импорта Torch проверяет фактический dispatch через:

```python
torch.backends.cpu.get_cpu_capability()
```

Controlled profiles создают `AdamW` с явно заданными `foreach` и `fused`.
Значение `None` в controlled investigation mode запрещено.

### Named profiles

#### historical-default

Воспроизводит historical quality training defaults:

```text
ATEN_CPU_CAPABILITY: absent
MKL_CBWR: absent
foreach: None
fused: None
profile_kind: historical
```

Пример:

```bash
env -u ATEN_CPU_CAPABILITY -u MKL_CBWR \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  python -m scripts.run_canonical_training_probe run \
    --profile historical-default \
    --output historical-default.json
```

#### avx2-single-tensor

Controlled investigation alternative:

```bash
ATEN_CPU_CAPABILITY=avx2 \
MKL_CBWR=COMPATIBLE \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python -m scripts.run_canonical_training_probe run \
  --profile avx2-single-tensor \
  --optimizer-foreach false \
  --optimizer-fused false \
  --output avx2-single-tensor.json
```

#### default-single-tensor

Controlled investigation alternative:

```bash
ATEN_CPU_CAPABILITY=default \
MKL_CBWR=COMPATIBLE \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python -m scripts.run_canonical_training_probe run \
  --profile default-single-tensor \
  --optimizer-foreach false \
  --optimizer-fused false \
  --output default-single-tensor.json
```

Controlled profiles не являются historical quality path и не объявляются
кандидатами, принятыми для runtime/1.1.

### Contract-aware comparison

`compare_probes()` сначала проверяет полное совпадение execution contract.
Artifacts с разными contracts не сравниваются численно:

```json
{
  "comparable": false,
  "reason": "EXECUTION_CONTRACT_MISMATCH",
  "equal": null
}
```

Hardware fingerprint может отличаться между hosts и не входит в критерий
contract compatibility. Он сохраняется как observation evidence.

## Parity с frozen quality training

Probe содержит isolated one-update parity harness для A2. На одном process и
одинаковых seed/task он exact-сравнивает historical quality update и probe под
`historical-default` по:

```text
parameter names and order
initial parameters
encoded task
forward logits
loss components
raw gradients
gradient norm
gradients after clipping
AdamW exp_avg
AdamW exp_avg_sq
parameters after optimizer.step
```

Harness использует historical optimizer defaults и отдельную реализацию
quality loss/update sequence. Дополнительный source-contract guard ломает test
при изменении parameter selection, optimizer construction, loss construction,
clipping или порядка `zero_grad → forward → loss → backward → clip → step` в
`quality._train`.

`quality._train` не изменяется.

## Hardware/runtime fingerprint integrity

Retained fingerprint содержит:

- OS, kernel, architecture и `/etc/os-release`;
- CPU vendor, family, model, stepping, model name и microcode;
- canonical hash полного набора CPU flags;
- AVX, AVX2, AVX-512 и FMA capabilities;
- logical CPU count;
- runner OS, architecture, environment, image и image version;
- Azure region, когда metadata доступна;
- Python version, implementation, compiler и build;
- PyTorch version и полный build configuration;
- MKL, OpenMP и oneDNN availability/configuration;
- фактический ATen CPU dispatch capability;
- execution environment values.

Hostname, process ID и timestamps не входят в fingerprint identity.

Функция:

```text
validate_hardware_runtime_fingerprint
```

проверяет:

- exact top-level field set;
- fingerprint version;
- canonical `sha256:<64 lowercase hex>` format;
- пересчёт `observation_identity_sha256`;
- rejection observation mutation с прежним hash.

Она валидирует целостность observation artifact, но не объявляет host принятым
canonical target.

В PR нет partial fixed-target validator. Требования к будущему policy остаются
только decision record до provisioning реального target.

## Investigation evidence и временные workflows

Временные workflows использовались только для сбора cross-host evidence. Они
удалены из итогового diff. Retained probe теперь воспроизводит их software
contracts напрямую через named profiles и не зависит от workflow monkeypatch.

### AVX2 controlled investigation

```text
workflow run: 30872595002
commit: df126eab5e2da8b47a3d006406f7dbe2d3ed7268
profile: avx2-single-tensor
ATEN_CPU_CAPABILITY=avx2
actual_atten_cpu_capability=AVX2
MKL_CBWR=COMPATIBLE
foreach=false
fused=false
```

AMD и Intel workers использовали одинаковые pinned Python/PyTorch/thread
settings и отключённый MKLDNN.

Первое наблюдаемое расхождение:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.bos_embedding
stage: parameters_after_optimizer_step
```

До `optimizer.step()` совпали encoded task, forward logits, loss components,
raw gradients, gradient norm и clipped gradients.

### DEFAULT fail-closed experiment

```text
workflow run: 30907124345
commit: 2978356a7384164b4388d81cc8e75a5c0e7d03a4
profile: provisional default contract
```

Заранее записанный execution sentinel не совпал на всех host types. Experiment
остановился fail-closed и не был принят как runtime contract.

### DEFAULT controlled observation

```text
workflow run: 30907486933
commit: f3f27ff8428088c29c126fac6cf59bdbae35b2bc
profile: default-single-tensor
ATEN_CPU_CAPABILITY=default
actual_atten_cpu_capability=DEFAULT
MKL_CBWR=COMPATIBLE
foreach=false
fused=false
```

16 independent AMD и Intel hosts дали две exact probe identities.

Первое наблюдаемое расхождение:

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

Evidence локализует первую divergence boundary, но не устанавливает конкретную
vectorized instruction, assembly path или instruction-level root cause.

Не обнаружено расхождение до update в:

- dataset/order;
- initialization;
- encoded task;
- forward;
- loss;
- backward gradients;
- gradient norm;
- gradient clipping.

## Decision

Общий byte-exact profile для heterogeneous standard GitHub-hosted x86 runners
не найден ни для controlled AVX2, ни для controlled DEFAULT experiment.

Следовательно, для будущей canonical acceptance требуется фиксированный
execution target. Конкретный target ещё не выбран и не provisioned.

Будущий versioned policy должен fail-closed закреплять как минимум:

- CPU vendor/family/model/stepping/model name;
- canonical CPU flags hash;
- OS/kernel/image;
- Python build;
- PyTorch build configuration;
- ATen dispatch;
- BLAS/MKL/OpenMP/oneDNN configuration;
- canonical environment variables;
- accepted update-level execution sentinel.

Обычный `ubuntu-24.04` label или container image сами по себе не фиксируют
physical CPU identity.

## Версионирование и migration

Существующий:

```text
toy-quality-canonical-cpu-runtime/1.0
```

не переинтерпретируется.

`runtime/1.1` не объявлен готовым или принятым. Новый runtime contract и новая
versioned evaluation возможны только после provisioning fixed target и
отдельной acceptance-фазы.

Historical v0.1 не сохранял достаточный hardware fingerprint, поэтому его
нельзя доказуемо отнести к будущему target.

## Frozen scope

Не изменяются и не перегенерируются:

```text
planner_toy/canonical_runtime.py
.github/workflows/ci.yml
docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md
docs/evaluations/data/a2_a3_a4_heldout_summary.json
docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md
```

## CI и acceptance status

Investigation validation и canonical runtime acceptance — разные статусы:

```text
investigation tests: passed or pending on current head
canonical runtime acceptance: failed / unavailable
```

Hosted `canonical-compare` не ослабляется, tolerance не добавляется. Он может
оставаться красным как воспроизведённый known blocker.

Требование будущей acceptance-фазы:

```text
3/3 independent full runs green on one provisioned fixed target and one head
```

Текущий статус:

```text
0/3
```

Cross-host equality не требуется от retained investigation profiles до
provisioning fixed target.
