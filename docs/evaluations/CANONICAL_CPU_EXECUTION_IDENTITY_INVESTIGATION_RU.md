# Investigation: canonical CPU execution identity across hosts

## Статус

```text
Investigation:
COMPLETED

Common GitHub-hosted byte-exact profile:
NOT FOUND

Selected outcome:
VARIANT B — FIXED EXECUTION TARGET REQUIRED

Fixed target provisioning:
NOT AVAILABLE IN THIS REPOSITORY

PR status:
INCOMPLETE / MIGRATION REQUIRED
```

Этот документ не объявляет новый canonical runtime принятым. Он фиксирует
результат расследования cross-host numerical nondeterminism и требования к
следующему versioned execution target.

## Исходный сбой

На implementation head:

```text
33d0ff62c597170d46d1bdb40a05687326aa9369
```

CI run:

```text
30860485483
```

получил:

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

После этого разошлись trained checkpoints и replay artifacts.

## Update-level probe

Добавлен read-only probe:

```text
scripts/run_canonical_training_probe.py
```

Он не изменяет основной training loop и сохраняет exact hashes по каждому
update и parameter name для:

```text
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

Сравнение probe artifacts остаётся exact. В нём нет tolerance, rounding,
quantization или исключения optimizer/checkpoint evidence.

## Hardware/runtime fingerprint

Probe сохраняет machine-readable fingerprint:

- OS, kernel, architecture и `/etc/os-release`;
- CPU vendor, family, model, stepping, model name и microcode;
- canonical hash полного набора CPU flags;
- AVX, AVX2, AVX-512 и FMA capabilities;
- logical CPU count;
- runner OS, architecture, environment, image и image version;
- Azure region, когда metadata доступна;
- Python version, implementation, compiler и build;
- PyTorch version и полный build configuration;
- BLAS, MKL, OpenMP и oneDNN availability/configuration;
- фактический ATen CPU dispatch capability;
- canonical environment variables.

Hostname, process ID и timestamps не входят в fingerprint или его canonical
identity.

## Официально поддерживаемый dispatch control

PyTorch 2.12.0 читает:

```text
ATEN_CPU_CAPABILITY
```

в `aten/src/ATen/native/DispatchStub.cpp`. Для x86 source поддерживает значения:

```text
default
avx2
avx512
```

Invalid values только предупреждаются и игнорируются, поэтому runtime обязан
дополнительно проверять фактический:

```python
torch.backends.cpu.get_cpu_capability()
```

`MKL_CBWR=COMPATIBLE` применялся только вместе с fail-closed probe и
persisted fingerprint.

## AVX2 investigation

Run:

```text
30872595002
```

Профиль:

```text
ATEN_CPU_CAPABILITY=avx2
MKL_CBWR=COMPATIBLE
foreach=false
fused=false
```

Были получены AMD и Intel workers под одинаковыми:

```text
Ubuntu 24.04
Python 3.11.15
PyTorch 2.12.0+cpu
one Torch thread
one interop thread
MKLDNN disabled
```

Результат:

```text
all_numerical_probes_equal=false
```

Первое расхождение:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.bos_embedding
stage: parameters_after_optimizer_step
```

До `optimizer.step()` совпали:

- encoded task;
- forward logits;
- loss components;
- raw gradients;
- gradient norm;
- gradients after clipping.

Следовательно, AVX2 не создаёт общий byte-exact path на heterogeneous
GitHub-hosted x86 CPUs.

## DEFAULT investigation

Fail-closed contract run:

```text
30907124345
```

показал, что заранее записанный execution sentinel не совпадает на всех
host types. Runtime остановился до acceptance, как и требовалось.

Observation run:

```text
30907486933
```

измерил 16 independent hosts без принятия динамического sentinel как
canonical contract.

Профиль:

```text
ATEN_CPU_CAPABILITY=default
MKL_CBWR=COMPATIBLE
foreach=false
fused=false
```

В выборке присутствовали:

- AMD EPYC 7763;
- AMD EPYC 9V74;
- Intel Xeon Platinum 8573C;
- Intel Xeon 6973P-C.

Использовались одинаковые:

- Ubuntu 24.04.4;
- kernel `6.17.0-1020-azure`;
- runner image `20260720.247.2`;
- Python 3.11.15 / GCC 13.3;
- PyTorch 2.12.0+cpu;
- oneAPI MKL 2024.2;
- oneDNN 3.11.2;
- OpenMP 4.5;
- один intra-op и inter-op thread;
- отключённый MKLDNN.

Получены две exact probe identities:

```text
sha256:3f5ef16415f532dc570bda69097e5637aab38ba8fb6c049213a25ce962c7970c
sha256:c639261bf9f4fcb25f1915e9d4d66e2e3a6efc0dbca393336e8f92ce29a85d45
```

Первое расхождение:

```text
epoch: 1
update: 1
task: bw-00000001
parameter: concept_packer.output_norm.bias
stage: parameters_after_optimizer_step
```

Forward, loss, raw gradients, clipping и clipped gradients снова совпали.
Расхождение впервые появилось в AdamW parameter update.

MKL verbose evidence показывает `SGEMM` и `SGEMM_BATCH` с
`CNR:COMPATIBLE`, однако различие возникает после уже совпавших gradients.
Доступные artifacts не называют конкретную внутреннюю vectorized instruction
AdamW pointwise kernel, поэтому документ не приписывает расхождение
конкретной assembly-инструкции.

## Root cause

Поддерживаемый вывод:

> Heterogeneous GitHub-hosted x86 CPUs выполняют AdamW pointwise parameter
> update с различными exact float32 bytes даже при одинаковых Python,
> PyTorch, thread settings, disabled MKLDNN, fixed ATen dispatch class,
> compatible MKL mode и explicitly disabled foreach/fused optimizer paths.

Это не расхождение:

- dataset;
- initialization;
- forward;
- loss;
- backward gradients;
- gradient norm;
- gradient clipping.

Оно впервые наблюдается в `optimizer.step()` на update 1.

## Решение

### Вариант A

```text
Common byte-exact GitHub-hosted CPU path:
REJECTED
```

Проверены как минимум:

- `AVX2`;
- `DEFAULT`;
- `MKL_CBWR=COMPATIBLE`;
- single-thread execution;
- `foreach=false`;
- `fused=false`.

Оба dispatch path дали cross-host drift.

### Вариант B

```text
Fixed execution target:
REQUIRED
```

Обычный `ubuntu-24.04` и container image недостаточны: они не фиксируют
physical CPU vendor/model/stepping/flags.

Следующий canonical target должен быть provisioned как self-hosted runner или
фиксированная VM policy и fail-closed закреплять:

- CPU vendor/family/model/stepping/model name;
- canonical CPU flags hash;
- OS/kernel/image;
- Python build;
- PyTorch build configuration;
- ATen dispatch;
- BLAS/MKL/OpenMP/oneDNN configuration;
- all canonical environment variables;
- accepted update-level execution sentinel.

В текущем репозитории такого target нет. Поэтому runtime/1.1 не принимается и
production CI не переводится на неподтверждённый profile.

## Версионирование и migration

Существующий:

```text
toy-quality-canonical-cpu-runtime/1.0
```

не переинтерпретируется.

Новый контракт может получить версию:

```text
toy-quality-canonical-cpu-runtime/1.1
```

только после provisioning fixed target и трёх независимых полностью зелёных
CI runs на одном implementation head.

Новый stable target может выдавать bytes, отличающиеся от historical v0.1.
Historical run не сохранял достаточный hardware fingerprint, поэтому его
нельзя доказуемо отнести к новому target.

Итог:

```text
migration required
new versioned evaluation required
```

## Frozen quality-v0.1

Не изменяются:

```text
docs/evaluations/A2_A3_A4_HELDOUT_DIAGNOSTIC_RU.md
docs/evaluations/data/a2_a3_a4_heldout_summary.json
docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md
```

Historical artifacts не перегенерируются под новый target.

## CI evidence status

Требование:

```text
3/3 independent full runs green
```

Текущий статус:

```text
0/3
```

Полные acceptance runs не запускаются, потому что accepted fixed target ещё
не provisioned. Повторные hosted runs не считаются решением обнаруженного
hardware mismatch.
