# Canonical CPU execution identity: investigation and migration policy

## Статус документа

```text
investigation_version: canonical-cpu-execution-identity/1.0
runtime_contract: toy-quality-canonical-cpu-runtime/1.1
historical_runtime_contract: toy-quality-canonical-cpu-runtime/1.0
```

Документ фиксирует отдельное расследование cross-host numerical nondeterminism,
обнаруженного после Development Quality Evaluation v0.1. Он не меняет модель,
обучение, метрики или исторические результаты.

## Исходное наблюдение

На implementation head
`33d0ff62c597170d46d1bdb40a05687326aa9369` CI run `30860485483`
успешно завершил оба независимых canonical workers, но exact semantic comparison
не прошёл. Worker A и worker B получили разные replay identities. Первое
зафиксированное итоговое различие находилось в A2, seed 17, в AdamW
`exp_avg[8567]`, после чего расходились trained checkpoints и replay artifacts.

Одинаковыми были Python 3.11.15, PyTorch 2.12.0+cpu, single-thread settings,
deterministic algorithms и отключённый MKLDNN. Отличались физические hosted
workers и Azure regions. Следовательно, runtime/1.0 фиксировал software и
threading, но не фиксировал CPU ISA dispatch и BLAS conditional numerical path.

Исходный cross-worker diagnostic artifact был скачан и разобран. Полные
`canonical-a` и `canonical-b` artifacts имеют размер более 1 GiB каждый и не
прошли через используемый artifact transport с жёстким лимитом 512 MiB. Для
update-level локализации в этом PR добавлена воспроизводимая двухworkerная проба,
а не вывод по одному итоговому tensor index.

## Update-level deterministic probe

`scripts/run_canonical_training_probe.py` зеркалирует A2 training policy, не
изменяя production training loop. Для каждого update и parameter name она
сохраняет exact SHA-256 следующих значений:

- initial parameters;
- canonical encoded task;
- action, arg1, arg2 и semantic forward logits;
- action, arg1, arg2 и total loss components;
- raw gradients до clipping;
- gradient norm;
- gradients после clipping;
- AdamW `exp_avg`;
- AdamW `exp_avg_sq`;
- parameters после `optimizer.step()`.

Проба также сохраняет CPU operator trace из public PyTorch profiler. В CI
`MKL_VERBOSE=1` пишет фактически выбранные oneMKL kernels. Comparator exact:
он не использует tolerance, rounding, post-hoc quantization или исключение
optimizer/checkpoint state.

Проба выполняется в двух режимах:

- `legacy`: семантика runtime/1.0 без новых dispatch variables;
- `canonical`: runtime/1.1 с новым execution profile.

Так определяется первый divergent update, стадия и parameter, а также отдельно
проверяется влияние миграции runtime/1.0 → runtime/1.1.

## Hardware и runtime fingerprint

Runtime/1.1 сохраняет две различные сущности.

### Semantic execution identity

Она одинакова для совместимых hosts и включает только контракт исполнения:

- runtime version;
- PyTorch version и build-configuration hash;
- ATen CPU dispatch capability;
- execution sentinel hash;
- deterministic/thread/MKLDNN settings;
- canonical environment.

Hostname, process ID и timestamps запрещены.

### Observed hardware evidence

Она не используется для требования равенства hosts, но сохраняется для аудита:

- OS, kernel, architecture и `/etc/os-release`;
- CPU vendor, family, model, stepping, model name и microcode;
- canonical CPU flags hash и AVX/AVX2/AVX-512 capabilities;
- logical CPU count;
- runner OS/architecture/environment, image и version;
- Azure region, если metadata endpoint доступен;
- Python implementation/build/compiler;
- PyTorch version и полный build configuration;
- MKL/OpenMP/MKLDNN availability;
- фактический public ATen CPU capability;
- все canonical environment variables.

Fingerprint сериализуется canonical JSON с отсортированными ключами.

## Runtime/1.1 execution profile

Runtime/1.1 добавляет к runtime/1.0:

```text
ATEN_CPU_CAPABILITY=default
MKL_CBWR=COMPATIBLE
```

`ATEN_CPU_CAPABILITY=default` принудительно выбирает generic ATen CPU dispatch
вместо автоматического AVX2/AVX-512 выбора. `MKL_CBWR=COMPATIBLE` принудительно
выбирает conditional numerical reproducibility path oneMKL. Оба значения должны
быть установлены до первого tensor/BLAS operation.

Runtime fail-closed требует:

- PyTorch `2.12.x+cpu`;
- CPU dispatch capability `DEFAULT` через public
  `torch.backends.cpu.get_cpu_capability()`;
- MKL и OpenMP availability;
- deterministic algorithms без warn-only;
- intra-op и inter-op thread count 1;
- MKLDNN disabled;
- требуемые build markers `BLAS_INFO=mkl`, `USE_MKL=ON`, `USE_OPENMP=ON`;
- exact execution sentinel identity.

Sentinel отдельно упражняет matrix multiplication, layer normalization, softmax,
backward, gradient clipping и AdamW moments. Он не заменяет cross-host probe, а
не позволяет молча принять unsupported или проигнорированный runtime path.

## Совместимость с runtime/1.0

`toy-quality-canonical-cpu-runtime/1.0` остаётся историческим контрактом и не
переинтерпретируется как runtime/1.1. Runtime validator/1.1 отклоняет fingerprint
версии 1.0.

Frozen quality-v0.1 regeneration выполняется из исторического implementation
commit в отдельном worktree с удалёнными `ATEN_CPU_CAPABILITY` и `MKL_CBWR`.
Полученные historical JSON и Markdown сравниваются byte-for-byte с committed
frozen artifacts.

## Migration policy

Новый profile может менять numerical bytes относительно runtime/1.0, потому что
фиксирует другой ISA/BLAS execution path. CI сравнивает legacy и canonical probes
на каждом worker.

Если probe показывает изменение numerical bytes:

1. frozen quality-v0.1 artifacts не переписываются;
2. runtime/1.1 не используется для заднего переопределения v0.1;
3. новая evaluation должна получить отдельную versioned identity;
4. decision status фиксируется как `migration required`;
5. scientific conclusions v0.1 остаются привязаны к historical runtime/1.0.

Этот PR определяет execution contract и evidence. Он не публикует новую
quality evaluation и не меняет frozen decision.

## Acceptance evidence

Final implementation head должен иметь три независимых полных CI runs. В каждом
обязательны:

```text
checks (3.11)
checks (3.13)
canonical-run-a
canonical-run-b
canonical-compare
```

Для каждого run сохраняются worker fingerprints, operator/MKL evidence,
legacy/canonical probe comparisons, numerical probe identities, replay hashes и
exact comparison report. Rerun отдельного failed job не считается независимым
run. При любом расхождении PR остаётся `INCOMPLETE`.
