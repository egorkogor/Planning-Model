# A2 gradient clipping causal discrimination — development spec

Версия: `development-a2-gradient-clipping/0.1`  
Статус: `development-only-scientific-microexperiment`  
Typed reviewer-bridge task: `a2-gradient-clipping-v1`

## Вопрос

Эксперимент изолированно проверяет, является ли canonical gradient clipping `max_norm=1.0` дополнительным причинным фактором задержки rescue A2 при уже достаточном optimization budget. Producer не формирует научный verdict; интерпретация принадлежит reviewer.

`GO_LATENT = NOT EVALUATED`.

## Causal contract

Единственная разрешённая интервенция между arms — gradient clipping policy:

- `clip_1_0`: `torch.nn.utils.clip_grad_norm_(..., 1.0)`;
- `clip_5_0`: `torch.nn.utils.clip_grad_norm_(..., 5.0)`;
- `no_clip`: clipping primitive не вызывается.

Во всех arms неизменны A2 architecture, initialization/seed semantics, train-only dataset, canonical task order `bw-00000001 → bw-00000002 → bw-00000003`, labels/objective, AdamW contract, learning rate, betas, epsilon, weight decay, optimizer parameter order, 100 epochs, 300 updates, evaluation/rescue semantics и canonical runtime. Early stopping запрещён.

Held-out `bw-00000004` и `bw-00000005` не входят в evaluated dataset и не используются producer/validator.

## Control equivalence

Для каждого seed 17/29/43 новый `clip_1_0` исполняется отдельно от существующего accepted sufficient-budget `canonical_order` reference path. До публикации evidence producer требует exact equality whole-trajectory projection на 300 updates.

Projection связывает initialization, все historical trace fields (loss decomposition, target counts, task/update order, legacy pre-clip norm, canonical clip norm/flag, operator position weight), checkpoint model/optimizer identities, final model/optimizer identities, rescue events и persistence. Полная reference projection на 300 updates сохраняется в JSON evidence; independent validator отдельно строит candidate projection из raw `clip_1_0` evidence и требует exact equality. Reference path дополнительно обязан пройти существующую frozen 3-epoch / 9-update historical prefix equivalence. Любое расхождение — fail closed.

## Gradient intervention evidence

На каждом update сохраняются:

- update/epoch/task identity;
- operator/arg1/arg2/total losses и target counts;
- `pre_intervention_global_l2_norm`;
- policy и threshold;
- legacy/reference `gradient_norm` и `clipping_occurred`;
- `clip_primitive_return_norm`;
- `threshold_exceeded`;
- `gradient_mutated`;
- `intervention_applied`;
- post-intervention global L2 norm;
- exact deterministic SHA-256 named gradients before/after intervention;
- canonical gradient-parameter manifest (ordered index/name/dtype/shape) и его SHA-256;
- per-update `gradient_parameter_manifest_sha256`;
- explicit ordered `gradient_activity` и его SHA-256;
- gradient hash encoding version;
- operator position weight.

### Named-gradient wire encoding

Current generated evidence использует только `a2-named-gradients-exact/1.1`. Идентификатор `a2-named-gradients-exact/1.0` является историческим идентификатором прежнего byte encoding и не является alias формата 1.1.

SHA-256 input для `a2-named-gradients-exact/1.1` строится строго в canonical optimizer parameter order:

1. domain separator: ASCII bytes строки `a2-named-gradients-exact/1.1`, затем один byte `0x00`;
2. для каждого parameter:
   - parameter name: unsigned 64-bit big-endian длина UTF-8 bytes, затем сами UTF-8 bytes;
   - activity marker: unsigned 64-bit big-endian длина ASCII marker, затем ровно `GRAD` или `NO_GRAD`;
3. если marker = `GRAD`:
   - gradient берётся read-only как `detach().cpu().contiguous()`;
   - dtype: unsigned 64-bit big-endian длина ASCII `str(dtype)`, затем dtype bytes;
   - ndim: unsigned 64-bit big-endian integer;
   - каждая shape dimension: unsigned 64-bit big-endian integer;
   - tensor payload: unsigned 64-bit big-endian длина bytes, затем exact `tensor.numpy().tobytes()`;
4. если marker = `NO_GRAD`:
   - после marker не кодируются dtype, ndim, shape, tensor-byte length или tensor bytes;
   - `NO_GRAD` означает `parameter.grad is None` и не эквивалентен существующему numeric zero gradient tensor.

Hashing не материализует gradients и не изменяет autograd state.

### Gradient activity

`gradient_activity` — ordered list вида:

```text
[{"index": 0, "name": "...", "state": "GRAD"}, ...]
```

Она обязана быть exact-aligned с `gradient_parameter_manifest`: один entry на каждый optimizer parameter, те же index/name/order, без duplicate/missing/extra entries. Allowlist state: только `GRAD` и `NO_GRAD`. Activity фиксируется до intervention; clipping не имеет права менять presence/absence gradient tensor. `gradient_activity_sha256` — deterministic commitment к exact activity list, но сам по себе не считается semantic proof.

Independent validator выводит ожидаемую activity не из producer-owned activity/hash, а из frozen A2 objective graph и independently reconstructed target counts. `heads.arg1_pointer.weight` обязан иметь `GRAD` iff `arg1_target_count > 0`; `heads.arg2_pointer.weight` обязан иметь `GRAD` iff `arg2_target_count > 0`; остальные canonical optimizer parameters лежат на always-present action-loss path и обязаны иметь `GRAD`. Поэтому self-consistent подмена `NO_GRAD → GRAD` с zero tensor и пересчётом producer-owned hashes/commitments всё равно fail closed.

### Norm и intervention semantics

`pre_intervention_global_l2_norm` — общий read-only global L2 measurement по одному и тому же ordered набору только фактически active (`GRAD`) gradients до любой intervention. Он является cross-arm measurement source.

Legacy/reference `gradient_norm` для clipping arms — return canonical `torch.nn.utils.clip_grad_norm_`; поле сохраняется исключительно для accepted reference equivalence. Для `no_clip` оно равно common pre-intervention norm. Оно не используется как authoritative cross-arm measurement source.

`threshold_exceeded` — только predicate `pre_intervention_global_l2_norm > configured threshold`. Это не доказательство фактической mutation.

`gradient_mutated` определяется только exact before/after named-gradient commitment difference. Before/after commitments являются authoritative evidence фактической gradient mutation.

`intervention_applied` для clipping arms истинно только при фактической gradient mutation. Для `no_clip` оно всегда false.

Legacy `clipping_occurred` сохраняет accepted reference semantics (`gradient_norm > threshold`) и не заменяет current actual-intervention evidence.

### Gradient evidence commitment

Current commitment version — `a2-gradient-evidence-commitment/1.1`. Для каждого arm/seed commitment покрывает все 300 raw update records, включая gradient hash version, manifest SHA, explicit activity/activity SHA, before/after hashes, norms и intervention fields. Commitment также несёт gradient hash version и result-level manifest SHA. Любая activity tampering меняет commitment, но validator дополнительно обязан независимо проверить activity semantics, а не только self-consistency hashes.

Semantic consistency:

- `no_clip`: before == after и actual intervention=false;
- inactive clip: before == after;
- active clip: before != after и post norm ограничен threshold;
- clipping не меняет `GRAD`/`NO_GRAD` activity;
- если `clip_5_0` не клипает ни одного update, его non-policy trajectory обязана быть exact-equivalent `no_clip`.

## Outcomes

Raw epoch evidence сохраняется после каждого completed epoch. Primary outcomes:

1. first position-0 operator rescue на nontrivial train tasks 02/03;
2. first full free-running rescue на train tasks 02/03 по existing final-goal-success semantics.

Если event не наступил к epoch 100, raw event остаётся `null`. Persistence фиксируется для epochs 10/30/100.

Clipping summaries считаются для первых 9 updates, до observation первого position-0 rescue, до observation первого full free-running rescue и полной 300-update trajectory. Censored windows маркируются явно. Дополнительно сохраняются cross-seed descriptive aggregates по каждому arm без inferential statistics.

Paired descriptive contrasts считаются по seed для `clip_5_0` и `no_clip` относительно `clip_1_0`. P-values и автоматический scientific verdict отсутствуют.

## Evidence / validation boundary

Claim-bearing источник — `a2-gradient-clipping.json`. Markdown — только deterministic derivative.

Independent validator отдельно пересчитывает из persisted lower-level evidence coverage arms/seeds/tasks, schedule/budget, loss decomposition, clipping semantics/counts, gradient-evidence commitments, gradient-parameter manifests, explicit activity structure, independent objective-derived activity semantics, rescue/persistence, clipping windows, cross-seed aggregates, paired contrasts, control-equivalence binding, dataset/source identity, runtime identity и implementation commit. Validator не вызывает producer summary helpers и не выбирает scientific conclusion.

Source identity строится из exact git object at implementation commit, который обязан быть commit object и ancestor текущего checkout, и закрывает transitive reference/science/runtime files, producer/validator/CLI/spec, fixed-target contract и reviewer bridge execution mapping. Runtime evidence отдельно связывает текущий canonical CPU fingerprint и semantic identity `configs/fixed-cpu-target-1.0.json`.

## Execution lifecycle

Canonical execution разрешается только через typed reviewer bridge task `a2-gradient-clipping-v1` после fresh fixed-target preflight. Во время implementation PR scientific producer не запускается: допускаются только reduced/synthetic tests, static validation и natural PR CI.
