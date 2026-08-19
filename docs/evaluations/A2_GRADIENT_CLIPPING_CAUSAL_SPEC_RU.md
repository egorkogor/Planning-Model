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

Projection связывает initialization, все historical trace fields (loss decomposition, target counts, task/update order, pre-clip norm, canonical clip norm/flag, operator position weight), checkpoint model/optimizer identities, final model/optimizer identities, rescue events и persistence. Полная reference projection на 300 updates сохраняется в JSON evidence; independent validator отдельно строит candidate projection из raw `clip_1_0` evidence и требует exact equality. Reference path дополнительно обязан пройти существующую frozen 3-epoch / 9-update historical prefix equivalence. Любое расхождение — fail closed.

## Gradient intervention evidence

На каждом update сохраняются:

- update/epoch/task identity;
- operator/arg1/arg2/total losses и target counts;
- pre-intervention global L2 norm;
- policy и threshold;
- `clipping_occurred`;
- post-intervention global L2 norm;
- exact deterministic SHA-256 named gradients before/after intervention;
- canonical gradient-parameter manifest (ordered names, dtype, shape) и его SHA-256;
- gradient hash encoding version;
- operator position weight.

Named-gradient hash использует canonical optimizer parameter order, parameter name, dtype, shape и exact contiguous CPU bytes. Hashing не изменяет gradients. Для каждого arm/seed сохраняется отдельный commitment над 300 raw gradient-evidence records; validator пересчитывает его независимо.

Semantic consistency:

- `no_clip`: before == after и clipping=false;
- inactive clip: before == after;
- active clip: before != after и post norm ограничен threshold;
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

Independent validator отдельно пересчитывает из persisted lower-level evidence coverage arms/seeds/tasks, schedule/budget, loss decomposition, clipping semantics/counts, gradient-evidence commitments, gradient-parameter manifests, rescue/persistence, clipping windows, cross-seed aggregates, paired contrasts, control-equivalence binding, dataset/source identity, runtime identity и implementation commit. Validator не вызывает producer summary helpers и не выбирает scientific conclusion.

Source identity строится из exact git object at implementation commit, который обязан быть commit object и ancestor текущего checkout, и закрывает transitive reference/science/runtime files, новый producer/validator/CLI/spec, fixed-target contract и reviewer bridge execution mapping. Runtime evidence отдельно связывает текущий canonical CPU fingerprint и semantic identity `configs/fixed-cpu-target-1.0.json`.

## Execution lifecycle

Canonical execution разрешается только через typed reviewer bridge task `a2-gradient-clipping-v1` после fresh fixed-target preflight. Во время implementation PR scientific producer не запускается: допускаются только reduced/synthetic tests, static validation и natural PR CI.
