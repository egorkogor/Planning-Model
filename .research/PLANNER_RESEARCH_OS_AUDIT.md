# Planning-Model Research OS v1 audit

Status: advisory governance record for Draft PR #45. It is not a scientific contract and does not override existing project authority.

## Authority and scope

- Authoritative base reviewed: `ee773479cbbc84eab52f93996bc8beb64a51453c` (`main`).
- Current scientific line: A2 causal-discrimination, currently gradient-clipping follow-up.
- Current gradient-clipping study remains `legacy-bound` and finishes under its existing project contracts.
- Existing scientific specs, fixed-target/runtime contracts, reviewer bridge, validators, held-out policy, evidence/provenance bindings, consumed-run semantics and reviewer decisions retain precedence over `.research/` metadata.
- No current scientific protocol or evidence is migrated or rewritten by this audit.

## Scientific lineage reconstructed

1. **Development Quality Evaluation v0.1** established an end-to-end reproducible evaluation path but produced END-only behavior across A2/A3/A4, so the scientific decision was `REDESIGN`, not a latent-channel conclusion. Source: `docs/evaluations/A2_A3_A4_V0_1_DECISION_RU.md`.
2. **A2 optimization-budget trajectory** asked whether the historical 3-epoch/9-update budget was sufficient. It froze train-only A2 conditions, preserved held-out exclusion, bound the historical 9-update prefix exactly, and treated rescue under additional canonical updates as evidence supporting under-training as a major cause, with interpretation bounded to `SUPPORTED HYPOTHESIS / NOT PROVEN`. Source: `docs/evaluations/A2_OPTIMIZATION_BUDGET_TRAJECTORY_SPEC_RU.md`.
3. **A2 sufficient-budget task-order study** isolated within-epoch task order at the sufficient budget. The accepted reviewer result classified task order as a secondary trajectory modifier rather than a stable primary controller of rescue; evidence was accepted under manual fixed-target provenance, while prior invalid/infra runs remained non-results. Source contract: `docs/evaluations/A2_SUFFICIENT_BUDGET_TASK_ORDER_SPEC_RU.md`; accepted scientific review: PR #34 comment `5318288809`.
4. **A2 gradient-clipping study** now isolates clipping policy while holding canonical order, model/objective/data/optimizer/budget/runtime fixed. Primary outcomes are pre-specified rescue events; held-out 04/05 remain excluded; producer cannot issue a scientific verdict; independent validation reconstructs lower-level semantics. Source: `docs/evaluations/A2_GRADIENT_CLIPPING_CAUSAL_SPEC_RU.md`.
5. PR #44 is reviewability infrastructure only: a bounded derivative summary bound back to full claim-bearing evidence. It does not alter sealed evidence or scientific execution semantics.

Current bounded claim graph:

- `CLM-PLN-A2-BUDGET`: insufficient optimization budget is a major/dominant supported explanation for the historical A2 END-collapse in the tested toy regime; status remains bounded by existing reviewer language and is not a universal causal proof.
- `CLM-PLN-A2-TASK-ORDER`: within-epoch placement of task01 can modify transient optimization trajectory, but accepted evidence does not show a stable primary control of eventual rescue at sufficient budget.
- `CLM-PLN-A2-CLIPPING`: canonical gradient clipping at norm 1.0 may be an additional causal contributor to rescue delay; active legacy-bound study, scientific verdict pending external reviewer interpretation.
- `CLM-PLN-LATENT`: latent/A3 superiority or usefulness is not evaluated by the current A2 causal-discrimination line; `GO_LATENT = NOT EVALUATED`.

## Gap matrix

| Research OS practice | Existing Planner mechanism | True gap | Proposed change | Class | Risk |
|---|---|---|---|---|---|
| Research questions | Questions are embedded in versioned evaluation/spec Markdown | No single index linking questions across rounds | Add read-only `lineage.yaml` with RQ/claim/study references | ADDITIVE | Low; metadata can become stale |
| Competing hypotheses | Budget, task-order and clipping are explicitly isolated in successive specs | Relationships are reconstructed from multiple docs/PR reviews | Encode bounded competing-claim graph by reference, without strengthening statuses | ADDITIVE | Medium if summaries overstate evidence |
| Versioned study protocol | Strongly present: causal contract, arms, frozen scope, outcomes, execution boundary | None for legacy studies | Reuse existing specs as `STU-*` authority; do not duplicate | ADDITIVE adapter only | Low |
| Evaluation Contract distinct from Study | Measurement semantics exist inside each study spec/producer/validator | No reusable/versioned construct-level contract independent of an intervention | Add template for future `EVAL-*`; do not retrofit current study | ADDITIVE | Low |
| Measurement calibration | Strong semantic/tamper validation exists at study level | No explicit lifecycle saying when a measurement construct itself is calibrated/validated versus merely implemented | Add future evaluation-contract calibration fields and adversarial validity checklist | ADDITIVE | Low |
| Independent evidence validation | Strongly present: persisted lower-level independent recomputation, source/runtime binding, tamper tests | No gap | Preserve existing validators; do not create duplicate Research OS validator | NONE | Duplication would raise risk |
| Validity audit distinct from validator | Reviewer performs some interpretation/validity reasoning ad hoc; specs list boundaries | No first-class post-validation audit for shortcut/confounder/construct validity | Add `validity-audit` template for future studies and optional legacy review notes | ADDITIVE | Medium; must not masquerade as evidence validation |
| Statistical design | Seeds, paired arms, descriptive contrasts and censoring semantics are pre-specified | No explicit unit-of-analysis, dependence, uncertainty/MDE/power rationale; current clipping intentionally avoids inferential statistics | Add statistical-plan template for future confirmatory studies; current legacy studies unchanged | ADDITIVE | Low now; future methodological choice requires reviewer approval |
| Run identity/provenance | Strong fixed-target, implementation SHA, runtime/source identity, consumed-run semantics | No Research OS gap | Index by reference only | NONE | Low |
| Invalid vs negative result | Decision docs and reviewer comments distinguish invalid/infra/no-result from valid scientific findings | No centralized decision registry | Add lineage/decision references with explicit `invalid`, `negative`, `supported`, `pending` vocabulary for future use | ADDITIVE | Medium if legacy status is reclassified post hoc |
| Replication | Same-host/canonical equivalence and repeated seeds exist, but not a general replication registry | Future robustness studies lack a common metadata slot | Add future study template guidance and lineage relation `replicates` | ADDITIVE | Low |
| Literature/prior art | Architecture/research docs exist but are not consistently linked to claims/evals | No structured linkage from external evidence to RQ/CLM/EVAL/STU | Add literature-record template only; source ingestion is a later advisory round | ADDITIVE | Low |
| Large-artifact review | PR #44 introduces validator-bound bounded summaries and large-artifact protocol | Protocol exists only on PR #44 until merged | Record dependency/reference; do not copy or make it required in #45 | ADDITIVE documentation | Low |
| Required OS gates | Existing scientific gates are already strict | Enabling new OS-required CI now could interfere with legacy work | Keep all OS validation advisory until an OS-native future study is piloted | ADDITIVE policy | High if made required prematurely |

## Compatibility findings

### Already stronger than Research OS defaults

Planner already has strong controls that should remain authoritative:

- exact implementation/source identity binding;
- fixed canonical CPU target and runtime provenance;
- reviewer-bridge execution/security boundaries;
- frozen/versioned causal microexperiment contracts;
- exact intervention semantics and control equivalence;
- lower-level evidence persistence;
- independent recomputation instead of producer self-consistency;
- adversarial tamper tests;
- held-out exclusion for train-only causal diagnostics;
- explicit no-rerun/consumed-run handling;
- reviewer-only scientific interpretation.

Research OS must reference these mechanisms, not replace them.

### Genuine upper-layer gaps

The useful additions are primarily governance/navigation for future work:

1. one research-question / claim / study graph;
2. reusable `EVAL-*` contracts independent of interventions;
3. explicit measurement-calibration state and validity threats;
4. validity audit separated from evidence validator;
5. explicit statistical rationale before future claim-bearing runs;
6. structured decision/replication/literature linkage.

## Deferred proposals requiring reviewer approval

The following are **not implemented** in this round because they would be behavioral or scientific:

- making Research OS validation a required CI gate;
- requiring an `EVAL-*` object for the already-running gradient-clipping study;
- changing current seeds/repetitions or adding inferential statistics/power targets to legacy studies;
- changing retry/no-rerun semantics;
- changing existing scientific verdict labels;
- introducing new held-out evaluation or replication execution;
- changing reviewer-bridge task registry or scientific runtime.

## Recommended migration

1. Accept this advisory registry/templates as read-only metadata.
2. Finish the current gradient-clipping line under existing authority.
3. Start the next genuinely new scientific question as the first OS-native path: `RQ -> CLM -> EVAL -> STU`.
4. Keep OS checks advisory through that first complete cycle.
5. Only after external review consider required governance gates for future studies.
