# Planning-Model Research OS adapter

Status: advisory, additive, non-runtime.

This directory indexes research state without replacing authoritative project contracts, evidence, validators, workflows, or reviewer decisions.

## Authority

For legacy work, repository scientific specs, accepted reviewer decisions, exact Git/provenance bindings, sealed evidence, and consumed-run semantics remain authoritative. Research OS metadata is derivative and must never silently strengthen a scientific conclusion.

Current A2 gradient-clipping work is `legacy-bound` and completes under the existing A2/reviewer-bridge rules.

## Object model

Future OS-native research may use:

- `RQ-*`: research question
- `CLM-*`: hypothesis/claim
- `EVAL-*`: measurement/evaluation contract
- `STU-*`: study protocol
- `RUN-*`: immutable execution instance
- `EVD-*`: evidence reference
- `AUD-*`: validity audit
- `DEC-*`: scientific decision

For legacy A2 lineage, `registry.yaml` is an index only. It points to existing authoritative documents instead of copying or rewriting them.

## Four-loop policy

1. Discovery: adaptive and non-claim-bearing.
2. Measurement: validate that tasks/metrics/harness distinguish the intended construct from shortcuts or confounders.
3. Confirmatory: freeze study-level scientific degrees of freedom before claim-bearing execution.
4. Knowledge: retain supported, contradicted, negative, invalid and replication outcomes.

Discovery and pilot evidence must not be promoted retroactively into confirmatory evidence.

## Existing Planner mechanisms already covering Research OS

The repository already has strong equivalents for much of the confirmatory loop:

- versioned A2 evaluation/causal specs;
- exact intervention/control contracts;
- fixed-target runtime;
- strict held-out restrictions;
- planned seeds/budgets/checkpoints;
- raw update/epoch evidence;
- independently reconstructed/validated evidence;
- source and implementation provenance;
- reviewer-controlled execution and scientific interpretation;
- historical result freeze / no silent replacement.

Research OS must not duplicate or weaken these.

## Advisory additions

The bootstrap adds only:

- a cross-study research/claim/evaluation index;
- a gap audit;
- templates for future Evaluation Contracts, statistical design and validity audits.

No required CI gate is introduced. No existing producer, validator, workflow or scientific result imports this directory.

## Operational layer

Research OS v1.1 adds reusable operational semantics without adding a task database or runtime dependency:

- persistent worker role: `.agents/roles/planner-research-worker.md`;
- queue and single-writer/multi-reader semantics: `OPERATIONS.md`;
- bounded GitHub Work Package pattern: `WORK_PACKAGE_CONTRACT.md`;
- resumable GitHub Checkpoint/handoff pattern: `CHECKPOINT_CONTRACT.md`.

Live queue state, work-package instances, checkpoints, blockers, and handoffs stay in GitHub Issues, PRs, and comments. Do not mirror them as populated files under `.research/`.

## Future transition

A genuinely new confirmatory study should normally establish:

`RQ -> CLM -> validated EVAL -> frozen STU -> existing execution backend -> EVD -> independent validator -> AUD -> reviewer DEC`

The existing execution backend remains the preferred execution/provenance layer unless a separately reviewed change is justified.
