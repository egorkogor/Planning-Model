# Research OS v1.1 operations

Status: additive governance/coordination semantics. No runtime dependency.

## Authority and transport

GitHub remains the authoritative operational system:

- **Issue / explicitly authorized issue comment** — work-package definition and current queue state;
- **PR** — implementation branch, review transport, and code-level handoff;
- **Issue/PR comments** — checkpoints, blockers, resume state, and final handoff.

`.research/` stores reusable scientific/governance semantics and blank contracts only. It is **not** a task database. Do not commit populated per-issue work packages, worker leases, queue snapshots, checkpoints, or duplicate GitHub state here.

Repository scientific contracts, accepted evidence, validators, reviewer decisions, and stricter legacy rules remain authoritative over this operational layer.

## Queue states

The only Research OS operational queue states are:

| State | Meaning |
| --- | --- |
| `READY` | Explicitly authorized, bounded, and available to be claimed. No worker currently owns its writable surfaces. |
| `ACTIVE` | A worker has claimed the package and its declared writable surfaces and is executing it. |
| `PARTIAL` | Useful progress is persisted and resumable; more work remains, but there is no known external blocker. |
| `BLOCKED` | Safe progress cannot continue because an external prerequisite, authority ambiguity, ownership conflict, or required reviewer decision is unresolved. |
| `REVIEW` | Implementation/checkpoint work is complete and stopped for external review. This does not imply scientific acceptance or PR readiness/merge. |
| `DONE` | The authorized reviewer/owner has accepted or closed the operational package. Workers do not use `DONE` to self-approve scientific or review outcomes. |
| `CANCELLED` | The authorized owner/reviewer has abandoned or superseded the package without completion. |

Normal transitions are:

```text
READY -> ACTIVE
ACTIVE -> PARTIAL | BLOCKED | REVIEW
PARTIAL -> ACTIVE | BLOCKED
BLOCKED -> READY | ACTIVE
REVIEW -> ACTIVE | DONE
READY | ACTIVE | PARTIAL | BLOCKED | REVIEW -> CANCELLED
```

A transition is recorded in GitHub, not by editing a repository queue file.

## Selection and resume rule

A persistent worker:

1. resumes an explicitly authorized `ACTIVE` or `PARTIAL` package assigned to its role before taking another package;
2. otherwise takes only an explicitly authorized `READY` package;
3. does not infer work from repository TODOs, open PRs, or `.research/` metadata alone;
4. executes one coherent package per worker window where practical;
5. checkpoints before stopping or changing package.

## 1-hour-safe package rule

A work package should be small enough that a worker can normally do one of these inside a single worker window:

- finish the package and enter `REVIEW`; or
- persist a coherent intermediate commit/state and publish a complete `PARTIAL`/`BLOCKED` checkpoint.

Bound packages by **one objective and explicit writable surfaces**, not by a vague topic. If the safe next step would expand scope, change scientific semantics, cross an ownership boundary, or require an unplanned execution, checkpoint and split a new GitHub package instead.

Every package must define a stop condition and an exact next atomic action so a fresh worker can resume without reconstructing hidden chat state.

## Write ownership: single-writer / multi-reader

### Rule

Shared scientific/evaluation/runtime surfaces are **single-writer, multi-reader** while a package holds ownership.

Multiple workers may inspect the same surface. Only the package that explicitly owns that writable surface may modify it. A second worker must use a disjoint surface or remain read-only.

### Shared surfaces include

At minimum:

- scientific/evaluation specs and frozen contracts;
- producer, validator, and scientific-analysis implementation;
- fixed-target/runtime contracts and workflow mappings;
- reviewer bridge and security-principal surfaces;
- dataset/held-out policy and split definitions;
- provenance/source-binding logic;
- sealed or claim-bearing evidence and its manifests;
- consumed-run/request ledgers and no-rerun state;
- shared schemas/configs whose change can alter scientific or evaluation semantics.

Documentation is not automatically disjoint: a document that changes the meaning of one of these surfaces is governed by the same owner.

### Ownership declaration

A Work Package must list:

- exact writable paths or a narrowly defined writable surface;
- read-only authoritative references;
- active PR/branch that holds the write claim, when applicable.

Ownership is operational coordination, not permission to weaken scientific authority. Existing reviewer/security/provenance controls still apply.

If two packages need the same writable surface, the later package becomes `BLOCKED` until ownership is released or the reviewer explicitly rebases/reassigns the work. Do not solve the conflict with concurrent edits and later merge conflict resolution on scientific surfaces.

`ACTIVE` and `PARTIAL` retain ownership by default. Shared scientific/evaluation/runtime surfaces also remain owned through `REVIEW` by default so repair work cannot race another writer. `BLOCKED` must state whether ownership is retained or released. `DONE` and `CANCELLED` release ownership. An authorized reviewer may explicitly release or reassign a surface earlier.

## Current Planner compatibility boundary

This layer does not alter current A2/gradient-clipping science. The existing line remains legacy-bound, including fixed-target execution, reviewer bridge/security principal, provenance/source binding, independent validation, sealed evidence, held-out restrictions, consumed-run/no-rerun semantics, reviewer-only interpretation, and `GO_LATENT = NOT EVALUATED`.

PR #44 is a separate derivative-only reviewability/infrastructure work line. Other Research OS work treats its branch and touched A2 surfaces as read-only unless an explicit package says otherwise.

## Required GitHub handoff

Every stop publishes the fields defined in `.research/CHECKPOINT_CONTRACT.md`. A completed implementation uses `REVIEW`, not `DONE`, and stops for external reviewer judgment.
