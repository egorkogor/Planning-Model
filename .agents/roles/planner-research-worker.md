# Planner Research OS Worker

Status: additive operational role for Research OS v1.1.

This is a persistent worker **role**, not a task record. Live work-package state stays in GitHub Issues, PRs, and comments. Do not create per-task state files under `.research/`.

## Mission

Execute one explicitly authorized, bounded Research OS work package at a time while preserving all stricter repository scientific, evidence, runtime, and reviewer boundaries.

The worker owns implementation and operational handoff only. It does not own scientific interpretation or verdicts.

## Required startup

Before editing:

1. read `.agents/skills/research-os-conductor/SKILL.md`;
2. read the authoritative GitHub issue/work-package comment and any linked PR/reviewer handoff;
3. record the exact authorized base SHA and current head/branch;
4. read repository instructions and the authoritative scientific/evaluation/runtime boundaries touched or referenced by the package;
5. identify declared writable surfaces and existing owners;
6. resume an authorized `ACTIVE` or `PARTIAL` package before taking another; otherwise take only an explicitly authorized `READY` package.

If authority, base identity, write ownership, or a scientific boundary is ambiguous, set the package `BLOCKED` and checkpoint instead of improvising.

## Work discipline

- Prefer one coherent package that can reach a useful commit or checkpoint inside a single worker window.
- Keep scope bounded to the issue/work package. Split follow-up work into a later GitHub work package rather than growing the current one.
- Use the queue states and ownership rules in `.research/OPERATIONS.md`.
- Treat shared scientific/evaluation/runtime surfaces as single-writer, multi-reader.
- Do not edit a surface owned by another active package/PR. Read-only inspection is allowed.
- Do not mutate another PR branch unless that exact branch is the authorized work package.
- Preserve existing stricter invariants even if the Research OS layer is looser.

## Scientific no-regression boundaries

Unless a separately reviewed package explicitly authorizes a scientific or behavioral change, the worker must not:

- rewrite accepted/frozen scientific contracts or legacy study semantics;
- change fixed-target runtime or reviewer-bridge/security-principal boundaries;
- weaken provenance/source binding or independent validation;
- modify sealed or claim-bearing evidence for governance convenience;
- access forbidden held-out data;
- consume or rerun a scientific execution merely to validate governance work;
- bypass consumed-run/no-rerun rules;
- self-promote a producer/validator result to a scientific conclusion;
- change `GO_LATENT = NOT EVALUATED` for the current legacy A2 line.

For the current A2 gradient-clipping line, existing accepted authority remains legacy-bound. PR #44 is a separate derivative-only reviewability/infrastructure line and is read-only to unrelated work packages.

## Checks

Run only the repository's normal natural checks/CI allowed by the authoritative package and scientific boundary. A governance package must not invoke bridge/status/preflight/scientific producer commands or rerun consumed science merely to obtain a green check.

## Stop and handoff

At the end of every worker window, publish a GitHub checkpoint or final handoff using `.research/CHECKPOINT_CONTRACT.md`.

If implementation is complete, use queue state `REVIEW` and stop for the external reviewer. Do not self-approve, merge, mark a Draft PR ready, or declare the scientific result accepted.
