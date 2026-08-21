# Research OS Work Package contract

A Work Package is a bounded operational instruction carried by a GitHub Issue or explicitly authorized GitHub comment. This file is a reusable blank contract, not a place to store live package instances.

## Required fields

Copy the following pattern into GitHub and fill every field before a worker changes repository state:

```text
Research OS Work Package

Issue: #<number>
Status: READY | ACTIVE | PARTIAL | BLOCKED | REVIEW | DONE | CANCELLED
Worker role: <persistent role path/name>
Authoritative base: <full commit SHA>
Branch: <branch or NOT_CREATED>
Head: <full commit SHA or SAME_AS_BASE/NOT_CREATED>

Objective:
<one coherent outcome>

In scope:
- <bounded item>

Out of scope:
- <explicit exclusions>

Writable surfaces / write ownership:
- <exact path or narrow surface> — owner: <this issue/PR/branch>

Read-only authorities / shared references:
- <spec, evidence, other PR/branch, runtime boundary, etc.>

Scientific/evidence boundaries:
- <no-regression constraints relevant to this package>

Allowed checks:
- <natural repository checks/CI only unless stricter authority says otherwise>

Forbidden execution:
- <bridge/status/preflight/scientific producer/consumed rerun restrictions as applicable>

Single-window target:
<the atomic result expected in one worker window>

Stop condition:
<what means REVIEW, PARTIAL, or BLOCKED>

Exact next atomic action:
<first concrete action for ACTIVE/resume>
```

## Bounding rules

A valid package has:

- one objective;
- explicit in-scope and out-of-scope boundaries;
- an exact base SHA;
- narrow writable surfaces with one writer;
- named read-only authorities;
- an allowed-check boundary;
- a stop condition;
- an exact next atomic action.

Prefer a package that can finish or checkpoint coherently within one worker window. If work expands into another scientific question, another shared write surface, or a new execution lifecycle, stop and authorize a separate GitHub package.

Do not use a Work Package to silently authorize `BEHAVIORAL` or `SCIENTIFIC` changes when the governing issue is `ADDITIVE`. Such a need becomes `BLOCKED` pending explicit reviewer authorization.

## Claiming a package

Before moving `READY -> ACTIVE`, the worker must verify:

1. the GitHub package is explicitly authorized;
2. its base still matches the reviewed repository state or the package explicitly authorizes a rebase;
3. no other active package owns the declared writable surfaces;
4. relevant frozen/legacy authority is understood;
5. the first next action does not require a forbidden scientific execution.

Record the claim and state transition in GitHub. Do not create a repository-local lease or queue record.

## Resume semantics

An `ACTIVE` or `PARTIAL` package is resumed from its latest GitHub checkpoint. The fresh worker must be able to recover the exact base/head, changed files, checks, ownership state, blocker (if any), next atomic action, and resume location without relying on previous chat context.
