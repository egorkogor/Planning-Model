# Research OS Checkpoint contract

A Checkpoint is a resumable GitHub Issue/PR comment written whenever a worker stops. It records operational continuity; it is not scientific evidence and is not stored as a populated `.research/` task record.

## Required checkpoint pattern

```text
Research OS Checkpoint

Status: PARTIAL | BLOCKED | REVIEW
Issue: #<number>
PR: #<number or NOT_CREATED>
Worker role: <persistent role>
Authoritative base: <full commit SHA>
Branch: <branch>
Head: <full commit SHA>

Completed work:
- <what is actually persisted>

Changed files:
- <path>

Checks:
- <command/check/status and result>

Scientific execution performed: YES | NO
Claim-bearing evidence modified: YES | NO
Consumed run rerun: YES | NO
Existing scientific behavior changed: YES | NO

Write ownership:
- <surface> — RETAINED | RELEASED

Blocker:
<exact blocker or NONE>

Exact next atomic action:
<one concrete next action; NONE when REVIEW>

Resume location:
<file:line, PR thread, failing check, or other exact starting point>

Scientific/reviewer boundary:
<state any required reviewer-only interpretation and current scientific status>
```

All SHAs must be full commit identities. Report only work/checks that actually occurred.

## State-specific requirements

### `PARTIAL`

Use when useful progress is persisted and the package can continue without an external prerequisite.

- retain write ownership by default;
- provide a non-empty exact next atomic action;
- provide an exact resume location.

### `BLOCKED`

Use when safe progress requires an external prerequisite, ownership release, authority clarification, or reviewer decision.

- name the blocker precisely;
- state whether each write surface remains retained or is released;
- the next atomic action must be conditional only on the named blocker being resolved.

### `REVIEW`

Use when worker implementation and allowed local/natural checks are complete enough for external review.

- release normal write ownership unless the reviewer explicitly reserves it;
- set blocker to `NONE` unless review itself is waiting on a known prerequisite;
- set exact next atomic action to `NONE — external reviewer decision required`;
- do not convert `REVIEW` into `DONE`, merge, self-approve, or mark a Draft PR ready on the worker's own authority.

## Mandatory Research OS reviewer handoff

A final `REVIEW` checkpoint must additionally include the conductor report fields:

```text
Project:
Authoritative base:
Current scientific round:
Legacy/frozen authorities preserved:
Research OS gaps found:
Changes made:
Files changed:
Existing scientific behavior changed: YES/NO
Claim-bearing evidence modified: YES/NO
Scientific execution performed: YES/NO
Consumed run rerun: YES/NO
Natural CI/checks:
Open risks/blockers:
Recommended next action:
Reviewer decision requested: ACCEPT / REPAIR / REJECT
```

For legacy Planner scientific lines, a worker handoff must preserve the current reviewer-owned scientific status rather than deriving a new verdict. In particular, operational completion cannot change `GO_LATENT = NOT EVALUATED`.
