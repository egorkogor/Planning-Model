# Changelog v1.21 / v2.21

- Narrowed A3↔A2c claim to an architecture-bundle comparison; active capacity is not equal and a pure representation-effect claim is forbidden.
- Replaced ambiguous P04 “exhaustive n≤5” wording with exhaustive n≤4 plus deterministic 4,096-case n=5 coverage and explicit manifest/capacity limits.
- Strengthened P06 SAME_INFORMATION to 21 cases with three cases per intent.
- Strengthened RAW_ROLLOUT to a 36-case variant×intent matrix with byte-bound logits, frozen plan and event log artifacts.
- Made resource-plan checks fail closed and plan_hash recomputable.
- Removed optimized-mode assert bypasses from the static validator.
- Made isolated test runner reject skipped or empty test files; removed duplicate CI suite execution.
- Rejected NFC-normalized JSON-key collisions.
