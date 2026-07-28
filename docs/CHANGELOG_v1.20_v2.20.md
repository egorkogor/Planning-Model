# Changelog v1.20 / v2.20

## Gate evidence hardening

- P08 accepts only canonical P07 checkpoint manifests and their canonical training reports.
- Checkpoint manifests are validated end-to-end before confirmatory execution.
- Initialization tensors are recomputed from the locked name-derived PCG64 initializer.
- Trained checkpoints must change at least one active tensor while dormant tensors remain byte-identical to initialization.
- AdamW states must be finite, non-negative for second moments, non-zero in aggregate, and carry locked optimizer metadata.
- P06 checks now have check-specific recomputation for parameter inventory, same-information cases, raw rollout invariants, and dormant gradients.

## Operational fixes

- `validation/validate_bundle.py` runs both `validation/` and `tests/` suites.
- P02 bundle testing runs the complete repository test suite.
- Test files run in isolated subprocesses with per-file timeouts, separate basetemp directories, and retained failure logs.
- Git bundle restoration is documented with `git clone -b main` and release bundles publish `HEAD` plus `main`.
