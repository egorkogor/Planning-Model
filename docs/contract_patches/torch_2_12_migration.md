# Runtime contract patch: PyTorch 2.12.0

Status: **DEVELOPMENT IMPLEMENTATION PATCH**  
Scope: training runtime dependency only; no scientific hypothesis, arm, dataset, metric,
or confirmatory execution rule changes.

## Evidence and reason

The available, exercised CPU runtime is `torch==2.12.0` (`2.12.0+cpu`). It successfully
performs tensor autograd, the locked 177-tensor A2 forward/backward pass, AdamW with the
contract betas `(0.9, 0.95)`, checkpoint reload, and two clean deterministic replays.
The former broad `2.10.x` declaration was inconsistent with the exact reproducibility pin
requested by the implementation handoff and with `requirements.lock`.

## Patch

`docs/infrastructure/runtime_dependency_contract_v1.yaml#training_environment.torch` is
migrated from `2.10.x` to exact `2.12.0`; `requirements.lock`, the optional `ml`
dependency, runtime fail-fast validation, release manifest, and repository hashes move in
the same patch. FINAL_CONFIRMATORY remains prohibited until the normal environment lock
and sealing gates are satisfied. This patch itself is not confirmatory evidence.
