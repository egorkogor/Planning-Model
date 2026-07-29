# Toy A2 implementation progress (non-normative)

## Domain and toy dataset

- Implemented: canonical BlocksWorld facts/goals, contract actions, preconditions, deterministic application, goal verification, BFS oracle, canonical hashing, and seed-17 train/validation generation for `n=1..3`.
- Passing tests: `python -m pytest -q tests/toy/test_domain_dataset.py` (3 passed).
- Last commit before this block: `1cbe957`.
- Known limitations: model, training, execution, and lineage remain; PyTorch is absent from the configured environment and the approved CPU wheel index is unreachable (HTTP tunnel 403).
- Contract discrepancy: `episode_plan_manifest.schema.json` excludes toy seed 17 while WorkPlan and training contracts require/permit it; toy evidence will be kept explicitly non-confirmatory rather than weakening the normative schema.
- Next step: make PyTorch available in the environment, then implement the locked A2 state-dict inventory and deterministic initialization.
- Continue with: `python -c "import torch; print(torch.__version__)"`, then `python -m pytest -q tests/toy`.
