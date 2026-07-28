# Changelog v1.19 / v2.19

- Added an exact machine-readable PyTorch module/state_dict inventory.
- Replaced declarative training hashes with structured, recomputed sidecars and safetensors header validation.
- Added exact A2c/A3 FLOPs-sensitivity training and confirmatory lineage arms.
- Made P06 model-audit checks evidence-bound and added the independent architecture/initialization/dormant-parameter audit check.
- Added P08 checkpoint lineage binding from every Planner arm to the exact P07 regime/variant/seed/model SHA.
