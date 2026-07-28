# Changelog v1.17 / v2.17

Focused architecture-freeze release. No new trust or gate layers.

1. A1 and step-level arms now share one 85-position decoder parameter inventory. A1 uses all 85 positions and a 24-token grammar head; step-level arms use positions 0..16 through ConceptPacker.
2. Loss semantics are fully masked and reduced: END is trained only through action CE, pointer heads use action-conditional targets, semantic losses exclude END, and empty contrastive-positive batches have deterministic zero loss.
3. `planner_seed` is required in EpisodePlanManifest and EpisodeLog and is checked against lineage even when plan generation fails.
4. Block counts 7–8 are forbidden in all training/development expansion and remain sealed size-OOD evaluation only.
5. The sensitivity run is explicitly train-FLOPs-matched A3↔A2c; inference FLOPs are reporting guardrails only.
