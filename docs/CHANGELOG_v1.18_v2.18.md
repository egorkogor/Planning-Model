# Changelog v1.18 / v2.18

Focused final architecture-evidence release. No new experimental arms, lock layers, or scientific gates.

1. Added deterministic name-derived tensor initialization, created once per seed as a common superset state dict and copied identically to A1/A2/A2b/A2c/A3/A3r.
2. Specified A3r inference exactly: raw predicted normalized z remains autoregressive feedback; nearest frozen random-code entry resolves only the external semantic signature by cosine similarity with lexicographic tie-break.
3. Removed the undefined A1 equal-compute retraining requirement. A1 compute is reported only as train/inference FLOPs, positions, latency and memory guardrails.
4. Replaced ambiguous best-checkpoint evidence with exact final-step evidence. Final reports require optimizer step 12000, FINAL_STEP_ONLY selection, architecture/initialization/inventory hashes, ordered examples, dormant-gradient audit and final checkpoint hashes.
5. Model audit requires the exact unique eight-variant set A1/A2/A2b/A2c/A3/A3r/A4/A5.
6. Existing P06/P07 gates now machine-check model audit and the exact six-variant × five-seed final training report matrix.
