# Sources and pinned external dependencies

External identifiers below are bootstrap inputs, not moving runtime references. Phase P01 resolves immutable revisions and writes `ModelLock` artifacts.

## Models

- Frozen LLM: `Qwen/Qwen3.5-0.8B`.
- Semantic target encoder: `sentence-transformers/all-MiniLM-L6-v2`.

## Required lock contents

For each model:

- exact repository ID and immutable commit SHA;
- file-level hashes for config, tokenizer, chat template and weights manifest;
- library versions;
- accepted license metadata;
- load configuration, dtype and device;
- generated lock hash.

A moving branch such as `main` is forbidden after P01.

## Algorithms and libraries

- normative Oracle: repository BFS implementation, not an external solver;
- Fast Downward: optional cross-check only;
- JSON Schema: Draft 2020-12;
- PyTorch and Transformers/Sentence Transformers versions are fixed in environment lock;
- statistical procedures are implemented in versioned scripts whose hashes are included in freezes.

## Citation boundary

The experiment only claims outcomes produced by the frozen code, tasks and model revisions. Model-card performance claims are not imported into the experimental conclusion.
