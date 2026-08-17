# A2 sufficient-budget task-order causal discrimination

- Version: `development-a2-sufficient-budget-task-order/0.1`
- Implementation: `2a3843b1c5774cb73de1c7bc3f9b724f7a697ba7`
- Source: `sha256:67053aa64e7fe8c9ef3e4f493c57ef3d0d4c066c448882a9a921f6e2a67be229`
- Seeds: `[17, 29, 43]`
- Arms: `{'canonical_order': ['bw-00000001', 'bw-00000002', 'bw-00000003'], 'task01_middle': ['bw-00000002', 'bw-00000001', 'bw-00000003'], 'task01_last': ['bw-00000002', 'bw-00000003', 'bw-00000001']}`
- Budget: `100 epochs / 300 updates per seed-arm`
- Canonical frozen first-9-update equivalence: `PASS` for every seed
- Held-out accessed: `false`
- GO_LATENT: `NOT EVALUATED`

## Rescue events

### canonical_order
- order: `['bw-00000001', 'bw-00000002', 'bw-00000003']`
- first position-0 rescue by seed: `{'17': 33, '29': 33, '43': 15}`
- first full free-running rescue by seed: `{'17': 33, '29': 51, '43': 54}`

### task01_middle
- order: `['bw-00000002', 'bw-00000001', 'bw-00000003']`
- first position-0 rescue by seed: `{'17': 36, '29': 33, '43': 39}`
- first full free-running rescue by seed: `{'17': 60, '29': 36, '43': 42}`

### task01_last
- order: `['bw-00000002', 'bw-00000003', 'bw-00000001']`
- first position-0 rescue by seed: `{'17': 39, '29': 30, '43': 30}`
- first full free-running rescue by seed: `{'17': 45, '29': 48, '43': 33}`

## Interpretation boundary

`SUPPORTED HYPOTHESIS / NOT PROVEN`

This round tests only whether sufficient-budget rescue depends materially on
within-epoch placement of task01. It does not support A3/latent/semantic claims.
