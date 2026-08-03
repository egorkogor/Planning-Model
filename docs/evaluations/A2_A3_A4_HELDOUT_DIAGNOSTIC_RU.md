# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Evaluator source SHA256: `sha256:58ce15ddcf99f706b43523c9d96c55497b2cd1a7e826d71442bca310d30eec6e`
- Implementation commit: `85673afa433bfe8774b11d94a988c2b3028fb358`
- Requirements lock: `sha256:883c8e262c6f3ea917239010be08ac0064ab67de70c1caad7bb98fe9f0b68401`
- Runtime: `{"cuda_available": false, "cuda_version": null, "execution_device": "cpu", "numpy": "2.3.5", "python": "3.11.15", "torch": "2.12.0+cpu"}`
- Evaluator: `development-quality-evaluation/0.1`
- Dataset hash: `sha256:60e4ce06d6cfc90dc467fb4e82b2eb71cf2d92d37471eee3aeda64f864c541df`
- Train tasks: 3; held-out tasks: 2
- Seeds: 17, 29, 43
- Budget: 3 epochs × 3 canonical train tasks = 9 updates/run; final checkpoint only

## Variant × seed

| Variant | Seed | Success | Rate | Executable | Action applicable | Mean predicted length |
|---|---:|---:|---:|---:|---:|---:|
| A2-structured-baseline | 17 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A2-structured-baseline | 29 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A2-structured-baseline | 43 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-codebook | 17 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-codebook | 29 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-codebook | 43 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-zero | 17 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-zero | 29 | 0/2 | 0.000 | 0.000 | null | 0.00 |
| A3a-zero | 43 | 0/2 | 0.000 | 0.000 | null | 0.00 |

## Aggregate

```json
{
  "A2": {
    "action_applicable_rate": null,
    "applicable_action_count": 0,
    "attempted_action_count": 0,
    "end_only_rate": 1.0,
    "max_success_rate": 0.0,
    "mean_executability": 0.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "nonempty_plan_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "evaluation_unit_count": 6,
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 0.0,
        "mean_predicted_plan_length": 0,
        "seed_count": 3,
        "task_success_rate": 0.0,
        "unique_task_count": 2
      }
    },
    "success_counts": {
      "17": 0,
      "29": 0,
      "43": 0
    }
  },
  "A3": {
    "action_applicable_rate": null,
    "applicable_action_count": 0,
    "attempted_action_count": 0,
    "end_only_rate": 1.0,
    "max_success_rate": 0.0,
    "mean_executability": 0.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "nonempty_plan_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "evaluation_unit_count": 6,
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 0.0,
        "mean_predicted_plan_length": 0,
        "seed_count": 3,
        "task_success_rate": 0.0,
        "unique_task_count": 2
      }
    },
    "success_counts": {
      "17": 0,
      "29": 0,
      "43": 0
    }
  },
  "A4": {
    "action_applicable_rate": null,
    "applicable_action_count": 0,
    "attempted_action_count": 0,
    "end_only_rate": 1.0,
    "max_success_rate": 0.0,
    "mean_executability": 0.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "nonempty_plan_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "evaluation_unit_count": 6,
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 0.0,
        "mean_predicted_plan_length": 0,
        "seed_count": 3,
        "task_success_rate": 0.0,
        "unique_task_count": 2
      }
    },
    "success_counts": {
      "17": 0,
      "29": 0,
      "43": 0
    }
  }
}
```

## Paired comparisons

```json
[
  {
    "both_fail": 6,
    "both_succeed": 0,
    "first": "A3",
    "only_first_succeeds": 0,
    "only_second_succeeds": 0,
    "pair_count": 6,
    "second": "A2"
  },
  {
    "both_fail": 6,
    "both_succeed": 0,
    "first": "A4",
    "only_first_succeeds": 0,
    "only_second_succeeds": 0,
    "pair_count": 6,
    "second": "A2"
  },
  {
    "both_fail": 6,
    "both_succeed": 0,
    "first": "A3",
    "only_first_succeeds": 0,
    "only_second_succeeds": 0,
    "pair_count": 6,
    "second": "A4"
  }
]
```

## Детерминированно выбранные примеры

# Human-readable held-out examples



Development diagnostic only; not a Stage 2A semantic gate.



## bw-00000004 (seed 17)



Initial state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal: `[["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Gold/reference plan: `[["UNSTACK", "@B0", "@B1"], ["PUT_DOWN", "@B0"], ["UNSTACK", "@B1", "@B2"], ["STACK", "@B1", "@B0"], ["PICK_UP", "@B2"], ["STACK", "@B2", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:17d905961303942b6cd5d3b4cd032dfd2dad97b2ef0543ed35c1f3a48c4801d3`

Canonical checkpoint hash: `sha256:3c912f9552afa8fa8f90df845904287caf318e52268f9dec2c69f092aaf5a358`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:f47d490106595df3d45ecc6e1d67ea853ef6e4a0aacce5589fdf131213b2f7c8`

Canonical checkpoint hash: `sha256:602eb680b0e17faf3404e5ffbe01284f99b4c5477069bb3bdf04a1a7eaa93c8f`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:4c3a615f82650864fcf952d9fe035a462cbca24122745cc420d025af2ad8e66a`

Canonical checkpoint hash: `sha256:6b34cccfc5b027806076720ad9dc4fb27e72110840a3eeed91040f39076f0d6e`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`



## bw-00000005 (seed 17)



Initial state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal: `[["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Gold/reference plan: `[["UNSTACK", "@B2", "@B1"], ["PUT_DOWN", "@B2"], ["UNSTACK", "@B1", "@B0"], ["STACK", "@B1", "@B2"], ["PICK_UP", "@B0"], ["STACK", "@B0", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:de6fa3845d333a29f11a2dd25167ea65de6831878f773bed820dde58d8bae10d`

Canonical checkpoint hash: `sha256:3c912f9552afa8fa8f90df845904287caf318e52268f9dec2c69f092aaf5a358`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:519f748ab0e04432bbf62710845554144e473afdd4a71b00544be0a2be393135`

Canonical checkpoint hash: `sha256:602eb680b0e17faf3404e5ffbe01284f99b4c5477069bb3bdf04a1a7eaa93c8f`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:c37326f6327bd2b2fdb25fe5a7dafcf839dddbfaeb7608590ecd435783de7715`

Canonical checkpoint hash: `sha256:6b34cccfc5b027806076720ad9dc4fb27e72110840a3eeed91040f39076f0d6e`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`



## bw-00000004 (seed 29)



Initial state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal: `[["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Gold/reference plan: `[["UNSTACK", "@B0", "@B1"], ["PUT_DOWN", "@B0"], ["UNSTACK", "@B1", "@B2"], ["STACK", "@B1", "@B0"], ["PICK_UP", "@B2"], ["STACK", "@B2", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:17d905961303942b6cd5d3b4cd032dfd2dad97b2ef0543ed35c1f3a48c4801d3`

Canonical checkpoint hash: `sha256:9670e7ba14740ba2dcb0e5734c6e5ad54bf8872c55f959f94bbdfc58b0f5dd60`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:85e2e013ef2ea0bfdb31ae64a02791f7784d9aef2dc995e9cf76b245bde80f83`

Canonical checkpoint hash: `sha256:acc29f3bb36136b21a9502621c549cc17b6581faa26642d175a7ba4a5e2b680b`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:80e2b77cde21f67db344493fc2a57ae522dc7851524bad57372cb1ba1d968aa3`

Canonical checkpoint hash: `sha256:22c07023fcc1353c3b6f713dc75950aecfc8008430e1be54e00233e1fcdcf675`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`



## bw-00000005 (seed 29)



Initial state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal: `[["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Gold/reference plan: `[["UNSTACK", "@B2", "@B1"], ["PUT_DOWN", "@B2"], ["UNSTACK", "@B1", "@B0"], ["STACK", "@B1", "@B2"], ["PICK_UP", "@B0"], ["STACK", "@B0", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:de6fa3845d333a29f11a2dd25167ea65de6831878f773bed820dde58d8bae10d`

Canonical checkpoint hash: `sha256:9670e7ba14740ba2dcb0e5734c6e5ad54bf8872c55f959f94bbdfc58b0f5dd60`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:969c253ff54d0dafb84a96f4340a8f0949db85c3b4fa9f6ff703b8b77f15aa4e`

Canonical checkpoint hash: `sha256:acc29f3bb36136b21a9502621c549cc17b6581faa26642d175a7ba4a5e2b680b`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:4d5423acf9b63fcb50a3d16669611ddf3156ab2cf6fe605c9f95686af0edf840`

Canonical checkpoint hash: `sha256:22c07023fcc1353c3b6f713dc75950aecfc8008430e1be54e00233e1fcdcf675`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`



## bw-00000004 (seed 43)



Initial state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal: `[["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Gold/reference plan: `[["UNSTACK", "@B0", "@B1"], ["PUT_DOWN", "@B0"], ["UNSTACK", "@B1", "@B2"], ["STACK", "@B1", "@B0"], ["PICK_UP", "@B2"], ["STACK", "@B2", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:17d905961303942b6cd5d3b4cd032dfd2dad97b2ef0543ed35c1f3a48c4801d3`

Canonical checkpoint hash: `sha256:9fdcb0546a88b9402db0462dbbce71c3e8bf7b18eaa488f7c7bed494b56828ad`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:feaac8eec32575185b68819ffa84e3c6e707edf9a59ac06474fd6bfafac30f83`

Canonical checkpoint hash: `sha256:269068f1feed45b75bf1484b295880870b1a6c808e0d24fa2e3e3858ac6a4a3f`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:6c9efeab41974ae9228ce270108d69d419e5c92643499da673eef9ed5dc5bf6a`

Canonical checkpoint hash: `sha256:569c433276f009e08054a97bfeaa6dbd4303e9a0fb4a26ce6ad6728939e84d5c`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`



## bw-00000005 (seed 43)



Initial state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal: `[["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Gold/reference plan: `[["UNSTACK", "@B2", "@B1"], ["PUT_DOWN", "@B2"], ["UNSTACK", "@B1", "@B0"], ["STACK", "@B1", "@B2"], ["PICK_UP", "@B0"], ["STACK", "@B0", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:de6fa3845d333a29f11a2dd25167ea65de6831878f773bed820dde58d8bae10d`

Canonical checkpoint hash: `sha256:9fdcb0546a88b9402db0462dbbce71c3e8bf7b18eaa488f7c7bed494b56828ad`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:d90347a630f94885c8ccc91e957821d88bfd728b3aafd3e1f52d297a97b0e68f`

Canonical checkpoint hash: `sha256:269068f1feed45b75bf1484b295880870b1a6c808e0d24fa2e3e3858ac6a4a3f`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

END-only: `true`

Model forwards: `1`

Generated/attempted/applicable: `0/0/0`

Execution evidence hash: `sha256:d8fa278a51c71a596593801d01a6821a7c264a822c9844ea42ddc6b34f8f0551`

Canonical checkpoint hash: `sha256:569c433276f009e08054a97bfeaa6dbd4303e9a0fb4a26ce6ad6728939e84d5c`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`




## Ограничения

Все failures включены в denominator. Hyperparameters не подбирались после просмотра held-out результата. A3a-shuffled, A3a-foreign, A3s, A3b и Verbalizer не реализованы. Observed feedback application positions: 0. Observed nonzero feedback application positions: 0. Observed nonzero downstream semantic-component positions: 0. Because all runs terminated before a downstream semantic component was observed, this diagnostic is non-diagnostic for feedback-channel causality.

## Воспроизведение

```bash
python -m scripts.run_toy_quality_evaluation \
  --output-dir .quality-eval \
  --implementation-commit <IMPLEMENTATION_SHA>

python -m scripts.run_toy_quality_evaluation \
  --output-dir .quality-eval \
  --validate-only \
  --compact-dir docs/evaluations
```
