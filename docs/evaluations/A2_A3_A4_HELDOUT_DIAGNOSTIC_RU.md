# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Evaluator source SHA256: `sha256:8dab8fc734ea17fc583148e3f289ae4eb9a9f7ec6eb97a1ea53750b3fe1cc93b`
- Implementation commit: `ef8ff7b630c11cea103f777d1ccadfa73d09321c`
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

Execution evidence hash: `sha256:2d5ac82f6f5c59a6a206df19d9ea8a71e596e041fcd5f11d434b813ec92f68cd`

Checkpoint hash: `sha256:1aefe1aaaa7421f4db277777c690d65185835416baeb8273ed5bc15e1737027d`

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

Execution evidence hash: `sha256:488236514fc28c717c2fd703c37b4a5dcf7dd3418b7edaa0a15a7e2f359ac56c`

Checkpoint hash: `sha256:cf647d8593bbfcdbb95ae6fe91e54e398058842a6b7cc9b6069b3351fb0db8a7`

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

Execution evidence hash: `sha256:d25ca0df852a199a332eee8b71f5a8806388ac0c16c0c1fb034057f7507f4a72`

Checkpoint hash: `sha256:4859805f699fd004cb064f3be36256004e114bbb86401a906566e16f13638cb9`

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

Execution evidence hash: `sha256:26aa5500e3f246606d16ff12bf80b34c641d77c53ebddff90bab6ea7b13b91ee`

Checkpoint hash: `sha256:1aefe1aaaa7421f4db277777c690d65185835416baeb8273ed5bc15e1737027d`

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

Execution evidence hash: `sha256:34eea180178072b7c261286fec42b0f2926e8bcbb84a845b4dcca693a26f39a2`

Checkpoint hash: `sha256:cf647d8593bbfcdbb95ae6fe91e54e398058842a6b7cc9b6069b3351fb0db8a7`

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

Execution evidence hash: `sha256:2d8a4eb9c033f272116e96ffc88074bcdaa4da5419e577394bacc54f82918c23`

Checkpoint hash: `sha256:4859805f699fd004cb064f3be36256004e114bbb86401a906566e16f13638cb9`

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

Execution evidence hash: `sha256:7283723719c42af7ca40c6b6294c717740313557789782fe24c846f78c6a69b4`

Checkpoint hash: `sha256:e9ed494e452f7921ff0d18b0a3304c1f8ab8a89f83bf009a0dcafc69f9f56830`

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

Execution evidence hash: `sha256:5d1e302cb6faa7c2f69c6079d537ae9a28f4328bb66c52d7ac95eaea8f7fd9e0`

Checkpoint hash: `sha256:4620a077a0d5620298ef5b6df52c6456beb77e6fe0d96940778af1b9361eb6d5`

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

Execution evidence hash: `sha256:d5251927f14568f25a29a1a03510d222d260c334c6cb6a8700c2b42cfa60152c`

Checkpoint hash: `sha256:210939a2a363705119d596ca5b311b18d58e60882f67ce03b916783fd4268ba9`

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

Execution evidence hash: `sha256:8f75d60dfc0d6ea1d2af66f39dfe31e8b68e08c4660b188238d16db5fd52db25`

Checkpoint hash: `sha256:e9ed494e452f7921ff0d18b0a3304c1f8ab8a89f83bf009a0dcafc69f9f56830`

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

Execution evidence hash: `sha256:416344af1d3cc1c1b590e4e0f1b6881f4a5b0b5f889e40155dd03d80e68f5fca`

Checkpoint hash: `sha256:4620a077a0d5620298ef5b6df52c6456beb77e6fe0d96940778af1b9361eb6d5`

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

Execution evidence hash: `sha256:03b23073e63e878b2e6d4bf7044e5d18b61b2646753c261845a3484feb953e10`

Checkpoint hash: `sha256:210939a2a363705119d596ca5b311b18d58e60882f67ce03b916783fd4268ba9`

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

Execution evidence hash: `sha256:d430def184c91cdb868709b04e82962ed030c90c89ffc1311c29a013d985480a`

Checkpoint hash: `sha256:587a0def4e63619fa21e59c4f03ec2c7f55084fcf9832bdf21544445a4fdc49b`

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

Execution evidence hash: `sha256:6ae087bfd259a19b6e8e64b7c2d27d72f9758d16c815d1746f7183e636ba0e3a`

Checkpoint hash: `sha256:b7e07d708076110e63f89bedce9d65d624d47c28978818fd5ff40666ffad454b`

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

Execution evidence hash: `sha256:90fda5080b55ffc0364eef9702ddb12c3ba7ed01627b47d56861810c1ce1ffa4`

Checkpoint hash: `sha256:7daec6553785464eb9ffa4425d962084fb5babdcc534ae8d166e04de01244e1e`

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

Execution evidence hash: `sha256:d792cee50fa61e9e524974e23cef8ff8e74b6642713d78e9eecef1c258f82cff`

Checkpoint hash: `sha256:587a0def4e63619fa21e59c4f03ec2c7f55084fcf9832bdf21544445a4fdc49b`

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

Execution evidence hash: `sha256:a714d7db21dda6f08bca5e8764ee852ff65731b7216d4afc881a3852cdacd77e`

Checkpoint hash: `sha256:b7e07d708076110e63f89bedce9d65d624d47c28978818fd5ff40666ffad454b`

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

Execution evidence hash: `sha256:e18a0451f0f9bf8650ac8a5de8027a063b682b0b7ece564de8a60e96c4de7a39`

Checkpoint hash: `sha256:7daec6553785464eb9ffa4425d962084fb5babdcc534ae8d166e04de01244e1e`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`




## Ограничения

Все failures включены в denominator. Hyperparameters не подбирались после просмотра held-out результата. A3a-shuffled, A3a-foreign, A3s, A3b и Verbalizer не реализованы. Observed feedback application positions: 0. Because every run terminated at the first END token, this diagnostic is non-diagnostic for feedback-channel causality and for the difference between active feedback and compute-then-zero at downstream decoding positions.

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
