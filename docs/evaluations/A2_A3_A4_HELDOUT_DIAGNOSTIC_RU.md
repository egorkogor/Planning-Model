# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Optional Git commit: `8f8c4bd37b86689d5e5d83b0e7cff117412c9621`
- Evaluator source SHA256: `sha256:0df6539ab7c72c3836fa05831e7598417051e1d8a9cae5358e977c70d05ee715`
- Implementation commit: `8f8c4bd37b86689d5e5d83b0e7cff117412c9621`
- Requirements lock: `sha256:883c8e262c6f3ea917239010be08ac0064ab67de70c1caad7bb98fe9f0b68401`
- Runtime: `{"python": "3.11.15", "torch": "2.12.0+cpu"}`
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

Execution evidence hash: `sha256:b85f561e67f80fa66ed0b84cf5e63b4c32ff33d9e6eab1c7cb4ca0ae8b765ce0`

Checkpoint hash: `sha256:c9bee3fc484e60042fc4d44bde0fbfb9bf3795989fe791fec21b59d5fc55d11f`

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

Execution evidence hash: `sha256:f3e8fcc62084294d9b72e7f62799acb9ee99aad47d9d141ad17105320997451e`

Checkpoint hash: `sha256:ccc75704dde7618dac6744a8ff23d74d124f0bd8803621a8268f9e1ac1a3b076`

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

Execution evidence hash: `sha256:baa1dcb26e9d0d32485cb9b662c736b887a524c5929733abca231a64e47ee7dc`

Checkpoint hash: `sha256:5add7d18fb018944e7226b626a015fa04ea501600fabe991731fa0d01d0468d7`

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

Execution evidence hash: `sha256:b9e67a730559b57125543c55abae857bd9c62c42b2029a6f8a70f4aff4b33405`

Checkpoint hash: `sha256:c9bee3fc484e60042fc4d44bde0fbfb9bf3795989fe791fec21b59d5fc55d11f`

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

Execution evidence hash: `sha256:e135a650c837cec03ab7846f7ae99b10d79cce758cd0c63c37f9c1e5c65b76e3`

Checkpoint hash: `sha256:ccc75704dde7618dac6744a8ff23d74d124f0bd8803621a8268f9e1ac1a3b076`

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

Execution evidence hash: `sha256:464538c0ad5185354a39584827eb5003c7063d1720d6c3559e4b7271d349afe7`

Checkpoint hash: `sha256:5add7d18fb018944e7226b626a015fa04ea501600fabe991731fa0d01d0468d7`

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

Execution evidence hash: `sha256:442772a5f5b35fd695bbe7e10911690c6ce4366f38bc01157af44023b1c2d6bd`

Checkpoint hash: `sha256:9d720279ec79e7586b3a9011f37dd503d195db04def95434a72722b497724237`

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

Execution evidence hash: `sha256:d13a4344b8127e2f1f6d9c885a0a0232a390ba5619eef4c5d00b11bdb7be289a`

Checkpoint hash: `sha256:9a2a65fa9054189cd5f4c4ef83fe8363bc7a8b77191a63cf1fe95012eeb87fc7`

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

Execution evidence hash: `sha256:2ff31631e8c8c4ea408fee86ee7c9439dc67831c9ea1627616eec06a7c74ae43`

Checkpoint hash: `sha256:83340d5522ea4e60e535193a01581974331f303311068b712402e3263e135937`

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

Execution evidence hash: `sha256:b90f490741d776d45bf5c16d1435c32c6d6b60ffe8a83eae18b5f1c35ef56735`

Checkpoint hash: `sha256:9d720279ec79e7586b3a9011f37dd503d195db04def95434a72722b497724237`

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

Execution evidence hash: `sha256:703f17eff79c7f0aac97f49b8a4d0211aed1a588a2a80c39ca21c504beb83b21`

Checkpoint hash: `sha256:9a2a65fa9054189cd5f4c4ef83fe8363bc7a8b77191a63cf1fe95012eeb87fc7`

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

Execution evidence hash: `sha256:0ccfce36f67b26d4df02ed80164fbbd1bd789f4df21d382e577d051aca6f5098`

Checkpoint hash: `sha256:83340d5522ea4e60e535193a01581974331f303311068b712402e3263e135937`

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

Execution evidence hash: `sha256:b3c673cede48dd620f30b1c959e230b1c098b9612b724adb89541f9eededb622`

Checkpoint hash: `sha256:1c2dcf4d70083f1259437e4482c84c20916afcf34910e4d1efd01c25be36dad1`

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

Execution evidence hash: `sha256:083a89f5119111aa94c80313ffe3855fa460d72dc65e5342c151731f093a180d`

Checkpoint hash: `sha256:41239e44bad9ddd550011b9706825b1d6af5eaf12a43d1bbabdc95b6b78ad8e5`

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

Execution evidence hash: `sha256:6a4b90387497e5ffdc5beee3f82ae3618a9d8bd3ae47d609e3fdbb72d5f98cbb`

Checkpoint hash: `sha256:04a69af34921558fafd3bd419505a0d2a0e5e23bf3bd94ec90160c5fa90d272c`

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

Execution evidence hash: `sha256:cac958a942367de958909d5ce2ab1a7a7040da78fc442c4a7943db2463ec9d76`

Checkpoint hash: `sha256:1c2dcf4d70083f1259437e4482c84c20916afcf34910e4d1efd01c25be36dad1`

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

Execution evidence hash: `sha256:35fbd325cd88e972aa84c67fd3f42713a9fd78bdc8bee0603714b90f70acfe85`

Checkpoint hash: `sha256:41239e44bad9ddd550011b9706825b1d6af5eaf12a43d1bbabdc95b6b78ad8e5`

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

Execution evidence hash: `sha256:d9f6958d6dadc3aa37ef74fbe9cb0be08c8197a5c93631878b481f38338f4722`

Checkpoint hash: `sha256:04a69af34921558fafd3bd419505a0d2a0e5e23bf3bd94ec90160c5fa90d272c`

Execution: `[]`

Initial goal satisfied: `false`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`




## Ограничения

Все failures включены в denominator. Hyperparameters не подбирались после просмотра held-out результата. A3a-shuffled, A3a-foreign, A3s, A3b и Verbalizer не реализованы.

## Воспроизведение

```bash
python -m scripts.run_toy_quality_evaluation --output-dir .quality-eval
python -m scripts.run_toy_quality_evaluation --output-dir .quality-eval --validate-only
```
