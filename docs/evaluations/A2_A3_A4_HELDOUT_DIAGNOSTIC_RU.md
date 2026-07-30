# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Optional Git commit: `None`
- Evaluator source SHA256: `sha256:db735a8e050da6c63c47baf95d9cd482699f331906026277469af3e0d08cc4d8`
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

Execution evidence hash: `sha256:8d0f9826b46517999311daff3d3c230f7582b19642942e17d904af61c3493930`

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

Execution evidence hash: `sha256:68f821a7ad6f30c77a8698798f6845c71e9573edfc8f2792550a6466f7a0aad7`

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

Execution evidence hash: `sha256:50f9e1ea980d046ead25fc1defad10952ed3d0d5037ee0743fa3920efe04c38b`

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

Execution evidence hash: `sha256:b7535fbf493b930be6eca693aed75d7b2aeb976e083e730100c3ac3890e96655`

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

Execution evidence hash: `sha256:f6e736c553668d4c60810d567c6dfd0acbd89fd0c69572c0da6cee8b331cf62c`

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

Execution evidence hash: `sha256:62e9723f74c4802bc7736e8436a64f46504c2916fb18d99df931df7d303f717e`

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

Execution evidence hash: `sha256:a0cae5f59e8ed7598eeb50491a947cf6d7c9ee34bc75883086e7ac11d2502777`

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

Execution evidence hash: `sha256:b6564a186325520f04004358d172526ae50e478cb343504f85fa255a259b6011`

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

Execution evidence hash: `sha256:30bc51d8af6e70550346bdc49ab1a2bb2c4b8ebf9bee3d0be7979233e77e8477`

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

Execution evidence hash: `sha256:1355fcf2bebf14ab5c9c09f98eed98e0809e3d50d267a6d5ff6b35e9a2f71564`

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

Execution evidence hash: `sha256:a2e85dcbe2989c9602b187e819731a1b7987ccde4d19f07ec6d0346d59e7187b`

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

Execution evidence hash: `sha256:fb0c0b684bdca23e1cc4d00565c03e3fbfd0a8e6056359971d30fdc9a2a9ce6e`

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
