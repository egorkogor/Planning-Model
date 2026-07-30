# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Implementation commit at generation: `f9ab208dca1801739ef1f287bb6336848bf0bb9d`
- Evaluator: `development-quality-evaluation/0.1`
- Dataset hash: `sha256:60e4ce06d6cfc90dc467fb4e82b2eb71cf2d92d37471eee3aeda64f864c541df`
- Train tasks: 3; held-out tasks: 2
- Seeds: 17, 29, 43
- Budget: 3 epochs × 3 canonical train tasks = 9 updates/run; final checkpoint only

## Variant × seed

| Variant | Seed | Success | Rate | Executable | Action applicable | Mean predicted length |
|---|---:|---:|---:|---:|---:|---:|
| A2-structured-baseline | 17 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A2-structured-baseline | 29 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A2-structured-baseline | 43 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-codebook | 17 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-codebook | 29 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-codebook | 43 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-zero | 17 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-zero | 29 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |
| A3a-zero | 43 | 0/2 | 0.000 | 1.000 | 0.000 | 0.00 |

## Aggregate

```json
{
  "A2": {
    "max_success_rate": 0.0,
    "mean_action_validity": 0.0,
    "mean_executability": 1.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 1.0,
        "heldout_task_count": 6,
        "mean_predicted_plan_length": 0,
        "task_success_rate": 0.0
      }
    },
    "success_counts": {
      "17": 0,
      "29": 0,
      "43": 0
    }
  },
  "A3": {
    "max_success_rate": 0.0,
    "mean_action_validity": 0.0,
    "mean_executability": 1.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 1.0,
        "heldout_task_count": 6,
        "mean_predicted_plan_length": 0,
        "task_success_rate": 0.0
      }
    },
    "success_counts": {
      "17": 0,
      "29": 0,
      "43": 0
    }
  },
  "A4": {
    "max_success_rate": 0.0,
    "mean_action_validity": 0.0,
    "mean_executability": 1.0,
    "mean_plan_length_difference": 6,
    "mean_success_rate": 0.0,
    "min_success_rate": 0.0,
    "structural_breakdown": {
      "5+": {
        "failure_code_distribution": {
          "GOAL_NOT_ACHIEVED": 6
        },
        "fully_executable_plan_rate": 1.0,
        "heldout_task_count": 6,
        "mean_predicted_plan_length": 0,
        "task_success_rate": 0.0
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

Execution: `[]`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

Execution: `[]`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

Execution: `[]`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B0"], ["HAND_EMPTY"], ["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Goal reached: `false`



## bw-00000005 (seed 17)



Initial state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal: `[["ON", "@B0", "@B1"], ["ON", "@B1", "@B2"], ["ON_TABLE", "@B2"]]`

Gold/reference plan: `[["UNSTACK", "@B2", "@B1"], ["PUT_DOWN", "@B2"], ["UNSTACK", "@B1", "@B0"], ["STACK", "@B1", "@B2"], ["PICK_UP", "@B0"], ["STACK", "@B0", "@B1"]]`

### A2-structured-baseline

Predicted plan: `[]`

Execution: `[]`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-codebook

Predicted plan: `[]`

Execution: `[]`

Failure: `GOAL_NOT_ACHIEVED`

Final state: `[["CLEAR", "@B2"], ["HAND_EMPTY"], ["ON", "@B1", "@B0"], ["ON", "@B2", "@B1"], ["ON_TABLE", "@B0"]]`

Goal reached: `false`

### A3a-zero

Predicted plan: `[]`

Execution: `[]`

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
