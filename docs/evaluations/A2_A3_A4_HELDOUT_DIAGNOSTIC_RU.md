# Диагностика качества A2/A3/A4 на held-out задачах

> Development-only diagnostic. Это не confirmatory experiment, не прохождение Stage 2A semantic gate, не доказательство semantic reasoning или superiority A3 и не разрешение A3b.

- Optional Git commit: `None`
- Evaluator source SHA256: `sha256:5db66277abdf55901f978b26c424c58f50ce092cdaa42aa6419e99bc72423d42`
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

Execution evidence hash: `sha256:4ffaabff99242f2f1bc9586082bc4466aefa75930b329aa38fc2a5e4ada00b42`

Checkpoint hash: `sha256:7ccad001eee657c9dca1a7211e486baeac5b9f452635ed55bbf819eb98c5a5d3`

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

Execution evidence hash: `sha256:862039734c042797cdc8f18c2035f642138720e23418298c50afbdbcfa5616a7`

Checkpoint hash: `sha256:10e93ed28c519a8324fc4c54900736c224b6f5abd5915d9e56e0d49261a52412`

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

Execution evidence hash: `sha256:402bf901a076bf5d027c527a807df1244ed1e65ad3deb2ec6b6afc798966799d`

Checkpoint hash: `sha256:8d6a306ba0ed3665251723aa407cb6d5614d710735e9b50e1995fe2866bc7ca7`

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

Execution evidence hash: `sha256:7468f89e7893e4dcbc0c8b7a18c6338bc5a3d93795887f105ddcd58bfbc9cfb7`

Checkpoint hash: `sha256:7ccad001eee657c9dca1a7211e486baeac5b9f452635ed55bbf819eb98c5a5d3`

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

Execution evidence hash: `sha256:e1b2ff4ca2617af4af20d1525eaa9122736eaf14d9e67ac0dcbb9ba98cd0ac04`

Checkpoint hash: `sha256:10e93ed28c519a8324fc4c54900736c224b6f5abd5915d9e56e0d49261a52412`

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

Execution evidence hash: `sha256:9ac239e4d7c5169950c6d5ffd5d99b173f5296a235e8cb61a8a19a1d001221cf`

Checkpoint hash: `sha256:8d6a306ba0ed3665251723aa407cb6d5614d710735e9b50e1995fe2866bc7ca7`

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

Execution evidence hash: `sha256:62b85b21ae0e2f95d001275fef4b8010d7cbfab97f479b6c83ff7249115e0955`

Checkpoint hash: `sha256:a3fa60784456d335a7695651e921901e5ef957fb58cee32b08f2fef35260dcef`

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

Execution evidence hash: `sha256:ca05940a4d527e5cec5f678dfad186827c9156b823ecc92389147cc85f69aae0`

Checkpoint hash: `sha256:de8bfb81da57d7474d9915e3ece6a6654b5a941a7da436b8331ad6fd1f173a0e`

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

Execution evidence hash: `sha256:d7fad63745c2a4e47addb0a3a334651da3e545dd1436b9e017d4b810d62333e9`

Checkpoint hash: `sha256:feba66fe986912cbf40a9d9ab110300bd8c192bb5547d3460cbc3c9a3cc6ffff`

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

Execution evidence hash: `sha256:02a0a6cc17717614b27bca1240e38dc3960688dc971e5858e68c348d90db8d81`

Checkpoint hash: `sha256:a3fa60784456d335a7695651e921901e5ef957fb58cee32b08f2fef35260dcef`

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

Execution evidence hash: `sha256:0e19a485e90c5776c69cdd85e81de2a6b477e95b9c7bf53b8148e6b9bc4717c4`

Checkpoint hash: `sha256:de8bfb81da57d7474d9915e3ece6a6654b5a941a7da436b8331ad6fd1f173a0e`

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

Execution evidence hash: `sha256:246d1631b0863d4b3845dc6922cba1c396a059e919e3111d39e413a14f326d50`

Checkpoint hash: `sha256:feba66fe986912cbf40a9d9ab110300bd8c192bb5547d3460cbc3c9a3cc6ffff`

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

Execution evidence hash: `sha256:39d9f3f448d6a9911d1113d92536d6266c8dc5e2c511a3683bd6626b0c9ed308`

Checkpoint hash: `sha256:5e5b8020ee3ea60fc9a1ab7ad5c6c1ff74a8bcc42a4a941b039262090248425f`

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

Execution evidence hash: `sha256:4605d2700c658a5c8c4a151d568a69b7176b08a8138fd35008412c2358aee530`

Checkpoint hash: `sha256:bedccfd0137cfecfbfb31d568b709328f4b48a301ab4e1682d1785afd9665b7a`

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

Execution evidence hash: `sha256:ce892028aca85991df82c051fd21d3cab5b76df3930032eb50c1a0b10e7914cb`

Checkpoint hash: `sha256:d1211eb94449dddf65464ed90f3c3b8456ec69f4313897af3bd7d55497f8014b`

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

Execution evidence hash: `sha256:10755baf7fdfbf21923439592cd9a23d8f6d0bb154b943117a41c90e5e2ac9a6`

Checkpoint hash: `sha256:5e5b8020ee3ea60fc9a1ab7ad5c6c1ff74a8bcc42a4a941b039262090248425f`

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

Execution evidence hash: `sha256:cee6d47cd3d8371d65942aa0ba3a0c16fb7423399c13215f9810e825669e7262`

Checkpoint hash: `sha256:bedccfd0137cfecfbfb31d568b709328f4b48a301ab4e1682d1785afd9665b7a`

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

Execution evidence hash: `sha256:5379fb6dcf0c65d3eada85a053617fd61b62966972c7b2dfbc847eac925e0b4c`

Checkpoint hash: `sha256:d1211eb94449dddf65464ed90f3c3b8456ec69f4313897af3bd7d55497f8014b`

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
