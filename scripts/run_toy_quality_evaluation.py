# ruff: noqa: E501
"""CLI for Development Quality Evaluation v0.1."""
from __future__ import annotations

import argparse
from pathlib import Path

from planner_toy.quality import SEEDS, VARIANTS, export_compact, run, validate_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-eval-tasks", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--compact-dir", type=Path)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--reuse-checkpoint-root", type=Path)
    args = parser.parse_args()
    if args.skip_training and args.reuse_checkpoint_root is None:
        parser.error("--skip-training requires --reuse-checkpoint-root")
    result = validate_evaluation(args.output_dir) if args.validate_only else run(args.output_dir, variants=args.variants, seeds=args.seeds, max_eval_tasks=args.max_eval_tasks, reuse_checkpoint_root=args.reuse_checkpoint_root if args.skip_training else None)
    print(result)
    if args.compact_dir:
        export_compact(args.output_dir, args.compact_dir, args.implementation_commit)

if __name__ == "__main__":
    main()
