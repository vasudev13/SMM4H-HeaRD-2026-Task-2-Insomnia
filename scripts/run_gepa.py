#!/usr/bin/env python3
"""Run baseline vs GEPA-optimized prompt comparison on validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insomnia.evaluate import evaluate
from insomnia.gepa_optimize import (
    _evaluate_program,
    _extract_seed_prompt_from_baml,
    _project_root,
    _run_eval_from_dicts,
    load_examples,
    run_optimization,
)
from insomnia.inference import run as run_baseline_inference


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description="Run GEPA prompt optimization with baseline comparison."
    )
    parser.add_argument("--task-model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--reflection-model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--train-limit", type=int, default=30)
    parser.add_argument("--baseline-max-rows", type=int, default=None)
    parser.add_argument(
        "--baseline-neighbors",
        type=int,
        default=0,
        help="Few-shot neighbors for baseline inference (0 disables few-shot).",
    )
    parser.add_argument(
        "--baseline-out-dir",
        type=Path,
        default=root / "outputs/gepa/baseline",
    )
    parser.add_argument(
        "--gepa-out-dir",
        type=Path,
        default=root / "outputs/gepa/optimized",
    )
    args = parser.parse_args()

    validation_csv = root / "data/validation/validation_corpus.csv"
    val_st1_gold = root / "data/validation/subtask_1.json"
    val_st2_gold = root / "data/validation/subtask_2.json"

    args.baseline_out_dir.mkdir(parents=True, exist_ok=True)
    args.gepa_out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Baseline: current BAML pipeline
    run_baseline_inference(
        input_csv=validation_csv,
        out_dir=args.baseline_out_dir,
        max_rows=args.baseline_max_rows,
        use_few_shot=args.baseline_neighbors > 0,
        few_shot_k=max(1, args.baseline_neighbors),
    )
    baseline_eval = evaluate(
        gold_subtask1_path=val_st1_gold,
        gold_subtask2_path=val_st2_gold,
        pred_subtask1_path=args.baseline_out_dir / "subtask_1.json",
        pred_subtask2_path=args.baseline_out_dir / "subtask_2.json",
    )

    # 2) GEPA optimization
    seed_prompt = _extract_seed_prompt_from_baml(root / "baml_src/insomnia.baml")
    trainset = load_examples(
        root / "data/training/train_corpus.csv",
        root / "data/training/subtask_1.json",
        root / "data/training/subtask_2.json",
    )
    valset = load_examples(
        validation_csv,
        val_st1_gold,
        val_st2_gold,
    )
    optimized_program, best_prompt = run_optimization(
        trainset=trainset,
        valset=valset,
        seed_prompt=seed_prompt,
        task_model=args.task_model,
        reflection_model=args.reflection_model,
        max_metric_calls=args.max_metric_calls,
        train_limit=args.train_limit,
    )
    (args.gepa_out_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    optimized_program.save(str(args.gepa_out_dir / "optimized_program.json"))

    pred_st1, pred_st2 = _evaluate_program(optimized_program, valset)
    optimized_eval = _run_eval_from_dicts(
        gold_st1_path=val_st1_gold,
        gold_st2_path=val_st2_gold,
        pred_st1=pred_st1,
        pred_st2=pred_st2,
    )

    # 3) Comparison report
    comparison = {
        "baseline": baseline_eval.to_dict(),
        "optimized": optimized_eval.to_dict(),
        "delta": {
            "subtask1_f1": optimized_eval.subtask1_f1 - baseline_eval.subtask1_f1,
            "subtask2a_micro_f1": optimized_eval.subtask2a_micro_f1
            - baseline_eval.subtask2a_micro_f1,
            "subtask2b_rougeL_f1_macro": optimized_eval.subtask2b_rougeL_f1_macro
            - baseline_eval.subtask2b_rougeL_f1_macro,
        },
    }
    out_path = args.gepa_out_dir / "comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"Baseline Subtask 1 F1: {baseline_eval.subtask1_f1:.6f}")
    print(f"Optimized Subtask 1 F1: {optimized_eval.subtask1_f1:.6f}")
    print(f"Delta Subtask 1 F1: {comparison['delta']['subtask1_f1']:.6f}")
    print(f"Baseline Subtask 2A micro-F1: {baseline_eval.subtask2a_micro_f1:.6f}")
    print(f"Optimized Subtask 2A micro-F1: {optimized_eval.subtask2a_micro_f1:.6f}")
    print(
        f"Delta Subtask 2A micro-F1: {comparison['delta']['subtask2a_micro_f1']:.6f}"
    )
    print(
        "Baseline Subtask 2B macro ROUGE-L F1: "
        f"{baseline_eval.subtask2b_rougeL_f1_macro:.6f}"
    )
    print(
        "Optimized Subtask 2B macro ROUGE-L F1: "
        f"{optimized_eval.subtask2b_rougeL_f1_macro:.6f}"
    )
    print(
        "Delta Subtask 2B macro ROUGE-L F1: "
        f"{comparison['delta']['subtask2b_rougeL_f1_macro']:.6f}"
    )
    print(f"Wrote comparison report: {out_path}")


if __name__ == "__main__":
    main()
