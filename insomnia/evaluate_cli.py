"""CLI for evaluating Task 2 predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insomnia.evaluate import evaluate


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate SMM4H-HeaRD Task 2 outputs.")
    parser.add_argument(
        "--gold-subtask1",
        type=Path,
        default=root / "data/training/subtask_1.json",
        help="Gold labels for Subtask 1",
    )
    parser.add_argument(
        "--gold-subtask2",
        type=Path,
        default=root / "data/training/subtask_2.json",
        help="Gold labels for Subtask 2",
    )
    parser.add_argument(
        "--pred-subtask1",
        type=Path,
        default=root / "outputs/inference/subtask_1.json",
        help="Predictions for Subtask 1",
    )
    parser.add_argument(
        "--pred-subtask2",
        type=Path,
        default=root / "outputs/inference/subtask_2.json",
        help="Predictions for Subtask 2",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="train-or-validation",
        help="Label shown in output summary",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write metrics JSON",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="v0 - gemini-2.5-flash baseline",
        metavar="NAME",
        help=(
            "Top-level key for --json-out (metrics nested under that name, plus split_name inside). "
            "Pass empty string for a flat JSON object (metrics only)."
        ),
    )
    args = parser.parse_args()

    result = evaluate(
        gold_subtask1_path=args.gold_subtask1,
        gold_subtask2_path=args.gold_subtask2,
        pred_subtask1_path=args.pred_subtask1,
        pred_subtask2_path=args.pred_subtask2,
    )

    print(f"Split: {args.split_name}")
    print(f"Compared note_ids: {result.compared_note_ids}")
    print(f"Missing in predictions: {result.missing_in_predictions}")
    print(f"Extra in predictions: {result.extra_in_predictions}")
    print(f"Subtask 1 F1: {result.subtask1_f1:.6f}")
    print(f"Subtask 2A micro-F1: {result.subtask2a_micro_f1:.6f}")
    print(f"Subtask 2B macro ROUGE-L Precision: {result.subtask2b_rougeL_precision_macro:.6f}")
    print(f"Subtask 2B macro ROUGE-L Recall: {result.subtask2b_rougeL_recall_macro:.6f}")
    print(f"Subtask 2B macro ROUGE-L F1: {result.subtask2b_rougeL_f1_macro:.6f}")
    print(f"Subtask 2B scored items: {result.subtask2b_items}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {**result.to_dict(), "split_name": args.split_name}
        name = (args.experiment_name or "").strip()
        existing: dict = {}
        if args.json_out.exists():
            with args.json_out.open(encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded

        # Named experiment mode: keep a top-level map of experiment_name -> metrics.
        # Flat mode (empty experiment name): merge keys into the root object.
        out_obj: dict
        if name:
            out_obj = {**existing, name: payload}
        else:
            out_obj = {**existing, **payload}
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)
        print(f"Wrote metrics JSON: {args.json_out}")


if __name__ == "__main__":
    main()
