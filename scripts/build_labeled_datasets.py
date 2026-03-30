#!/usr/bin/env python3
"""Build labeled artifacts for training/validation splits (JSONL + CSV)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

RULE_COLUMN_MAP = {
    "Definition 1": "def1_spans",
    "Definition 2": "def2_spans",
    "Rule B": "ruleb_spans",
    "Rule C": "rulec_spans",
}

SPLIT_DEFAULTS = {
    "training": {
        "corpus": "data/training/train_corpus.csv",
        "subtask1": "data/training/subtask_1.json",
        "subtask2": "data/training/subtask_2.json",
    },
    "validation": {
        "corpus": "data/validation/validation_corpus.csv",
        "subtask1": "data/validation/subtask_1.json",
        "subtask2": "data/validation/subtask_2.json",
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _validate_subtask2_schema(sub2: dict[str, dict], split_name: str) -> None:
    for nid, blocks in sub2.items():
        for rule_name in RULE_COLUMN_MAP:
            if rule_name not in blocks:
                raise SystemExit(
                    f"{split_name}: note_id={nid!r} missing block {rule_name!r} in subtask_2"
                )


def _build_split(
    *,
    split_name: str,
    corpus_path: Path,
    subtask1_path: Path,
    subtask2_path: Path,
    split_out_dir: Path,
) -> int:
    with subtask1_path.open(encoding="utf-8") as f:
        sub1: dict[str, dict] = json.load(f)
    with subtask2_path.open(encoding="utf-8") as f:
        sub2: dict[str, dict] = json.load(f)

    labeled_ids = set(sub1.keys()) | set(sub2.keys())
    if set(sub1.keys()) != set(sub2.keys()):
        only1 = set(sub1.keys()) - set(sub2.keys())
        only2 = set(sub2.keys()) - set(sub1.keys())
        raise SystemExit(
            f"{split_name}: subtask key mismatch: only in 1={only1!r} only in 2={only2!r}"
        )
    _validate_subtask2_schema(sub2, split_name)

    df = pd.read_csv(corpus_path, dtype={"note_id": str})
    if "note_id" not in df.columns or "text" not in df.columns:
        raise SystemExit(f"{split_name}: corpus must have columns note_id, text")

    df["note_id"] = df["note_id"].astype(str)
    labeled = df[df["note_id"].isin(labeled_ids)].copy()
    missing = labeled_ids - set(labeled["note_id"].tolist())
    if missing:
        raise SystemExit(
            f"{split_name}: {len(missing)} labeled note_ids missing from corpus, "
            f"e.g. {next(iter(missing))!r}"
        )

    split_out_dir.mkdir(parents=True, exist_ok=True)
    sub1_path = split_out_dir / "subtask1_labeled.jsonl"
    sub2_path = split_out_dir / "subtask2_labeled.jsonl"
    sub2_text_only_path = split_out_dir / "subtask2_labeled_text_only.jsonl"
    csv_path = split_out_dir / f"{split_name}_labeled.csv"

    csv_rows: list[dict[str, str]] = []
    with sub1_path.open("w", encoding="utf-8") as f1, sub2_path.open(
        "w", encoding="utf-8"
    ) as f2, sub2_text_only_path.open("w", encoding="utf-8") as f3:
        for _, row in labeled.iterrows():
            nid = row["note_id"]
            text = row["text"]
            if not isinstance(text, str):
                text = "" if pd.isna(text) else str(text)

            rec1 = {
                "note_id": nid,
                "text": text,
                "insomnia": sub1[nid]["Insomnia"],
            }
            f1.write(json.dumps(rec1, ensure_ascii=False) + "\n")

            labels_full = sub2[nid]
            rec2 = {"note_id": nid, "text": text, "labels": labels_full}
            f2.write(json.dumps(rec2, ensure_ascii=False) + "\n")

            labels_text_only = {}
            for block_name, block in labels_full.items():
                labels_text_only[block_name] = {
                    "label": block["label"],
                    "text": block.get("text", []),
                }
            rec3 = {"note_id": nid, "text": text, "labels": labels_text_only}
            f3.write(json.dumps(rec3, ensure_ascii=False) + "\n")

            csv_row = {
                "note_id": nid,
                "text": text,
                "subtask1_label": sub1[nid]["Insomnia"],
            }
            for rule_name, col_name in RULE_COLUMN_MAP.items():
                csv_row[col_name] = json.dumps(
                    labels_full[rule_name].get("span", []), ensure_ascii=False
                )
            csv_rows.append(csv_row)

    csv_df = pd.DataFrame(
        csv_rows,
        columns=[
            "note_id",
            "text",
            "subtask1_label",
            "def1_spans",
            "def2_spans",
            "ruleb_spans",
            "rulec_spans",
        ],
    )
    csv_df.to_csv(csv_path, index=False)

    print(f"[{split_name}] Wrote {len(labeled)} rows to {sub1_path}")
    print(f"[{split_name}] Wrote {len(labeled)} rows to {sub2_path}")
    print(f"[{split_name}] Wrote {len(labeled)} rows to {sub2_text_only_path}")
    print(f"[{split_name}] Wrote {len(labeled)} rows to {csv_path}")
    return len(labeled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=["training", "validation", "all"],
        default="all",
        help="Which split to build (default: all)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Override corpus CSV path (valid only when --split is training or validation)",
    )
    parser.add_argument(
        "--subtask1",
        type=Path,
        default=None,
        help="Override subtask_1.json path (valid only when --split is training or validation)",
    )
    parser.add_argument(
        "--subtask2",
        type=Path,
        default=None,
        help="Override subtask_2.json path (valid only when --split is training or validation)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output base directory (default: <repo>/outputs/labeled)",
    )
    args = parser.parse_args()
    root = _project_root()
    out_dir = args.out_dir or (root / "outputs/labeled")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "all" and (args.corpus or args.subtask1 or args.subtask2):
        raise SystemExit(
            "--corpus/--subtask1/--subtask2 overrides require --split training or validation"
        )

    splits = list(SPLIT_DEFAULTS.keys()) if args.split == "all" else [args.split]
    total = 0
    for split_name in splits:
        split_cfg = SPLIT_DEFAULTS[split_name]
        corpus_path = args.corpus or (root / split_cfg["corpus"])
        subtask1_path = args.subtask1 or (root / split_cfg["subtask1"])
        subtask2_path = args.subtask2 or (root / split_cfg["subtask2"])
        split_out_dir = out_dir / split_name

        total += _build_split(
            split_name=split_name,
            corpus_path=corpus_path,
            subtask1_path=subtask1_path,
            subtask2_path=subtask2_path,
            split_out_dir=split_out_dir,
        )

    print(f"Wrote {total} total labeled rows across {len(splits)} split(s)")


if __name__ == "__main__":
    main()
