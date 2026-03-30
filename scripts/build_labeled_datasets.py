#!/usr/bin/env python3
"""Join train_corpus.csv with subtask_1 / subtask_2 labels and write JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Training corpus CSV (default: <repo>/data/training/train_corpus.csv)",
    )
    parser.add_argument(
        "--subtask1",
        type=Path,
        default=None,
        help="subtask_1.json path",
    )
    parser.add_argument(
        "--subtask2",
        type=Path,
        default=None,
        help="subtask_2.json path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/outputs/labeled)",
    )
    args = parser.parse_args()
    root = _project_root()
    corpus_path = args.corpus or (root / "data/training/train_corpus.csv")
    p1 = args.subtask1 or (root / "data/training/subtask_1.json")
    p2 = args.subtask2 or (root / "data/training/subtask_2.json")
    out_dir = args.out_dir or (root / "outputs/labeled")
    out_dir.mkdir(parents=True, exist_ok=True)

    with p1.open(encoding="utf-8") as f:
        sub1: dict[str, dict] = json.load(f)
    with p2.open(encoding="utf-8") as f:
        sub2: dict[str, dict] = json.load(f)

    labeled_ids = set(sub1.keys()) | set(sub2.keys())
    if set(sub1.keys()) != set(sub2.keys()):
        only1 = set(sub1.keys()) - set(sub2.keys())
        only2 = set(sub2.keys()) - set(sub1.keys())
        raise SystemExit(f"subtask key mismatch: only in 1={only1!r} only in 2={only2!r}")

    df = pd.read_csv(corpus_path, dtype={"note_id": str})
    if "note_id" not in df.columns or "text" not in df.columns:
        raise SystemExit("corpus must have columns note_id, text")

    df["note_id"] = df["note_id"].astype(str)
    labeled = df[df["note_id"].isin(labeled_ids)].copy()
    missing = labeled_ids - set(labeled["note_id"].tolist())
    if missing:
        raise SystemExit(f"{len(missing)} labeled note_ids missing from corpus, e.g. {next(iter(missing))!r}")

    sub1_path = out_dir / "subtask1_labeled.jsonl"
    sub2_path = out_dir / "subtask2_labeled.jsonl"
    sub2_text_only_path = out_dir / "subtask2_labeled_text_only.jsonl"

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

    print(f"Wrote {len(labeled)} rows to {sub1_path}")
    print(f"Wrote {len(labeled)} rows to {sub2_path}")
    print(f"Wrote {len(labeled)} rows to {sub2_text_only_path}")


if __name__ == "__main__":
    main()
