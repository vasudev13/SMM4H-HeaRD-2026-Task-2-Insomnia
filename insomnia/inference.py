"""CSV → BAML → subtask_1.json / subtask_2.json (training submission shape)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from baml_client import b
from baml_client.types import InsomniaEvidenceExtraction, RuleEvidence

from insomnia.spans import rule_block_from_quotes

SUBTASK2_BLOCK_ATTRS: tuple[tuple[str, str], ...] = (
    ("definition_1", "Definition 1"),
    ("definition_2", "Definition 2"),
    ("rule_b", "Rule B"),
    ("rule_c", "Rule C"),
)


def _text_cell(x: object) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


def extraction_to_subtask2_block(note_text: str, ext: InsomniaEvidenceExtraction) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for attr, display in SUBTASK2_BLOCK_ATTRS:
        block: RuleEvidence | None = getattr(ext, attr, None)
        if block is None:
            out[display] = {"label": "no", "span": [], "text": []}
            continue
        quotes = list(block.evidence_quotes or [])
        label = block.label
        out[display] = rule_block_from_quotes(label, quotes, note_text)
    return out


def run(
    input_csv: Path,
    out_dir: Path,
    *,
    max_rows: int | None = None,
) -> None:
    df = pd.read_csv(input_csv, dtype={"note_id": str})
    if "note_id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns: note_id, text")
    df["note_id"] = df["note_id"].astype(str)
    if max_rows is not None:
        df = df.head(max_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    subtask_1: dict[str, dict] = {}
    subtask_2: dict[str, dict] = {}

    for _, row in df.iterrows():
        nid = row["note_id"]
        note_text = _text_cell(row["text"])
        cls = b.ClassifyInsomnia(note_text)
        subtask_1[nid] = {"Insomnia": cls.insomnia}
        ext = b.ExtractInsomniaEvidence(note_text)
        subtask_2[nid] = extraction_to_subtask2_block(note_text, ext)

    p1 = out_dir / "subtask_1.json"
    p2 = out_dir / "subtask_2.json"
    with p1.open("w", encoding="utf-8") as f:
        json.dump(subtask_1, f, ensure_ascii=False, indent=4)
    with p2.open("w", encoding="utf-8") as f:
        json.dump(subtask_2, f, ensure_ascii=False, indent=4)
    print(f"Wrote {p1} ({len(subtask_1)} notes)")
    print(f"Wrote {p2} ({len(subtask_2)} notes)")


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run BAML classification + extraction; write subtask JSON files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root / "data/validation/validation_corpus.csv",
        help="Input corpus CSV with note_id and text",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "outputs/inference",
        help="Directory for subtask_1.json and subtask_2.json",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process only the first N rows (for smoke tests)",
    )
    args = parser.parse_args(argv)
    try:
        run(args.input_csv, args.out_dir, max_rows=args.max_rows)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
