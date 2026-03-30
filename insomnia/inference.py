"""CSV → BAML → subtask_1.json / subtask_2.json (training submission shape)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


def _default_min_interval_sec() -> float:
    env = os.environ.get("GEMINI_MIN_INTERVAL_SEC", "").strip()
    if env:
        return max(0.0, float(env))
    return 12.0


def _resolve_min_interval_sec(
    min_interval_sec: float | None,
    rpm: float | None,
) -> float:
    if min_interval_sec is not None:
        return max(0.0, min_interval_sec)
    if rpm is not None:
        if rpm <= 0:
            return 0.0
        return 60.0 / rpm
    return _default_min_interval_sec()


class _RequestPacer:
    """Enforce a minimum time between the start of consecutive LLM calls."""

    def __init__(self, min_interval_sec: float) -> None:
        self._min = min_interval_sec
        self._last_start: float | None = None

    def before_request(self) -> None:
        if self._min <= 0:
            return
        now = time.monotonic()
        if self._last_start is not None:
            wait = self._min - (now - self._last_start)
            if wait > 0:
                time.sleep(wait)
        self._last_start = time.monotonic()


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
    min_interval_sec: float | None = None,
    rpm: float | None = None,
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
    pacer = _RequestPacer(_resolve_min_interval_sec(min_interval_sec, rpm))

    for _, row in df.iterrows():
        nid = row["note_id"]
        note_text = _text_cell(row["text"])
        pacer.before_request()
        cls = b.ClassifyInsomnia(note_text)
        subtask_1[nid] = {"Insomnia": cls.insomnia}
        pacer.before_request()
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
    pace = parser.add_mutually_exclusive_group()
    pace.add_argument(
        "--min-interval-sec",
        type=float,
        default=None,
        metavar="SEC",
        help=(
            "Minimum seconds between consecutive LLM request starts "
            "(default 12 unless GEMINI_MIN_INTERVAL_SEC is set; "
            "0 disables throttling for higher API limits)"
        ),
    )
    pace.add_argument(
        "--rpm",
        type=float,
        default=None,
        metavar="N",
        help="Cap ~N Gemini requests per minute (interval = 60/N; 0 or negative means no throttle)",
    )
    args = parser.parse_args(argv)
    try:
        run(
            args.input_csv,
            args.out_dir,
            max_rows=args.max_rows,
            min_interval_sec=args.min_interval_sec,
            rpm=args.rpm,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
