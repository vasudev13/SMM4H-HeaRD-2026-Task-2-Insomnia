"""Unit tests for quote→span alignment and optional gold consistency checks."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


from insomnia.spans import spans_from_quotes

ROOT = Path(__file__).resolve().parent.parent


def _span_pairs(span_field: str) -> list[tuple[int, int]]:
    """Parse one JSON ``span`` entry: may be ``\"a b\"`` or ``\"a b;c d\"``."""
    pairs: list[tuple[int, int]] = []
    for piece in span_field.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        parts = piece.split()
        if len(parts) != 2:
            raise ValueError(f"bad span fragment: {piece!r}")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


class TestSpansFromQuotes(unittest.TestCase):
    def test_sequential_first_match(self) -> None:
        note = "hello world hello"
        spans, texts = spans_from_quotes(note, ["hello", "hello"])
        self.assertEqual(spans, ["0 5", "12 17"])
        self.assertEqual(texts, ["hello", "hello"])

    def test_skips_empty_quote(self) -> None:
        note = "abc"
        spans, texts = spans_from_quotes(note, ["", "bc"])
        self.assertEqual(spans, ["1 3"])
        self.assertEqual(texts, ["bc"])

    def test_not_found_skipped(self) -> None:
        note = "short"
        spans, texts = spans_from_quotes(note, ["nope", "ort"])
        self.assertEqual(spans, ["2 5"])
        self.assertEqual(texts, ["ort"])


class TestGoldIntegrity(unittest.TestCase):
    """Sanity: stored spans slice to the stored text snippets."""

    @classmethod
    def setUpClass(cls) -> None:
        corpus_path = ROOT / "data/training/train_corpus.csv"
        if not corpus_path.exists():
            raise unittest.SkipTest(f"missing {corpus_path}")
        import pandas as pd

        df = pd.read_csv(corpus_path, dtype={"note_id": str})
        df["note_id"] = df["note_id"].astype(str)
        cls._text_by_id = dict(zip(df["note_id"], df["text"].map(lambda x: str(x) if x == x else "")))

        with (ROOT / "data/training/subtask_2.json").open(encoding="utf-8") as f:
            cls._sub2 = json.load(f)

    def test_gold_spans_match_snippets(self) -> None:
        for nid, blocks in self._sub2.items():
            note = self._text_by_id.get(nid)
            if note is None:
                continue
            for block_name, block in blocks.items():
                spans = block.get("span", [])
                texts = block.get("text", [])
                for sp in spans:
                    for a, b in _span_pairs(sp):
                        self.assertLessEqual(0, a < b <= len(note), msg=f"{nid}/{block_name}")
                        _ = note[a:b]
                if len(spans) == len(texts):
                    for sp, tx in zip(spans, texts):
                        pairs = _span_pairs(sp)
                        if len(pairs) == 1:
                            a, b = pairs[0]
                            self.assertEqual(note[a:b], tx, msg=f"{nid}/{block_name}")


class TestGoldQuoteRecovery(unittest.TestCase):
    """Optional: sequential quote policy recovers gold spans when quotes match document order."""

    @classmethod
    def setUpClass(cls) -> None:
        corpus_path = ROOT / "data/training/train_corpus.csv"
        if not corpus_path.exists():
            raise unittest.SkipTest(f"missing {corpus_path}")
        import pandas as pd

        df = pd.read_csv(corpus_path, dtype={"note_id": str})
        df["note_id"] = df["note_id"].astype(str)
        cls._text_by_id = dict(zip(df["note_id"], df["text"].map(lambda x: str(x) if x == x else "")))

        with (ROOT / "data/training/subtask_2.json").open(encoding="utf-8") as f:
            cls._sub2 = json.load(f)

    def test_recover_spans_for_gold_text_order(self) -> None:
        exact = 0
        total = 0
        for nid, blocks in self._sub2.items():
            note = self._text_by_id.get(nid)
            if note is None:
                continue
            for block_name, block in blocks.items():
                gold_spans = block.get("span", [])
                gold_texts = block.get("text", [])
                if not gold_texts:
                    continue
                total += 1
                got_spans, _ = spans_from_quotes(note, gold_texts)
                if got_spans == gold_spans:
                    exact += 1
        # Many gold files order quotes by character position; sequential policy matches often.
        self.assertGreater(total, 0)
        if exact < total * 0.5:
            self.fail(
                f"Expected most gold blocks to match sequential policy; got {exact}/{total} exact"
            )


if __name__ == "__main__":
    unittest.main()
