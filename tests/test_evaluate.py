"""Tests for Task 2 evaluation metrics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from insomnia.evaluate import evaluate


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


class TestEvaluate(unittest.TestCase):
    def _run_eval(
        self,
        gold1: dict,
        gold2: dict,
        pred1: dict,
        pred2: dict,
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            g1 = root / "g1.json"
            g2 = root / "g2.json"
            p1 = root / "p1.json"
            p2 = root / "p2.json"
            _write_json(g1, gold1)
            _write_json(g2, gold2)
            _write_json(p1, pred1)
            _write_json(p2, pred2)
            return evaluate(g1, g2, p1, p2)

    def test_perfect_match_scores_one(self) -> None:
        gold1 = {"n1": {"Insomnia": "yes"}, "n2": {"Insomnia": "no"}}
        gold2 = {
            "n1": {
                "Definition 1": {"label": "yes", "text": ["sleep onset insomnia"]},
                "Definition 2": {"label": "no", "text": []},
                "Rule B": {"label": "yes", "text": ["zolpidem"]},
                "Rule C": {"label": "no", "text": []},
            },
            "n2": {
                "Definition 1": {"label": "no", "text": []},
                "Definition 2": {"label": "no", "text": []},
                "Rule B": {"label": "no", "text": []},
                "Rule C": {"label": "no", "text": []},
            },
        }
        got = self._run_eval(gold1, gold2, gold1, gold2)
        self.assertAlmostEqual(got.subtask1_f1, 1.0)
        self.assertAlmostEqual(got.subtask2a_micro_f1, 1.0)
        self.assertAlmostEqual(got.subtask2b_rougeL_precision_macro, 1.0)
        self.assertAlmostEqual(got.subtask2b_rougeL_recall_macro, 1.0)
        self.assertAlmostEqual(got.subtask2b_rougeL_f1_macro, 1.0)

    def test_partial_predictions_reduce_scores(self) -> None:
        gold1 = {"n1": {"Insomnia": "yes"}, "n2": {"Insomnia": "no"}}
        pred1 = {"n1": {"Insomnia": "yes"}, "n2": {"Insomnia": "yes"}}
        gold2 = {
            "n1": {
                "Definition 1": {"label": "yes", "text": ["poor sleep quality"]},
                "Definition 2": {"label": "no", "text": []},
                "Rule B": {"label": "yes", "text": ["zolpidem use"]},
                "Rule C": {"label": "no", "text": []},
            },
            "n2": {
                "Definition 1": {"label": "no", "text": []},
                "Definition 2": {"label": "no", "text": []},
                "Rule B": {"label": "no", "text": []},
                "Rule C": {"label": "no", "text": []},
            },
        }
        pred2 = {
            "n1": {
                "Definition 1": {"label": "yes", "text": ["poor quality sleep"]},
                "Definition 2": {"label": "yes", "text": ["wrong positive"]},
                "Rule B": {"label": "no", "text": []},
                "Rule C": {"label": "no", "text": []},
            },
            "n2": {
                "Definition 1": {"label": "no", "text": []},
                "Definition 2": {"label": "no", "text": []},
                "Rule B": {"label": "yes", "text": ["hallucinated"]},
                "Rule C": {"label": "no", "text": []},
            },
        }
        got = self._run_eval(gold1, gold2, pred1, pred2)
        self.assertLess(got.subtask1_f1, 1.0)
        self.assertLess(got.subtask2a_micro_f1, 1.0)
        self.assertGreaterEqual(got.subtask2b_rougeL_f1_macro, 0.0)
        self.assertLess(got.subtask2b_rougeL_f1_macro, 1.0)

    def test_missing_note_ids_are_reported(self) -> None:
        gold1 = {"n1": {"Insomnia": "yes"}, "n2": {"Insomnia": "no"}}
        gold2 = {
            "n1": {"Definition 1": {"label": "yes", "text": ["insomnia symptoms"]}},
            "n2": {"Definition 1": {"label": "no", "text": []}},
        }
        pred1 = {"n1": {"Insomnia": "yes"}, "n3": {"Insomnia": "no"}}
        pred2 = {
            "n1": {"Definition 1": {"label": "yes", "text": ["insomnia symptoms"]}},
            "n3": {"Definition 1": {"label": "no", "text": []}},
        }
        got = self._run_eval(gold1, gold2, pred1, pred2)
        self.assertEqual(got.compared_note_ids, 1)
        self.assertEqual(got.missing_in_predictions, 1)
        self.assertEqual(got.extra_in_predictions, 1)


if __name__ == "__main__":
    unittest.main()
