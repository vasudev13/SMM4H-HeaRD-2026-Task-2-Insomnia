"""Evaluation helpers for SMM4H-HeaRD Task 2 outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

EVIDENCE_RULES: tuple[str, ...] = ("Definition 1", "Definition 2", "Rule B", "Rule C")


def _load_json(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at {path}, got {type(data).__name__}")
    return data


def _yn(value: object) -> str:
    s = str(value).strip().lower()
    return "yes" if s == "yes" else "no"


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[-1]


def _rouge_l_prf(reference: str, prediction: str) -> tuple[float, float, float]:
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    lcs = _lcs_len(ref_tokens, pred_tokens)
    precision = lcs / len(pred_tokens) if pred_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return precision, recall, f1


def _text_from_block(block: dict | None) -> str:
    if not isinstance(block, dict):
        return ""
    text = block.get("text", [])
    if isinstance(text, list):
        return " ".join(str(x).strip() for x in text if str(x).strip())
    if isinstance(text, str):
        return text.strip()
    return ""


@dataclass(frozen=True)
class EvaluationResult:
    subtask1_f1: float
    subtask2a_micro_f1: float
    subtask2b_rougeL_precision_macro: float
    subtask2b_rougeL_recall_macro: float
    subtask2b_rougeL_f1_macro: float
    compared_note_ids: int
    missing_in_predictions: int
    extra_in_predictions: int
    subtask2b_items: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "subtask1_f1": self.subtask1_f1,
            "subtask2a_micro_f1": self.subtask2a_micro_f1,
            "subtask2b_rougeL_precision_macro": self.subtask2b_rougeL_precision_macro,
            "subtask2b_rougeL_recall_macro": self.subtask2b_rougeL_recall_macro,
            "subtask2b_rougeL_f1_macro": self.subtask2b_rougeL_f1_macro,
            "compared_note_ids": self.compared_note_ids,
            "missing_in_predictions": self.missing_in_predictions,
            "extra_in_predictions": self.extra_in_predictions,
            "subtask2b_items": self.subtask2b_items,
        }


def evaluate(
    gold_subtask1_path: Path,
    gold_subtask2_path: Path,
    pred_subtask1_path: Path,
    pred_subtask2_path: Path,
) -> EvaluationResult:
    gold1 = _load_json(gold_subtask1_path)
    gold2 = _load_json(gold_subtask2_path)
    pred1 = _load_json(pred_subtask1_path)
    pred2 = _load_json(pred_subtask2_path)

    gold_ids = set(gold1.keys()) & set(gold2.keys())
    pred_ids = set(pred1.keys()) & set(pred2.keys())
    common_ids = sorted(gold_ids & pred_ids)

    # Subtask 1: binary F1, "yes" positive class.
    tp1 = fp1 = fn1 = 0
    for nid in common_ids:
        y_true = _yn(gold1.get(nid, {}).get("Insomnia", "no"))
        y_pred = _yn(pred1.get(nid, {}).get("Insomnia", "no"))
        if y_pred == "yes" and y_true == "yes":
            tp1 += 1
        elif y_pred == "yes" and y_true == "no":
            fp1 += 1
        elif y_pred == "no" and y_true == "yes":
            fn1 += 1
    subtask1_f1 = _safe_f1(tp1, fp1, fn1)

    # Subtask 2A: micro-F1 over binary decisions for all available rule labels per note.
    tp2a = fp2a = fn2a = 0
    for nid in common_ids:
        g_blocks = gold2.get(nid, {})
        p_blocks = pred2.get(nid, {})
        if not isinstance(g_blocks, dict):
            g_blocks = {}
        if not isinstance(p_blocks, dict):
            p_blocks = {}
        rule_keys = sorted(set(g_blocks.keys()) | set(p_blocks.keys()))
        for rule in rule_keys:
            y_true = _yn((g_blocks.get(rule) or {}).get("label", "no"))
            y_pred = _yn((p_blocks.get(rule) or {}).get("label", "no"))
            if y_pred == "yes" and y_true == "yes":
                tp2a += 1
            elif y_pred == "yes" and y_true == "no":
                fp2a += 1
            elif y_pred == "no" and y_true == "yes":
                fn2a += 1
    subtask2a_micro_f1 = _safe_f1(tp2a, fp2a, fn2a)

    # Subtask 2B: macro-average ROUGE-L over evidence rules where gold label is "yes".
    # This avoids crediting empty-empty pairs for non-evidence negatives.
    p_sum = r_sum = f_sum = 0.0
    items = 0
    for nid in common_ids:
        g_blocks = gold2.get(nid, {})
        p_blocks = pred2.get(nid, {})
        if not isinstance(g_blocks, dict):
            continue
        if not isinstance(p_blocks, dict):
            p_blocks = {}
        for rule in EVIDENCE_RULES:
            g_block = g_blocks.get(rule, {})
            if _yn((g_block or {}).get("label", "no")) != "yes":
                continue
            g_text = _text_from_block(g_block)
            p_text = _text_from_block(p_blocks.get(rule, {}))
            p_val, r_val, f_val = _rouge_l_prf(g_text, p_text)
            p_sum += p_val
            r_sum += r_val
            f_sum += f_val
            items += 1

    if items:
        rouge_p = p_sum / items
        rouge_r = r_sum / items
        rouge_f = f_sum / items
    else:
        rouge_p = rouge_r = rouge_f = 0.0

    missing_in_predictions = len(gold_ids - pred_ids)
    extra_in_predictions = len(pred_ids - gold_ids)
    return EvaluationResult(
        subtask1_f1=subtask1_f1,
        subtask2a_micro_f1=subtask2a_micro_f1,
        subtask2b_rougeL_precision_macro=rouge_p,
        subtask2b_rougeL_recall_macro=rouge_r,
        subtask2b_rougeL_f1_macro=rouge_f,
        compared_note_ids=len(common_ids),
        missing_in_predictions=missing_in_predictions,
        extra_in_predictions=extra_in_predictions,
        subtask2b_items=items,
    )
