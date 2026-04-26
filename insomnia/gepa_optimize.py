"""GEPA prompt optimization for SMM4H-HeaRD Task 2.

This module mirrors the two-pass pipeline used in `insomnia.inference`:
1) LLM returns structured extraction JSON
2) deterministic rules derive labels and submission blocks

GEPA evolves the extraction instructions while keeping deterministic logic intact.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dspy
import pandas as pd
from pydantic import ValidationError

from baml_client.types import ClinicalNoteExtraction
from insomnia.evaluate import EvaluationResult, evaluate
from insomnia.evaluate import _rouge_l_prf as rouge_l_prf
from insomnia.evaluate import _yn as yn
from insomnia.inference import derive_labels, format_submission


class ClinicalEvidenceSignature(dspy.Signature):
    """Extract structured clinical evidence for insomnia detection."""

    note_text: str = dspy.InputField(desc="Clinical note text")
    extraction_json: str = dspy.OutputField(
        desc="JSON object matching ClinicalNoteExtraction schema"
    )


class InsomniaPipeline(dspy.Module):
    """Single-predictor DSPy module so GEPA can evolve instructions."""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = dspy.Predict(ClinicalEvidenceSignature)

    def forward(self, note_text: str) -> dspy.Prediction:
        result = self.extractor(note_text=note_text)
        return dspy.Prediction(extraction_json=result.extraction_json)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _extract_seed_prompt_from_baml(path: Path) -> str:
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"function\s+ExtractClinicalEvidence\([^)]*\)\s*->\s*ClinicalNoteExtraction\s*\{.*?prompt\s*#\"(.*?)\"#",
        flags=re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        raise ValueError(f"Could not find ExtractClinicalEvidence prompt in {path}")
    return match.group(1).strip()


def _clean_json_blob(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_extraction_json(raw: str) -> ClinicalNoteExtraction:
    payload = json.loads(_clean_json_blob(raw))
    if isinstance(payload, dict):
        payload.setdefault("sleep_difficulty", [])
        payload.setdefault("daytime_impairment", [])
        payload.setdefault("medications", [])
        payload.setdefault("reasoning", "No exclusions applied")
        # GEPA may evolve schema-adjacent names; coerce them back to the expected schema.
        for sd in payload.get("sleep_difficulty", []) or []:
            if isinstance(sd, dict) and "criterion" not in sd and "criterion_key" in sd:
                sd["criterion"] = sd["criterion_key"]
        for di in payload.get("daytime_impairment", []) or []:
            if isinstance(di, dict) and "criterion" not in di and "criterion_key" in di:
                di["criterion"] = di["criterion_key"]
        for med in payload.get("medications", []) or []:
            if not isinstance(med, dict):
                continue
            if "name" not in med:
                med["name"] = med.get("medication_name") or med.get("normalized_name") or ""
            if "normalized_name" not in med:
                med["normalized_name"] = med.get("name", "")
            if "citation" not in med:
                med["citation"] = med.get("quote") or med.get("name", "")
    return ClinicalNoteExtraction.model_validate(payload)


def load_examples(
    corpus_csv: Path,
    gold_st1_json: Path,
    gold_st2_json: Path,
) -> list[dspy.Example]:
    df = pd.read_csv(corpus_csv)
    with gold_st1_json.open(encoding="utf-8") as f:
        gold1 = json.load(f)
    with gold_st2_json.open(encoding="utf-8") as f:
        gold2 = json.load(f)

    examples: list[dspy.Example] = []
    for _, row in df.iterrows():
        note_id = str(row["note_id"])
        if note_id not in gold1 or note_id not in gold2:
            continue
        examples.append(
            dspy.Example(
                note_id=note_id,
                note_text=str(row["text"]),
                gold_st1=gold1[note_id],
                gold_st2=gold2[note_id],
            ).with_inputs("note_text")
        )
    return examples


def _rule_label_score(st2: dict[str, Any], gold_st2: dict[str, Any]) -> float:
    all_rules = sorted(set(st2.keys()) | set(gold_st2.keys()))
    if not all_rules:
        return 0.0
    correct = 0
    for rule in all_rules:
        pred_label = yn((st2.get(rule) or {}).get("label", "no"))
        gold_label = yn((gold_st2.get(rule) or {}).get("label", "no"))
        if pred_label == gold_label:
            correct += 1
    return correct / len(all_rules)


def _evidence_rouge_f1(st2: dict[str, Any], gold_st2: dict[str, Any]) -> float:
    evidence_rules = ("Definition 1", "Definition 2", "Rule B", "Rule C")
    values: list[float] = []
    for rule in evidence_rules:
        gold_block = gold_st2.get(rule, {})
        if yn((gold_block or {}).get("label", "no")) != "yes":
            continue
        gold_text = " ".join((gold_block.get("text") or []))
        pred_text = " ".join(((st2.get(rule) or {}).get("text") or []))
        _, _, f1 = rouge_l_prf(gold_text, pred_text)
        values.append(f1)
    return (sum(values) / len(values)) if values else 0.0


def smm4h_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """Combined objective for GEPA.

    30% insomnia binary label agreement
    40% rule-level label agreement (Subtask 2A proxy)
    30% evidence ROUGE-L F1 (Subtask 2B)
    """
    feedback: list[str] = []
    try:
        extraction = _parse_extraction_json(prediction.extraction_json)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        feedback.append(f"Invalid extraction JSON: {exc}")
        if isinstance(trace, dict):
            trace["feedback"] = "; ".join(feedback)
        return 0.0

    labels = derive_labels(extraction)
    st1, st2 = format_submission(example.note_text, extraction, labels)

    pred_insomnia = yn(st1.get("Insomnia", "no"))
    gold_insomnia = yn((example.gold_st1 or {}).get("Insomnia", "no"))
    s1 = 1.0 if pred_insomnia == gold_insomnia else 0.0
    if s1 < 1.0:
        feedback.append(f"Insomnia mismatch (pred={pred_insomnia}, gold={gold_insomnia})")

    s2a = _rule_label_score(st2, example.gold_st2 or {})
    if s2a < 1.0:
        feedback.append(f"Rule label score below perfect: {s2a:.3f}")

    s2b = _evidence_rouge_f1(st2, example.gold_st2 or {})
    if s2b < 1.0:
        feedback.append(f"Evidence ROUGE-L F1 below perfect: {s2b:.3f}")

    score = 0.3 * s1 + 0.4 * s2a + 0.3 * s2b
    if isinstance(trace, dict) and feedback:
        trace["feedback"] = "; ".join(feedback)
    if isinstance(pred_trace, dict) and feedback:
        pred_trace["feedback"] = "; ".join(feedback)
    return float(score)


def _configure_lm(model_name: str, temperature: float) -> dspy.LM:
    lm = dspy.LM(model=model_name, temperature=temperature)
    dspy.configure(lm=lm)
    return lm


def _evaluate_program(
    program: InsomniaPipeline,
    examples: list[dspy.Example],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    subtask_1: dict[str, dict[str, Any]] = {}
    subtask_2: dict[str, dict[str, Any]] = {}

    for ex in examples:
        pred = program(note_text=ex.note_text)
        extraction = _parse_extraction_json(pred.extraction_json)
        labels = derive_labels(extraction)
        st1, st2 = format_submission(ex.note_text, extraction, labels)
        subtask_1[ex.note_id] = st1
        subtask_2[ex.note_id] = st2

    return subtask_1, subtask_2


def _run_eval_from_dicts(
    gold_st1_path: Path,
    gold_st2_path: Path,
    pred_st1: dict[str, dict[str, Any]],
    pred_st2: dict[str, dict[str, Any]],
) -> EvaluationResult:
    with TemporaryDirectory(prefix="gepa_eval_") as tmp_dir:
        tmp = Path(tmp_dir)
        pred_st1_path = tmp / "subtask_1.json"
        pred_st2_path = tmp / "subtask_2.json"
        pred_st1_path.write_text(json.dumps(pred_st1, indent=2), encoding="utf-8")
        pred_st2_path.write_text(json.dumps(pred_st2, indent=2), encoding="utf-8")
        return evaluate(
            gold_subtask1_path=gold_st1_path,
            gold_subtask2_path=gold_st2_path,
            pred_subtask1_path=pred_st1_path,
            pred_subtask2_path=pred_st2_path,
        )


def run_optimization(
    trainset: list[dspy.Example],
    valset: list[dspy.Example],
    seed_prompt: str,
    task_model: str,
    reflection_model: str,
    max_metric_calls: int,
    train_limit: int,
) -> tuple[InsomniaPipeline, str]:
    _configure_lm(task_model, temperature=0.0)
    reflection_lm = dspy.LM(model=reflection_model, temperature=1.0)

    program = InsomniaPipeline()
    program.extractor.signature = program.extractor.signature.with_instructions(seed_prompt)

    optimizer = dspy.GEPA(
        metric=smm4h_metric,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
    )
    optimized = optimizer.compile(
        student=program,
        trainset=trainset[:train_limit],
        valset=valset,
    )
    best_prompt = optimized.extractor.signature.instructions
    return optimized, best_prompt


def main() -> None:
    root = _project_root()
    parser = argparse.ArgumentParser(description="Run GEPA prompt optimization for SMM4H Task 2.")
    parser.add_argument("--task-model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--reflection-model", default="gemini/gemini-2.5-flash")
    parser.add_argument("--max-metric-calls", type=int, default=150)
    parser.add_argument("--train-limit", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "outputs/gepa",
    )
    args = parser.parse_args()

    seed_prompt = _extract_seed_prompt_from_baml(root / "baml_src" / "insomnia.baml")
    trainset = load_examples(
        root / "data/training/train_corpus.csv",
        root / "data/training/subtask_1.json",
        root / "data/training/subtask_2.json",
    )
    valset = load_examples(
        root / "data/validation/validation_corpus.csv",
        root / "data/validation/subtask_1.json",
        root / "data/validation/subtask_2.json",
    )

    optimized, best_prompt = run_optimization(
        trainset=trainset,
        valset=valset,
        seed_prompt=seed_prompt,
        task_model=args.task_model,
        reflection_model=args.reflection_model,
        max_metric_calls=args.max_metric_calls,
        train_limit=args.train_limit,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    optimized.save(str(args.out_dir / "optimized_program.json"))

    pred_st1, pred_st2 = _evaluate_program(optimized, valset)
    result = _run_eval_from_dicts(
        root / "data/validation/subtask_1.json",
        root / "data/validation/subtask_2.json",
        pred_st1,
        pred_st2,
    )
    (args.out_dir / "validation_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )

    print(f"Saved best prompt to: {args.out_dir / 'best_prompt.txt'}")
    print(f"Saved optimized program to: {args.out_dir / 'optimized_program.json'}")
    print(f"Validation Subtask 1 F1: {result.subtask1_f1:.6f}")
    print(f"Validation Subtask 2A micro-F1: {result.subtask2a_micro_f1:.6f}")
    print(f"Validation Subtask 2B macro ROUGE-L F1: {result.subtask2b_rougeL_f1_macro:.6f}")


if __name__ == "__main__":
    main()
