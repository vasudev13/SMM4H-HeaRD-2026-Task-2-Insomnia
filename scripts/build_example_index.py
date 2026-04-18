#!/usr/bin/env python3
"""Build FAISS index and metadata for adaptive few-shot retrieval (kNN-ICL).

Run once during setup:
    python scripts/build_example_index.py

Outputs (written to --out-dir, default: data/):
    faiss_index.bin       — FAISS IndexFlatIP over L2-normalized note embeddings
    example_metadata.json — list of {note_id, insomnia_label, cot_reasoning, note_preview}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
# Secondary medication names for distractor scanning
# ──────────────────────────────────────────────────────────────

_SECONDARY_MEDS: list[tuple[str, str]] = [
    ("acamprosate", "Acamprosate"),
    ("campral", "Acamprosate"),
    ("alprazolam", "Alprazolam"),
    ("xanax", "Alprazolam"),
    ("clonazepam", "Clonazepam"),
    ("klonopin", "Clonazepam"),
    ("clonidine", "Clonidine"),
    ("catapres", "Clonidine"),
    ("diazepam", "Diazepam"),
    ("valium", "Diazepam"),
    ("diphenhydramine", "Diphenhydramine"),
    ("benadryl", "Diphenhydramine"),
    ("doxepin", "Doxepin"),
    ("silenor", "Doxepin"),
    ("gabapentin", "Gabapentin"),
    ("neurontin", "Gabapentin"),
    ("hydroxyzine", "Hydroxyzine"),
    ("vistaril", "Hydroxyzine"),
    ("atarax", "Hydroxyzine"),
    ("lorazepam", "Lorazepam"),
    ("ativan", "Lorazepam"),
    ("melatonin", "Melatonin"),
    ("mirtazapine", "Mirtazapine"),
    ("remeron", "Mirtazapine"),
    ("olanzapine", "Olanzapine"),
    ("zyprexa", "Olanzapine"),
    ("quetiapine", "Quetiapine"),
    ("seroquel", "Quetiapine"),
    ("trazodone", "Trazodone"),
    ("desyrel", "Trazodone"),
]

_MOOD_FATIGUE_TERMS: list[str] = [
    "fatigue",
    "fatigued",
    "tired",
    "lethargic",
    "lethargy",
    "depression",
    "depressed",
    "anxiety",
    "anxious",
    "agitated",
    "irritable",
    "irritability",
]

_SLEEPINESS_TERMS: list[str] = [
    "sleepy",
    "somnolent",
    "napping",
    "drowsy",
    "drowsiness",
]


# ──────────────────────────────────────────────────────────────
# Keyword scanners (case-insensitive, whole-word matching)
# ──────────────────────────────────────────────────────────────


def _scan_secondary_meds(note_text: str) -> list[str]:
    """Return canonical names of secondary meds found in the note (deduplicated)."""
    note_lower = note_text.lower()
    found: dict[str, str] = {}
    for keyword, canonical in _SECONDARY_MEDS:
        if re.search(r"\b" + re.escape(keyword) + r"\b", note_lower):
            found[canonical] = canonical
    return list(found.values())


def _scan_mood_fatigue(note_text: str) -> list[str]:
    """Return distractor mood/fatigue terms found in the note."""
    note_lower = note_text.lower()
    return [
        term
        for term in _MOOD_FATIGUE_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", note_lower)
    ]


def _scan_sleepiness(note_text: str) -> list[str]:
    """Return distractor sleepiness terms found in the note."""
    note_lower = note_text.lower()
    return [
        term
        for term in _SLEEPINESS_TERMS
        if re.search(r"\b" + re.escape(term) + r"\b", note_lower)
    ]


# ──────────────────────────────────────────────────────────────
# CoT building helpers
# ──────────────────────────────────────────────────────────────


def _build_def2_no_reasoning(note_text: str) -> str:
    """Explain why Definition 2 is 'no', calling out any distractors."""
    mood_fatigue = _scan_mood_fatigue(note_text)
    sleepiness = _scan_sleepiness(note_text)

    parts: list[str] = []

    if mood_fatigue:
        terms_str = "', '".join(mood_fatigue[:3])
        parts.append(
            f"Words such as '{terms_str}' appear in the note but are attributed to "
            f"pre-existing comorbidities or acute illness context, not sleep disruption."
        )

    if sleepiness:
        terms_str = "', '".join(sleepiness[:2])
        parts.append(
            f"Patient sleepiness/napping ('{terms_str}') is an expected feature of "
            f"acute illness or ICU care, not a daytime consequence of insomnia."
        )

    if not parts:
        parts.append(
            "no evidence of fatigue, mood disturbance, impaired performance, "
            "or other daytime impairment criteria."
        )

    return "Daytime impairment: No — " + " ".join(parts)


def _build_rulec_no_reasoning(
    note_text: str, def1_label: str, def2_label: str
) -> str:
    """Explain why Rule C is 'no'."""
    secondary_meds = _scan_secondary_meds(note_text)
    has_symptoms = def1_label == "yes" or def2_label == "yes"

    if secondary_meds:
        meds_str = ", ".join(secondary_meds[:4])
        if has_symptoms:
            return (
                f"Rule C: No — secondary medication(s) ({meds_str}) present and "
                f"symptoms present, but Rule C not satisfied per gold annotation."
            )
        else:
            return (
                f"Rule C: No — secondary medication(s) ({meds_str}) present "
                f"but no Definition 1 or 2 symptoms found."
            )
    else:
        return "Rule C: No — no secondary insomnia medications prescribed."


# ──────────────────────────────────────────────────────────────
# Main CoT builder
# ──────────────────────────────────────────────────────────────


def build_cot_reasoning(
    note_text: str,
    sub1_entry: dict,
    sub2_entry: dict,
) -> str:
    """Build CoT reasoning chain deterministically from gold annotations.

    No LLM calls — uses only gold labels and evidence texts from subtask JSON files.
    For 'no' examples, scans the note for known distractor keywords and explains
    why they don't count as Definition 2 evidence or Rule C evidence.
    """
    def1_label = sub2_entry["Definition 1"]["label"]
    def1_texts = sub2_entry["Definition 1"].get("text", [])
    def2_label = sub2_entry["Definition 2"]["label"]
    def2_texts = sub2_entry["Definition 2"].get("text", [])
    ruleb_label = sub2_entry["Rule B"]["label"]
    ruleb_texts = sub2_entry["Rule B"].get("text", [])
    rulec_label = sub2_entry["Rule C"]["label"]
    rulec_texts = sub2_entry["Rule C"].get("text", [])
    insomnia_label = sub1_entry["Insomnia"]
    rule_a = "yes" if (def1_label == "yes" and def2_label == "yes") else "no"

    lines: list[str] = []

    # ── Definition 1 line ──────────────────────────────────────
    if def1_label == "yes":
        if def1_texts:
            preview = f'"{def1_texts[0]}"'
            if len(def1_texts) > 1:
                preview += f" (and {len(def1_texts) - 1} more)"
            lines.append(f"Sleep difficulty: Yes — {preview}")
        else:
            lines.append("Sleep difficulty: Yes")
    else:
        lines.append(
            "Sleep difficulty: No — no evidence of trouble initiating/maintaining "
            "sleep, early awakening, or explicit insomnia mention."
        )

    # ── Definition 2 line (most critical — teach exclusion logic) ──
    if def2_label == "yes":
        if def2_texts:
            preview = f'"{def2_texts[0]}"'
            if len(def2_texts) > 1:
                preview += f" (and {len(def2_texts) - 1} more)"
            lines.append(f"Daytime impairment: Yes — {preview}")
        else:
            lines.append("Daytime impairment: Yes")
    else:
        lines.append(_build_def2_no_reasoning(note_text))

    # ── Medications ────────────────────────────────────────────
    if ruleb_label == "yes" and ruleb_texts:
        meds_str = ", ".join(ruleb_texts[:4])
        lines.append(f"Primary medications: {meds_str} → Rule B satisfied.")
    else:
        lines.append("Primary medications: None → Rule B: No.")

    if rulec_label == "yes" and rulec_texts:
        meds_str = ", ".join(rulec_texts[:4])
        symptom_basis = "Definition 1" if def1_label == "yes" else "Definition 2"
        lines.append(
            f"Secondary medications: {meds_str} AND {symptom_basis} symptoms present "
            f"→ Rule C satisfied."
        )
    elif rulec_label == "no":
        lines.append(_build_rulec_no_reasoning(note_text, def1_label, def2_label))

    # ── Rules summary ──────────────────────────────────────────
    lines.append(f"Rule A: {rule_a} — Def1 {def1_label} AND Def2 {def2_label}")
    lines.append(f"Rule B: {ruleb_label}")
    lines.append(f"Rule C: {rulec_label}")
    lines.append(f"Insomnia: {insomnia_label}")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data/training",
        help="Directory containing train_corpus.csv, subtask_1.json, subtask_2.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data",
        help="Output directory for faiss_index.bin and example_metadata.json (default: data/)",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="Sentence-transformers model name for embedding (default: BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=2000,
        help="Max characters of note text stored as preview for prompt injection (default: 2000)",
    )
    args = parser.parse_args(argv)

    corpus_path = args.data_dir / "train_corpus.csv"
    sub1_path = args.data_dir / "subtask_1.json"
    sub2_path = args.data_dir / "subtask_2.json"

    for p in (corpus_path, sub1_path, sub2_path):
        if not p.exists():
            print(f"error: required file not found: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading training data from {args.data_dir} ...")
    corpus_df = pd.read_csv(corpus_path, dtype={"note_id": str})
    with sub1_path.open(encoding="utf-8") as f:
        sub1: dict = json.load(f)
    with sub2_path.open(encoding="utf-8") as f:
        sub2: dict = json.load(f)

    # Only keep notes that have gold labels in both subtask files
    labeled_ids = set(sub1.keys()) & set(sub2.keys())
    df = corpus_df[corpus_df["note_id"].isin(labeled_ids)].copy()
    df = df.reset_index(drop=True)
    print(f"Found {len(df)} labeled notes.")

    # Build CoT reasoning chains (no LLM calls)
    print("Building CoT reasoning chains from gold annotations ...")
    metadata: list[dict] = []
    texts_for_embedding: list[str] = []

    for _, row in df.iterrows():
        nid = str(row["note_id"])
        note_text = str(row["text"]) if not pd.isna(row["text"]) else ""
        cot = build_cot_reasoning(note_text, sub1[nid], sub2[nid])
        metadata.append(
            {
                "note_id": nid,
                "insomnia_label": sub1[nid]["Insomnia"],
                "cot_reasoning": cot,
                "note_preview": note_text[: args.preview_chars],
            }
        )
        texts_for_embedding.append(note_text)

    # Embed all notes
    print(f"Loading embedding model: {args.model} ...")
    model = SentenceTransformer(args.model)
    print(f"Embedding {len(texts_for_embedding)} notes (batch_size=32) ...")
    embeddings = model.encode(
        texts_for_embedding,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS IndexFlatIP (dot product = cosine sim on normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "faiss_index.bin"
    meta_path = args.out_dir / "example_metadata.json"

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index ({index.ntotal} vectors, dim={dim}) → {index_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata ({len(metadata)} entries) → {meta_path}")

    n_yes = sum(1 for m in metadata if m["insomnia_label"] == "yes")
    n_no = len(metadata) - n_yes
    print(f"Label distribution: {n_yes} yes, {n_no} no")


if __name__ == "__main__":
    main()
