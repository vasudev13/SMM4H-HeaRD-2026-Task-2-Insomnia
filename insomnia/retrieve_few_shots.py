"""Adaptive few-shot retrieval using FAISS nearest-neighbour search.

Usage:
    from insomnia.retrieve_few_shots import retrieve_few_shots
    examples = retrieve_few_shots(note_text, k=4, balance=True)

Module-level caches avoid reloading the model and index on every call.
The model (~435 MB for bge-base-en-v1.5) is loaded once and reused.
"""

from __future__ import annotations

import json
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────────
# Module-level caches (keyed by resolved path / model name)
# ──────────────────────────────────────────────────────────────

_index_cache: dict[str, Any] = {}
_metadata_cache: dict[str, list[dict]] = {}
_model_cache: dict[str, SentenceTransformer] = {}

_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def _load_index(index_path: str) -> Any:
    if index_path not in _index_cache:
        _index_cache[index_path] = faiss.read_index(index_path)
    return _index_cache[index_path]


def _load_metadata(metadata_path: str) -> list[dict]:
    if metadata_path not in _metadata_cache:
        with open(metadata_path, encoding="utf-8") as f:
            _metadata_cache[metadata_path] = json.load(f)
    return _metadata_cache[metadata_path]


def _get_model(model_name: str = _DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def batch_encode(
    texts: list[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode multiple note texts in one model call batch."""
    model = _get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    return np.array(vectors, dtype="float32")


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────


def retrieve_few_shots(
    note_text: str,
    k: int = 4,
    balance: bool = True,
    exclude_note_id: str | None = None,
    index_path: str = "data/faiss_index.bin",
    metadata_path: str = "data/example_metadata.json",
    model_name: str = _DEFAULT_MODEL,
    query_vec: np.ndarray | None = None,
) -> list[dict]:
    """Retrieve k most similar training examples for few-shot injection.

    If balance=True (default), returns k//2 "yes" examples and k//2 "no"
    examples from the top 3*k candidates. Ensures the model sees both classes
    regardless of which class dominates the nearest neighbours.

    Falls back to best available if one class has fewer than k//2 in the
    top 3*k results (e.g., tiny training set or extreme class imbalance).

    Args:
        note_text:       Query clinical note text.
        k:               Total examples to return (default 4).
        balance:         If True, return equal yes/no examples.
        exclude_note_id: Skip this note_id (use when querying on training data
                         to avoid returning the query note itself).
        index_path:      Path to the FAISS index file.
        metadata_path:   Path to the metadata JSON file.
        model_name:      Sentence-transformers model used at index build time.

    Returns:
        List of dicts with keys:
            note_id, label, cot_reasoning, note_preview, similarity_score
    """
    index = _load_index(index_path)
    metadata = _load_metadata(metadata_path)
    if query_vec is None:
        model = _get_model(model_name)
        # Embed query note (normalize for cosine similarity via dot product)
        query_vec = model.encode([note_text], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
    else:
        # Normalize shape to (1, d) for FAISS search.
        query_vec = np.array(query_vec, dtype="float32")
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

    n_candidates = min(3 * k, index.ntotal)
    scores, indices = index.search(query_vec, n_candidates)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    # Build candidate list, optionally excluding the query note itself
    candidates: list[dict] = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx]
        if exclude_note_id is not None and entry["note_id"] == exclude_note_id:
            continue
        candidates.append(
            {
                "note_id": entry["note_id"],
                "label": entry["insomnia_label"],
                "cot_reasoning": entry["cot_reasoning"],
                "note_preview": entry["note_preview"],
                "similarity_score": float(score),
            }
        )

    if not balance:
        return candidates[:k]

    # Balanced selection: top k//2 from each class
    half = k // 2
    yes_pool = [c for c in candidates if c["label"] == "yes"]
    no_pool = [c for c in candidates if c["label"] == "no"]

    selected_yes = yes_pool[:half]
    selected_no = no_pool[:half]

    # If one class is short, fill extra from the other
    shortfall_yes = half - len(selected_yes)
    shortfall_no = half - len(selected_no)
    if shortfall_yes > 0:
        selected_no = no_pool[: half + shortfall_yes]
    if shortfall_no > 0:
        selected_yes = yes_pool[: half + shortfall_no]

    return selected_yes + selected_no
