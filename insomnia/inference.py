"""CSV → BAML → subtask_1.json / subtask_2.json (training submission shape).

Two-pass architecture:
  Pass 1 — LLM extracts structured evidence via ExtractClinicalEvidence (BAML).
            One LLM call per note; no label decisions made by the model.
  Pass 2 — derive_labels() deterministically computes all subtask labels.
            Pure Python; zero additional LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from baml_client import b
from baml_client.types import ClinicalNoteExtraction

from insomnia.spans import rule_block_from_quotes

try:
    from insomnia.retrieve_few_shots import batch_encode, retrieve_few_shots
    from insomnia.format_few_shots import format_few_shots
    _FEW_SHOT_AVAILABLE = True
except ImportError:
    _FEW_SHOT_AVAILABLE = False

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


class _AsyncRequestPacer:
    """Enforce a minimum time between the start of consecutive LLM calls (async)."""

    def __init__(self, min_interval_sec: float) -> None:
        self._min = min_interval_sec
        self._lock = asyncio.Lock()
        self._last_start: float | None = None

    async def before_request(self) -> None:
        if self._min <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if self._last_start is not None:
                wait = self._min - (now - self._last_start)
                if wait > 0:
                    await asyncio.sleep(wait)
            self._last_start = time.monotonic()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _few_shot_paths(few_shot_index: Path | None) -> tuple[Path, Path]:
    root = _project_root()
    if few_shot_index is not None:
        return few_shot_index, few_shot_index.parent / "example_metadata.json"
    return root / "data/faiss_index.bin", root / "data/example_metadata.json"


def prepare_note_jobs(
    df: pd.DataFrame,
    *,
    use_few_shot: bool = True,
    few_shot_k: int = 4,
    few_shot_index: Path | None = None,
) -> list[tuple[str, str, str]]:
    """Build (note_id, note_text, few_shot_context) rows for v2 extraction.

    Emits the same few-shot warnings as the CLI when the index is missing.
    """
    _index_path, _meta_path = _few_shot_paths(few_shot_index)
    _few_shot_ready = (
        use_few_shot
        and _FEW_SHOT_AVAILABLE
        and _index_path.exists()
        and _meta_path.exists()
    )
    if use_few_shot and not _few_shot_ready:
        if not _FEW_SHOT_AVAILABLE:
            print(
                "Warning: few-shot modules unavailable (import failed), "
                "falling back to zero-shot.",
                file=sys.stderr,
            )
        else:
            print(
                f"Warning: few-shot index not found at {_index_path}, "
                "falling back to zero-shot. "
                "Run: python scripts/build_example_index.py",
                file=sys.stderr,
            )

    jobs: list[tuple[str, str, str]] = []
    note_texts: list[str] = []
    note_ids: list[str] = []
    for _, row in df.iterrows():
        note_ids.append(str(row["note_id"]))
        note_texts.append(_text_cell(row["text"]))

    query_vecs = None
    if _few_shot_ready:
        query_vecs = batch_encode(note_texts)

    for i, (nid, note_text) in enumerate(zip(note_ids, note_texts)):
        few_shot_str = ""
        if _few_shot_ready:
            examples = retrieve_few_shots(
                note_text,
                k=few_shot_k,
                balance=True,
                index_path=str(_index_path),
                metadata_path=str(_meta_path),
                query_vec=query_vecs[i] if query_vecs is not None else None,
            )
            few_shot_str = format_few_shots(examples)
        jobs.append((nid, note_text, few_shot_str))
    return jobs


# ──────────────────────────────────────────────────────────────
# Pass 2: deterministic label derivation (no LLM calls)
# ──────────────────────────────────────────────────────────────

def derive_labels(extraction: ClinicalNoteExtraction) -> dict:
    """Two-pass architecture — Pass 2: derive all subtask labels from structured extraction.

    Pure Python — zero LLM calls. All classification logic lives here;
    Pass 1 (ExtractClinicalEvidence) provides the evidence.

    Label logic:
      Definition 1  — any sleep difficulty criterion extracted
      Definition 2  — any daytime impairment criterion extracted
      Rule A        — Definition 1 AND Definition 2
      Rule B        — any primary insomnia medication present
      Rule C        — any secondary medication AND (Definition 1 OR Definition 2)
      Insomnia      — Rule A OR Rule B OR Rule C

    Returns a dict with keys: insomnia, definition_1, definition_2, rule_a, rule_b, rule_c
    (all values "yes" or "no").
    """
    def1 = len(extraction.sleep_difficulty) > 0
    def2 = len(extraction.daytime_impairment) > 0
    rule_a = def1 and def2
    rule_b = any(m.med_type == "primary" for m in extraction.medications)
    has_secondary = any(m.med_type == "secondary" for m in extraction.medications)
    rule_c = has_secondary and (def1 or def2)
    insomnia = "yes" if (rule_a or rule_b or rule_c) else "no"

    return {
        "insomnia": insomnia,
        "definition_1": "yes" if def1 else "no",
        "definition_2": "yes" if def2 else "no",
        "rule_a": "yes" if rule_a else "no",
        "rule_b": "yes" if rule_b else "no",
        "rule_c": "yes" if rule_c else "no",
    }


def format_submission(
    note_text: str,
    extraction: ClinicalNoteExtraction,
    labels: dict,
) -> tuple[dict, dict]:
    """Build subtask_1 and subtask_2 submission dicts from extraction + derived labels.

    subtask_1 entry:
      {"Insomnia": "yes"/"no"}

    subtask_2 entry:
      {
        "Definition 1": {"label": ..., "span": [...], "text": [...]},
        "Definition 2": {"label": ..., "span": [...], "text": [...]},
        "Rule B":       {"label": ..., "span": [...], "text": [...]},
        "Rule C":       {"label": ..., "span": [...], "text": [...]},
      }

    Evidence assembly per block:
      Definition 1 — citation fields from sleep_difficulty items
      Definition 2 — citation fields from daytime_impairment items
      Rule B       — citation fields from primary medication mentions
      Rule C       — citation fields from secondary medication mentions

    `rule_block_from_quotes` enforces scorer-facing constraints:
      - label "no" yields empty span/text lists
      - label "yes" requires non-empty span (otherwise downgraded to "no")
    """
    st1 = {"Insomnia": labels["insomnia"]}

    def1_quotes = [item.citation for item in extraction.sleep_difficulty]
    def2_quotes = [item.citation for item in extraction.daytime_impairment]
    rule_b_quotes = [m.citation for m in extraction.medications if m.med_type == "primary"]
    rule_c_quotes = [m.citation for m in extraction.medications if m.med_type == "secondary"]

    st2 = {
        "Definition 1": rule_block_from_quotes(labels["definition_1"], def1_quotes, note_text),
        "Definition 2": rule_block_from_quotes(labels["definition_2"], def2_quotes, note_text),
        "Rule B": rule_block_from_quotes(labels["rule_b"], rule_b_quotes, note_text),
        "Rule C": rule_block_from_quotes(labels["rule_c"], rule_c_quotes, note_text),
    }

    return st1, st2


async def _run_one_v2(
    nid: str,
    note_text: str,
    few_shot_str: str,
    sem: asyncio.Semaphore,
    pacer: _AsyncRequestPacer,
) -> tuple[str, dict, dict]:
    async with sem:
        await pacer.before_request()
        extraction = await b.ExtractClinicalEvidence(
            note_text=note_text,
            few_shot_context=few_shot_str,
        )
    labels = derive_labels(extraction)
    st1, st2 = format_submission(note_text, extraction, labels)
    return nid, st1, st2


def _flush_results(
    out_dir: Path,
    subtask_1: dict[str, dict],
    subtask_2: dict[str, dict],
) -> None:
    """Atomically write current results to disk (tmp + rename)."""
    for name, data in (("subtask_1.json", subtask_1), ("subtask_2.json", subtask_2)):
        target = out_dir / name
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        tmp.replace(target)


async def _run_pipeline_async(
    jobs: list[tuple[str, str, str]],
    *,
    concurrency: int,
    min_interval_sec: float,
    out_dir: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    sem = asyncio.Semaphore(concurrency)
    pacer = _AsyncRequestPacer(min_interval_sec)
    total = len(jobs)
    subtask_1: dict[str, dict] = {}
    subtask_2: dict[str, dict] = {}
    done = 0
    errors = 0
    write_lock = asyncio.Lock()

    async def _run_and_save(nid: str, text: str, fs: str) -> None:
        nonlocal done, errors
        try:
            _, st1, st2 = await _run_one_v2(nid, text, fs, sem, pacer)
        except Exception as exc:
            async with write_lock:
                errors += 1
                print(
                    f"[inference] FAILED {nid} ({errors} error(s)): {exc}",
                    file=sys.stderr,
                )
            return
        async with write_lock:
            subtask_1[nid] = st1
            subtask_2[nid] = st2
            done += 1
            print(f"[inference] done {done}/{total}", file=sys.stderr)
            _flush_results(out_dir, subtask_1, subtask_2)

    tasks = [_run_and_save(nid, text, fs) for nid, text, fs in jobs]
    await asyncio.gather(*tasks)

    if errors:
        print(
            f"[inference] completed with {errors} failed note(s) out of {total}",
            file=sys.stderr,
        )
    return subtask_1, subtask_2


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def run(
    input_csv: Path,
    out_dir: Path,
    *,
    max_rows: int | None = None,
    min_interval_sec: float | None = None,
    rpm: float | None = None,
    use_few_shot: bool = True,
    few_shot_k: int = 4,
    few_shot_index: Path | None = None,
    concurrency: int = 10,
) -> None:
    """Run the full inference pipeline and write subtask JSON files.

    Two-pass architecture:
      Pass 1: one LLM call per note via ExtractClinicalEvidence
      Pass 2: derive_labels() + format_submission() with no further LLM calls

    use_few_shot (default True) — Inject adaptive few-shot examples retrieved
      from the FAISS index. Falls back to zero-shot if the index files are
      missing or if few-shot modules are unavailable.

    concurrency — Maximum in-flight LLM requests (v2: one call per note).
      Combine with --min-interval-sec / --rpm to avoid provider rate limits.
    """
    df = pd.read_csv(input_csv, dtype={"note_id": str})
    if "note_id" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain columns: note_id, text")
    df["note_id"] = df["note_id"].astype(str)
    if max_rows is not None:
        df = df.head(max_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    interval = _resolve_min_interval_sec(min_interval_sec, rpm)
    jobs = prepare_note_jobs(
        df,
        use_few_shot=use_few_shot,
        few_shot_k=few_shot_k,
        few_shot_index=few_shot_index,
    )
    subtask_1, subtask_2 = asyncio.run(
        _run_pipeline_async(
            jobs,
            concurrency=concurrency,
            min_interval_sec=interval,
            out_dir=out_dir,
        )
    )

    _flush_results(out_dir, subtask_1, subtask_2)
    print(f"Wrote {out_dir / 'subtask_1.json'} ({len(subtask_1)} notes)")
    print(f"Wrote {out_dir / 'subtask_2.json'} ({len(subtask_2)} notes)")


def main(argv: list[str] | None = None) -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run BAML evidence extraction; write subtask JSON files."
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
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        metavar="N",
        help="Max in-flight LLM requests (default 10). Raise for higher throughput if quotas allow.",
    )
    few_shot_grp = parser.add_mutually_exclusive_group()
    few_shot_grp.add_argument(
        "--use-few-shot",
        dest="use_few_shot",
        action="store_true",
        default=True,
        help=(
            "Inject adaptive few-shot examples from the FAISS index "
            "(default: enabled; falls back to zero-shot if index is missing)"
        ),
    )
    few_shot_grp.add_argument(
        "--no-few-shot",
        dest="use_few_shot",
        action="store_false",
        help="Disable few-shot injection — run zero-shot (useful for A/B comparison)",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=4,
        metavar="K",
        help="Number of few-shot examples to retrieve per note (default 4; yields 2 yes + 2 no)",
    )
    parser.add_argument(
        "--few-shot-index",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Path to faiss_index.bin (metadata JSON expected alongside it). "
            "Defaults to data/faiss_index.bin relative to the project root."
        ),
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
            use_few_shot=args.use_few_shot,
            few_shot_k=args.few_shot_k,
            few_shot_index=args.few_shot_index,
            concurrency=args.concurrency,
        )
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
