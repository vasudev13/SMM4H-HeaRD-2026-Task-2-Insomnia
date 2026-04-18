"""Map verbatim evidence quotes to character spans for submission JSON.

Policy (documented): process quotes **in order**. For each non-empty quote, take the
**first** occurrence in ``note_text`` starting at or after the end of the previous
match (non-overlapping, left-to-right). This matches typical annotation where quotes
are listed in document order.

Empty strings are skipped. If a quote is not found from the current search position,
it is skipped (no span), so downstream code may log or count misses.
"""

from __future__ import annotations


def _yn(label: object) -> str:
    return "yes" if str(label).strip().lower() == "yes" else "no"


def spans_from_quotes(
    note_text: str, quotes: list[str] | tuple[str, ...]
) -> tuple[list[str], list[str]]:
    """Return parallel ``span`` (``"start end"`` strings) and ``text`` snippets.

    Each span is half-open ``[start, end)`` in Python slice terms, using character
    offsets into ``note_text``.
    """
    span_strs: list[str] = []
    texts: list[str] = []
    cursor = 0
    for q in quotes:
        if not q:
            continue
        pos = note_text.find(q, cursor)
        if pos < 0:
            continue
        start, end = pos, pos + len(q)
        span_strs.append(f"{start} {end}")
        texts.append(note_text[start:end])
        cursor = end
    return span_strs, texts


def rule_block_from_quotes(
    label: str, quotes: list[str] | tuple[str, ...], note_text: str
) -> dict:
    """Build one submission block and enforce scorer-facing invariants.

    Submission constraints:
      - label == "no"  -> span/text must be empty lists
      - label == "yes" -> span must be non-empty
    """
    normalized = _yn(label)
    if normalized == "no":
        return {"label": "no", "span": [], "text": []}

    span, text = spans_from_quotes(note_text, quotes)
    if not span:
        # If no quote could be grounded, downgrade to a valid negative block.
        return {"label": "no", "span": [], "text": []}
    return {"label": "yes", "span": span, "text": text}
