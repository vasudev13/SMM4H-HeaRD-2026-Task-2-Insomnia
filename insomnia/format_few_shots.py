"""Format retrieved few-shot examples into a prompt-injectable string.

Usage:
    from insomnia.format_few_shots import format_few_shots
    few_shot_str = format_few_shots(examples)
"""

from __future__ import annotations

_MAX_TOTAL_CHARS = 9_500
_MAX_PREVIEW_CHARS_DEFAULT = 1_800
_DIVIDER = "═" * 19


def format_few_shots(examples: list[dict]) -> str:
    """Format retrieved examples into a few-shot demonstration string.

    Ordering: yes examples first (most similar first), then no examples
    (most similar first). This ensures the model sees a positive example
    before the negative rejection reasoning.

    Each example block shows:
      1. A truncated clinical note preview
      2. Evidence found as bullet points (from cot_reasoning)
      3. Derived labels with Insomnia verdict

    The total output is capped at ~9,500 characters to leave headroom
    in the context window for the clinical note and instructions.
    """
    if not examples:
        return ""

    # Order: yes first (similarity desc), then no (similarity desc)
    yes_examples = sorted(
        [e for e in examples if e.get("label") == "yes"],
        key=lambda e: e.get("similarity_score", 0.0),
        reverse=True,
    )
    no_examples = sorted(
        [e for e in examples if e.get("label") == "no"],
        key=lambda e: e.get("similarity_score", 0.0),
        reverse=True,
    )
    ordered = yes_examples + no_examples

    n = len(ordered)
    # Budget preview chars per example to stay within total limit
    # Reserve ~250 chars overhead per example (headers/labels) + ~400 chars for CoT
    overhead_per_example = 250
    cot_budget_per_example = 400
    remaining_for_previews = _MAX_TOTAL_CHARS - (overhead_per_example + cot_budget_per_example) * n
    preview_per_example = max(500, remaining_for_previews // n) if n > 0 else 500
    preview_per_example = min(preview_per_example, _MAX_PREVIEW_CHARS_DEFAULT)

    parts: list[str] = []
    for i, ex in enumerate(ordered, start=1):
        note_preview = (ex.get("note_preview") or "")[:preview_per_example]
        cot = ex.get("cot_reasoning") or ""
        label = ex.get("label", "?")

        # Split CoT into evidence lines (bullet-pointed) and rules summary
        evidence_lines, rules_lines = _split_cot(cot)
        evidence_block = "\n".join(f"• {line}" for line in evidence_lines)
        rules_block = _format_rules_summary(rules_lines, label)

        parts.append(
            f"═══ EXAMPLE {i} ═══\n"
            f"Clinical note (excerpt):\n"
            f"{note_preview}\n"
            f"\n"
            f"Evidence found:\n"
            f"{evidence_block}\n"
            f"\n"
            f"{rules_block}\n"
            f"{_DIVIDER}"
        )

    result = "\n\n".join(parts)

    # Hard truncation guard
    if len(result) > _MAX_TOTAL_CHARS:
        result = result[:_MAX_TOTAL_CHARS] + "\n[...truncated]"

    return result


def _split_cot(cot: str) -> tuple[list[str], list[str]]:
    """Split CoT reasoning into evidence lines and rules-summary lines."""
    evidence_lines: list[str] = []
    rules_lines: list[str] = []
    for line in cot.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("Rule A:", "Rule B:", "Rule C:", "Insomnia:")):
            rules_lines.append(line)
        else:
            evidence_lines.append(line)
    return evidence_lines, rules_lines


def _format_rules_summary(rules_lines: list[str], label: str) -> str:
    """Format the rules summary into a compact 'Derived labels' block."""
    if not rules_lines:
        return f"Insomnia: {label}"

    # Separate the Insomnia line from the rule lines
    insomnia_lines = [l for l in rules_lines if l.startswith("Insomnia:")]
    other_lines = [l for l in rules_lines if not l.startswith("Insomnia:")]

    insomnia_line = insomnia_lines[0] if insomnia_lines else f"Insomnia: {label}"

    if other_lines:
        derived = ", ".join(other_lines)
        return f"Derived labels: {derived}\n{insomnia_line}"
    return insomnia_line
