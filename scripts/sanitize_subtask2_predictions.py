"""Sanitize subtask_2 predictions to strict label/span constraints.

Rules enforced per component block:
  - label == "no"  -> span == [] and text == []
  - label == "yes" -> span must be non-empty, else downgrade to "no"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RULE_KEYS: tuple[str, ...] = ("Definition 1", "Definition 2", "Rule B", "Rule C")


def _yn(label: object) -> str:
    return "yes" if str(label).strip().lower() == "yes" else "no"


def _as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    return []


def sanitize_block(block: object) -> tuple[dict, bool]:
    changed = False
    src = block if isinstance(block, dict) else {}
    label = _yn(src.get("label", "no"))
    span = _as_str_list(src.get("span", []))
    text = _as_str_list(src.get("text", []))

    if label == "no":
        out = {"label": "no", "span": [], "text": []}
    elif not span:
        out = {"label": "no", "span": [], "text": []}
    else:
        out = {"label": "yes", "span": span, "text": text}

    if out != {"label": label, "span": span, "text": text}:
        changed = True
    return out, changed


def sanitize_subtask2(data: object) -> tuple[dict[str, dict], int, int]:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object keyed by note_id")

    cleaned: dict[str, dict] = {}
    changed_notes = 0
    changed_blocks = 0

    for note_id, note_block in data.items():
        note_src = note_block if isinstance(note_block, dict) else {}
        note_out: dict[str, dict] = {}
        note_changed = False

        for key in RULE_KEYS:
            fixed, was_changed = sanitize_block(note_src.get(key, {}))
            note_out[key] = fixed
            if was_changed:
                note_changed = True
                changed_blocks += 1

        cleaned[str(note_id)] = note_out
        if note_changed:
            changed_notes += 1

    return cleaned, changed_notes, changed_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize subtask_2 prediction JSON.")
    parser.add_argument("input_json", type=Path, help="Path to subtask_2.json")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output path (default: <input>.sanitized.json)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place",
    )
    args = parser.parse_args()

    if args.in_place and args.output_json is not None:
        raise ValueError("Use either --in-place or --output-json, not both.")

    in_path = args.input_json
    out_path = (
        in_path
        if args.in_place
        else (
            args.output_json
            if args.output_json is not None
            else in_path.with_name(f"{in_path.stem}.sanitized{in_path.suffix}")
        )
    )

    with in_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    cleaned, changed_notes, changed_blocks = sanitize_subtask2(raw)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    print(f"Wrote {out_path}")
    print(f"Changed notes: {changed_notes}")
    print(f"Changed blocks: {changed_blocks}")


if __name__ == "__main__":
    main()
