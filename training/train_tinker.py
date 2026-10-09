"""
TrailTutor Tinker training preparation script.

This file validates and formats the project dataset. It intentionally does not
hard-code a Tinker training API call because supported model IDs and SDK details
can change. After selecting a supported model in your Tinker account, wire this
prepared data into the official current Tinker SFT example.

This design prevents the repository from pretending a training run occurred.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "dataset" / "train.jsonl"

REQUIRED_OUTPUT_KEYS = {"mission", "observe", "questions", "safety", "reflection"}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def validate_row(row, i):
    if "input" not in row or "output" not in row:
        raise ValueError(f"row {i}: input/output required")

    missing = REQUIRED_OUTPUT_KEYS - set(row["output"])
    if missing:
        raise ValueError(f"row {i}: missing output keys: {sorted(missing)}")

    questions = row["output"]["questions"]
    if not isinstance(questions, list) or len(questions) != 2:
        raise ValueError(f"row {i}: exactly two questions required")


def to_messages(row):
    inp = row["input"]
    out = row["output"]

    system = (
        "You are TrailTutor. Create one concise, safe outdoor learning mission "
        "that minimizes screen time and returns strict JSON."
    )

    user = (
        f"Age: {inp['age']}\n"
        f"Environment: {inp['environment']}\n"
        f"Topic: {inp['topic']}\n"
        f"Duration: {inp['duration_minutes']} minutes"
    )

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
        ]
    }


if __name__ == "__main__":
    rows = load_jsonl(TRAIN_PATH)

    for i, row in enumerate(rows, start=1):
        validate_row(row, i)

    formatted = [to_messages(row) for row in rows]
    out_path = ROOT / "dataset" / "train_messages.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for row in formatted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Validated {len(rows)} training examples.")
    print(f"Wrote: {out_path}")
    print("Next: connect this file to the current official Tinker SFT workflow.")
