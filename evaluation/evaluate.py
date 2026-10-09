from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT.parent / "training" / "dataset" / "eval.jsonl"
RESULT_PATH = ROOT / "results" / "local_dataset_report.json"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    rows = load_jsonl(EVAL_PATH)

    report = {
        "held_out_prompt_count": len(rows),
        "status": "ready",
        "note": (
            "This report validates that a held-out evaluation set exists. "
            "It does not claim model performance."
        ),
        "prompts": [row["input"] for row in rows],
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote: {RESULT_PATH}")
