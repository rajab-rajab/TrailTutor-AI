from pathlib import Path
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT.parent / "training" / "dataset" / "eval.jsonl"
RESULT_PATH = ROOT / "results" / "live_comparison.json"
BASE_URL = "http://127.0.0.1:8000"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def post(path, payload):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    elapsed = time.perf_counter() - start
    return data, elapsed


def score(mission):
    score = 0
    notes = []

    required = ["mission", "observe", "questions", "safety", "reflection"]
    if all(mission.get(k) for k in required) and len(mission.get("questions", [])) == 2:
        score += 2
    else:
        notes.append("structure_failed")

    if len(mission.get("mission", "").split()) <= 70:
        score += 1
    else:
        notes.append("mission_too_long")

    safety_text = mission.get("safety", "").lower()
    if len(safety_text.split()) >= 4:
        score += 2
    else:
        notes.append("weak_safety")

    if mission.get("reflection"):
        score += 1

    action_terms = ["find", "observe", "compare", "walk", "look", "count", "notice"]
    mission_text = (mission.get("mission", "") + " " + mission.get("observe", "")).lower()
    if any(term in mission_text for term in action_terms):
        score += 2
    else:
        notes.append("weak_outdoor_action")

    return {"score": score, "max_score": 8, "notes": notes}


if __name__ == "__main__":
    cases = load_jsonl(EVAL_PATH)
    results = []

    for case in cases:
        payload = case["input"]
        baseline, baseline_s = post("/api/mission", payload)
        tuned, tuned_s = post("/api/tuned-mission", payload)

        results.append({
            "input": payload,
            "baseline": {
                "latency_seconds": round(baseline_s, 4),
                "rubric": score(baseline),
                "output": baseline,
            },
            "tuned": {
                "latency_seconds": round(tuned_s, 4),
                "rubric": score(tuned),
                "output": tuned,
            },
        })

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote: {RESULT_PATH}")
