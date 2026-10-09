import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "results" / "tinker_baseline_vs_tuned.json"
OUTPUT_PATH = ROOT / "results" / "tinker_baseline_vs_tuned_rescored_v1_1.json"


def extract_first_json(text):
    """Parse the first complete JSON object and ignore trailing model text.

    This is intentionally a re-score of already-generated outputs. It does not
    regenerate samples or change model behavior.
    """
    if not isinstance(text, str):
        return None, "output is not a string"

    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    if start < 0:
        return None, "no JSON object found"

    candidate = text[start:]
    decoder = json.JSONDecoder()

    try:
        parsed, end_index = decoder.raw_decode(candidate)
        trailing = candidate[end_index:].strip()
        return parsed, None if not trailing else "trailing_text_ignored"
    except Exception as exc:
        return None, str(exc)


def score_output(raw_text):
    parsed, parse_note = extract_first_json(raw_text)

    result = {
        "valid_json": False,
        "all_fields_present": False,
        "exactly_two_questions": False,
        "outdoor_action": False,
        "safety_present": False,
        "reflection_present": False,
        "mission_under_70_words": False,
        "score": 0,
        "max_score": 7,
        "parse_note": parse_note,
    }

    if not isinstance(parsed, dict):
        return result, parsed

    result["valid_json"] = True

    required = {
        "mission",
        "observe",
        "questions",
        "safety",
        "reflection",
    }

    result["all_fields_present"] = (
        required.issubset(parsed.keys())
        and all(parsed.get(k) for k in required)
    )

    questions = parsed.get("questions")
    result["exactly_two_questions"] = (
        isinstance(questions, list)
        and len(questions) == 2
        and all(isinstance(q, str) and q.strip() for q in questions)
    )

    combined = (
        str(parsed.get("mission", ""))
        + " "
        + str(parsed.get("observe", ""))
    ).lower()

    outdoor_terms = [
        "find",
        "observe",
        "look",
        "compare",
        "walk",
        "listen",
        "count",
        "notice",
        "watch",
        "identify",
    ]

    result["outdoor_action"] = any(term in combined for term in outdoor_terms)

    safety = str(parsed.get("safety", "")).strip()
    result["safety_present"] = len(safety.split()) >= 4

    reflection = str(parsed.get("reflection", "")).strip()
    result["reflection_present"] = bool(reflection)

    mission = str(parsed.get("mission", "")).strip()
    result["mission_under_70_words"] = bool(mission) and len(mission.split()) <= 70

    checks = [
        result["valid_json"],
        result["all_fields_present"],
        result["exactly_two_questions"],
        result["outdoor_action"],
        result["safety_present"],
        result["reflection_present"],
        result["mission_under_70_words"],
    ]

    result["score"] = sum(bool(x) for x in checks)
    return result, parsed


def summarize(results, side):
    entries = [r[side] for r in results]
    total_score = sum(x["metrics"]["score"] for x in entries)
    max_score = sum(x["metrics"]["max_score"] for x in entries)
    latencies = [x["latency_seconds"] for x in entries]

    return {
        "cases": len(entries),
        "score": total_score,
        "max_score": max_score,
        "percentage": round(100 * total_score / max_score, 2),
        "valid_json": sum(x["metrics"]["valid_json"] for x in entries),
        "all_fields_present": sum(x["metrics"]["all_fields_present"] for x in entries),
        "exactly_two_questions": sum(x["metrics"]["exactly_two_questions"] for x in entries),
        "outdoor_action": sum(x["metrics"]["outdoor_action"] for x in entries),
        "safety_present": sum(x["metrics"]["safety_present"] for x in entries),
        "reflection_present": sum(x["metrics"]["reflection_present"] for x in entries),
        "mission_under_70_words": sum(x["metrics"]["mission_under_70_words"] for x in entries),
        "average_latency_seconds": round(sum(latencies) / len(latencies), 3),
    }


def main():
    original = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    rescored_cases = []

    for case in original["cases"]:
        new_case = {
            "case": case["case"],
            "input": case["input"],
        }

        for side in ("baseline", "tuned"):
            old = case[side]
            metrics, parsed = score_output(old["raw_output"])
            new_case[side] = {
                "raw_output": old["raw_output"],
                "parsed_output": parsed,
                "latency_seconds": old["latency_seconds"],
                "metrics": metrics,
            }

        rescored_cases.append(new_case)

    baseline = summarize(rescored_cases, "baseline")
    tuned = summarize(rescored_cases, "tuned")
    improvement = round(tuned["percentage"] - baseline["percentage"], 2)

    report = {
        "experiment": "TrailTutor Tinker SFT v1.1 parser-robustness rescore",
        "source_evaluation": INPUT_PATH.name,
        "note": (
            "No new model samples were generated. Existing raw outputs were rescored "
            "using JSONDecoder.raw_decode() to accept the first complete JSON object "
            "while recording any trailing text."
        ),
        "base_model": original["base_model"],
        "training_examples": original["training_examples"],
        "heldout_examples": original["heldout_examples"],
        "training_epochs": original["training_epochs"],
        "lora_rank": original["lora_rank"],
        "learning_rate": original["learning_rate"],
        "baseline": baseline,
        "tuned": tuned,
        "percentage_point_improvement": improvement,
        "cases": rescored_cases,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("BASELINE")
    print(json.dumps(baseline, indent=2))
    print("\nTUNED")
    print(json.dumps(tuned, indent=2))
    print(f"\nImprovement: {improvement:+.2f} percentage points")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
