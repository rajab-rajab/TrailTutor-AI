import json
import time
from pathlib import Path

import tinker


BASE_MODEL = "Qwen/Qwen3.5-4B"

TUNED_CHECKPOINT = (
    "tinker://8bb8578c-2d0c-5a5d-b8e2-e311ac13e8c5:train:0/"
    "sampler_weights/trailtutor-sft-v1-sampler"
)

ROOT = Path(__file__).resolve().parent
EVAL_PATH = ROOT.parent / "training" / "dataset" / "eval.jsonl"
OUTPUT_PATH = ROOT / "results" / "tinker_baseline_vs_tuned.json"


SYSTEM_PROMPT = """
You are TrailTutor, an outdoor learning mission designer.

Create one concise, safe, age-appropriate outdoor learning mission.

The goal is to minimize screen time and encourage direct observation
of the physical world.

Return ONLY valid JSON using exactly this structure:

{
  "mission": "string",
  "observe": "string",
  "questions": ["string", "string"],
  "safety": "string",
  "reflection": "string"
}

Rules:
- exactly two questions
- include a clear safety instruction
- mission must involve real-world observation
- mission should be under 70 words
- do not encourage tasting plants
- do not touch wildlife
- do not damage plants or property
- do not enter unsafe or restricted areas
""".strip()


def load_eval():
    rows = []

    with EVAL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def make_prompt(inp):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Age: {inp['age']}\n"
        f"Environment: {inp['environment']}\n"
        f"Topic: {inp['topic']}\n"
        f"Available time: {inp['duration_minutes']} minutes\n\n"
        "JSON:"
    )


def generate(client, tokenizer, prompt_text):
    prompt_tokens = tokenizer.encode(prompt_text)

    prompt = tinker.ModelInput.from_ints(prompt_tokens)

    params = tinker.SamplingParams(
        max_tokens=400,
        temperature=0.2,
    )

    start = time.perf_counter()

    result = client.sample(
        prompt=prompt,
        sampling_params=params,
        num_samples=1,
    ).result()

    latency = time.perf_counter() - start

    sequence = result.sequences[0]
    text = tokenizer.decode(sequence.tokens).strip()

    return text, latency


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text), None
    except Exception as exc:
        return None, str(exc)


def score_output(raw_text):
    parsed, json_error = extract_json(raw_text)

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
        "json_error": json_error,
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

    result["outdoor_action"] = any(
        term in combined for term in outdoor_terms
    )

    safety = str(parsed.get("safety", "")).strip()

    result["safety_present"] = len(safety.split()) >= 4

    reflection = str(parsed.get("reflection", "")).strip()

    result["reflection_present"] = bool(reflection)

    mission = str(parsed.get("mission", "")).strip()

    result["mission_under_70_words"] = (
        bool(mission)
        and len(mission.split()) <= 70
    )

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
        "percentage": round(
            100 * total_score / max_score, 2
        ),
        "valid_json": sum(
            x["metrics"]["valid_json"] for x in entries
        ),
        "all_fields_present": sum(
            x["metrics"]["all_fields_present"] for x in entries
        ),
        "exactly_two_questions": sum(
            x["metrics"]["exactly_two_questions"] for x in entries
        ),
        "outdoor_action": sum(
            x["metrics"]["outdoor_action"] for x in entries
        ),
        "safety_present": sum(
            x["metrics"]["safety_present"] for x in entries
        ),
        "reflection_present": sum(
            x["metrics"]["reflection_present"] for x in entries
        ),
        "mission_under_70_words": sum(
            x["metrics"]["mission_under_70_words"] for x in entries
        ),
        "average_latency_seconds": round(
            sum(latencies) / len(latencies), 3
        ),
    }


def main():
    cases = load_eval()

    print(f"Held-out cases: {len(cases)}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Tuned checkpoint: {TUNED_CHECKPOINT}")

    service = tinker.ServiceClient()

    print("\nCreating untuned baseline sampler...")

    baseline_client = service.create_sampling_client(
        base_model=BASE_MODEL
    )

    print("Creating tuned sampler...")

    tuned_client = service.create_sampling_client(
        model_path=TUNED_CHECKPOINT
    )

    tokenizer = baseline_client.get_tokenizer()

    results = []

    for i, case in enumerate(cases, start=1):

        inp = case["input"]
        prompt = make_prompt(inp)

        print(
            f"\nCase {i:02d}/{len(cases)} | "
            f"age={inp['age']} | "
            f"topic={inp['topic']}"
        )

        baseline_text, baseline_latency = generate(
            baseline_client,
            tokenizer,
            prompt,
        )

        tuned_text, tuned_latency = generate(
            tuned_client,
            tokenizer,
            prompt,
        )

        baseline_metrics, baseline_json = score_output(
            baseline_text
        )

        tuned_metrics, tuned_json = score_output(
            tuned_text
        )

        print(
            f"  baseline: "
            f"{baseline_metrics['score']}/7 | "
            f"{baseline_latency:.2f}s"
        )

        print(
            f"  tuned:    "
            f"{tuned_metrics['score']}/7 | "
            f"{tuned_latency:.2f}s"
        )

        results.append(
            {
                "case": i,
                "input": inp,

                "baseline": {
                    "raw_output": baseline_text,
                    "parsed_output": baseline_json,
                    "latency_seconds": round(
                        baseline_latency, 3
                    ),
                    "metrics": baseline_metrics,
                },

                "tuned": {
                    "raw_output": tuned_text,
                    "parsed_output": tuned_json,
                    "latency_seconds": round(
                        tuned_latency, 3
                    ),
                    "metrics": tuned_metrics,
                },
            }
        )

    baseline_summary = summarize(results, "baseline")
    tuned_summary = summarize(results, "tuned")

    improvement = round(
        tuned_summary["percentage"]
        - baseline_summary["percentage"],
        2
    )

    report = {
        "experiment": "TrailTutor Tinker SFT v1",
        "base_model": BASE_MODEL,
        "training_examples": 80,
        "heldout_examples": len(cases),
        "training_epochs": 1,
        "lora_rank": 16,
        "learning_rate": 1e-4,
        "baseline": baseline_summary,
        "tuned": tuned_summary,
        "percentage_point_improvement": improvement,
        "cases": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print("\nUNTUNED BASELINE")
    print(json.dumps(
        baseline_summary,
        indent=2
    ))

    print("\nTINKER-TUNED")
    print(json.dumps(
        tuned_summary,
        indent=2
    ))

    print(
        "\nImprovement:",
        f"{improvement:+.2f} percentage points"
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()