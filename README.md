# TrailTutor AI

**TrailTutor AI** is an open-source outdoor learning companion built for the DEV Hacktoberfest Week 1 2026 **Touch Grass** challenge.

> Generate one short learning mission, put the screen away, explore the real world, then return only to reflect.

## Live demo

https://trailtutor-ai.onrender.com/

## What it does

A learner selects an age, environment, topic, and available time. TrailTutor generates one concise outdoor mission with a real-world observation task, exactly two guiding questions, a safety instruction, and a short reflection prompt.

The design goal is simple: **the screen should be the shortest part of the experience.**

## Technology stack

- **Gemma:** `google/diffusiongemma-26b-a4b-it`, served through the NVIDIA API
- **Tinker:** LoRA fine-tuning of `Qwen/Qwen3.5-4B`
- **Render:** public Docker deployment of the FastAPI application
- **FastAPI:** backend API
- **HTML / CSS / JavaScript:** lightweight frontend

## Architecture

```text
Browser
  |
  v
TrailTutor FastAPI app on Render
  |
  +--> NVIDIA API
  |      google/diffusiongemma-26b-a4b-it
  |
  +--> Tinker experiment
         Qwen/Qwen3.5-4B
              |
              +--> untuned baseline
              +--> LoRA-tuned TrailTutor checkpoint
```

## Verified Tinker experiment

### Training configuration

- Base model: `Qwen/Qwen3.5-4B`
- Training examples: **80**
- Held-out evaluation cases: **20**
- LoRA rank: **16**
- Epochs: **1**
- Batch size: **4**
- Learning rate: **1e-4**

### Corrected held-out results

The first evaluation used a strict JSON parser that rejected outputs when a model emitted two consecutive valid JSON objects. A documented v1.1 rescore parses the first complete JSON object while preserving the original raw output and original evaluation file.

| Metric | Untuned baseline | Tinker-tuned |
|---|---:|---:|
| Overall rubric score | 95% | **100%** |
| Valid JSON | 19/20 | **20/20** |
| All required fields | 19/20 | **20/20** |
| Exactly two questions | 19/20 | **20/20** |
| Outdoor action present | 19/20 | **20/20** |
| Safety present | 19/20 | **20/20** |
| Reflection present | 19/20 | **20/20** |
| Mission under 70 words | 19/20 | **20/20** |
| Average latency | 3.355 s | **2.907 s** |

The tuned model improved the held-out structured-output score by **5 percentage points** and reduced average latency by about **13.4%**.

Evidence:

```text
evaluation/results/tinker_baseline_vs_tuned.json
evaluation/results/tinker_baseline_vs_tuned_rescored_v1_1.json
```

## Dataset

The dataset includes:

- **80** supervised training examples
- **20** held-out evaluation prompts
- ages **7–17**
- **8** outdoor environments
- **20** topics
- **9** activity durations

The held-out prompts have no exact input overlap with the training set.

Key files:

```text
training/dataset/train.jsonl
training/dataset/eval.jsonl
training/dataset/train_messages.jsonl
```

## API

### Health

```http
GET /api/health
```

### Generate a Gemma mission

```http
POST /api/mission
Content-Type: application/json
```

Example request:

```json
{
  "age": 13,
  "environment": "school playground",
  "topic": "plants",
  "duration_minutes": 10
}
```

### Tuned-model adapter

```http
POST /api/tuned-mission
```

### Compare model paths

```http
POST /api/compare
```

## Local setup

```powershell
cd TrailTutor-AI
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Tinker workflow

The repository contains:

```text
training/run_tinker_sft.py
evaluation/evaluate_tinker.py
evaluation/rescore_existing_outputs.py
```

The held-out evaluation compares the same base model before and after LoRA fine-tuning.

## Render deployment

The public app is deployed from this GitHub repository using the root `Dockerfile`. Render hosts the FastAPI web service while Gemma inference is handled by the NVIDIA API.

Live URL: https://trailtutor-ai.onrender.com/

## Challenge categories

TrailTutor genuinely implements technology relevant to:

- Overall **Touch Grass** challenge
- **Best Use of Gemma**
- **Best Use of Tinker**
- **Best Use of Render**

## Why this project matters

TrailTutor uses AI to shorten screen time rather than extend it. The model gives the learner one clear mission, then the learner leaves the device and engages with the physical environment.

The project also treats specialization as an experiment rather than a marketing claim: training data, held-out prompts, raw outputs, parser behavior, and corrected evaluation evidence are kept in the repository.

## License

MIT
