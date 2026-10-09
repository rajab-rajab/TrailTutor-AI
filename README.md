# TrailTutor AI

**TrailTutor AI** is an open-source outdoor learning companion for the DEV Hacktoberfest Week 1 "Touch Grass" challenge.

The core product idea is simple:

> Generate one short learning mission, put the screen away, explore the real world, then return only to reflect.

## Target stack

- **Gemma** — open-weight baseline mission generator
- **Tinker** — fine-tuning workflow for a specialized outdoor-learning model
- **DigitalOcean** — deployment target for the public app/API
- **FastAPI** — backend
- **Plain HTML/CSS/JavaScript** — lightweight frontend

## What is included

- Working FastAPI app
- Browser UI
- `/api/mission` endpoint for the Gemma baseline
- `/api/tuned-mission` endpoint for a tuned-model endpoint
- `/api/compare` endpoint for side-by-side comparison
- Deterministic demo mode when no model endpoint is configured
- Tinker-oriented training dataset
- Evaluation script and rubric
- Dockerfile
- DigitalOcean App Platform spec
- `.env.example`
- GitHub-ready `.gitignore`

## Quick start on Windows PowerShell

```powershell
cd TrailTutor-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt

Copy-Item .env.example .env

uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Model modes

TrailTutor supports three practical modes.

### 1. Demo mode

No AI credentials are needed.

```env
AI_MODE=demo
```

This lets you build, test, record screenshots, and deploy the basic app before configuring external model services.

### 2. Gemma through an OpenAI-compatible endpoint

Use a local or hosted endpoint that exposes an OpenAI-compatible `/v1/chat/completions` API.

Example:

```env
AI_MODE=remote
GEMMA_BASE_URL=http://127.0.0.1:8080/v1
GEMMA_MODEL=gemma
GEMMA_API_KEY=
```

You can point this at a compatible Gemma server running locally or on infrastructure you control.

### 3. Tuned model endpoint

After training/exporting your specialized model, set:

```env
TUNED_BASE_URL=https://your-endpoint.example/v1
TUNED_MODEL=trailtutor-tuned
TUNED_API_KEY=
```

The app deliberately keeps the tuned-model serving layer separate from the training workflow. This avoids coupling the product to one serving vendor.

## API

### Health

```http
GET /api/health
```

### Baseline mission

```http
POST /api/mission
Content-Type: application/json
```

Example:

```json
{
  "age": 13,
  "environment": "school playground",
  "topic": "plants",
  "duration_minutes": 10
}
```

### Tuned-model mission

```http
POST /api/tuned-mission
```

### Baseline vs tuned comparison

```http
POST /api/compare
```

## DigitalOcean deployment

The repository contains:

```text
.do/app.yaml
Dockerfile
```

Recommended path:

1. Push the repository to GitHub.
2. Create a DigitalOcean App Platform app from the repository.
3. Configure the environment variables in DigitalOcean.
4. Deploy.
5. If you later serve Gemma from a GPU Droplet or another model service, set `GEMMA_BASE_URL` to that endpoint.

Do **not** commit API keys.

## Tinker experiment

The `training/` folder contains:

- `dataset/train.jsonl`
- `dataset/eval.jsonl`
- `train_tinker.py`
- `README.md`

The training script is intentionally conservative: it prepares the experiment and validates data without pretending a training run happened.

For the final hackathon submission, record:

- exact base model
- training configuration
- number of examples
- evaluation prompts
- baseline results
- tuned results
- latency/cost measurements if available

Do not publish invented improvements.

## Evaluation

Run:

```powershell
python evaluation/evaluate.py
```

This validates the held-out examples and writes:

```text
evaluation/results/local_dataset_report.json
```

Once both baseline and tuned APIs are configured:

```powershell
python evaluation/compare_live.py
```

That creates:

```text
evaluation/results/live_comparison.json
```

## Suggested hackathon categories

Use only categories that are genuinely implemented and demonstrated.

- Overall "Touch Grass"
- Best Use of Gemma
- Best Use of Tinker
- Best Use of DigitalOcean

## Project story

TrailTutor is deliberately not a long-form chatbot.

A user selects:

- age
- environment
- topic
- available time

TrailTutor generates a short outdoor task. The user leaves the screen, completes it, and comes back for a brief reflection.

This makes the AI useful precisely because it helps the user stop using the AI for a while.

## License

MIT
