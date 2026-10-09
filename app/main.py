from pathlib import Path
import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="TrailTutor AI",
    version="0.1.0",
    description="A short-screen-time outdoor learning companion.",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class MissionRequest(BaseModel):
    age: int = Field(default=13, ge=5, le=100)
    environment: str = Field(default="school playground", min_length=2, max_length=120)
    topic: str = Field(default="plants", min_length=2, max_length=120)
    duration_minutes: int = Field(default=10, ge=3, le=60)


class Mission(BaseModel):
    mission: str
    observe: str
    questions: list[str]
    safety: str
    reflection: str
    provider: str


def system_prompt() -> str:
    return (
        "You are TrailTutor, an outdoor learning mission designer. "
        "Your job is to minimize screen time. Create one concise, age-appropriate, "
        "safe outdoor mission that can be completed in the user's stated environment. "
        "Do not encourage tasting plants, entering unsafe areas, touching wildlife, "
        "or damaging property. Return strict JSON with keys: mission, observe, "
        "questions (array of 2 strings), safety, reflection."
    )


def user_prompt(req: MissionRequest) -> str:
    return (
        f"Age: {req.age}\n"
        f"Environment: {req.environment}\n"
        f"Topic: {req.topic}\n"
        f"Available time: {req.duration_minutes} minutes\n"
        "Keep the mission practical and short enough that the learner can read it quickly and put the screen away."
    )


def demo_mission(req: MissionRequest, provider: str) -> Mission:
    topic = req.topic.strip().lower()
    environment = req.environment.strip()

    return Mission(
        mission=(
            f"Spend {min(req.duration_minutes, 10)} minutes in the {environment}. "
            f"Find three real examples connected to {topic}. Do not photograph them yet; first observe with your own eyes."
        ),
        observe=(
            "Compare shape, size, colour, position, texture, movement, or pattern. "
            "Notice one similarity and one difference."
        ),
        questions=[
            f"Which example best represents {topic}, and what evidence supports your choice?",
            "What detail did you notice only after looking carefully?"
        ],
        safety=(
            "Stay in a permitted area, avoid roads and hazards, do not taste unknown plants, "
            "do not touch wildlife, and do not damage anything you observe."
        ),
        reflection=(
            "Return after the mission and write one sentence about something you noticed that you would probably have missed on a screen."
        ),
        provider=provider,
    )


async def call_openai_compatible(
    base_url: str,
    model: str,
    api_key: str,
    req: MissionRequest,
    provider: str,
) -> Mission:
    import json

    if not base_url:
        raise HTTPException(status_code=503, detail=f"{provider} endpoint is not configured.")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": system_prompt() + "\n\n" + user_prompt(req)
        }
    ],
    "max_tokens": 700,
    "stream": False,
    "temperature": 0.4,
    "top_p": 0.95,
    "chat_template_kwargs": {
        "enable_thinking": False
    }
}

    timeout = float(os.getenv("MODEL_TIMEOUT_SECONDS", "60"))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"{provider} request failed: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        parsed = json.loads(content)
        return Mission(
            mission=parsed["mission"],
            observe=parsed["observe"],
            questions=parsed["questions"][:2],
            safety=parsed["safety"],
            reflection=parsed["reflection"],
            provider=provider,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"{provider} returned an invalid structured response."
        ) from exc


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "ai_mode": os.getenv("AI_MODE", "demo"),
        "gemma_configured": bool(os.getenv("GEMMA_BASE_URL")),
        "tuned_configured": bool(os.getenv("TUNED_BASE_URL")),
    }


@app.post("/api/mission", response_model=Mission)
async def mission(req: MissionRequest):
    if os.getenv("AI_MODE", "demo").lower() == "demo":
        return demo_mission(req, "gemma-demo")

    return await call_openai_compatible(
        base_url=os.getenv("GEMMA_BASE_URL", ""),
        model=os.getenv("GEMMA_MODEL", "gemma"),
        api_key=os.getenv("GEMMA_API_KEY") or os.getenv("NVIDIA_API_KEY", ""),
        req=req,
        provider="gemma-baseline",
    )


@app.post("/api/tuned-mission", response_model=Mission)
async def tuned_mission(req: MissionRequest):
    if not os.getenv("TUNED_BASE_URL"):
        return demo_mission(req, "tuned-demo")

    return await call_openai_compatible(
        base_url=os.getenv("TUNED_BASE_URL", ""),
        model=os.getenv("TUNED_MODEL", "trailtutor-tuned"),
        api_key=os.getenv("TUNED_API_KEY", ""),
        req=req,
        provider="tuned-model",
    )


@app.post("/api/compare")
async def compare(req: MissionRequest):
    baseline = await mission(req)
    tuned = await tuned_mission(req)
    return {
        "input": req.model_dump(),
        "baseline": baseline.model_dump(),
        "tuned": tuned.model_dump(),
    }
