"""Public API gateway.

The orchestrator runs no model itself. It only forwards requests to the right
internal service, addressed by its compose service name (never localhost):

    POST /predict      -> http://classifier:8000/predict
    POST /get_message  -> http://llm:11434/v1/chat/completions
"""
from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI, HTTPException

from models import (
    GetMessageRequestModel,
    GetMessageResponseModel,
    IncomingMessage,
    Prediction,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("orchestrator")

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-0.5b-instruct")
CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "30"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# Turing-test persona: reply like a human, match the user's language, stay short.
SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a participant in a 'human or bot' chat game. Reply to the other "
    "person the way a real human would: casual, friendly and brief (one or two "
    "short sentences). Always answer in the SAME language the person used "
    "(Russian or English). Never say or hint that you are an AI, a bot, or a "
    "language model. Do not use lists or markdown, just chat naturally.",
)

app = FastAPI(title="YouAreBot Orchestrator")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
async def predict(msg: IncomingMessage) -> Prediction:
    """Forward the message to the classifier service and relay its prediction."""
    try:
        async with httpx.AsyncClient(timeout=CLASSIFIER_TIMEOUT) as client:
            response = await client.post(
                f"{CLASSIFIER_URL}/predict", json=msg.model_dump(mode="json")
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("classifier request failed: %s", exc)
        raise HTTPException(status_code=502, detail="classifier request failed") from exc

    return Prediction(**response.json())


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel) -> GetMessageResponseModel:
    """Forward the user's text to the LLM (OpenAI chat API) and return its reply."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": body.last_msg_text},
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
    }
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            response = await client.post(
                f"{LLM_URL}/v1/chat/completions", json=payload
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("llm request failed: %s", exc)
        raise HTTPException(status_code=502, detail="llm request failed") from exc

    data = response.json()
    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("unexpected LLM response: %s", data)
        raise HTTPException(status_code=502, detail="invalid LLM response") from exc

    logger.info("get_message dialog_id=%s -> %s", body.dialog_id, reply)
    return GetMessageResponseModel(new_msg_text=reply, dialog_id=body.dialog_id)
