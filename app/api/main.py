from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI

from app.core.llm import LLM_BASE_URL, LLM_MODEL, generate_reply
from app.core.logging import app_logger
from app.models import (
    GetMessageRequestModel,
    GetMessageResponseModel,
    IncomingMessage,
    Prediction,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The LLM lives in its own container (the `llm` service); nothing heavy to
    # load here, so the bot backend starts instantly.
    app_logger.info("Bot backend up. LLM at %s (model=%s)", LLM_BASE_URL, LLM_MODEL)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    Receive a message from the platform and answer with a real LLM reply.

    The user's text is sent to the LLM container (OpenAI-compatible llama.cpp
    server) which returns a human-like response. If the LLM is unreachable we
    fall back to echoing so the chat never hard-fails.
    """
    app_logger.info(
        "get_message dialog_id=%s last_msg_id=%s", body.dialog_id, body.last_message_id
    )
    try:
        reply = generate_reply(body.last_msg_text)
        if not reply:
            raise ValueError("empty LLM reply")
    except Exception as exc:  # noqa: BLE001 - keep the chat alive on any LLM error
        app_logger.error("LLM call failed (%s); falling back to echo", exc)
        reply = body.last_msg_text

    app_logger.info("reply: %s", reply)
    return GetMessageResponseModel(new_msg_text=reply, dialog_id=body.dialog_id)


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Classify a single message. The zero-shot classifier (torch/transformers) is
    imported lazily so this lean bot image stays small; when those deps are not
    installed we return a neutral 0.5 instead of crashing.
    """
    try:
        from app.core.classifier import classify_message

        is_bot_probability = classify_message(msg.text)
    except Exception as exc:  # noqa: BLE001
        app_logger.warning("classifier unavailable (%s); returning neutral 0.5", exc)
        is_bot_probability = 0.5

    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability,
    )
