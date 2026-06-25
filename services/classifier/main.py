"""Classifier microservice.

On startup it makes sure a champion model is registered in MLflow, then loads
that champion from the registry and serves predictions. The model itself is
owned by MLflow; this service only loads and runs it.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from uuid import uuid4

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

from model import CHAMPION_ALIAS, REGISTERED_MODEL_NAME, ensure_champion
from models import IncomingMessage, Prediction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("classifier")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking URI: %s", MLFLOW_TRACKING_URI)
    ensure_champion()
    logger.info("Loading champion model from %s ...", MODEL_URI)
    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info("Champion model loaded; classifier ready.")
    yield
    app.state.model = None


app = FastAPI(title="YouAreBot Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    ready = getattr(app.state, "model", None) is not None
    return {"status": "ok" if ready else "loading"}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """Return the probability in [0, 1] that this message was written by a bot."""
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")

    probability = float(model.predict(pd.DataFrame({"text": [msg.text]}))[0])
    logger.info("is_bot_probability=%.3f for message %s", probability, msg.id)

    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=probability,
    )
