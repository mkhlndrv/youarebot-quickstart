"""The champion bot-detector and the code that registers it in MLflow.

The model is a zero-shot NLI classifier (the same mDeBERTa setup from Session
8/9). We wrap it as an MLflow pyfunc so the registry owns it: the classifier
service never hard-codes a checkpoint, it just loads
``models:/youarebot-classifier@champion`` and serves whatever MLflow returns.

We compare two explicit hypotheses (single-label softmax) and read the bot
score; the wording below was the strongest of several tested on the Kaggle
train set.
"""
from __future__ import annotations

import logging
import os

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient

logger = logging.getLogger("classifier.model")

# Override the checkpoint with MODEL_NAME (e.g. facebook/bart-large-mnli).
MODEL_NAME = os.getenv("MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "youarebot-classifier")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "youarebot-classifier")
CHAMPION_ALIAS = "champion"

HUMAN_HYPOTHESIS = "This text was written by a real person."
BOT_HYPOTHESIS = "This text was written by an AI chatbot."
MAX_CHARS = 1500


class BotZeroShotClassifier(mlflow.pyfunc.PythonModel):
    """Zero-shot 'human or bot' classifier, served as an MLflow model.

    ``load_context`` builds the HuggingFace pipeline once; ``predict`` takes a
    pandas DataFrame with a ``text`` column (or a plain list of strings) and
    returns the bot probability in [0, 1] for each row.
    """

    def load_context(self, context):  # noqa: D102 - MLflow hook
        from transformers import pipeline

        logger.info("Loading zero-shot checkpoint '%s'...", MODEL_NAME)
        self._pipe = pipeline(
            "zero-shot-classification", model=MODEL_NAME, device=-1
        )

    def _score_one(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.5  # nothing to judge yet -> stay neutral
        text = text[:MAX_CHARS]
        result = self._pipe(
            text,
            candidate_labels=[HUMAN_HYPOTHESIS, BOT_HYPOTHESIS],
            hypothesis_template="{}",
            multi_label=False,
        )
        scores = dict(zip(result["labels"], result["scores"]))
        human = float(scores.get(HUMAN_HYPOTHESIS, 0.0))
        bot = float(scores.get(BOT_HYPOTHESIS, 0.0))
        probability = bot / max(human + bot, 1e-9)
        return max(0.0, min(1.0, probability))

    def predict(self, context, model_input, params=None):  # noqa: D102
        if hasattr(model_input, "columns"):  # pandas DataFrame
            texts = model_input["text"].tolist()
        elif isinstance(model_input, dict):
            texts = list(model_input["text"])
        else:
            texts = list(model_input)
        return [self._score_one(t) for t in texts]


def _client() -> MlflowClient:
    return MlflowClient()


def champion_exists() -> bool:
    """True if a model is already registered under the champion alias."""
    try:
        _client().get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
        return True
    except Exception:  # noqa: BLE001 - any miss means "not registered yet"
        return False


def register_champion() -> str:
    """Log the zero-shot model to MLflow and tag the new version as champion."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="register-champion"):
        mlflow.log_params(
            {
                "checkpoint": MODEL_NAME,
                "approach": "zero-shot-nli",
                "human_hypothesis": HUMAN_HYPOTHESIS,
                "bot_hypothesis": BOT_HYPOTHESIS,
            }
        )
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=BotZeroShotClassifier(),
            code_paths=[__file__],
            registered_model_name=REGISTERED_MODEL_NAME,
            pip_requirements=[
                "mlflow==2.17.0",
                "transformers>=4.40.0,<4.45.0",
                "torch",
                "sentencepiece",
                "protobuf<5",
            ],
        )

    client = _client()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME, CHAMPION_ALIAS, latest.version
    )
    logger.info(
        "Registered %s v%s as @%s", REGISTERED_MODEL_NAME, latest.version, CHAMPION_ALIAS
    )
    return latest.version


def ensure_champion() -> None:
    """Make sure a champion exists in the registry (idempotent bootstrap)."""
    if champion_exists():
        logger.info("Champion already present in MLflow; skipping registration.")
        return
    logger.info("No champion found in MLflow; registering one...")
    register_champion()
