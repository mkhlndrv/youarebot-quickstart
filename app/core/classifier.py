import logging
import os
import time

from transformers import pipeline

logger = logging.getLogger(__name__)

# Pretrained zero-shot NLI classifier (Session 8, Route D).
#
# Instead of the previous English-only DistilBERT-MNLI baseline, we default to a
# multilingual NLI model so the RU + EN chats on youare.bot are both handled. We
# compare two explicit hypotheses (single-label softmax) and read the bot score.
# The wording below was the strongest of several tested locally on the Kaggle
# train set (see ../you-are-bot-2/sweep_zeroshot.py: AUC 0.60, log-loss 0.72).
#
# Override the checkpoint with the MODEL_NAME env var (e.g. facebook/bart-large-mnli).
MODEL_NAME = os.getenv("MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
HUMAN_HYPOTHESIS = "This text was written by a real person."
BOT_HYPOTHESIS = "This text was written by an AI chatbot."
MAX_CHARS = 1500

_classifier = None


def load_model():
    """Load the zero-shot model once and reuse it on later calls."""
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot model '%s'...", MODEL_NAME)
        start = time.perf_counter()
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,
        )
        logger.info("Model '%s' loaded in %.1fs", MODEL_NAME, time.perf_counter() - start)
    return _classifier


def classify_message(text: str) -> float:
    """Probability in [0, 1] that `text` was written by a bot."""
    classifier = load_model()

    text = (text or "").strip()
    if not text:
        return 0.5  # nothing to judge yet -> stay neutral
    text = text[:MAX_CHARS]

    start = time.perf_counter()
    result = classifier(
        text,
        candidate_labels=[HUMAN_HYPOTHESIS, BOT_HYPOTHESIS],
        hypothesis_template="{}",
        multi_label=False,
    )
    scores = dict(zip(result["labels"], result["scores"]))
    human = float(scores.get(HUMAN_HYPOTHESIS, 0.0))
    bot = float(scores.get(BOT_HYPOTHESIS, 0.0))
    probability = bot / max(human + bot, 1e-9)

    logger.info(
        "classified (%.1f ms): bot=%.3f human=%.3f -> is_bot_probability=%.3f",
        (time.perf_counter() - start) * 1000,
        bot,
        human,
        probability,
    )
    return max(0.0, min(1.0, probability))
