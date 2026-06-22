"""LLM client for the bot's /get_message replies.

Talks to an OpenAI-compatible server (a local llama.cpp `--server`, see the
`llm` service in docker-compose.yaml). Configured entirely via env vars so the
same code works locally and inside Docker.
"""
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

# Inside docker-compose the llm service is reachable as http://llm:8080/v1.
# Locally (llama-cpp.sh) it is http://127.0.0.1:8080/v1.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-0.5b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "128"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))

# Turing-test persona: reply like a human, match the user's language, stay short.
SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a participant in a 'human or bot' chat game. Reply to the other "
    "person the way a real human would: casual, friendly and brief (one or two "
    "short sentences). Always answer in the SAME language the person used "
    "(Russian or English). Never say or hint that you are an AI, a bot, or a "
    "language model. Do not use lists or markdown — just chat naturally.",
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        logger.info("Creating LLM client for %s (model=%s)", LLM_BASE_URL, LLM_MODEL)
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)
    return _client


def generate_reply(user_text: str) -> str:
    """Ask the LLM for a human-like reply to `user_text`. Raises on failure."""
    client = _get_client()
    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return (completion.choices[0].message.content or "").strip()
