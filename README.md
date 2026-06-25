# YouAreBot — microservice architecture

The HumanOrBot quick-start, split into four small services that each do one
job. The public API is a thin gateway; the model serving, the LLM and the
experiment tracking each live in their own container.

```
                         ┌───────────────────────────┐
   client  ──POST──────► │   orchestrator (gateway)  │   public API, no model
                         │   :8000                   │
                         └─────────┬─────────┬───────┘
              /predict             │         │            /get_message
                         ┌─────────▼──┐   ┌──▼──────────┐
                         │ classifier │   │     llm     │
                         │  :8000     │   │   :11434    │
                         │ /predict   │   │ /v1/chat/.. │  llama.cpp (OpenAI API)
                         └─────┬──────┘   └─────────────┘
              load champion    │
                         ┌─────▼──────┐
                         │   mlflow   │   tracking + model registry
                         │  :5000     │
                         └────────────┘
```

| Service        | Build / image                     | Internal port | Published | Role |
| -------------- | --------------------------------- | ------------- | --------- | ---- |
| `mlflow`       | `services/mlflow`                 | 5000          | 5001      | Experiment tracking + model registry (SQLite + local artifact store). |
| `classifier`   | `services/classifier`             | 8000          | —         | Loads the champion model from MLflow, exposes `POST /predict`. |
| `llm`          | `ghcr.io/ggml-org/llama.cpp`      | 11434         | 11434     | OpenAI-style `POST /v1/chat/completions` (Qwen2.5-0.5B). |
| `orchestrator` | `services/orchestrator`           | 8000          | 8000      | Public gateway. Forwards `/predict` and `/get_message`. Runs no model. |

Inside the compose network the services talk to each other **by service name**
(`http://classifier:8000`, `http://llm:11434`, `http://mlflow:5000`), never
`localhost`.

## How it works

- **MLflow** is the source of truth for the model. On its first start the
  `classifier` registers the zero-shot bot-detector as
  `youarebot-classifier@champion`, then loads it back with
  `mlflow.pyfunc.load_model("models:/youarebot-classifier@champion")`. The
  registration is idempotent, so later restarts just load the existing champion.
  The champion is a zero-shot NLI model
  (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`) that scores how likely a message
  was written by a bot.
- **The orchestrator only routes.** It builds the OpenAI chat request and reads
  the reply, but it never runs the classifier or the LLM itself:
  - `POST /predict` → `http://classifier:8000/predict`
  - `POST /get_message` → `http://llm:11434/v1/chat/completions`

## Run

```bash
docker compose up --build
```

That starts all four services. The first start is slow: it downloads the
classifier model (~560 MB from Hugging Face) and the GGUF LLM weights (~400 MB),
so give it a few minutes. The orchestrator waits until the `classifier` and
`llm` are healthy before it comes up. When everything is ready:

- Public API: <http://localhost:8000>
- MLflow UI: <http://localhost:5001> (look at the `youarebot-classifier`
  experiment and the registered model)

Stop and wipe everything (including the cached models) with:

```bash
docker compose down -v
```

## Test the endpoints

**1. `/predict` — returns a bot probability in `[0, 1]` from the classifier.**

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "text": "Hello, how are you doing today?",
        "dialog_id": "11111111-1111-1111-1111-111111111111",
        "id": "22222222-2222-2222-2222-222222222222",
        "participant_index": 0
      }'
```

Example response:

```json
{
  "id": "….",
  "message_id": "22222222-2222-2222-2222-222222222222",
  "dialog_id": "11111111-1111-1111-1111-111111111111",
  "participant_index": 0,
  "is_bot_probability": 0.31
}
```

**2. `/get_message` — returns a real reply from the LLM container.**

```bash
curl -s http://localhost:8000/get_message \
  -H "Content-Type: application/json" \
  -d '{
        "dialog_id": "11111111-1111-1111-1111-111111111111",
        "last_msg_text": "Hi! What is your favourite city?",
        "last_message_id": "33333333-3333-3333-3333-333333333333"
      }'
```

Example response:

```json
{
  "new_msg_text": "Honestly I love Lisbon, the food and the light are unbeatable!",
  "dialog_id": "11111111-1111-1111-1111-111111111111"
}
```

You can also hit the internal services directly while debugging, e.g.
`curl http://localhost:5001/health` (MLflow) or
`curl http://localhost:11434/health` (LLM). The classifier is not published; it
is only reachable from inside the compose network through the orchestrator.

## Configuration

Everything is configured with environment variables in `docker-compose.yaml`
(see `.env.example` for the full list). The important ones:

| Variable             | Service        | Default                                      |
| -------------------- | -------------- | -------------------------------------------- |
| `CLASSIFIER_URL`     | orchestrator   | `http://classifier:8000`                     |
| `LLM_URL`            | orchestrator   | `http://llm:11434`                           |
| `LLM_MODEL`          | orchestrator   | `qwen2.5-0.5b-instruct`                      |
| `MLFLOW_TRACKING_URI`| classifier     | `http://mlflow:5000`                         |
| `MODEL_NAME`         | classifier     | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`    |

The LLM runs on port **11434** (the OpenAI-compatible llama.cpp server). If you
swap in a different LLM on another port, update `LLM_URL` and keep the
`/v1/chat/completions` path.

## Secrets

No secrets, tokens or API keys are needed to run this stack, and none are
committed. `.env` is gitignored; `.env.example` only contains non-secret
defaults. The `LLM_API_KEY` value is the literal string `not-needed` (llama.cpp
ignores it).

## Fly.io

Each service is an independent container, so any one of them can be deployed to
Fly.io on its own (`fly launch` from `services/<name>`). Deployment is covered
in class; this repo is structured so a single service or the whole stack can be
shipped.
