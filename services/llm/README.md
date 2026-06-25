# `llm` service

This service is the LLM backend. It runs the upstream
[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) server image
directly (no custom build), so there is nothing to compile here — it is wired up
entirely in [`../../docker-compose.yaml`](../../docker-compose.yaml).

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M`, downloaded from Hugging
  Face on first start and cached in the `llm-models` Docker volume.
- **Internal port:** `11434`.
- **API:** OpenAI-compatible. The orchestrator calls
  `POST http://llm:11434/v1/chat/completions`.

Swap the model by editing the `-hf` argument of the `llm` service in the compose
file (any GGUF on Hugging Face works). If you point the orchestrator at a
different LLM (for example Ollama, which also defaults to port 11434), keep the
`/v1/chat/completions` path and update `LLM_URL` / `LLM_MODEL` accordingly.
