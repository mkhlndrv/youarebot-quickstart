# Shared image for the `fastapi` (bot backend) and `streamlit` (chat UI) services.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl is handy for container healthchecks / debugging.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app

# So `from app...` resolves under `streamlit run` (it puts the script dir, not
# the workdir, on sys.path).
ENV PYTHONPATH=/app

# Default command is overridden per-service in docker-compose.yaml.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "6872"]
