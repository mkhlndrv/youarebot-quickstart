# MLflow tracking server + model registry.
# Backend store is SQLite, artifacts live on a local directory; both are kept on
# the mlflow-data volume (see docker-compose.yaml) so runs and the registered
# champion survive restarts. `--serve-artifacts` lets the other services
# upload/download artifacts through this server, so they never touch the volume
# directly.
FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==2.17.0

WORKDIR /mlflow
EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
