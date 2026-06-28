# ============================================================
# Nexialog VR Dashboard — image Flask
# Build:  docker build -t nexialog-vr .
# Run:    docker run --rm -p 5000:5000 nexialog-vr
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# System deps: gcc/g++ pour wheels sources (catboost/xgboost fallback),
# libgomp1 requis par xgboost/catboost en runtime, curl pour healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Dépendances (couche cachée tant que pyproject/poetry.lock ne bougent pas)
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root

# Code + données + artefacts + assets
COPY app.py vr_pipeline.py ./
COPY *.py ./
COPY *.json ./
COPY *.csv ./
COPY data ./data
COPY artifacts ./artifacts
COPY static ./static
COPY templates ./templates

# Cache de prédictions rempli au premier démarrage
RUN mkdir -p /app/.web_cache

# Config runtime (override via -e ou docker-compose env)
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3.1:8b \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:5000/api/data > /dev/null || exit 1

CMD ["python", "app.py"]
