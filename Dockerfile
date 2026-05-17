# PROPHET — runtime image
# Multi-purpose: works for local dev (CPU) and GPU (RunPod) when CUDA is present
ARG BASE=python:3.11-slim
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    PROPHET_CACHE_DIR=/cache/prophet

# System packages — minimal
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better layer caching)
COPY pyproject.toml ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install -e ".[api]" --no-build-isolation || true
# ".[api]" may fail without source files; install source next

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
COPY tests ./tests
COPY README.md LICENSE ./

RUN pip install -e ".[api]"

# Cache mount for HF + prophet
RUN mkdir -p /cache/huggingface /cache/prophet

ENTRYPOINT ["prophet"]
CMD ["--help"]
