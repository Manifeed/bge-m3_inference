# syntax=docker/dockerfile:1.7

ARG PYTORCH_RUNTIME_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

FROM ${PYTORCH_RUNTIME_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    SOURCE_EMBEDDING_MODEL_ID=BAAI/bge-m3 \
    SOURCE_EMBEDDING_MODEL_PATH=/opt/models/bge-m3 \
    BGE_M3_INFERENCE_USE_FP16=true \
    BGE_M3_INFERENCE_MAX_LENGTH=512 \
    BGE_M3_INFERENCE_BATCH_MAX_ITEMS=64 \
    BGE_M3_INFERENCE_BATCH_MAX_TOKENS=16384 \
    BGE_M3_INFERENCE_BATCH_MAX_WAIT_MS=5 \
    BGE_M3_INFERENCE_QUEUE_MAX_ITEMS=1024 \
    BGE_M3_INFERENCE_REQUEST_TIMEOUT_SECONDS=300 \
    BGE_M3_INFERENCE_SHUTDOWN_GRACE_SECONDS=30 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/tmp/huggingface

RUN useradd --create-home --home-dir /home/appuser --shell /usr/sbin/nologin appuser

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
from FlagEmbedding import BGEM3FlagModel

model_id = os.environ["SOURCE_EMBEDDING_MODEL_ID"]
model_path = os.environ["SOURCE_EMBEDDING_MODEL_PATH"]

snapshot_download(repo_id=model_id, local_dir=model_path)
BGEM3FlagModel(model_path, use_fp16=False)
PY

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=appuser:appuser app /app/app

RUN find /app -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && find /app -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && find /opt/conda -type d \( -name __pycache__ -o -name tests -o -name test \) -prune -exec rm -rf '{}' + \
    && find /opt/conda -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete \
    && rm -rf /opt/conda/pkgs /root/.cache /tmp/requirements.txt /tmp/huggingface

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/internal/health').read()"

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
