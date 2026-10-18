# CPU deployment for instructor queries. No models or user data enter the image.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CATSTUDIO_DEVICE=cpu \
    CATSTUDIO_BASE_MODEL=/models/base_sd15 \
    CATSTUDIO_ADAPTER=/models/cat_lora \
    HF_HOME=/tmp/huggingface

WORKDIR /app
COPY requirements-lock-cpu.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
    && pip install -r requirements-lock-cpu.txt
COPY catstudio ./catstudio
COPY scripts ./scripts
COPY config ./config
COPY licenses ./licenses
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"
CMD ["uvicorn", "catstudio.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
