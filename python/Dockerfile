FROM python:3.14.10-slim-trixie@sha256:3cabfd7d7aaf394e6040e79fa1eec6ca1c52b74ed9188650472695d9a9326999

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DSPY_LOG_LEVEL=ERROR \
    LITELLM_LOCAL_MODEL_COST_MAP=True

COPY python/requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir -r /tmp/requirements.lock

WORKDIR /app
