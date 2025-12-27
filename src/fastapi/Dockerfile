FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    ffmpeg \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

ENV PYTHONPATH="/app:${PYTHONPATH}"

COPY pyproject.toml /app/pyproject.toml

RUN python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print('\n'.join(d['project']['dependencies']))" \
    > /tmp/requirements.txt \
 && uv pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

COPY stt_engine /app/stt_engine
COPY src/fastapi /app/src/fastapi

EXPOSE 8000
CMD ["uvicorn", "src.fastapi.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
