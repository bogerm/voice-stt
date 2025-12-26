FROM nvidia/cuda:12.8.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip3 install --no-cache-dir uv

# Copy project metadata first
COPY pyproject.toml /app/pyproject.toml

# Install deps
RUN uv pip install --system --no-cache-dir \
  -r <(python3 -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

# Copy source
COPY stt_engine /app/stt_engine
COPY src/standelone /app/src/standelone

EXPOSE 7860
CMD ["python3", "-m", "src.standelone.gradio_app"]
