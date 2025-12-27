FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv \
    ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# venv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy metadata first for caching
COPY pyproject.toml /app/pyproject.toml

# Install only what the client needs (gradio + requests)
# (We intentionally do NOT install faster-whisper / server deps here.)
RUN python3 -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); deps=d['project']['dependencies']; keep=[x for x in deps if x.startswith('gradio')]; print('\n'.join(keep + ['requests>=2.31.0']))" \
    > /tmp/requirements.txt \
 && uv pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -f /tmp/requirements.txt

COPY src/client-gradio /app/src/client-gradio

EXPOSE 7860

ENV FASTAPI_BASE_URL="http://api:8000"
CMD ["python3", "/app/src/client-gradio/client_app.py"]
