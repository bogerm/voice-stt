FROM nvidia/cuda:12.8.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ffmpeg \
    ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip3 install --no-cache-dir uv

COPY pyproject.toml /app/pyproject.toml

# Install dependencies (system-wide in container)
RUN uv pip install --system --no-cache-dir -r <(python3 -c "import tomllib;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")

COPY app.py /app/app.py

EXPOSE 7860
CMD ["python3", "app.py"]
