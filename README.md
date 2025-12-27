# Voice STT - Local Whisper Speech-to-Text

A fully local, GPU-accelerated speech-to-text system built on OpenAI Whisper (faster-whisper), designed as modular services:
- reusable STT engine package
- FastAPI backend (REST, streaming-ready)
- Gradio UI (standalone or client-to-API)
- Docker-first, CUDA-enabled, Kubernetes-friendly

## Architecture Overview

```graphql
voice-stt/
├── stt_engine/              # Core Whisper STT engine (lazy-loaded, reusable)
│   ├── __init__.py
│   └── stt_engine.py
│
├── src/
│   ├── standelone/          # Standalone Gradio app (local STT)
│   │   ├── gradio_app.py
│   │   └── Dockerfile
│   │
│   ├── fastapi/             # FastAPI STT server (REST / WS)
│   │   ├── api_server.py
│   │   └── Dockerfile
│   │
│   └── client-gradio/       # Gradio client → FastAPI server
│       ├── client_app.py
│       └── Dockerfile
│
├── tests/                   # Unit tests (engine + API, mocked)
│   ├── test_stt_engine.py
│   └── test_fastapi_api.py
│
├── docker-compose.yml       # API + Client orchestration
└── pyproject.toml
```

## Features

- 🎤 Local Whisper STT (no external APIs)
- ⚡ GPU acceleration (CUDA / NVIDIA)
- 🧠 Lazy model loading (fast startup, cached models)
- 🌐 FastAPI backend (clean REST interface)
- 🎛️ Gradio UI
- standalone (all-in-one)
- or client → API architecture
- 📦 Docker & Docker Compose
- 🧪 Unit-tested (engine + API with mocks)
- 🔧 Ubuntu 24.04 + PEP-668 safe Python setup

## Requirements

### Local Setup
- Python 3.13+
- FFmpeg
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- uv (recommended)

### Docker Setup
- Docker installed
- Docker Desktop (for Windows/Mac) or Docker Engine (for Linux)
- NVIDIA Docker runtime (if using GPU with Docker)

## Installation (Local Dev)

```bash
git clone https://github.com/bogerm/voice-stt
cd voice-stt
```

### Create environment (recommended)

```bash
uv venv
source .venv/bin/activate
```

### Install dependencies

```bash
uv pip install -e . --group dev
```

### Run tests
```bash
uv run pytest
```

### Option 1: Standalone Gradio App (all-in-one)

Runs Whisper + UI in a single process.

### Run locally
```bash
uv run python src/standelone/gradio_app.py
```

### Docker
```bash
docker build -f src/standelone/Dockerfile -t voice-stt-standalone .
docker run --rm -it --gpus all -p 7860:7860 voice-stt-standalone
```

## Option 2: FastAPI Server (STT backend)

Provides a REST API for speech-to-text.

### Endpoints

- GET /health
- POST /v1/transcribe (audio upload)

### Run with Docker
```bash
docker build -f src/fastapi/Dockerfile -t voice-stt-api .
docker run --rm -it --gpus all -p 8000:8000 voice-stt-api
```

Docs:
- 👉 http://localhost:8000/docs


## Option 3: Gradio Client → FastAPI Server (recommended)

Decoupled UI + backend, production-friendly.

### Run with Docker Compose
```bash
docker compose up --build
```


Services:
- FastAPI → http://localhost:8000
- Gradio UI → http://localhost:7860

The Gradio client sends audio to the FastAPI server over HTTP.

---

## Supported Whisper Models

| Model     | Speed | Accuracy | VRAM |
|----------|---------|----------|------|
| tiny     | 🚀🚀🚀 | ⭐        | very low |
| base     | 🚀🚀   | ⭐⭐       | low |
| small    | 🚀     | ⭐⭐⭐     | moderate |
| medium   | 🐢     | ⭐⭐⭐⭐    | high |
| large-v3 | 🐢🐢   | ⭐⭐⭐⭐⭐ | very high |

---

## Environment Variables

### FastAPI
- `PYTHONUNBUFFERED=1`

### Gradio Client
- `FASTAPI_BASE_URL`  
  (default: `http://api:8000` in docker-compose)

---

## Docker & GPU Notes

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 nvidia-smi
```

If GPU is not detected:
- Ensure NVIDIA drivers are installed on host
- Ensure NVIDIA Container Toolkit is installed

---

## Testing Strategy

- **stt_engine tests**
  - no Whisper download
  - lazy init behavior
  - edge cases
- **FastAPI tests**
  - fully mocked engine
  - no GPU / no audio decoding
  - validates HTTP contract

Run all tests:

```bash
uv run pytest
```

---


## License

Uses OpenAI Whisper models.  
See: https://github.com/openai/whisper/blob/main/LICENSE

---

## Roadmap

- WebSocket live captions
- Streaming partial transcripts
- Kubernetes GPU manifests
- API authentication & rate limiting
- Model pre-warming & batching