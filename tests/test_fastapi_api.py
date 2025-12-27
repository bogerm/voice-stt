import io
from dataclasses import dataclass
from typing import Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.fastapi.api_server import app


@dataclass(frozen=True)
class FakeResult:
    text: str
    detected_language: Optional[str]
    language_probability: Optional[float]
    seconds: float


class FakeEngine:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio_path: str, language=None, beam_size=5, vad_filter=True):
        self.calls.append(
            {"audio_path": audio_path, "language": language, "beam_size": beam_size, "vad_filter": vad_filter}
        )
        return FakeResult(
            text="hello world",
            detected_language="en",
            language_probability=0.9,
            seconds=0.1234,
        )


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_transcribe_success(monkeypatch, client):
    fake = FakeEngine()

    # Patch get_engine in the module where it is used
    import src.fastapi.api_server as api

    monkeypatch.setattr(api, "get_engine", lambda model: fake)

    # upload a tiny "wav"
    files = {"file": ("a.wav", b"RIFFxxxxWAVE", "audio/wav")}
    r = client.post("/v1/transcribe?model=small&beam_size=5&vad_filter=true", files=files)

    assert r.status_code == 200
    data = r.json()
    assert data["text"] == "hello world"
    assert data["model"] == "small"
    assert data["detected_language"] == "en"
    assert data["language_probability"] == 0.9
    assert isinstance(data["seconds"], float)
    assert data["bytes"] > 0

    # Ensure engine was called with a temp file path
    assert len(fake.calls) == 1
    assert fake.calls[0]["beam_size"] == 5
    assert fake.calls[0]["vad_filter"] is True


def test_transcribe_rejects_bad_mime(client):
    files = {"file": ("a.wav", b"data", "application/pdf")}
    r = client.post("/v1/transcribe", files=files)
    assert r.status_code == 415
    assert "Unsupported content-type" in r.text


def test_transcribe_rejects_bad_extension(client):
    files = {"file": ("a.exe", b"data", "application/octet-stream")}
    r = client.post("/v1/transcribe", files=files)
    assert r.status_code == 415
    assert "Unsupported file extension" in r.text


def test_transcribe_rejects_too_large(monkeypatch, client):
    # To avoid uploading huge data, set max_upload_mb=1 and send a bit more than 1MB
    fake = FakeEngine()
    import src.fastapi.api_server as api
    monkeypatch.setattr(api, "get_engine", lambda model: fake)

    big = b"x" * (1024 * 1024 + 10)
    files = {"file": ("a.wav", big, "audio/wav")}
    r = client.post("/v1/transcribe?max_upload_mb=1", files=files)

    assert r.status_code == 413
    assert "File too large" in r.text
