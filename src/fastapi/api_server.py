from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from stt_engine import WhisperModelName, WhisperSTT

app = FastAPI(title="Local Whisper STT API", version="0.1.0")

# Engine cache so you don't reload weights repeatedly.
ENGINES: dict[WhisperModelName, WhisperSTT] = {}


def get_engine(model: WhisperModelName) -> WhisperSTT:
    if model not in ENGINES:
        ENGINES[model] = WhisperSTT(model)  # lazy-loads on first transcribe()
    return ENGINES[model]


ALLOWED_MIME = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/flac",
    "audio/ogg",
    "video/mp4",
    "application/octet-stream",
}
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    model: WhisperModelName = Query(WhisperModelName.small),
    language: Optional[str] = Query(None),
    beam_size: int = Query(5, ge=1, le=10),
    vad_filter: bool = Query(True),
    max_upload_mb: int = Query(50, ge=1, le=500),
):
    # Basic input validation
    if file.content_type and file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported content-type: {file.content_type}")

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix and suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail=f"Unsupported file extension: {suffix}")

    max_bytes = max_upload_mb * 1024 * 1024
    total = 0

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".bin") as tmp:
        tmp_path = tmp.name
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File too large (>{max_upload_mb}MB).")
                tmp.write(chunk)
        finally:
            await file.close()

    try:
        engine = get_engine(model)
        res = engine.transcribe(
            audio_path=tmp_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        return JSONResponse(
            {
                "text": res.text,
                "model": model.value,
                "detected_language": res.detected_language,
                "language_probability": res.language_probability,
                "seconds": round(res.seconds, 3),
                "bytes": total,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
