from __future__ import annotations

import io
import threading
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from faster_whisper import WhisperModel


class WhisperModelName(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large_v3 = "large-v3"


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    detected_language: Optional[str]
    language_probability: Optional[float]
    seconds: float


class WhisperSTT:
    """
    Lazy-loading Whisper STT engine.

    - Model name is fixed at init.
    - Actual model weights are loaded on first transcribe().
    - Thread-safe initialization for multi-request servers.
    """

    def __init__(
        self,
        model_name: WhisperModelName,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device, self.compute_type = self._default_device_config(device, compute_type)

        self._model: Optional[WhisperModel] = None
        self._init_lock = threading.Lock()

    @staticmethod
    def _default_device_config(
        device: Optional[str],
        compute_type: Optional[str],
    ) -> Tuple[str, str]:
        d = device or "cuda"
        ct = compute_type or "float16"
        return d, ct

    def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        with self._init_lock:
            if self._model is None:
                self._model = WhisperModel(
                    self.model_name.value,
                    device=self.device,
                    compute_type=self.compute_type,
                )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        import os
        import time

        if not audio_path or not os.path.exists(audio_path):
            return TranscriptionResult(
                text="",
                detected_language=None,
                language_probability=None,
                seconds=0.0,
            )

        if beam_size < 1 or beam_size > 10:
            raise ValueError("beam_size must be between 1 and 10")

        model = self._ensure_model()
        lang = language.strip() if language and language.strip() else None

        t0 = time.time()
        segments, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=int(beam_size),
            vad_filter=bool(vad_filter),
        )
        seconds = time.time() - t0

        text = "".join(seg.text for seg in segments).strip()
        detected_language = getattr(info, "language", None)
        language_probability = getattr(info, "language_probability", None)

        return TranscriptionResult(
            text=text,
            detected_language=detected_language,
            language_probability=language_probability,
            seconds=seconds,
        )

    def transcribe_pcm16(
        self,
        pcm16le: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe raw PCM16 little-endian mono audio by wrapping it into an in-memory WAV file.
        """
        if not pcm16le:
            return TranscriptionResult("", None, None, 0.0)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16le)

        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(buf.getvalue())
            tmp_path = tmp.name

        try:
            return self.transcribe(
                audio_path=tmp_path,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
