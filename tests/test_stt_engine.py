import os
from types import SimpleNamespace

import pytest

from stt_engine import WhisperSTT, WhisperModelName, TranscriptionResult


def test_transcribe_missing_path_returns_empty_result():
    engine = WhisperSTT(WhisperModelName.tiny)
    res = engine.transcribe(audio_path="does_not_exist.wav")
    assert isinstance(res, TranscriptionResult)
    assert res.text == ""
    assert res.seconds == 0.0
    assert res.detected_language is None


def test_transcribe_empty_path_returns_empty_result():
    engine = WhisperSTT(WhisperModelName.tiny)
    res = engine.transcribe(audio_path="")
    assert res.text == ""
    assert res.seconds == 0.0


def test_transcribe_beam_size_out_of_range_raises(tmp_path):
    # Create an empty file just so os.path.exists passes
    p = tmp_path / "a.wav"
    p.write_bytes(b"")

    engine = WhisperSTT(WhisperModelName.tiny)
    with pytest.raises(ValueError):
        engine.transcribe(str(p), beam_size=0)
    with pytest.raises(ValueError):
        engine.transcribe(str(p), beam_size=11)


def test_lazy_initialization_not_called_for_missing_file(monkeypatch):
    engine = WhisperSTT(WhisperModelName.small)

    called = {"n": 0}

    def fake_ensure_model():
        called["n"] += 1
        raise AssertionError("_ensure_model should not be called when file does not exist")

    monkeypatch.setattr(engine, "_ensure_model", fake_ensure_model)
    res = engine.transcribe("missing.wav")
    assert res.text == ""
    assert called["n"] == 0


def test_transcribe_calls_model_transcribe(monkeypatch, tmp_path):
    # Prepare a dummy file path that exists
    p = tmp_path / "audio.wav"
    p.write_bytes(b"RIFF....WAVE")  # not a valid wav, but we won't decode it because we mock model

    # Fake segments + info returned by model.transcribe
    class Seg:
        def __init__(self, text):
            self.text = text

    fake_segments = [Seg(" hello"), Seg(" world")]
    fake_info = SimpleNamespace(language="en", language_probability=0.9)

    class FakeModel:
        def transcribe(self, audio_path, language, beam_size, vad_filter):
            assert audio_path == str(p)
            assert beam_size == 5
            return fake_segments, fake_info

    engine = WhisperSTT(WhisperModelName.base)
    monkeypatch.setattr(engine, "_ensure_model", lambda: FakeModel())

    res = engine.transcribe(str(p), language="en", beam_size=5, vad_filter=True)
    assert res.text == "hello world"
    assert res.detected_language == "en"
    assert res.language_probability == 0.9
    assert res.seconds >= 0.0


def test_transcribe_pcm16_empty_returns_empty():
    engine = WhisperSTT(WhisperModelName.tiny)
    res = engine.transcribe_pcm16(b"")
    assert res.text == ""
    assert res.seconds == 0.0


def test_transcribe_pcm16_writes_temp_and_calls_transcribe(monkeypatch):
    engine = WhisperSTT(WhisperModelName.tiny)

    # 1 second of silence PCM16 @16kHz: 16000 samples * 2 bytes
    pcm = b"\x00\x00" * 16000

    seen = {"path": None}

    def fake_transcribe(audio_path, language=None, beam_size=5, vad_filter=True):
        # Ensure a temp wav was created and exists at time of call
        seen["path"] = audio_path
        assert os.path.exists(audio_path)
        return TranscriptionResult("ok", "en", 1.0, 0.01)

    monkeypatch.setattr(engine, "transcribe", fake_transcribe)

    res = engine.transcribe_pcm16(pcm, sample_rate=16000)
    assert res.text == "ok"
    assert seen["path"] is not None
    # file should be removed after
    assert not os.path.exists(seen["path"])
