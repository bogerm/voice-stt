from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import gradio as gr
from faster_whisper import WhisperModel


@dataclass
class LoadedModel:
    name: str
    model: WhisperModel


_MODEL: Optional[LoadedModel] = None


def get_device() -> Tuple[str, str]:
    """
    Prefer CUDA if available. faster-whisper/ctranslate2 will use CUDA if the wheel supports it.
    compute_type:
      - "float16" is typical for NVIDIA GPU
      - "int8_float16" can reduce VRAM further (often slightly slower / small accuracy changes)
    """
    return ("cuda", "float16")


def load_model(model_name: str) -> WhisperModel:
    global _MODEL

    if _MODEL is not None and _MODEL.name == model_name:
        return _MODEL.model

    device, compute_type = get_device()

    # You can also set download_root to control where models cache:
    # download_root=os.getenv("WHISPER_CACHE", "./models")
    m = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )
    _MODEL = LoadedModel(name=model_name, model=m)
    return m


def transcribe(
    audio_path: str,
    model_name: str,
    language: str,
    beam_size: int,
) -> str:
    if not audio_path or not os.path.exists(audio_path):
        return "No audio received."

    model = load_model(model_name)

    # language: "" means auto-detect
    lang = None if language.strip() == "" else language.strip()

    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        beam_size=int(beam_size),
        language=lang,
        vad_filter=True,  # helps cut long silences for mic audio
    )

    text_parts = [seg.text for seg in segments]
    text = "".join(text_parts).strip()

    dt = time.time() - t0
    detected = getattr(info, "language", None)
    prob = getattr(info, "language_probability", None)

    meta = []
    if detected:
        meta.append(f"Detected language: {detected}" + (f" (p={prob:.2f})" if prob is not None else ""))
    meta.append(f"Time: {dt:.2f}s")

    return (("\n".join(meta) + "\n\n") if meta else "") + (text or "[No speech detected]")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local Whisper STT") as demo:
        gr.Markdown("# 🎙️ Local Speech-to-Text (Whisper)\nUpload audio or use your microphone, then transcribe locally.")

        with gr.Row():
            audio = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Audio (wav/mp3/etc)",
            )

        with gr.Row():
            model_name = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="small",
                label="Whisper model",
            )
            language = gr.Textbox(
                value="",
                label="Language (optional, e.g. 'en', 'de'). Leave blank for auto.",
            )

        with gr.Row():
            beam_size = gr.Slider(1, 10, value=5, step=1, label="Beam size")

        btn = gr.Button("Transcribe", variant="primary")
        out = gr.Textbox(label="Transcript", lines=12)

        btn.click(
            fn=transcribe,
            inputs=[audio, model_name, language, beam_size],
            outputs=[out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861)


if __name__ == "__main__":
    main()
