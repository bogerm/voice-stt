from __future__ import annotations

import os
import requests
import gradio as gr

# In docker-compose, the FastAPI service will be reachable as http://api:8000
FASTAPI_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://api:8000")


def transcribe_via_api(
    audio_path: str,
    model: str,
    language: str,
    beam_size: int,
    vad_filter: bool,
    max_upload_mb: int,
) -> str:
    if not audio_path:
        return "No audio received."

    params: dict[str, str] = {
        "model": model,
        "beam_size": str(int(beam_size)),
        "vad_filter": "true" if vad_filter else "false",
        "max_upload_mb": str(int(max_upload_mb)),
    }
    if language.strip():
        params["language"] = language.strip()

    with open(audio_path, "rb") as f:
        files = {
            "file": (os.path.basename(audio_path), f, "application/octet-stream")
        }
        r = requests.post(
            f"{FASTAPI_BASE_URL}/v1/transcribe",
            params=params,
            files=files,
            timeout=300,
        )

    if r.status_code != 200:
        return f"API error {r.status_code}:\n{r.text}"

    data = r.json()
    meta = []
    if data.get("detected_language"):
        p = data.get("language_probability")
        if isinstance(p, (int, float)):
            meta.append(f"Detected language: {data['detected_language']} (p={p:.2f})")
        else:
            meta.append(f"Detected language: {data['detected_language']}")
    meta.append(f"Model: {data.get('model', '?')}")
    meta.append(f"Time: {data.get('seconds', '?')}s")
    meta.append(f"Bytes: {data.get('bytes', '?')}")

    text = data.get("text") or "[No speech detected]"
    return "\n".join(meta) + "\n\n" + text


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Whisper STT Client") as demo:
        gr.Markdown(
            "# 🎙️ Whisper STT (Gradio Client)\n"
            "This UI sends audio to the FastAPI server and shows the transcription.\n\n"
            f"**Server:** `{FASTAPI_BASE_URL}`"
        )

        with gr.Row():
            model = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large-v3"],
                value="small",
                label="Model",
            )
            language = gr.Textbox(value="", label="Language (optional, e.g. en). Blank = auto-detect")

        with gr.Row():
            beam_size = gr.Slider(1, 10, value=5, step=1, label="Beam size")
            vad_filter = gr.Checkbox(value=True, label="VAD filter")
            max_upload_mb = gr.Slider(1, 200, value=50, step=1, label="Max upload (MB)")

        audio = gr.Audio(
            sources=["microphone", "upload"],
            type="filepath",
            label="Audio (wav/mp3/m4a/...)",
        )

        btn = gr.Button("Transcribe", variant="primary")
        out = gr.Textbox(label="Transcript", lines=14)

        btn.click(
            fn=transcribe_via_api,
            inputs=[audio, model, language, beam_size, vad_filter, max_upload_mb],
            outputs=[out],
        )

    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
