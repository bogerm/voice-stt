from __future__ import annotations

import gradio as gr

from stt_engine import WhisperSTT, WhisperModelName


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Local Whisper STT") as demo:
        gr.Markdown("# 🎙️ Local Speech-to-Text (Whisper)\nMic or file upload → local transcription.")

        model_dd = gr.Dropdown(
            choices=[m.value for m in WhisperModelName],
            value=WhisperModelName.small.value,
            label="Model",
        )
        language = gr.Textbox(value="", label="Language (optional, e.g. en). Blank = auto-detect")
        beam = gr.Slider(1, 10, value=5, step=1, label="Beam size")
        vad = gr.Checkbox(value=True, label="VAD filter")

        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Audio (wav/mp3/m4a/...)")
        btn = gr.Button("Transcribe", variant="primary")
        out = gr.Textbox(label="Transcript", lines=12)

        engines: dict[str, WhisperSTT] = {}

        def do_transcribe(audio_path: str, model_value: str, lang: str, beam_size: int, vad_filter: bool) -> str:
            if not audio_path:
                return "No audio received."

            if model_value not in engines:
                engines[model_value] = WhisperSTT(WhisperModelName(model_value))

            res = engines[model_value].transcribe(
                audio_path=audio_path,
                language=lang,
                beam_size=int(beam_size),
                vad_filter=bool(vad_filter),
            )

            meta = []
            if res.detected_language:
                p = f"{res.language_probability:.2f}" if res.language_probability is not None else "?"
                meta.append(f"Detected language: {res.detected_language} (p={p})")
            meta.append(f"Time: {res.seconds:.2f}s")

            header = "\n".join(meta)
            body = res.text if res.text else "[No speech detected]"
            return f"{header}\n\n{body}"

        btn.click(
            fn=do_transcribe,
            inputs=[audio, model_dd, language, beam, vad],
            outputs=[out],
        )

    return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
