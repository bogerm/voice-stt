# Voice STT - Local Whisper Speech-to-Text

A local speech-to-text application powered by OpenAI's Whisper model with a Gradio web interface. Features GPU acceleration support (CUDA) for faster transcription.

## Features

- 🎤 Real-time speech-to-text transcription using Whisper
- ⚡ GPU acceleration with CUDA support (NVIDIA GPUs)
- 🎯 Faster transcription with faster-whisper implementation
- 🌐 Web-based interface built with Gradio
- 🔒 Runs locally - no external API calls required
- 📦 Docker support for easy deployment

## Requirements

### Local Setup
- Python 3.13+
- FFmpeg
- NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- pip or uv package manager

### Docker Setup
- Docker installed
- Docker Desktop (for Windows/Mac) or Docker Engine (for Linux)
- NVIDIA Docker runtime (if using GPU with Docker)

## Installation & Setup

### Option 1: Local Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd voice-stt
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -e .
   ```

5. **Run the application:**
   ```bash
   python app.py
   ```

   The web interface will be available at `http://localhost:7860`

### Option 2: Docker Installation

#### Prerequisites
For GPU support on Docker, ensure you have:
- NVIDIA GPU drivers installed on your host machine
- Docker with NVIDIA runtime support

#### Build the Docker Image

1. **Build the image:**
   ```bash
   docker build -t voice-stt:latest .
   ```

#### Run the Docker Container

**For CPU-only (no GPU):**
```bash
docker run -p 7860:7860 voice-stt:latest
```

**For GPU support (NVIDIA GPUs):**
```bash
docker run --gpus all -p 7860:7860 voice-stt:latest
```

Or with specific GPU:
```bash
docker run --gpus '"device=0"' -p 7860:7860 voice-stt:latest
```

**With volume mounting for persistent model cache:**
```bash
docker run --gpus all -p 7860:7860 -v whisper_cache:/root/.cache/huggingface voice-stt:latest
```

#### Access the Application

Once the container is running, open your browser and navigate to:
```
http://localhost:7860
```

## Docker Common Commands

### Stop a running container:
```bash
docker stop <container_id>
```

### Remove a container:
```bash
docker rm <container_id>
```

### View running containers:
```bash
docker ps
```

### View all containers (including stopped):
```bash
docker ps -a
```

### View container logs:
```bash
docker logs <container_id>
```

### Remove the image:
```bash
docker rmi voice-stt:latest
```

## Configuration

The application uses the following environment variables (optional):
- `WHISPER_CACHE`: Custom cache directory for downloaded Whisper models (default: `./models`)

### Setting environment variables with Docker:
```bash
docker run --gpus all -p 7860:7860 -e WHISPER_CACHE=/models voice-stt:latest
```

## Usage

1. Open the Gradio interface in your browser (http://localhost:7860)
2. Upload an audio file or record audio directly in the interface
3. Select the Whisper model size (tiny, base, small, medium, large)
4. Click "Transcribe" to start the speech-to-text process
5. View the transcribed text in the output

## Supported Model Sizes

- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Balanced (~74M parameters)
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy (~1550M parameters)

## Dependencies

- **faster-whisper** (>=1.2.1): Optimized Whisper implementation
- **gradio** (>=6.2.0): Web interface framework

## Performance Notes

- GPU acceleration significantly speeds up transcription
- First run downloads the selected model (1-3GB depending on model size)
- Subsequent runs use cached models
- Volume mounts help preserve models between container runs

## Troubleshooting

### Docker GPU not detected
- Ensure NVIDIA Docker runtime is installed: `docker run --rm --gpus all nvidia/cuda:12.8.2-cudnn8-runtime-ubuntu22.04 nvidia-smi`
- Check NVIDIA Docker setup on [NVIDIA's documentation](https://github.com/NVIDIA/nvidia-docker)

### Out of memory errors
- Use a smaller model (tiny, base, or small)
- Increase Docker memory allocation
- Use `int8_float16` compute type in the code for reduced VRAM usage

### Models taking long to download
- Models are downloaded on first use
- Use volume mounts to persist cached models: `-v whisper_cache:/root/.cache/huggingface`

## License

This project uses OpenAI's Whisper model. Refer to the [Whisper license](https://github.com/openai/whisper/blob/main/LICENSE) for usage terms.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues or questions, please open an issue on the project repository.
