# Video Studio (Pinokio)

One-click [Pinokio](https://pinokio.co) app for local AI video generation on Ubuntu with an NVIDIA GPU. It uses Hugging Face Diffusers and a Gradio web UI.

- **Text to Video**: generate a clip from a prompt.
- **Image to Video**: animate a still image, guided by a prompt.
- **Music Video**: upload a song and it is split into beat-aligned clips. Each clip gets a prompt from your storyline plus the song's energy (calm, mid or energetic), and the prompts can be edited before rendering. The clips are generated, cut to the beat and muxed with the original audio into `music_video.mp4`. You can also start from an image and enable **Continuity**, which seeds each clip from the previous clip's last frame.

## Enhance prompt

**Enhance prompt** buttons (Text to Video and Image to Video tabs) and **Enhance all scene prompts** (Music Video plan) rewrite short ideas into detailed video prompts: subject, motion, setting, camera, lighting and style. Pick the backend under **Prompt enhancer settings**:

- **Local (Qwen2.5 1.5B)**, the default: downloads about 3 GB once and moves off the GPU after each use. Override the model with `VIDEO_STUDIO_ENHANCER_MODEL`.
- **Ollama**: base URL (default `http://127.0.0.1:11434`) plus a model name (default `llama3.2`).
- **OpenAI-compatible**: base URL (default `https://api.openai.com/v1`), a model name and an API key (or `OPENAI_API_KEY`). This also works with LM Studio, vLLM, OpenRouter and similar servers.

## Models

| Model | Modes | Native res / fps | VRAM (with CPU offload) |
|---|---|---|---|
| [LTX-Video](https://huggingface.co/Lightricks/LTX-Video) (default) | T2V, I2V | 768x512 @ 24fps, up to 257 frames | ~12 GB+ |
| [Wan 2.2 TI2V 5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers) | T2V, I2V | 1280x704 @ 24fps, up to 121 frames | ~16 GB+ |
| [Wan 2.1 T2V 1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | T2V only | 832x480 @ 16fps, up to 81 frames | ~8 GB+ |

Each model downloads into the Pinokio Hugging Face cache the first time you use it (about 10–35 GB per model). On cards with little VRAM, tick **Low VRAM** to use sequential offload. It is much slower.

## Install

1. Install Pinokio on Ubuntu, along with an NVIDIA driver that supports CUDA 12.8.
2. In Pinokio, click **Download from URL** and paste this repository's git URL.
3. Click **Install**, then **Start**, then **Open Web UI**.

Outputs are saved to `app/outputs/`. You can open that folder from the Pinokio menu with **Outputs**.

## Manual run (without Pinokio)

```bash
cd app
python -m venv env && source env/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python app.py --port 7860
```

Set `VIDEO_STUDIO_FAKE=1` to run the whole UI and the music-video pipeline with a CPU stand-in generator instead of a real model. This is useful for testing without a GPU.

## Music video tips

- Each clip's length is capped by the model's maximum frames: about 10.7 s for LTX, 5 s for Wan. Shorter clips (2–4 s) cut more often on the beat.
- Use **Audio start** and **Max duration** to render a short test section before you commit to the full song.
- Scenes are spread across the song in order, so writing 4–8 lines gives you a storyline.
