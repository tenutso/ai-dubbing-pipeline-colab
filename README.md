# 🎙️ AI Dubbing Pipeline

> Translate and dub any MP4 or Vimeo video into **French (OQLF standard)** — with
> automatic transcription, speaker diarization, voice matching, subtitles and a
> re-muxed video — all running on **Google Colab's free tier**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&logoColor=white)
![Package manager: uv](https://img.shields.io/badge/deps-uv-261230?logo=astral&logoColor=white)
![WhisperX](https://img.shields.io/badge/STT-WhisperX-orange.svg)
![Gemini](https://img.shields.io/badge/translation-Gemini-4285F4?logo=google&logoColor=white)
![Google TTS](https://img.shields.io/badge/TTS-Google%20Cloud-EA4335?logo=googlecloud&logoColor=white)

---

## 📖 Description

The **AI Dubbing Pipeline** takes a source video (a local MP4 or a Vimeo URL),
understands *who said what and when*, translates the dialogue into Quebec/OQLF
French, and produces a fully dubbed video where each on-screen speaker is matched
to a distinct, tonally-appropriate synthetic voice.

It is engineered to fit within the memory and time constraints of the **free
Google Colab GPU tier**, and uses [`uv`](https://github.com/astral-sh/uv) for
fast, reproducible dependency management.

## ✨ Features

- 🎬 **Flexible input** — local `.mp4` files or Vimeo (and most `yt-dlp` supported) URLs.
- 🗣️ **Speech-to-text + alignment** with [WhisperX](https://github.com/m-bain/whisperX) for accurate word-level timestamps.
- 👥 **Speaker diarization** via pyannote — automatically separates multiple speakers.
- 🎚️ **Tone-based voice matching** — analyzes pitch, RMS energy and rate of speech to pick a fitting Google TTS voice per speaker.
- 🇫🇷 **OQLF French translation** powered by Google Gemini, with optional custom glossary.
- 🔊 **Natural dub mixing** — original audio is ducked and the synthesized track is overlaid in sync.
- 📝 **SRT subtitles** generated automatically, labeled per speaker.
- 🎞️ **Final muxed MP4** with the dubbed audio track.
- 🧾 **JSON manifest** describing every speaker profile and utterance.
- ⚙️ **Speaker range forcing** — pin the diarizer to a known speaker count.

## 🚀 Quick Start

### Google Colab (recommended)

```python
# Cell 1 — clone & install (installs ffmpeg + Python deps via pip)
!git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git
%cd ai-dubbing-pipeline-colab
!bash setup_colab.sh

# Cell 2 — upload credentials and write .env
from google.colab import files
uploaded = files.upload()          # upload google_tts_service_account.json
!mv google_tts_service_account.json creds/

# Cell 3 — create .env (fill in your keys)
%%writefile .env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
HF_TOKEN=your_hf_token_here
GOOGLE_APPLICATION_CREDENTIALS=./creds/google_tts_service_account.json

# Cell 4 — run
!bash run_dub.sh --input https://vimeo.com/123456789 --glossary examples/oqlf_glossary.txt
```

### Local

```bash
git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git
cd ai-dubbing-pipeline-colab

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# ...or plain pip
pip install -r requirements.txt

cp .env.example .env   # then edit .env with your keys
python dubbing_pipeline.py --input path/to/video.mp4
```

> ⚠️ A CUDA GPU is strongly recommended. For CPU-only runs pass `--device cpu`
> and expect significantly slower transcription.

## 🔧 Installation

You need three credentials (all have free tiers):

| Service | Purpose | Where to get it |
|---------|---------|-----------------|
| **Gemini API key** | Translation | https://aistudio.google.com/apikey |
| **Hugging Face token** | Speaker diarization | https://huggingface.co/settings/tokens |
| **Google Cloud TTS service account** | Voice synthesis | https://console.cloud.google.com |

Then create your environment file:

```bash
cp .env.example .env
```

and fill in the values. Full step-by-step instructions (including accepting the
pyannote model license and enabling the TTS API) are in **[docs/SETUP.md](docs/SETUP.md)**.

## 💻 Basic Usage

```bash
# Local MP4 → French dub
python dubbing_pipeline.py --input inputs/interview.mp4

# Vimeo URL with a custom OQLF glossary
python dubbing_pipeline.py \
    --input https://vimeo.com/123456789 \
    --glossary examples/oqlf_glossary.txt \
    --output_dir outputs/

# Force exactly two speakers, use a larger model
python dubbing_pipeline.py --input clip.mp4 --min_speakers 2 --max_speakers 2 --model medium
```

Outputs are written to `--output_dir` (default `outputs/`):

| File | Description |
|------|-------------|
| `final_dubbed_video.mp4` | The video with the new French audio track. |
| `dubbed_audio.wav` | The mixed dub audio (ducked original + synthesized voices). |
| `subtitles.srt` | Per-speaker French subtitles. |
| `manifest.json` | Speaker profiles, voice assignments and translated utterances. |
| `original_audio.wav` | Extracted source audio (intermediate). |

See **[docs/USAGE.md](docs/USAGE.md)** for every CLI flag and advanced workflows.

## 📚 Documentation

- 📦 **[Setup Guide](docs/SETUP.md)** — API keys, credentials, local & Colab install.
- 🛠️ **[Usage Guide](docs/USAGE.md)** — CLI options, advanced options, outputs, workflows.
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** — pipeline flow, components, design decisions.
- 🧪 **[Examples](examples/README.md)** — sample glossary and how to use it.

## 🤝 Contributing

Contributions are welcome! To propose a change:

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Make your changes with clear commit messages.
3. Ensure the code runs and follows the existing style.
4. Open a Pull Request describing **what** and **why**.

Please open an issue first for large features so we can discuss the approach.

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<sub>Built with WhisperX · Google Gemini · Google Cloud Text-to-Speech · uv</sub>
