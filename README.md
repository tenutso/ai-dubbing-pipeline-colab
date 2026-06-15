# AI Dubbing Pipeline

> Translate and dub any MP4 or Vimeo video into **Quebec French (OQLF standard)** — with
> automatic transcription, speaker diarization, **XTTS-V2 voice cloning**, anchored subtitles
> and a re-muxed video — all running on **Google Colab's free tier**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-F9AB00?logo=googlecolab&logoColor=white)
![WhisperX](https://img.shields.io/badge/STT-WhisperX-orange.svg)
![Gemini](https://img.shields.io/badge/translation-Gemini%202.5-4285F4?logo=google&logoColor=white)
![XTTS-V2](https://img.shields.io/badge/TTS-XTTS--V2%20voice%20cloning-blueviolet)

---

## Description

The **AI Dubbing Pipeline** takes a source video (a local MP4 or a Vimeo URL),
understands *who said what and when*, translates the dialogue into Quebec/OQLF
French, and produces a fully dubbed video where each speaker's voice is **cloned**
from the source audio using XTTS-V2 — no stock voices, no API fees for synthesis.

It is engineered to fit within the memory and time constraints of the **free
Google Colab GPU tier**. On Colab, credentials are loaded automatically from
**Colab Secrets** — no file uploads or `.env` files required.

## Features

- **Flexible input** — local `.mp4` files or Vimeo (and most `yt-dlp`-supported) URLs.
- **Speech-to-text + alignment** with [WhisperX](https://github.com/m-bain/whisperX) for accurate word-level timestamps.
- **Speaker diarization** via pyannote — automatically separates multiple speakers.
- **XTTS-V2 voice cloning** — clones each speaker's voice directly from their audio in the source video; no stock voices, no Google TTS API key needed.
- **Multi-speaker aware** — extracts a high-quality reference sample per speaker (scored by duration × RMS), loads the XTTS-V2 model once, and synthesizes every utterance with the correct cloned voice.
- **Temperature control** — `--tts_temperature` dial (0.1 = consistent, 1.0 = expressive) adjusts XTTS-V2's prosodic variation.
- **Time-stretching** — each dubbed clip is compressed or expanded via ffmpeg `atempo` to fit the original utterance window, keeping lip-sync plausible without phase-vocoder artifacts.
- **OQLF French translation** powered by Gemini 2.5 Flash with a tuned dubbing prompt (spoken register, conciseness, proper-noun protection, optional domain context and glossary).
- **Anchored subtitles** — SRT cue timecodes track the dubbed audio, not the original transcript, so subtitles appear exactly when the cloned voice speaks.
- **Checkpoint / resume** — pipeline writes JSON checkpoints after each expensive stage; `--resume` skips completed stages after a Colab disconnection.
- **GPU memory sequencing** — WhisperX models are unloaded before XTTS-V2 loads, preventing OOM on Colab's T4.
- **24 kHz mixing** — original audio is upsampled to 24 kHz with ffmpeg before mixing; XTTS-V2 clips are never downsampled, preserving full voice bandwidth.
- **Final muxed MP4** with the dubbed audio track.
- **JSON manifest** describing every speaker profile and utterance with original and dubbed timings.
- **Speaker range forcing** — pin the diarizer to a known speaker count.

## Quick Start

### Google Colab (recommended)

**Step 1 — Add credentials to Colab Secrets** (key icon in the left sidebar):

| Secret name | Value |
|-------------|-------|
| `GEMINI_API_KEY` | Your Google AI Studio key |
| `HF_TOKEN` | Your Hugging Face access token |

Only two keys are needed. XTTS-V2 runs locally on the Colab GPU — no synthesis API required.

**Step 2 — Run in a notebook** (GPU runtime required):

```python
# Cell 1 — clone & install (idempotent: safe to re-run)
import os
REPO = "ai-dubbing-pipeline-colab"
if not os.path.isdir(f"/content/{REPO}"):
    !git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git /content/{REPO}
%cd /content/{REPO}
!bash setup_colab.sh
```

```python
# Cell 2 — inject secrets into the session environment
# Colab Secrets are only accessible from the notebook kernel, not from bash
# subprocesses. Setting os.environ here makes them available to all subsequent
# ! commands in this session.
from google.colab import userdata
import os
os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
os.environ['HF_TOKEN']       = userdata.get('HF_TOKEN')
```

```python
# Cell 3 — run
# Use %%bash for multi-line commands — the ! magic doesn't reliably handle
# backslash line continuation in Colab and will treat each line as a new command.
%%bash
bash run_dub.sh \
    --input https://vimeo.com/123456789 \
    --tts_lang fr \
    --tts_temperature 0.65 \
    --context "a two-person interview about enterprise software" \
    --glossary examples/oqlf_glossary.txt
```

```python
# Cell 4 — download outputs
from google.colab import files
import os

output_dir = "outputs"  # adjust if you passed --output_dir
for f in ["final_dubbed_video.mp4", "dubbed_audio.wav", "subtitles.srt", "manifest.json"]:
    path = os.path.join(output_dir, f)
    if os.path.exists(path):
        files.download(path)
```

> **Tip — resume after disconnection:** if Colab drops mid-run, re-run cells 1–2 then add `--resume` to the `%%bash` cell. The pipeline reloads from the deepest available checkpoint and skips completed stages.

> **Tip — single-line alternative:** if you prefer `!`, keep all arguments on one line: `!bash run_dub.sh --input URL --tts_lang fr --tts_temperature 0.65`

### Local

```bash
git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git
cd ai-dubbing-pipeline-colab

# Using uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# ...or plain pip
pip install -r requirements.txt

cp .env.example .env   # fill in GEMINI_API_KEY and HF_TOKEN
python3 dubbing_pipeline.py --input path/to/video.mp4 --tts_lang fr
```

> A CUDA GPU is strongly recommended. For CPU-only runs pass `--device cpu` and expect significantly slower transcription and synthesis.

## Credentials

Only two credentials are required:

| Service | Purpose | Where to get it |
|---------|---------|-----------------|
| **Gemini API key** | Translation | https://aistudio.google.com/apikey |
| **Hugging Face token** | Speaker diarization | https://huggingface.co/settings/tokens |

XTTS-V2 runs locally — no Google Cloud TTS account or API key needed.

- **On Colab:** add them to Colab Secrets (see Quick Start above).
- **Locally:** copy `.env.example` to `.env` and fill in the values.

Full step-by-step instructions (including accepting the pyannote model license) are in **[docs/SETUP.md](docs/SETUP.md)**.

## Basic Usage

```bash
# Local MP4 → French dub
python3 dubbing_pipeline.py --input inputs/interview.mp4 --tts_lang fr

# With domain context (improves translation register and vocabulary)
python3 dubbing_pipeline.py \
    --input inputs/conference.mp4 \
    --tts_lang fr \
    --context "a corporate presentation about financial software" \
    --glossary examples/oqlf_glossary.txt

# Force exactly two speakers, use a larger Whisper model
python3 dubbing_pipeline.py \
    --input clip.mp4 \
    --min_speakers 2 --max_speakers 2 \
    --model medium \
    --tts_lang fr

# Resume after interruption
python3 dubbing_pipeline.py --input inputs/long.mp4 --tts_lang fr --resume

# Tune voice expressiveness (lower = cleaner, higher = more varied)
python3 dubbing_pipeline.py --input inputs/clip.mp4 --tts_lang fr --tts_temperature 0.3
```

Outputs are written to `--output_dir` (default `outputs/`):

| File / Directory | Description |
|------------------|-------------|
| `final_dubbed_video.mp4` | The video with the cloned-voice French audio track. |
| `dubbed_audio.wav` | The mixed dub audio: 24 kHz, ducked original + XTTS-V2 voices. |
| `subtitles.srt` | French subtitles anchored to dubbed audio timing. |
| `manifest.json` | Speaker profiles, clone sample paths, utterances with original and dubbed timings. |
| `speaker_samples/` | One WAV per speaker — the reference clip used for XTTS-V2 voice cloning. |
| `ckpt_*.json` | Stage checkpoints — keep these to enable `--resume` on reconnect. |
| `original_audio.wav` | Extracted source audio (intermediate). |

See **[docs/USAGE.md](docs/USAGE.md)** for every CLI flag and advanced workflows.

## Documentation

- **[Setup Guide](docs/SETUP.md)** — credentials, Colab Secrets, local and Colab install.
- **[Usage Guide](docs/USAGE.md)** — CLI options, advanced flags, outputs, workflows.
- **[Architecture](docs/ARCHITECTURE.md)** — pipeline flow, components, design decisions.
- **[Examples](examples/README.md)** — sample glossary and how to use it.

## Contributing

Contributions are welcome. To propose a change:

1. Fork the repository and create a feature branch (`git checkout -b feature/my-change`).
2. Make your changes with clear commit messages.
3. Ensure the code runs and follows the existing style.
4. Open a Pull Request describing **what** and **why**.

Please open an issue first for large features so we can discuss the approach.

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<sub>Built with WhisperX · Google Gemini 2.5 · XTTS-V2 (Coqui TTS)</sub>
