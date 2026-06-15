# Setup Guide

This guide walks you through obtaining the required credentials and installing
the pipeline both **locally** and on **Google Colab**.

---

## 1. Prerequisites

- Python **3.10–3.13**
- [`ffmpeg`](https://ffmpeg.org/) and [`espeak-ng`](https://github.com/espeak-ng/espeak-ng) on your `PATH`
- A CUDA-capable GPU (strongly recommended; Colab provides one for free)
- Two API credentials (both have free tiers):
  1. A **Gemini API key** (translation)
  2. A **Hugging Face token** (speaker diarization)

> XTTS-V2 runs locally on the GPU — no Google Cloud TTS account or API key is needed.

---

## 2. Obtaining API Keys

### 2.1 Gemini API Key

1. Go to **[Google AI Studio → API keys](https://aistudio.google.com/apikey)**.
2. Sign in with your Google account.
3. Click **Create API key** (attach it to a new or existing project).
4. Copy the key — paste it into `.env` as `GEMINI_API_KEY` or add it to Colab Secrets.

### 2.2 Hugging Face Token (for diarization)

WhisperX uses the **pyannote** diarization models, which are gated behind a license agreement.

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Visit **[Settings → Access Tokens](https://huggingface.co/settings/tokens)** and
   create a token with **Read** permission. Copy it — this is your `HF_TOKEN`.
3. Accept the model licenses (you must be logged in) on **both**:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Without accepting these licenses, diarization will fail and all audio will be
   attributed to a single speaker (`SPEAKER_00`).

---

## 3. Configure your `.env`

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=AIza...your_key...
GEMINI_MODEL=gemini-2.5-flash
HF_TOKEN=hf_...your_token...

# XTTS-V2 defaults (override with CLI flags)
TTS_LANG=fr
TTS_TEMPERATURE=0.65

# WhisperX default
WHISPER_MODEL=small
```

> `.env` is excluded by `.gitignore` — never commit your secrets.

---

## 4. Local Installation

### System dependencies

On Debian/Ubuntu:
```bash
sudo apt-get install -y ffmpeg espeak-ng
```

On macOS:
```bash
brew install ffmpeg espeak-ng
```

Verify:
```bash
ffmpeg -version
espeak-ng --version
```

### Option A — uv (recommended)

```bash
git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git
cd ai-dubbing-pipeline-colab

curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Option B — plain pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run a test

```bash
python dubbing_pipeline.py --input path/to/short_clip.mp4 --tts_lang fr --device cpu
```

> XTTS-V2 downloads its model weights (~1.8 GB) on first run. Subsequent runs use the local cache.

---

## 5. Google Colab Setup

### 5.1 Add Colab Secrets (one-time, per account)

Open the **Secrets** panel (key icon in the left sidebar) and add:

| Secret name | Value |
|-------------|-------|
| `GEMINI_API_KEY` | Your Google AI Studio API key |
| `HF_TOKEN` | Your Hugging Face access token |

Secrets are stored in your Google account and reused across sessions.

### 5.2 Run the pipeline

1. Open a new Colab notebook and set the runtime to **GPU**
   (*Runtime → Change runtime type → T4 GPU*).

2. Clone the repo and install dependencies using this **idempotent pattern**
   (safe to re-run without creating nested directories):

   ```python
   import os
   REPO = "ai-dubbing-pipeline-colab"
   if not os.path.isdir(f"/content/{REPO}"):
       !git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git /content/{REPO}
   %cd /content/{REPO}
   !bash setup_colab.sh
   ```

   `setup_colab.sh` installs `ffmpeg`, `espeak-ng`, and all Python dependencies
   into Colab's managed Python environment via `pip`.

3. Inject secrets into the session environment:

   ```python
   # Colab Secrets are only reachable from the notebook kernel, not bash
   # subprocesses. Setting os.environ here propagates them to all subsequent
   # ! commands in this session.
   from google.colab import userdata
   import os
   os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
   os.environ['HF_TOKEN']       = userdata.get('HF_TOKEN')
   ```

4. Run the pipeline:

   ```python
   !bash run_dub.sh \
       --input https://vimeo.com/123456789 \
       --tts_lang fr \
       --tts_temperature 0.65 \
       --context "a corporate interview about financial technology"
   ```

5. Resume after a disconnection by adding `--resume`:

   ```python
   !bash run_dub.sh \
       --input https://vimeo.com/123456789 \
       --tts_lang fr \
       --resume
   ```

6. Download the results:

   ```python
   from google.colab import files
   files.download('outputs/final_dubbed_video.mp4')
   files.download('outputs/subtitles.srt')
   ```

### 5.3 Cache XTTS-V2 model weights to Google Drive (optional but recommended)

XTTS-V2 downloads ~1.8 GB of model weights on first use. On Colab these are
lost when the runtime disconnects. To avoid re-downloading every session:

```python
# Run once after mounting Drive
from google.colab import drive
drive.mount('/content/drive')
```

Then uncomment the cache block in `setup_colab.sh`, or run manually:

```python
import os
DRIVE_CACHE = "/content/drive/MyDrive/tts_cache"
os.makedirs(DRIVE_CACHE, exist_ok=True)
os.makedirs(os.path.expanduser("~/.local/share"), exist_ok=True)
!ln -sfn {DRIVE_CACHE} ~/.local/share/tts
```

### 5.4 Optional — override with a `.env` file

If you prefer not to use Colab Secrets, create a `.env` file directly:

```python
%%writefile .env
GEMINI_API_KEY=AIza...
HF_TOKEN=hf_...
TTS_LANG=fr
TTS_TEMPERATURE=0.65
```

---

## 6. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| All lines attributed to `SPEAKER_00` | `HF_TOKEN` missing/invalid, or pyannote model licenses not accepted (see §2.2). |
| `CUDA out of memory` during WhisperX | Use a smaller `--model` (e.g. `small`), lower `--batch_size`, or restart the runtime. |
| `CUDA out of memory` during XTTS-V2 | The pipeline frees WhisperX memory before loading XTTS-V2. If it still OOMs, try `--model tiny` or `--device cpu`. |
| `ffmpeg: command not found` | Install ffmpeg (see §4). On Colab, `setup_colab.sh` handles it. |
| `espeak-ng: command not found` | Install espeak-ng (see §4). On Colab, `setup_colab.sh` handles it. |
| XTTS-V2 model not found / download fails | Check internet access; first run downloads ~1.8 GB. See §5.3 to cache to Drive. |
| Nested clone path in errors (`repo/repo/repo`) | Re-read §5.2 — use the idempotent clone pattern with `os.path.isdir` guard. |
| Gemini `quota`/`429` errors | You hit the free-tier rate limit; wait and retry. Add `--resume` so translation is not re-run. |
| Vimeo download fails | Update `yt-dlp` (`pip install -U yt-dlp`); some videos are private or region-locked. |
| `TypeError: Object of type float32 is not JSON serializable` | Update to the latest version — this was fixed by the `_NumpyEncoder` in `save_manifest`. |

Still stuck? Open an issue with the full error trace.
