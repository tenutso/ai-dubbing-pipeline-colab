# Setup Guide

This guide walks you through obtaining the required credentials and installing
the pipeline both **locally** and on **Google Colab**.

---

## 1. Prerequisites

- Python **3.10+**
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH`
- A CUDA-capable GPU (strongly recommended; Colab provides one for free)
- Three API credentials (all have free tiers):
  1. A **Gemini API key** (translation)
  2. A **Hugging Face token** (speaker diarization)
  3. A **Google Cloud TTS API key** (voice synthesis)

---

## 2. Obtaining API Keys

### 2.1 Gemini API Key

1. Go to **[Google AI Studio → API keys](https://aistudio.google.com/apikey)**.
2. Sign in with your Google account.
3. Click **Create API key** (you can attach it to a new or existing project).
4. Copy the key — you will paste it into `.env` as `GEMINI_API_KEY`.

### 2.2 Hugging Face Token (for diarization)

WhisperX uses the **pyannote** diarization models, which are gated.

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Visit **[Settings → Access Tokens](https://huggingface.co/settings/tokens)** and
   create a token with **Read** permission. Copy it (`HF_TOKEN`).
3. Accept the model licenses (you must be logged in) on **both**:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
4. Without accepting these licenses, diarization will fail and all audio will be
   attributed to a single speaker (`SPEAKER_00`).

### 2.3 Google Cloud TTS API Key

1. Open the **[Google Cloud Console](https://console.cloud.google.com)** and
   create (or select) a project.
2. Enable the **Cloud Text-to-Speech API**:
   *APIs & Services → Library → search "Text-to-Speech" → Enable*.
3. Create an API key:
   *APIs & Services → Credentials → Create Credentials → API key*.
4. (Recommended) Restrict the key to the **Cloud Text-to-Speech API** only.
5. Copy the key — it goes in `.env` as `GOOGLE_TTS_API_KEY` or in Colab Secrets.

---

## 3. Configure your `.env`

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=AIza...your_key...
GEMINI_MODEL=gemini-2.0-flash
HF_TOKEN=hf_...your_token...
GOOGLE_TTS_API_KEY=AIza...your_tts_key...
DEFAULT_TTS_LANG=fr-CA
WHISPER_MODEL=small
```

> 🔒 `.env` is excluded by `.gitignore` — never commit your secrets.

---

## 4. Local Installation

### Option A — uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/<your-username>/dubbing-pipeline-repo.git
cd dubbing-pipeline-repo

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

### Verify ffmpeg

```bash
ffmpeg -version    # should print version info
```

On Debian/Ubuntu: `sudo apt-get install -y ffmpeg`.
On macOS: `brew install ffmpeg`.

### Run a test

```bash
python dubbing_pipeline.py --input path/to/short_clip.mp4 --device cpu
```

---

## 5. Google Colab Setup

### 5.1 Add Colab Secrets (one-time, per account)

Open the **🔑 Secrets** panel in the Colab left sidebar and add:

| Secret name | Value |
|-------------|-------|
| `GEMINI_API_KEY` | Your Google AI Studio API key |
| `HF_TOKEN` | Your Hugging Face access token |
| `GOOGLE_TTS_API_KEY` | Your Google Cloud TTS API key |

Secrets are stored in your Google account and reused across sessions — you
only need to do this once.

### 5.2 Run the pipeline

1. Open a new Colab notebook and set the runtime to **GPU**
   (*Runtime → Change runtime type → T4 GPU*).

2. Clone the repo and install dependencies:

   ```python
   !git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git
   %cd ai-dubbing-pipeline-colab
   !bash setup_colab.sh
   ```

   `setup_colab.sh` installs `ffmpeg` and all Python dependencies directly
   into Colab's managed Python environment via `pip` (no virtualenv needed —
   Colab's runtime is already isolated per session).

3. Run the pipeline — secrets are loaded automatically:

   ```python
   !bash run_dub.sh --input https://vimeo.com/123456789 --glossary examples/oqlf_glossary.txt
   ```

4. Download the results from `outputs/` via the Colab file browser, or:

   ```python
   from google.colab import files
   files.download('outputs/final_dubbed_video.mp4')
   ```

### 5.3 Optional — override with a `.env` file

If you prefer not to use Colab Secrets, you can still create a `.env` file
manually and it will take precedence:

   ```python
   %%writefile .env
   GEMINI_API_KEY=AIza...
   GEMINI_MODEL=gemini-2.0-flash
   HF_TOKEN=hf_...
   GOOGLE_TTS_API_KEY=AIza...
   ```

---

## 6. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| All lines attributed to `SPEAKER_00` | `HF_TOKEN` missing/invalid, or pyannote licenses not accepted (see §2.2). |
| `CUDA out of memory` | Use a smaller `--model` (e.g. `small`), lower `--batch_size`, or restart the runtime. |
| `ffmpeg: command not found` | Install ffmpeg (see §4). On Colab, `setup_colab.sh` handles it. |
| `403 / PermissionDenied` from TTS | TTS API not enabled on the project, or the API key is restricted to a different API. |
| `GOOGLE_TTS_API_KEY is not set` | Add the key to Colab Secrets or your `.env` file. |
| Gemini `quota`/`429` errors | You hit the free-tier rate limit; wait and retry or switch `GEMINI_MODEL`. |
| Vimeo download fails | Update `yt-dlp` (`uv pip install -U yt-dlp`); some videos are private/region-locked. |

Still stuck? Open an issue with the full error trace.
