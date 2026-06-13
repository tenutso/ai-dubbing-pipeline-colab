# Setup Guide

This guide walks you through obtaining the required credentials and installing
the pipeline both **locally** and on **Google Colab**.

---

## 1. Prerequisites

- Python **3.10+**
- [`ffmpeg`](https://ffmpeg.org/) on your `PATH`
- A CUDA-capable GPU (strongly recommended; Colab provides one for free)
- Three API credentials (all free-tier friendly):
  1. A **Gemini API key** (translation)
  2. A **Hugging Face token** (speaker diarization)
  3. A **Google Cloud Text-to-Speech service-account JSON** (voice synthesis)

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

### 2.3 Google Cloud Text-to-Speech Service Account

1. Open the **[Google Cloud Console](https://console.cloud.google.com)** and
   create (or select) a project.
2. Enable the **Cloud Text-to-Speech API**:
   *APIs & Services → Library → search "Text-to-Speech" → Enable*.
3. Create a service account:
   *IAM & Admin → Service Accounts → Create service account*.
   Grant it the **Cloud Text-to-Speech User** role (or Editor for testing).
4. Create a **JSON key** for that service account and download it.
5. Place the file in the repo at `creds/google_tts_service_account.json`
   (the `creds/` folder is git-ignored).

---

## 3. Configure your `.env`

Copy the template and fill in your values:

```bash
cp .env.example .env
```

```env
GEMINI_API_KEY=AIza...your_key...
GEMINI_MODEL=gemini-1.5-flash
HF_TOKEN=hf_...your_token...
GOOGLE_APPLICATION_CREDENTIALS=./creds/google_tts_service_account.json
DEFAULT_TTS_LANG=fr-CA
WHISPER_MODEL=small
```

> 🔒 `.env` and everything under `creds/` are excluded by `.gitignore` — never
> commit your secrets.

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

1. Open a new Colab notebook and set the runtime to **GPU**
   (*Runtime → Change runtime type → T4 GPU*).
2. Clone the repo and run the bootstrap script:

   ```python
   !git clone https://github.com/<your-username>/dubbing-pipeline-repo.git
   %cd dubbing-pipeline-repo
   !bash setup_colab.sh
   ```

   `setup_colab.sh` installs `ffmpeg`, installs `uv`, creates the
   `dubbing/{creds,inputs,outputs}` layout and syncs all dependencies into a
   uv-managed virtualenv.

3. Upload your credentials and create `.env`:

   ```python
   from google.colab import files
   files.upload()   # upload google_tts_service_account.json → move it to creds/
   ```

   ```python
   %%writefile .env
   GEMINI_API_KEY=AIza...
   GEMINI_MODEL=gemini-1.5-flash
   HF_TOKEN=hf_...
   GOOGLE_APPLICATION_CREDENTIALS=./creds/google_tts_service_account.json
   DEFAULT_TTS_LANG=fr-CA
   WHISPER_MODEL=small
   ```

4. Run the pipeline:

   ```python
   !bash run_dub.sh --input https://vimeo.com/123456789 --glossary examples/oqlf_glossary.txt
   ```

5. Download the results from `outputs/` via the Colab file browser, or:

   ```python
   from google.colab import files
   files.download('outputs/final_dubbed_video.mp4')
   ```

---

## 6. Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| All lines attributed to `SPEAKER_00` | `HF_TOKEN` missing/invalid, or pyannote licenses not accepted (see §2.2). |
| `CUDA out of memory` | Use a smaller `--model` (e.g. `small`), lower `--batch_size`, or restart the runtime. |
| `ffmpeg: command not found` | Install ffmpeg (see §4). On Colab, `setup_colab.sh` handles it. |
| `403 / PermissionDenied` from TTS | TTS API not enabled, or service account lacks the Text-to-Speech role. |
| `google.auth ... could not automatically determine credentials` | `GOOGLE_APPLICATION_CREDENTIALS` path is wrong or the JSON is missing. |
| Gemini `quota`/`429` errors | You hit the free-tier rate limit; wait and retry or switch `GEMINI_MODEL`. |
| Vimeo download fails | Update `yt-dlp` (`uv pip install -U yt-dlp`); some videos are private/region-locked. |

Still stuck? Open an issue with the full error trace.
