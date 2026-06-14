#!/bin/bash
#
# setup_colab.sh
# --------------
# One-shot environment bootstrap for Google Colab (free tier).
# Installs system deps (ffmpeg), the uv package manager, creates the project
# folder layout and syncs Python dependencies into a uv-managed virtualenv.
#
# Usage (inside a Colab cell):
#   !bash setup_colab.sh
#
set -e

echo "==> Initializing environment..."
apt-get update && apt-get install -y ffmpeg git curl

echo "==> Installing uv (fast Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh
# uv installs to ~/.cargo/env or ~/.local/bin depending on version.
source "$HOME/.cargo/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

echo "==> Creating project structure..."
mkdir -p inputs outputs

echo "==> Creating virtualenv and syncing dependencies..."
uv venv
source .venv/bin/activate
uv pip install \
    whisperx==3.8.6 \
    google-genai==2.8.0 \
    google-cloud-texttospeech==2.36.0 \
    python-dotenv==1.0.1 \
    pydub==0.25.1 \
    "librosa>=0.10" \
    soundfile==0.13.1 \
    yt-dlp \
    tqdm

echo ""
echo "==> Setup complete."
echo "    1. Place your Google TTS service-account JSON in creds/"
echo "    2. Copy .env.example to .env and fill in your API keys."
echo "    3. Run: bash run_dub.sh --input <mp4-or-vimeo-url>"
