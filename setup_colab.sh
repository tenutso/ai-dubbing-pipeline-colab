#!/bin/bash
#
# setup_colab.sh
# --------------
# One-shot environment bootstrap for Google Colab (free tier).
# Installs system deps (ffmpeg, espeak-ng) and syncs Python dependencies.
#
# On Colab: installs directly into Colab's managed Python via pip (no venv
# needed — the Colab runtime is already isolated per session).
# On a local machine: if a .venv already exists it is used; otherwise packages
# are installed into the active environment via pip.
#
# Usage (inside a Colab cell):
#   !bash setup_colab.sh
#
set -e

echo "==> Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq ffmpeg git curl espeak-ng

echo "==> Creating project structure..."
mkdir -p inputs outputs

# ---------------------------------------------------------------------------
# Optional: cache XTTS-V2 model weights (~1.8 GB) to Google Drive so they
# survive Colab disconnections without re-downloading each session.
#
# Run once after mounting Drive:
#   from google.colab import drive; drive.mount('/content/drive')
#
# Then uncomment the block below (or run it manually in a cell):
#
# DRIVE_TTS_CACHE="/content/drive/MyDrive/tts_cache"
# mkdir -p "$DRIVE_TTS_CACHE"
# mkdir -p "$HOME/.local/share"
# ln -sfn "$DRIVE_TTS_CACHE" "$HOME/.local/share/tts"
# echo "XTTS-V2 model cache → $DRIVE_TTS_CACHE"
# ---------------------------------------------------------------------------

echo "==> Installing Python dependencies..."
# Activate local venv if present (local dev), otherwise use Colab's pip.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

echo ""
echo "==> Setup complete."
echo "    1. Copy .env.example to .env and fill in your API keys."
echo "    2. Run: bash run_dub.sh --input <mp4-or-vimeo-url> --tts_lang fr"
