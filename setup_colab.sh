#!/bin/bash
#
# setup_colab.sh
# --------------
# One-shot environment bootstrap for Google Colab (free tier).
# Installs system deps (ffmpeg) and syncs Python dependencies.
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
apt-get update -qq && apt-get install -y -qq ffmpeg git curl

echo "==> Creating project structure..."
mkdir -p inputs outputs

echo "==> Installing Python dependencies..."
# Activate local venv if present (local dev), otherwise use Colab's pip.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt

echo ""
echo "==> Setup complete."
echo "    1. Place your Google TTS service-account JSON in creds/"
echo "    2. Copy .env.example to .env and fill in your API keys."
echo "    3. Run: bash run_dub.sh --input <mp4-or-vimeo-url>"
