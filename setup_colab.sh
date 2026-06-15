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
# To avoid nested-clone drift on Colab, use this idempotent pattern in your
# notebook instead of a bare !git clone:
#
#   import os
#   REPO = "ai-dubbing-pipeline-colab"
#   if not os.path.isdir(f"/content/{REPO}"):
#       !git clone https://github.com/<user>/{REPO}.git /content/{REPO}
#   %cd /content/{REPO}
#   !bash setup_colab.sh
#
set -e

# ── Always run from the repo root ───────────────────────────────────────────
# Resolve this script's real location and cd to it so the script works
# correctly regardless of the notebook's current working directory.
# This prevents the common Colab pattern of re-running a git-clone cell
# pushing the CWD deeper into repo/repo/repo/... subdirectories.
REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"

# Sanity-check: abort if we landed inside a nested clone.
# (Two consecutive occurrences of the repo directory name in the path means
# git clone was run from inside the already-cloned directory.)
REPO_NAME="$(basename "$REPO_ROOT")"
if [[ "$(dirname "$REPO_ROOT")" == *"$REPO_NAME"* ]]; then
    echo ""
    echo "ERROR: Nested clone detected."
    echo "  Script is at : $REPO_ROOT"
    echo "  This looks like a clone-inside-a-clone."
    echo ""
    echo "Fix: delete the extra copy and run from the top-level clone:"
    echo "  cd /content/$REPO_NAME && bash setup_colab.sh"
    exit 1
fi

cd "$REPO_ROOT"
echo "==> Working directory: $PWD"

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
