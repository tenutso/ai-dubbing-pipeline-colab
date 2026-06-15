#!/bin/bash
#
# run_dub.sh
# ----------
# Thin wrapper that sets Colab/GPU-friendly environment variables, activates
# the uv virtualenv and forwards all CLI arguments to dubbing_pipeline.py.
#
# Examples:
#   bash run_dub.sh --input inputs/clip.mp4
#   bash run_dub.sh --input https://vimeo.com/123456789 --glossary examples/oqlf_glossary.txt
#   bash run_dub.sh --input clip.mp4 --min_speakers 2 --max_speakers 2
#

# ── Always run from the repo root ───────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$REPO_ROOT"

# Stability / performance tuning for Colab GPUs.
export MPLBACKEND=Agg
# Bypass the Coqui TOS interactive prompt — crashes in non-interactive
# environments (Colab subprocesses, CI).  By setting this you confirm you
# agree to the Coqui CPML non-commercial licence: https://coqui.ai/cpml
export COQUI_TOS_AGREED=1
# Suppress SyntaxWarnings from pydub (invalid escape sequences in its regex
# strings — a known pydub 0.25.1 issue under Python 3.12, not our code).
export PYTHONWARNINGS="ignore::SyntaxWarning:pydub"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME="${HF_HOME:-$REPO_ROOT/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate the virtualenv if it exists.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Prefer the venv python, then python3, then python.
PYTHON=$(command -v python 2>/dev/null || command -v python3 2>/dev/null)

"$PYTHON" dubbing_pipeline.py "$@"
