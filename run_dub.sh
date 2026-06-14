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

# Stability / performance tuning for Colab GPUs.
export MPLBACKEND=Agg
# Suppress SyntaxWarnings from pydub (invalid escape sequences in its regex
# strings — a known pydub 0.25.1 issue under Python 3.12, not our code).
export PYTHONWARNINGS="ignore::SyntaxWarning:pydub"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HF_HOME="${HF_HOME:-$PWD/cache/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/cache}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate the virtualenv if it exists.
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Prefer the venv python, then python3, then python.
PYTHON=$(command -v python 2>/dev/null || command -v python3 2>/dev/null)

"$PYTHON" dubbing_pipeline.py "$@"
