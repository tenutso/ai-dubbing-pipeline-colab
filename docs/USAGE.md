# Usage Guide

How to run the dubbing pipeline, every CLI option, advanced workflows and a
description of each output file.

---

## 1. Command overview

```bash
python dubbing_pipeline.py --input <MP4_OR_VIMEO_URL> [options]
```

On Colab, use the wrapper which sets GPU-friendly env vars and activates the venv:

```bash
bash run_dub.sh --input <MP4_OR_VIMEO_URL> [options]
```

All arguments after `run_dub.sh` are forwarded directly to `dubbing_pipeline.py`.

---

## 2. CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Path to a local `.mp4` **or** a Vimeo/`yt-dlp`-supported URL. |
| `--output_dir` | `outputs` | Directory where all artifacts are written (created if missing). |
| `--glossary` | *(none)* | Path to an OQLF glossary `.txt` file passed to Gemini for consistent terminology. |
| `--tts_lang` | `fr` | XTTS-V2 language code (ISO 639-1). Common values: `fr`, `en`, `es`, `de`, `it`, `pt`. |
| `--tts_temperature` | `0.65` | XTTS-V2 expressiveness dial. `0.1` = flat/consistent, `1.0` = varied/expressive. |
| `--sample_min_duration` | `3.0` | Minimum utterance length (seconds) to qualify as a speaker clone reference sample. |
| `--model` | `small` | WhisperX model size: `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger = more accurate, slower, more VRAM. |
| `--device` | `cuda` | `cuda` (GPU) or `cpu`. |
| `--batch_size` | `8` | WhisperX transcription batch size. Lower if you hit GPU OOM. |
| `--min_speakers` | *(none)* | **Speaker range forcing** — minimum number of speakers the diarizer may detect. |
| `--max_speakers` | *(none)* | **Speaker range forcing** — maximum number of speakers the diarizer may detect. |
| `--context` | *(none)* | One-sentence description of the video content (e.g. `"a corporate interview about HR software"`). Passed to Gemini to improve translation register and vocabulary. |
| `--resume` | `False` | Reload checkpoints from `--output_dir` and skip completed stages. Critical for long videos on free Colab. |

---

## 3. Parameter deep-dive

### `--context`

A one-sentence description of the video that is injected into the Gemini
translation prompt. It helps Gemini choose the right register, vocabulary, and
domain terminology without needing a full glossary.

```bash
--context "a two-person interview about enterprise HR software"
--context "a documentary about Quebec's forestry industry"
--context "a product demo for a mobile payments app"
```

Without `--context`, Gemini defaults to a neutral register. The flag is most
useful when the video uses specialised vocabulary or when the register (casual
vs formal) needs to be preserved accurately.

### `--tts_temperature`

Controls how much XTTS-V2 varies delivery across utterances.

| Value | Effect |
|-------|--------|
| `0.1–0.3` | Very consistent; robotic on long passages. |
| `0.5–0.7` | **Recommended range** — natural variation without instability. |
| `0.8–1.0` | Expressive; good for dramatic content, may drift on technical speech. |

### `--tts_lang`

The XTTS-V2 model supports: `en`, `es`, `fr`, `de`, `it`, `pt`, `pl`, `tr`,
`ru`, `nl`, `cs`, `ar`, `zh-cn`, `ja`, `ko`, `hu`.

### `--model`

Controls the Whisper transcription quality/speed trade-off.

| Model | VRAM (approx.) | When to use |
|-------|----------------|-------------|
| `tiny` / `base` | ~1–2 GB | Quick tests, very clean audio. |
| `small` | ~2–3 GB | **Default** — good balance on free Colab. |
| `medium` | ~5 GB | Noisy audio, accents, better accuracy. |
| `large-v3` | ~10 GB | Best accuracy; needs a high-VRAM GPU. |

> Note: XTTS-V2 needs ~3–4 GB of VRAM. The pipeline explicitly frees WhisperX
> GPU memory before loading XTTS-V2, so `small` + XTTS-V2 fits comfortably on
> Colab's T4.

### `--resume`

After each expensive stage the pipeline writes a JSON checkpoint:

| File | Stage saved |
|------|-------------|
| `ckpt_segments.json` | After WhisperX transcription |
| `ckpt_utterances.json` | After utterance merging |
| `ckpt_translated.json` | After Gemini translation *(most important)* |

If Colab disconnects, re-clone the repo, re-run `setup_colab.sh`, and add
`--resume` to your command. The pipeline loads from the deepest available
checkpoint and continues from there.

### Speaker range forcing (`--min_speakers` / `--max_speakers`)

Constrains pyannote's automatic speaker count estimate:

```bash
# Exactly two speakers (e.g. a two-person interview)
python dubbing_pipeline.py --input interview.mp4 --min_speakers 2 --max_speakers 2

# Between 2 and 4 speakers (panel discussion)
python dubbing_pipeline.py --input panel.mp4 --min_speakers 2 --max_speakers 4
```

> Requires a valid `HF_TOKEN`.

---

## 4. Output files

Everything lands in `--output_dir` (default `outputs/`):

| File | Description |
|------|-------------|
| `final_dubbed_video.mp4` | **Main deliverable** — original video with the cloned-voice French audio track muxed in. |
| `dubbed_audio.wav` | Voice-only dubbed audio: XTTS-V2 cloned voices placed on a silent timeline, no original audio mixed in. |
| `subtitles.srt` | SubRip subtitles anchored to dubbed audio timing (cues appear when the cloned voice speaks). |
| `manifest.json` | Full run metadata: speaker profiles, clone sample paths, and every utterance with original + dubbed timings + translation. |
| `speaker_samples/` | One WAV per speaker — the reference clip used for XTTS-V2 voice cloning. |
| `original_audio.wav` | Intermediate mono 16 kHz extraction of the source audio. |
| `ckpt_*.json` | Stage checkpoints; keep these to enable `--resume` on reconnect. |
| `input_video.mp4` | Only when `--input` is a URL: the downloaded source video. |

---

## 5. Common workflows

### A. Dub a local interview with glossary and domain context

```bash
python dubbing_pipeline.py \
    --input inputs/interview.mp4 \
    --tts_lang fr \
    --tts_temperature 0.65 \
    --context "a two-person interview about enterprise HR software" \
    --glossary examples/oqlf_glossary.txt \
    --output_dir outputs/interview
```

In a **Colab cell**, use `%%bash` for multi-line commands — the `!` magic does
not reliably handle backslash line continuation (each continued line becomes
its own command, causing `--tts_lang: command not found`):

```python
%%bash
bash run_dub.sh \
    --input inputs/interview.mp4 \
    --tts_lang fr \
    --tts_temperature 0.65 \
    --context "a two-person interview about enterprise HR software" \
    --glossary examples/oqlf_glossary.txt
```

Or put everything on one line with `!`:
```python
!bash run_dub.sh --input inputs/interview.mp4 --tts_lang fr --tts_temperature 0.65
```

### B. Dub a Vimeo video, force a known speaker count

```bash
python dubbing_pipeline.py \
    --input https://vimeo.com/123456789 \
    --min_speakers 2 --max_speakers 2 \
    --model medium \
    --tts_lang fr \
    --context "a documentary about Quebec infrastructure"
```

### C. Colab — idempotent setup (safe to re-run)

Use this pattern in your notebook instead of a bare `!git clone` to avoid
nested `repo/repo/repo/` directories when cells are re-run:

```python
import os
REPO = "ai-dubbing-pipeline-colab"
if not os.path.isdir(f"/content/{REPO}"):
    !git clone https://github.com/tenutso/ai-dubbing-pipeline-colab.git /content/{REPO}
%cd /content/{REPO}
!bash setup_colab.sh
```

### D. Resume after a Colab disconnection

```python
# After reconnecting — skips any completed stages:
%%bash
bash run_dub.sh --input inputs/long_video.mp4 --tts_lang fr --resume
```

### E. CPU-only quick test (no GPU)

```bash
python dubbing_pipeline.py --input inputs/short.mp4 --device cpu --model base --tts_lang fr
```

### F. Burn subtitles into the video

```bash
# After the pipeline finishes, burn the SRT into the MP4:
ffmpeg -i outputs/final_dubbed_video.mp4 \
       -vf subtitles=outputs/subtitles.srt \
       outputs/final_with_subs_burned.mp4
```

### G. Inspect the manifest

```bash
python -c "import json; m=json.load(open('outputs/manifest.json')); print(json.dumps(m['profiles'], indent=2))"
```

---

## 6. Tips

- **Start small.** Validate on a 30–60 second clip before processing a long video — it saves GPU minutes and catches glossary/context issues early.
- **Use `--context`.** Even a single sentence dramatically improves Gemini's register and vocabulary choices. It costs nothing and skips the need for a glossary in many cases.
- **Glossaries for overrides only.** Gemini already knows OQLF norms; a glossary is most valuable for brand names, product names, and client-specific terminology Gemini can't know.
- **Speaker sample quality.** XTTS-V2 clones best from clean, 3–10 second clips. Raise `--sample_min_duration` to 5 or 6 if your source has heavy background noise.
- **Temperature tuning.** `0.3–0.5` for flat, consistent narration (documentaries, presentations). `0.6–0.7` for conversational interviews. Above `0.75` for dramatic or expressive content.
- **Watch Gemini quotas.** The free tier has rate limits; long videos with many utterances consume more tokens. The checkpoint system ensures translation is never re-run on `--resume`.
- **Cache XTTS-V2 to Drive.** Follow the instructions in `setup_colab.sh` to symlink the model cache to Google Drive — saves ~3–5 minutes of download time on every Colab reconnect.
- **Avoid nested clones.** Always use the idempotent `os.path.isdir` guard (see workflow C) instead of a bare `!git clone` — re-running a bare clone from inside the repo creates `repo/repo/repo/` subdirectories.
