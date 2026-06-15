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
| `--resume` | `False` | Reload checkpoints from `--output_dir` and skip completed stages. Critical for long videos on free Colab. |

---

## 3. Parameter deep-dive

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
| `dubbed_audio.wav` | The mixed audio: original ducked by 15 dB + XTTS-V2 cloned voices overlaid in sync. |
| `subtitles.srt` | SubRip subtitles anchored to dubbed audio timing (cues appear when the cloned voice speaks). |
| `manifest.json` | Full run metadata: speaker profiles, clone sample paths, and every utterance with original + dubbed timings + translation. |
| `speaker_samples/` | One WAV per speaker — the reference clip used for XTTS-V2 voice cloning. |
| `original_audio.wav` | Intermediate mono 16 kHz extraction of the source audio. |
| `ckpt_*.json` | Stage checkpoints; keep these to enable `--resume` on reconnect. |
| `input_video.mp4` | Only when `--input` is a URL: the downloaded source video. |

---

## 5. Common workflows

### A. Dub a local interview with a glossary

```bash
python dubbing_pipeline.py \
    --input inputs/interview.mp4 \
    --glossary examples/oqlf_glossary.txt \
    --tts_lang fr \
    --tts_temperature 0.65 \
    --output_dir outputs/interview
```

### B. Dub a Vimeo video, force a known speaker count

```bash
python dubbing_pipeline.py \
    --input https://vimeo.com/123456789 \
    --min_speakers 2 --max_speakers 2 \
    --model medium \
    --tts_lang fr
```

### C. Resume after a Colab disconnection

```bash
# First run (disconnected mid-way):
bash run_dub.sh --input inputs/long_video.mp4 --tts_lang fr

# After reconnecting — skips any completed stages:
bash run_dub.sh --input inputs/long_video.mp4 --tts_lang fr --resume
```

### D. CPU-only quick test (no GPU)

```bash
python dubbing_pipeline.py --input inputs/short.mp4 --device cpu --model base --tts_lang fr
```

### E. Burn subtitles into the video

```bash
# After the pipeline finishes, burn the SRT into the MP4:
ffmpeg -i outputs/final_dubbed_video.mp4 \
       -vf subtitles=outputs/subtitles.srt \
       outputs/final_with_subs_burned.mp4
```

### F. Inspect the manifest

```bash
python -c "import json; m=json.load(open('outputs/manifest.json')); print(json.dumps(m['profiles'], indent=2))"
```

---

## 6. Tips

- **Start small.** Validate on a 30–60 second clip before processing a long video.
- **Glossaries matter.** A short, well-curated glossary keeps brand names and
  OQLF-preferred wording consistent across utterances.
- **Speaker sample quality.** XTTS-V2 clones best from clean, 3–10 second clips.
  Use `--sample_min_duration` to raise the bar if your source has background noise.
- **Temperature tuning.** Interview speech works well at `0.6–0.7`. Documentary
  narration or presentations can go lower (`0.4–0.5`) for a flatter, cleaner read.
- **Watch your quotas.** Gemini has a free-tier rate limit; long videos with many
  utterances consume more tokens. The checkpoint system ensures you only pay for
  translation once per run.
- **Cache XTTS-V2 to Drive.** On Colab, follow the instructions in `setup_colab.sh`
  to symlink the model cache to Google Drive — this saves ~3–5 minutes of download
  time on every reconnect.
