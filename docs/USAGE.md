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
| `--input` | *(required)* | Path to a local `.mp4` **or** a Vimeo/`yt-dlp`-supported URL. URLs are detected automatically (anything starting with `http`). |
| `--output_dir` | `outputs` | Directory where all artifacts are written (created if missing). |
| `--glossary` | *(none)* | Path to an OQLF glossary `.txt` file passed to Gemini for consistent terminology. |
| `--lang` | `fr-CA` | Target TTS language/locale code (BCP-47). Determines which Google voices are available. |
| `--model` | `small` | WhisperX model size: `tiny`, `base`, `small`, `medium`, `large-v3`. Bigger = more accurate, slower, more VRAM. |
| `--device` | `cuda` | `cuda` (GPU) or `cpu`. |
| `--batch_size` | `8` | WhisperX transcription batch size. Lower it if you hit GPU OOM. |
| `--min_speakers` | *(none)* | **Speaker range forcing** — minimum number of speakers the diarizer may detect. |
| `--max_speakers` | *(none)* | **Speaker range forcing** — maximum number of speakers the diarizer may detect. |

---

## 3. Parameter deep-dive

### `--model`
Controls the Whisper transcription quality/speed trade-off.

| Model | VRAM (approx.) | When to use |
|-------|----------------|-------------|
| `tiny` / `base` | ~1–2 GB | Quick tests, very clean audio. |
| `small` | ~2–3 GB | **Default** — good balance on free Colab. |
| `medium` | ~5 GB | Noisy audio, accents, better accuracy. |
| `large-v3` | ~10 GB | Best accuracy; needs a high-VRAM GPU. |

### `--lang`
Selects the Google TTS locale (e.g. `fr-CA` for Quebec French, `fr-FR` for
France French). The available voices for that locale are listed at runtime and
matched to speakers. Note: translation always targets **OQLF French** regardless,
but `fr-CA` voices give the most natural Quebec pronunciation.

### Speaker range forcing (`--min_speakers` / `--max_speakers`)
By default pyannote estimates the number of speakers automatically. If you
already know the count, constrain it to avoid over/under-segmentation:

```bash
# Exactly two speakers (e.g. a two-person interview)
python dubbing_pipeline.py --input interview.mp4 --min_speakers 2 --max_speakers 2

# Between 2 and 4 speakers (panel discussion)
python dubbing_pipeline.py --input panel.mp4 --min_speakers 2 --max_speakers 4
```

> Requires a valid `HF_TOKEN` (diarization must be active). Without it, these
> flags have no effect and everything is labeled `SPEAKER_00`.

---

## 4. Output files

Everything lands in `--output_dir` (default `outputs/`):

| File | Description |
|------|-------------|
| `final_dubbed_video.mp4` | **Main deliverable** — original video with the new French audio track muxed in. |
| `dubbed_audio.wav` | The mixed audio: original ducked by 15 dB + synthesized French voices overlaid in sync. |
| `subtitles.srt` | SubRip subtitles, each cue prefixed with the speaker label, e.g. `[SPEAKER_01] Bonjour…`. |
| `manifest.json` | Full run metadata: per-speaker voice assignments, prosody, and every utterance with timings + translation. |
| `original_audio.wav` | Intermediate mono 16 kHz extraction of the source audio. |
| `input_video.mp4` | Only when `--input` is a URL: the downloaded source video. |

### Inspecting the manifest

```bash
python -c "import json; m=json.load(open('outputs/manifest.json')); print(json.dumps(m['speakers'], indent=2))"
```

```json
{
  "SPEAKER_00": { "voice_name": "fr-CA-Neural2-D", "pitch": 0.0, "speaking_rate": 0.92 },
  "SPEAKER_01": { "voice_name": "fr-CA-Neural2-A", "pitch": 0.0, "speaking_rate": 1.05 }
}
```

---

## 5. Common workflows

### A. Dub a local interview with a glossary

```bash
python dubbing_pipeline.py \
    --input inputs/interview.mp4 \
    --glossary examples/oqlf_glossary.txt \
    --output_dir outputs/interview
```

### B. Dub a Vimeo video, force a known speaker count

```bash
python dubbing_pipeline.py \
    --input https://vimeo.com/123456789 \
    --min_speakers 2 --max_speakers 2 \
    --model medium
```

### C. CPU-only quick test (no GPU)

```bash
python dubbing_pipeline.py --input inputs/short.mp4 --device cpu --model base
```

### D. Subtitles only (reuse the SRT, ignore the video)

Run the pipeline normally, then grab `outputs/subtitles.srt`. You can burn it
into any player or re-mux it with ffmpeg:

```bash
ffmpeg -i source.mp4 -vf subtitles=outputs/subtitles.srt out_with_subs.mp4
```

---

## 6. Tips

- **Glossaries matter.** A short, well-curated glossary (see
  [`examples/oqlf_glossary.txt`](../examples/oqlf_glossary.txt)) keeps brand
  names, product terms and OQLF-preferred wording consistent.
- **Start small.** Validate the whole pipeline on a 30–60 second clip before
  processing a long video — it saves a lot of GPU minutes.
- **Watch your quotas.** Gemini and Google TTS both have free-tier limits; long
  videos with many utterances consume more requests/characters.
