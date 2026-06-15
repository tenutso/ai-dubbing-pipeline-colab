# Architecture & Design Notes

This document describes how the dubbing pipeline is structured, how each
component works, and the rationale behind the key technical decisions.

---

## 1. Pipeline flow

```
                        ┌──────────────────────────────┐
   --input (MP4/URL) ─▶ │ 1. Acquire media             │
                        │    download_vimeo / passthru │
                        └──────────────┬───────────────┘
                                       │ video.mp4
                        ┌──────────────▼───────────────┐
                        │ 2. Extract audio             │
                        │    extract_audio_to_wav      │  ffmpeg → mono 16kHz WAV
                        └──────────────┬───────────────┘
                                       │ original_audio.wav
                        ┌──────────────▼───────────────┐
                        │ 3. Transcribe + align + diar │
                        │    transcribe_with_whisperx  │  WhisperX + pyannote
                        │                              │  ← ckpt_segments.json
                        └──────────────┬───────────────┘
                                       │ segments[]
                        ┌──────────────▼───────────────┐
                        │ 4. Merge to utterances       │
                        │    merge_segments_to_uttr…   │  same-speaker coalescing
                        │                              │  ← ckpt_utterances.json
                        └──────────────┬───────────────┘
                                       │ utterances[]
                        ┌──────────────▼───────────────┐
                        │ 5. Build speaker profiles    │
                        │    build_speaker_profiles    │  pitch / RMS / rate (metadata)
                        └──────────────┬───────────────┘
                                       │ profiles{}
                        ┌──────────────▼───────────────┐
                        │ 5b. Extract speaker samples  │
                        │    extract_speaker_samples   │  best clip per speaker
                        │                              │  → speaker_samples/{SPK}.wav
                        └──────────────┬───────────────┘
                                       │ speaker_samples{}
                        ┌──────────────▼───────────────┐
                        │ 6. Translate (OQLF French)   │
                        │    translate_utterances      │  Gemini, batched JSON
                        │                              │  ← ckpt_translated.json
                        └──────────────┬───────────────┘
                                       │ utterances[].translated_text
                     [free WhisperX GPU memory]
                        ┌──────────────▼───────────────┐
                        │ 7. XTTS-V2 voice cloning     │
                        │    build_dub_track_xtts      │  clone + time-stretch + mix
                        │    synthesize_line_xtts      │  per-utterance synthesis
                        │    _time_stretch_to_fit      │  librosa time-stretch
                        └──────────────┬───────────────┘
                                       │ dubbed_audio.wav
                                       │ utterances[].dubbed_start/end  ← anchored timing
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼
   ┌────────────────────┐  ┌────────────────────┐   ┌────────────────────┐
   │ 8. write_srt       │  │ 9. mux_video_…    │   │ 10. save_manifest  │
   │  subtitles.srt     │  │  final_dubbed.mp4  │   │  manifest.json     │
   │  (anchored timing) │  └────────────────────┘   └────────────────────┘
   └────────────────────┘
```

---

## 2. Components

| Stage | Function(s) | Library | Responsibility |
|-------|-------------|---------|----------------|
| Media acquisition | `download_vimeo` | yt-dlp | Download a remote video to MP4. Local files pass through untouched. |
| Audio extraction | `extract_audio_to_wav` | ffmpeg | Produce a mono 16 kHz PCM WAV — the format WhisperX and librosa expect. |
| Transcription | `transcribe_with_whisperx` | WhisperX | ASR + word-level alignment + speaker diarization. WhisperX and alignment models are deleted immediately after use to free GPU memory. |
| Utterance merge | `merge_segments_to_utterances` | — | Coalesce consecutive same-speaker segments into natural utterances. |
| Speaker profiling | `build_speaker_profiles` | librosa / numpy | Compute per-speaker pitch, RMS energy and rate of speech for manifest metadata. |
| Speaker sample extraction | `extract_speaker_samples` | librosa / soundfile | Select the highest-quality utterance per speaker (scored by `duration × RMS`, min 3 s) and save it as a 24 kHz WAV for XTTS-V2 cloning. |
| Translation | `translate_batch_with_gemini`, `translate_utterances` | google-genai | Batch-translate utterances to OQLF French with optional glossary. |
| GPU memory flush | `_free_gpu_memory` | gc / torch | Release WhisperX GPU memory before loading XTTS-V2 to avoid OOM on Colab T4. |
| XTTS-V2 synthesis | `_make_xtts_model`, `synthesize_line_xtts` | Coqui TTS | Load XTTS-V2 once, clone each speaker's voice from their reference sample, synthesize translated text. |
| Time-stretching | `_time_stretch_to_fit` | librosa | Compress or expand each dubbed clip to fit its original utterance window (rate capped 0.5×–2.0×). |
| Dub track assembly | `build_dub_track_xtts` | pydub | Overlay all time-stretched clips on the 15 dB-ducked original; record `dubbed_start`/`dubbed_end` on each utterance. |
| Subtitles | `write_srt` | — | Emit SRT cues anchored to `dubbed_start`/`dubbed_end` so each subtitle appears exactly when the cloned voice speaks. |
| Muxing | `mux_video_with_audio` | ffmpeg | Replace the video's audio with the dub track. |
| Checkpoints | `save_checkpoint`, `load_checkpoint` | json | Write/read stage results so `--resume` can skip completed stages after a Colab disconnection. |
| Manifest | `save_manifest` | — | Persist run metadata as JSON. |
| Orchestration | `main` | argparse | Wire stages together; implement stage-gating logic for `--resume`. |

---

## 3. Voice cloning algorithm

XTTS-V2 replaces the previous Google Cloud TTS stock-voice approach.  No API
key is required — the model runs locally on the Colab GPU.

**Step 1 — Sample extraction** (`extract_speaker_samples`):

For every diarized speaker the pipeline scores each utterance by
`duration × RMS energy` and selects the highest-scoring clip that meets the
`--sample_min_duration` threshold (default 3 s).  This maximises:

- **Clarity** — higher RMS tends to mean less background noise.
- **Length** — XTTS-V2 encodes a richer speaker embedding from longer clips.

The selected chunk is resampled to XTTS-V2's native 24 kHz and saved to
`speaker_samples/{SPEAKER_XX}.wav`.

**Step 2 — Synthesis** (`synthesize_line_xtts`):

Each translated utterance is synthesized by calling `TTS.tts_to_file` with the
speaker's reference WAV.  XTTS-V2 encodes a speaker embedding from the reference
at inference time — no fine-tuning or separate training is needed.

**Step 3 — Temperature** (`--tts_temperature`):

Controls the softmax temperature of XTTS-V2's internal acoustic model.  Lower
values produce flatter, more predictable delivery; higher values produce more
natural prosodic variation.  Recommended range: `0.5–0.7` for interview or
presentation content.

---

## 4. Time-stretching strategy

After synthesis, each clip's duration is compared to the original utterance
window (`utt["end"] - utt["start"]`).  `librosa.effects.time_stretch` is applied
with a rate capped to `[0.5, 2.0]` to avoid perceptible distortion:

| Scenario | Rate | Effect |
|----------|------|--------|
| Translated text is shorter | < 1.0 | Clip is slowed down to fill the window |
| Translated text is longer | > 1.0 | Clip is sped up to fit the window |
| Difference < 2% | 1.0 | No processing — avoids unnecessary phase artifacts |
| Rate would exceed 2.0 | 2.0 (capped) | Clip is compressed as much as feasible; may overflow window |

The actual clip duration after stretching is recorded as `dubbed_end - dubbed_start`
on each utterance and used to generate anchored SRT cues.

---

## 5. Anchored subtitles

Traditional subtitling uses the source transcript's timestamps.  When a dubbed
clip is longer or shorter than the original utterance (due to translation length
or stretch-cap overflow), those timestamps no longer reflect when the voice is
actually speaking.

`write_srt` reads `utt["dubbed_start"]` and `utt["dubbed_end"]` (set by
`build_dub_track_xtts`) instead of the raw WhisperX timestamps.  This means
each cue is precisely timed to the synthesized audio in the output video.

---

## 6. Checkpoint / resume system

Three checkpoints are written inside `--output_dir`:

| File | Written after | Value |
|------|--------------|-------|
| `ckpt_segments.json` | WhisperX transcription | Avoids re-running ASR (most GPU-intensive step). |
| `ckpt_utterances.json` | Utterance merge | Minor benefit on its own, but enables partial resume from merge stage. |
| `ckpt_translated.json` | Gemini translation | Avoids re-paying Gemini API costs and waiting for LLM calls. |

On `--resume`, `main()` probes for the deepest available checkpoint and jumps
directly to the next uncompleted stage.  Speaker sample extraction and XTTS-V2
synthesis always run because their outputs (WAV files) are fast to regenerate
relative to ASR and translation.

---

## 7. GPU memory management

WhisperX (`small`) and XTTS-V2 together require ~5–7 GB of VRAM.  The pipeline
explicitly deletes each WhisperX model object and calls `torch.cuda.empty_cache()`
after the transcription and diarization stages complete.  This ensures XTTS-V2
can claim the freed memory without hitting OOM on Colab's T4 (16 GB).

---

## 8. Translation approach

- **Batched, structured I/O.** All utterance texts are sent to Gemini as a single
  JSON list with `response_mime_type: application/json`.
- **OQLF standard.** The prompt explicitly requests French following the *Office
  québécois de la langue française* norms.
- **Glossary injection.** When a glossary file is supplied, its contents are
  embedded in the prompt for consistent brand/term translation.
- **Graceful fallback.** If the model returns fewer items than expected, the
  original text is retained for any missing index.

---

## 9. Audio mixing strategy

`build_dub_track_xtts` ducks the original audio by **15 dB** to preserve
background sound and music, then overlays time-stretched synthesized clips at
their original start positions.  Temporary per-line WAVs are written and deleted
immediately to keep disk usage low.

---

## 10. Technical decisions & rationale

| Decision | Rationale |
|----------|-----------|
| **XTTS-V2** over Google Cloud TTS | Zero-shot voice cloning from a 6-second reference — no API key, no per-character billing, and the cloned voice is far more natural than stock voices. |
| **`float16` on CUDA, `int8` on CPU** | Maximizes WhisperX throughput while remaining runnable on CPU for testing. |
| **24 kHz speaker samples** | XTTS-V2's native sample rate; resampling at extraction time is cheaper than doing it at every synthesis call. |
| **Time-stretch cap [0.5, 2.0]** | Beyond these ratios, phase-vocoder artifacts become perceptible. Clips that hit the cap overflow their window slightly rather than sound distorted. |
| **Checkpoint JSON, not pickle** | Human-readable, git-diff-able, and safe to inspect mid-run. |
| **Anchored subtitles** | Subtitles synced to the original transcript timestamps drift when the dubbed clip is stretched or overflows; anchoring them to actual dubbed duration eliminates that drift. |
| **WhisperX model deletion before XTTS-V2** | Colab T4 has ~16 GB VRAM; keeping both models resident simultaneously risks OOM, especially with `medium`/`large-v3` Whisper. |
| **Mono 16 kHz WAV for ASR** | The canonical input format for Whisper/pyannote and a stable basis for librosa analysis. |
| **Ducking instead of source separation** | Removing the original requires a source-separation model (e.g. demucs); ducking is cheap, preserves ambience, and is good enough for most content. |
| **JSON manifest** | Makes runs auditable and enables downstream tooling (re-synthesis, QA, editing) without re-running ASR. |

---

## 11. Known limitations & future work

- **Source separation** — replacing ducking with demucs would allow full removal
  of the original voices while preserving music and ambience.
- **Lip-sync** — dubbed clips are placed at the original start time but there is
  no per-phoneme alignment to lip movements.
- **Per-line retry / partial resume** — synthesis failures on individual utterances
  are currently skipped; a retry mechanism would improve robustness on long videos.
- **Multi-reference speaker samples** — XTTS-V2 supports multiple reference WAVs
  per speaker; using several clips could improve clone consistency.
