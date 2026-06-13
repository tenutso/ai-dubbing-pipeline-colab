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
                        └──────────────┬───────────────┘
                                       │ segments[] (text, start, end, speaker)
                        ┌──────────────▼───────────────┐
                        │ 4. Merge to utterances       │
                        │    merge_segments_to_uttr…   │  same-speaker coalescing
                        └──────────────┬───────────────┘
                                       │ utterances[]
                        ┌──────────────▼───────────────┐
                        │ 5. Build speaker profiles    │
                        │    build_speaker_profiles    │  pitch / RMS / rate
                        └──────────────┬───────────────┘
                                       │ profiles{}
                        ┌──────────────▼───────────────┐
                        │ 6. Assign voices + prosody   │
                        │    assign_voices_and_prosody │  Google TTS voice picker
                        └──────────────┬───────────────┘
                                       │ speaker_configs{}
                        ┌──────────────▼───────────────┐
                        │ 7. Translate (OQLF French)   │
                        │    translate_utterances      │  Gemini, batched JSON
                        └──────────────┬───────────────┘
                                       │ utterances[].translated_text
                        ┌──────────────▼───────────────┐
                        │ 8. Synthesize + mix dub      │
                        │    build_dub_track           │  TTS + pydub overlay
                        └──────────────┬───────────────┘
                                       │ dubbed_audio.wav
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼
   ┌────────────────────┐  ┌────────────────────┐   ┌────────────────────┐
   │ 9. write_srt       │  │ 10. mux_video_…    │   │ 11. save_manifest  │
   │  subtitles.srt     │  │  final_dubbed.mp4  │   │  manifest.json     │
   └────────────────────┘  └────────────────────┘   └────────────────────┘
```

---

## 2. Components

| Stage | Function(s) | Library | Responsibility |
|-------|-------------|---------|----------------|
| Media acquisition | `download_vimeo` | yt-dlp | Download a remote video to MP4. Local files pass through untouched. |
| Audio extraction | `extract_audio_to_wav` | ffmpeg | Produce a mono 16 kHz PCM WAV — the format WhisperX and librosa expect. |
| Transcription | `transcribe_with_whisperx` | WhisperX | ASR + word-level alignment + speaker diarization. |
| Utterance merge | `merge_segments_to_utterances` | — | Coalesce consecutive same-speaker segments into natural utterances. |
| Speaker profiling | `build_speaker_profiles` | librosa / numpy | Compute per-speaker pitch, RMS energy and rate of speech. |
| Voice assignment | `assign_voices_and_prosody` | google-cloud-texttospeech | Map each speaker to a TTS voice + speaking rate. |
| Translation | `translate_batch_with_gemini`, `translate_utterances` | google-genai | Batch-translate utterances to OQLF French with optional glossary. |
| Synthesis & mix | `synthesize_line`, `build_dub_track` | google-cloud-texttospeech, pydub | Generate speech per line and overlay on the ducked original. |
| Subtitles | `write_srt` | — | Emit per-speaker SRT cues. |
| Muxing | `mux_video_with_audio` | ffmpeg | Replace the video's audio with the dub track. |
| Manifest | `save_manifest` | — | Persist run metadata as JSON. |
| Orchestration | `main` | argparse | Wire the stages together from CLI args. |

---

## 3. Speaker matching algorithm

The goal is to give each diarized speaker a *distinct, tonally plausible* voice
without any manual configuration.

**Step 1 — Acoustic profiling** (`build_speaker_profiles`):
For every utterance belonging to a speaker, we slice the corresponding audio
window and compute:

- **Pitch** via `librosa.piptrack`, averaged over voiced frames.
- **RMS energy** (`sqrt(mean(chunk²))`) as a loudness proxy.
- **Rate of speech** = total words ÷ total speaking duration (words/second).

These per-utterance values are aggregated per speaker. The mean pitch is bucketed
into a **pitch category**:

| Mean pitch (Hz) | Category |
|-----------------|----------|
| `< 120` | `low` |
| `120 – 200` | `medium` |
| `> 200` | `high` |

**Step 2 — Voice selection** (`assign_voices_and_prosody`):
1. Derive a gender preference from the pitch category — `high → FEMALE`,
   otherwise `MALE`.
2. List the available Google TTS voices for the target locale and filter by that
   gender.
3. Round-robin (`i % len(filtered)`) through the filtered list so that distinct
   speakers receive distinct voices, falling back to the full voice list if no
   gendered match exists.

**Step 3 — Prosody** : the measured rate of speech is normalized into the TTS
valid range with `max(0.25, min(4.0, rate / 2.5))`, so faster talkers get a
slightly faster synthetic delivery. Pitch shifting is left at `0.0` (neutral) to
avoid robotic artifacts — voice identity is carried by voice *selection*, not by
pitch manipulation.

---

## 4. Translation approach

- **Batched, structured I/O.** All utterance texts are sent to Gemini as a single
  JSON list with `response_mime_type: application/json`, and the model is
  instructed to return a list of the same length and order. This is far cheaper
  and more context-aware than translating line-by-line, and keeps indices aligned
  with the source utterances.
- **OQLF standard.** The prompt explicitly requests French following the *Office
  québécois de la langue française* norms, yielding Quebec-appropriate
  terminology.
- **Glossary injection.** When a glossary file is supplied, its contents are
  embedded in the prompt so brand names, jargon and preferred terms translate
  consistently.
- **Graceful fallback.** If the model returns fewer items than expected, the
  original text is retained for any missing index so the pipeline never crashes
  mid-run.

---

## 5. Audio mixing strategy

`build_dub_track` keeps the original audio bed for ambience and music but ducks
it by **15 dB**, then overlays each synthesized clip at its utterance start time
(`position = start * 1000 ms`) using `pydub`. This preserves background sound and
timing alignment while making the French voices clearly dominant. Temporary
per-line WAVs are written and deleted immediately to keep disk usage low on Colab.

---

## 6. Technical decisions & rationale

| Decision | Rationale |
|----------|-----------|
| **WhisperX** over vanilla Whisper | Provides word-level alignment *and* a built-in diarization path, which we need for per-speaker timing and voice assignment. |
| **`float16` on CUDA, `int8` on CPU** | Maximizes throughput on Colab GPUs while remaining runnable on CPU for testing. |
| **`uv` for dependencies** | Dramatically faster installs and reproducible resolution — important given Colab's ephemeral runtimes. |
| **Pinned versions** in `pyproject.toml` / `requirements.txt` | The ML stack (whisperx, google-genai, TTS) breaks easily across versions; pinning keeps Colab runs reproducible. |
| **Mono 16 kHz WAV** | The canonical input format for Whisper/pyannote and a stable basis for librosa analysis. |
| **Voice identity via selection, not pitch-shift** | Pitch-shifting synthetic speech sounds artificial; choosing an appropriate base voice is more natural. |
| **Ducking instead of full removal** | Removing the original loses music/ambience and requires source separation; ducking is cheap and good enough. |
| **JSON manifest** | Makes runs auditable and enables downstream tooling (re-synthesis, QA, editing) without re-running ASR. |

---

## 7. Known limitations & future work

- **No lip-sync / time-stretching** — synthesized lines may run longer than the
  original utterance window; a future version could time-stretch clips to fit.
- **Source separation** — replacing ducking with a music/speech separator would
  allow full removal of the original voices.
- **Voice cloning** — current matching uses stock Google voices; speaker-adaptive
  or cloned voices would improve fidelity.
- **Per-line retry/caching** — long videos would benefit from resumable runs and
  TTS/translation caching keyed by the manifest.
