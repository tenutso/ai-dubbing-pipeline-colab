"""
dubbing_pipeline.py
===================

AI-powered video dubbing pipeline optimized for Google Colab's free tier.

Given an MP4 file or a Vimeo URL, this pipeline will:
  1. Download / load the source video.
  2. Extract a mono 16 kHz WAV track.
  3. Transcribe + word-align + diarize speakers with WhisperX.
  4. Merge word/segment level results into speaker utterances.
  5. Build acoustic speaker profiles (pitch, RMS energy, rate of speech).
  6. Extract a high-quality audio sample per speaker for voice cloning.
  7. Translate every utterance to French (OQLF standard) with Gemini.
  8. Synthesize dubbed audio with XTTS-V2 (voice cloning) and time-stretch
     each clip to fit the original utterance window.
  9. Write an SRT subtitle file anchored to dubbed audio timing.
 10. Mux the dubbed audio back onto the original video.
 11. Save a JSON manifest describing speakers and utterances.

See docs/USAGE.md for CLI examples and docs/ARCHITECTURE.md for design notes.
"""

import gc
import os
import warnings
import argparse
import json
import subprocess
import tempfile
from datetime import timedelta

# Suppress torchaudio internal deprecation warnings that originate inside
# XTTS-V2 and are not actionable from user code.
warnings.filterwarnings(
    "ignore",
    message=".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*torchaudio.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*degrees of freedom is <= 0.*",
    category=UserWarning,
)

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from tqdm import tqdm
from pydub import AudioSegment

import whisperx
from google import genai

# Load environment variables from a local .env file (if present).
load_dotenv()


# --------------------------------------------------------------------------- #
# Credential helpers — Colab secrets → env vars → .env fallback
# --------------------------------------------------------------------------- #
def _get_secret(name, default=None):
    """Return a secret, preferring Colab userdata over environment variables."""
    try:
        from google.colab import userdata  # noqa: PLC0415
        try:
            val = userdata.get(name)
            if val:
                return val
        except userdata.SecretNotFoundError:
            pass
        except userdata.NotebookAccessError:
            print(
                f"[WARNING] Colab Secret '{name}' has notebook access disabled. "
                "Open the 🔑 Secrets panel and toggle the switch next to it."
            )
        except Exception:
            pass
    except ImportError:
        pass
    return os.getenv(name, default)


# --------------------------------------------------------------------------- #
# Device helpers
# --------------------------------------------------------------------------- #
def _resolve_device(requested):
    """Return the best available device, falling back to CPU if CUDA is unusable."""
    if requested != "cuda":
        return requested
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            print("[WARNING] CUDA not available — falling back to CPU (slower).")
            return "cpu"
        torch.zeros(1).cuda()
        return "cuda"
    except Exception as e:
        print(f"[WARNING] CUDA unusable ({e})\nFalling back to CPU (slower).")
        return "cpu"


def _free_gpu_memory():
    """Release GPU memory held by previously loaded models."""
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# JSON serialization helper
# --------------------------------------------------------------------------- #
class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars and arrays that json.dump can't handle natively."""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# --------------------------------------------------------------------------- #
# Checkpoint helpers — resume after Colab disconnection
# --------------------------------------------------------------------------- #
def save_checkpoint(data, path):
    """Persist a pipeline stage result to JSON for later resumption."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, cls=_NumpyEncoder)


def load_checkpoint(path):
    """Return parsed JSON from a checkpoint file, or None if it doesn't exist."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# --------------------------------------------------------------------------- #
# 1. Media acquisition & audio extraction
# --------------------------------------------------------------------------- #
def download_vimeo(url, output_path):
    """Download a Vimeo (or any yt-dlp supported) URL to ``output_path`` as MP4."""
    print(f"Downloading video: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        url,
        "-o", output_path,
    ]
    subprocess.run(cmd, check=True)


def extract_audio_to_wav(video_path, audio_path):
    """Extract a mono 16 kHz PCM WAV track from ``video_path``."""
    print(f"Extracting audio to: {audio_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
    ]
    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------------- #
# 2. Transcription, alignment & diarization (WhisperX)
# --------------------------------------------------------------------------- #
def transcribe_with_whisperx(
    audio_path,
    device="cuda",
    model_name="small",
    batch_size=8,
    hf_token=None,
    min_speakers=None,
    max_speakers=None,
):
    """Transcribe, word-align and (optionally) diarize an audio file.

    Parameters
    ----------
    audio_path : str
        Path to the WAV file produced by :func:`extract_audio_to_wav`.
    device : str
        ``"cuda"`` (GPU, recommended) or ``"cpu"``.
    model_name : str
        WhisperX model size: ``tiny``/``base``/``small``/``medium``/``large-v3``.
    batch_size : int
        Inference batch size. Lower this if you hit GPU OOM.
    hf_token : str, optional
        Hugging Face token required for the pyannote diarization model.
        If omitted, every segment is assigned ``SPEAKER_00``.
    min_speakers, max_speakers : int, optional
        Force the diarizer to a known speaker count range.
    """
    print(f"Transcribing {audio_path} with WhisperX ({model_name})...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Free Whisper model before alignment model loads.
    del model
    _free_gpu_memory()

    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )
    del model_a
    _free_gpu_memory()

    if hf_token:
        print("Performing speaker diarization...")
        try:
            PipelineCls = whisperx.DiarizationPipeline
        except AttributeError:
            from whisperx.diarize import DiarizationPipeline as PipelineCls  # noqa: PLC0415
        import inspect  # noqa: PLC0415
        _params = inspect.signature(PipelineCls.__init__).parameters
        _token_kwarg = next(
            (p for p in ("hf_token", "use_auth_token", "token") if p in _params),
            None,
        )
        if _token_kwarg:
            diarize_model = PipelineCls(**{_token_kwarg: hf_token, "device": device})
        else:
            os.environ.setdefault("HF_TOKEN", hf_token)
            diarize_model = PipelineCls(device=device)
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        del diarize_model
        _free_gpu_memory()
    else:
        print("HF_TOKEN missing. Skipping diarization. Assigning 'SPEAKER_00' to all.")
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"

    for seg in result["segments"]:
        seg.setdefault("speaker", "SPEAKER_00")

    return result["segments"]


# --------------------------------------------------------------------------- #
# 3. Utterance merging
# --------------------------------------------------------------------------- #
def merge_segments_to_utterances(segments, max_duration=15.0):
    """Merge consecutive same-speaker segments into utterances.

    Segments are merged only when they share a speaker AND the combined duration
    stays within ``max_duration`` seconds.  Without the cap, a single-speaker
    video (or a failed diarization that labels everything SPEAKER_00) collapses
    into one enormous utterance — XTTS-V2 then synthesizes a very short clip
    which gets stretched to fill minutes, producing slow/distorted audio.

    Parameters
    ----------
    segments : list[dict]
        Word-aligned segments from WhisperX, each with ``speaker``, ``start``,
        ``end`` and ``text`` keys.
    max_duration : float
        Maximum merged utterance length in seconds.  15 s is a good default —
        long enough to be a complete sentence, short enough for XTTS-V2 to
        handle cleanly and for time-stretching to stay within the safe range.
    """
    utterances = []
    if not segments:
        return utterances

    current = segments[0].copy()
    for next_seg in segments[1:]:
        same_speaker = next_seg.get("speaker") == current.get("speaker")
        would_be_duration = next_seg["end"] - current["start"]
        if same_speaker and would_be_duration <= max_duration:
            current["end"] = next_seg["end"]
            current["text"] = current["text"].strip() + " " + next_seg["text"].strip()
        else:
            utterances.append(current)
            current = next_seg.copy()
    utterances.append(current)
    return utterances


# --------------------------------------------------------------------------- #
# 4. Acoustic speaker profiling
# --------------------------------------------------------------------------- #
def build_speaker_profiles(utterances, audio_path):
    """Estimate per-speaker pitch category, energy and rate of speech."""
    audio, sr = librosa.load(audio_path, sr=None)
    profiles = {}

    for utt in utterances:
        spk = utt["speaker"]
        if spk not in profiles:
            profiles[spk] = {"pitches": [], "rms": [], "durations": [], "word_counts": []}

        start_samp = int(utt["start"] * sr)
        end_samp = int(utt["end"] * sr)
        chunk = audio[start_samp:end_samp]

        if len(chunk) > 0:
            pitches, _ = librosa.piptrack(y=chunk, sr=sr)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            profiles[spk]["pitches"].append(pitch)
            profiles[spk]["rms"].append(float(np.sqrt(np.mean(chunk ** 2))))
            profiles[spk]["durations"].append(utt["end"] - utt["start"])
            profiles[spk]["word_counts"].append(len(utt["text"].split()))

    for spk, data in profiles.items():
        avg_pitch = np.mean(data["pitches"]) if data["pitches"] else 150
        profiles[spk]["avg_pitch"] = float(avg_pitch)
        profiles[spk]["pitch_category"] = (
            "low" if avg_pitch < 120 else "high" if avg_pitch > 200 else "medium"
        )
        profiles[spk]["avg_rms"] = float(np.mean(data["rms"])) if data["rms"] else 0.0
        total_dur = sum(data["durations"])
        profiles[spk]["rate_of_speech"] = (
            sum(data["word_counts"]) / total_dur if total_dur > 0 else 1.0
        )

    return profiles


# --------------------------------------------------------------------------- #
# 5. Speaker sample extraction for voice cloning
# --------------------------------------------------------------------------- #
_XTTS_SR = 24000  # XTTS-V2 native sample rate


def extract_speaker_samples(utterances, audio_path, output_dir, min_duration=3.0):
    """Extract a high-quality reference audio sample per speaker for XTTS-V2 cloning.

    For each speaker the utterance with the best ``duration × RMS`` score that
    meets the minimum duration threshold is selected.  If no utterance is long
    enough, the longest available is used as a fallback.

    Parameters
    ----------
    utterances : list[dict]
        Merged utterances with ``speaker``, ``start``, ``end`` keys.
    audio_path : str
        Path to the original mono WAV (any sample rate — resampled to 24 kHz).
    output_dir : str
        Root output directory.  Samples land in ``{output_dir}/speaker_samples/``.
    min_duration : float
        Minimum clip length in seconds to be considered as a reference sample.

    Returns
    -------
    dict[str, str]
        Mapping of ``speaker_id → wav_path``.
    """
    samples_dir = os.path.join(output_dir, "speaker_samples")
    os.makedirs(samples_dir, exist_ok=True)

    audio, sr = librosa.load(audio_path, sr=_XTTS_SR)

    by_speaker: dict = {}
    for utt in utterances:
        spk = utt["speaker"]
        dur = utt["end"] - utt["start"]
        rms = float(np.sqrt(np.mean(
            audio[int(utt["start"] * sr): int(utt["end"] * sr)] ** 2
        ))) if dur > 0 else 0.0
        by_speaker.setdefault(spk, []).append((dur, rms, utt))

    speaker_samples = {}
    for spk, entries in by_speaker.items():
        qualified = [(d, r, u) for d, r, u in entries if d >= min_duration]
        pool = qualified if qualified else entries
        # Score: duration × RMS — prefer long, loud clips
        _, _, best_utt = max(pool, key=lambda x: x[0] * x[1])

        start_samp = int(best_utt["start"] * sr)
        end_samp = int(best_utt["end"] * sr)
        chunk = audio[start_samp:end_samp]

        sample_path = os.path.join(samples_dir, f"{spk}.wav")
        sf.write(sample_path, chunk, sr)
        speaker_samples[spk] = sample_path
        print(f"  {spk}: sample {best_utt['end'] - best_utt['start']:.1f}s → {sample_path}")

    return speaker_samples


# --------------------------------------------------------------------------- #
# 6. Translation (Gemini, OQLF French)
# --------------------------------------------------------------------------- #
_TRANSLATION_BATCH_SIZE = 25  # utterances per Gemini call — keeps output well within token limits


def _call_gemini_translate(texts, system_prompt, user_prefix, model):
    """Send one batch to Gemini and return a list of translated strings."""
    client = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
    user_prompt = user_prefix + f"Strings: {json.dumps(texts, ensure_ascii=False)}"
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "response_mime_type": "application/json",
            "system_instruction": system_prompt,
        },
    )
    result = json.loads(response.text)
    if not isinstance(result, list):
        raise ValueError(
            f"Gemini returned unexpected type {type(result).__name__}; expected a JSON list"
        )
    return result


def translate_batch_with_gemini(texts, glossary_text="", video_context=""):
    """Translate a list of strings to OQLF French in batches, preserving order.

    Sends at most ``_TRANSLATION_BATCH_SIZE`` utterances per Gemini call to
    avoid truncated JSON from output-token overflow on long videos.
    """
    context_line = (
        f"The video is about: {video_context}\n" if video_context.strip() else ""
    )
    glossary_line = (
        f"Project-specific terminology to apply: {glossary_text}\n"
        if glossary_text.strip() else ""
    )
    user_prefix = context_line + glossary_line

    system_prompt = (
        "You are a professional Quebec French dubbing translator. "
        "Your translations must follow OQLF standards and sound completely natural "
        "when spoken aloud — use spoken Quebec French register, not formal written French. "
        "Rules:\n"
        "- Keep each translation as concise as possible while preserving meaning; "
        "shorter is better for voice sync.\n"
        "- Preserve sentence fragments and incomplete utterances exactly as fragments "
        "— do not complete or restructure them.\n"
        "- Never translate proper nouns, brand names, people's names, or acronyms "
        "unless they appear in the project terminology list.\n"
        "- Match the energy and register of each utterance (casual stays casual, "
        "formal stays formal).\n"
        "- Return a JSON array of strings, same length and order as the input."
    )

    model = _get_secret("GEMINI_MODEL", "gemini-2.5-flash")
    results = []
    total = len(texts)
    for start in range(0, total, _TRANSLATION_BATCH_SIZE):
        batch = texts[start: start + _TRANSLATION_BATCH_SIZE]
        end = min(start + _TRANSLATION_BATCH_SIZE, total)
        print(f"  Translating utterances {start + 1}–{end} of {total}...")
        results.extend(_call_gemini_translate(batch, system_prompt, user_prefix, model))
    return results


def translate_utterances(utterances, glossary_path=None, video_context=""):
    """Attach a ``translated_text`` field to every utterance."""
    glossary = ""
    if glossary_path and os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = f.read()

    texts = [u["text"] for u in utterances]
    translated_texts = translate_batch_with_gemini(texts, glossary, video_context)

    for i, utt in enumerate(utterances):
        utt["translated_text"] = (
            translated_texts[i] if i < len(translated_texts) else utt["text"]
        )
    return utterances


# --------------------------------------------------------------------------- #
# 7. XTTS-V2 speech synthesis with voice cloning
# --------------------------------------------------------------------------- #
def _make_xtts_model(device):
    """Load the XTTS-V2 model onto ``device``.

    Weights (~1.8 GB) are cached under ``$XDG_CACHE_HOME/tts`` on first run.
    On Colab, symlink that directory to Google Drive to avoid re-downloading
    on each session (see setup_colab.sh for the recommended command).
    """
    from TTS.api import TTS  # noqa: PLC0415
    # agree_to_tos=True bypasses the interactive TOS prompt that crashes in
    # non-interactive environments (Colab subprocesses, CI, etc.).
    # COQUI_TOS_AGREED is set in run_dub.sh as a belt-and-suspenders fallback.
    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    print("Loading XTTS-V2 model (first run downloads ~1.8 GB)...")
    return TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


_XTTS_MAX_CHARS = 250  # conservative buffer below XTTS-V2's hard 273-char truncation limit


def _split_text_for_xtts(text, max_chars=_XTTS_MAX_CHARS):
    """Split ``text`` into chunks that each fit within XTTS-V2's character limit.

    Splits at sentence-ending punctuation first (preserving natural prosody
    boundaries), then falls back to word-boundary splitting for any sentence
    that is itself too long.
    """
    import re  # noqa: PLC0415
    if len(text) <= max_chars:
        return [text]

    raw_sentences = re.split(r'(?<=[.!?;])\s+', text)
    chunks, current = [], ""
    for sentence in raw_sentences:
        if len(sentence) > max_chars:
            # Sentence alone exceeds limit — split word by word.
            if current:
                chunks.append(current.strip())
                current = ""
            for word in sentence.split():
                candidate = (current + " " + word).strip()
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current.strip())
                    current = word
        else:
            candidate = (current + " " + sentence).strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def synthesize_utterance_xtts(tts_model, text, speaker_wav, language, output_file, temperature=0.65):
    """Synthesize ``text`` to ``output_file``, chunking if it exceeds XTTS-V2's limit.

    XTTS-V2 silently truncates input beyond 273 characters, producing a short
    clip that then gets over-stretched. This function splits long texts at
    sentence boundaries, synthesizes each chunk, and concatenates the results
    so no audio is dropped.
    """
    chunks = _split_text_for_xtts(text)
    if len(chunks) == 1:
        tts_model.tts_to_file(
            text=chunks[0],
            speaker_wav=speaker_wav,
            language=language,
            file_path=output_file,
            temperature=temperature,
        )
        return

    chunk_files = []
    try:
        for chunk in chunks:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                chunk_path = tf.name
            tts_model.tts_to_file(
                text=chunk,
                speaker_wav=speaker_wav,
                language=language,
                file_path=chunk_path,
                temperature=temperature,
            )
            chunk_files.append(chunk_path)
        combined = AudioSegment.from_wav(chunk_files[0])
        for chunk_path in chunk_files[1:]:
            combined += AudioSegment.from_wav(chunk_path)
        combined.export(output_file, format="wav")
    finally:
        for f in chunk_files:
            if os.path.exists(f):
                os.remove(f)


def _time_stretch_to_fit(audio_path, target_dur_s):
    """Time-stretch the WAV at ``audio_path`` in place using ffmpeg atempo.

    ffmpeg's atempo filter is formant-preserving and produces far less
    boxiness/phasiness than librosa's phase vocoder.  Rate is capped to
    [0.5, 2.0] — atempo's valid single-stage range.
    """
    info = sf.info(audio_path)
    current_dur = info.duration
    if current_dur <= 0 or target_dur_s <= 0:
        return
    rate = current_dur / target_dur_s   # > 1 compresses, < 1 expands
    rate = max(0.5, min(2.0, rate))
    if abs(rate - 1.0) < 0.02:         # skip trivial stretches
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        tmp_out = tf.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-filter:a", f"atempo={rate:.6f}", tmp_out],
            check=True, capture_output=True,
        )
        os.replace(tmp_out, audio_path)
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def build_dub_track_xtts(
    utterances,
    speaker_samples,
    lang_code,
    original_audio_path,
    output_path,
    device,
    temperature=0.65,
):
    """Synthesize each utterance with XTTS-V2, time-stretch to fit, and mix.

    Dubbed timing (``dubbed_start`` / ``dubbed_end``) is stored on each
    utterance so ``write_srt`` can anchor subtitle cues to the actual speech.

    Parameters
    ----------
    utterances : list[dict]
        Translated utterances (must have ``translated_text`` set).
    speaker_samples : dict[str, str]
        ``speaker_id → wav_path`` from :func:`extract_speaker_samples`.
    lang_code : str
        ISO 639-1 language code recognised by XTTS-V2 (e.g. ``"fr"``).
    original_audio_path : str
        Path to original audio — used only to determine total track duration.
    output_path : str
        Destination path for the final dubbed WAV.
    device : str
        ``"cuda"`` or ``"cpu"``.
    temperature : float
        XTTS-V2 temperature (0.1 = consistent, 1.0 = expressive).
    """
    tts_model = _make_xtts_model(device)

    # Read original duration to size the silent timeline correctly.
    # The original audio is not mixed into the output — the dubbed track
    # contains only synthesized voices with silence between utterances.
    # The original language audio remains available as the primary video track.
    original_duration_ms = int(sf.info(original_audio_path).duration * 1000)

    dub_layer = AudioSegment.silent(
        duration=original_duration_ms, frame_rate=_XTTS_SR
    ).set_channels(1).set_sample_width(2)

    print("Synthesizing clips and building dub track...")
    for utt in tqdm(utterances):
        text = utt.get("translated_text", "").strip()
        if not text:
            utt["dubbed_start"] = utt["start"]
            utt["dubbed_end"] = utt["end"]
            continue

        speaker_wav = speaker_samples.get(utt["speaker"])
        if not speaker_wav:
            print(f"[WARNING] No voice sample for {utt['speaker']} — skipping utterance.")
            utt["dubbed_start"] = utt["start"]
            utt["dubbed_end"] = utt["end"]
            continue

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_file = tf.name
        try:
            synthesize_utterance_xtts(
                tts_model, text, speaker_wav, lang_code, temp_file, temperature
            )

            target_dur = utt["end"] - utt["start"]
            _time_stretch_to_fit(temp_file, target_dur)

            # XTTS-V2 outputs 24 kHz mono — no resampling needed.
            dub_segment = AudioSegment.from_wav(temp_file)

            start_ms = int(utt["start"] * 1000)
            dub_layer = dub_layer.overlay(dub_segment, position=start_ms)

            # Anchored subtitle timing: record actual clip duration after stretch.
            actual_dur_s = len(dub_segment) / 1000.0
            utt["dubbed_start"] = utt["start"]
            utt["dubbed_end"] = utt["start"] + actual_dur_s
        finally:
            os.remove(temp_file)

    dub_layer.export(output_path, format="wav")


# --------------------------------------------------------------------------- #
# 8. Subtitles, muxing & manifest
# --------------------------------------------------------------------------- #
_SRT_MAX_CHARS = 42
_SRT_MAX_LINES = 2
_SRT_MAX_S = 7.0
_SRT_MIN_S = 1.0


def _srt_wrap(text):
    """Word-wrap text to _SRT_MAX_CHARS per line, chunk into _SRT_MAX_LINES groups."""
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) <= _SRT_MAX_CHARS:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        return []
    return [
        "\n".join(lines[i: i + _SRT_MAX_LINES])
        for i in range(0, len(lines), _SRT_MAX_LINES)
    ]


def write_srt(utterances, srt_path):
    """Write Netflix/BBC-style SubRip subtitles anchored to dubbed audio timing.

    Each cue uses ``dubbed_start`` / ``dubbed_end`` when available (set by
    :func:`build_dub_track_xtts`) so subtitles appear exactly when the cloned
    voice speaks.  Falls back to the original ``start`` / ``end`` values if
    dubbed timing was not recorded (e.g. utterances that were skipped).
    """
    def format_ts(seconds):
        td = timedelta(seconds=seconds)
        total_sec = int(td.total_seconds())
        ms = int(td.microseconds / 1000)
        h, m, s = total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    cues = []
    for utt in utterances:
        text = (utt.get("translated_text") or utt.get("text", "")).strip()
        chunks = _srt_wrap(text)
        if not chunks:
            continue

        # Prefer dubbed timing; fall back to original transcription timing.
        utt_start = utt.get("dubbed_start", utt["start"])
        utt_end = utt.get("dubbed_end", utt["end"])

        utt_dur = max(utt_end - utt_start, _SRT_MIN_S * len(chunks))
        slice_dur = utt_dur / len(chunks)
        cue_dur = max(_SRT_MIN_S, min(_SRT_MAX_S, slice_dur))

        for j, chunk in enumerate(chunks):
            start = utt_start + j * slice_dur
            end = min(start + cue_dur, utt_end + _SRT_MIN_S)
            cues.append({"start": start, "end": max(end, start + _SRT_MIN_S), "text": chunk})

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, cue in enumerate(cues):
            f.write(f"{i + 1}\n")
            f.write(f"{format_ts(cue['start'])} --> {format_ts(cue['end'])}\n")
            f.write(f"{cue['text']}\n\n")


def mux_video_with_audio(video_path, audio_path, output_path):
    """Replace the audio track of ``video_path`` with ``audio_path``."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def save_manifest(data, path):
    """Persist a JSON manifest of speakers + utterances."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)


# --------------------------------------------------------------------------- #
# 9. CLI entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="AI Dubbing Pipeline — XTTS-V2 voice cloning (Colab Optimized)"
    )
    parser.add_argument("--input", required=True, help="Path to MP4 file or a Vimeo URL")
    parser.add_argument("--output_dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--glossary", help="Path to an OQLF glossary text file")
    parser.add_argument(
        "--tts_lang",
        default=_get_secret("TTS_LANG", "fr"),
        help="XTTS-V2 language code (ISO 639-1: fr, en, es, …)",
    )
    parser.add_argument(
        "--tts_temperature",
        type=float,
        default=float(_get_secret("TTS_TEMPERATURE", "0.65")),
        help="XTTS-V2 temperature: 0.1 (consistent) → 1.0 (expressive). Default: 0.65",
    )
    parser.add_argument(
        "--sample_min_duration",
        type=float,
        default=3.0,
        help="Minimum utterance duration (seconds) to use as a speaker clone reference. Default: 3.0",
    )
    parser.add_argument(
        "--max_utterance_duration",
        type=float,
        default=15.0,
        help=(
            "Maximum merged utterance length in seconds (default: 15). "
            "Prevents a single-speaker video or failed diarization from "
            "collapsing the entire audio into one utterance."
        ),
    )
    parser.add_argument("--model", default=_get_secret("WHISPER_MODEL", "small"), help="WhisperX model size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="WhisperX batch size")
    parser.add_argument("--min_speakers", type=int, default=None,
                        help="Force a minimum speaker count (speaker range forcing)")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Force a maximum speaker count (speaker range forcing)")
    parser.add_argument(
        "--context",
        default="",
        help=(
            "One-sentence description of the video content "
            "(e.g. 'a corporate presentation about financial software'). "
            "Helps Gemini choose appropriate register and terminology."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent stage checkpoint if available",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Checkpoint paths (written after each expensive stage).
    ckpt_segments = os.path.join(args.output_dir, "ckpt_segments.json")
    ckpt_utterances = os.path.join(args.output_dir, "ckpt_utterances.json")
    ckpt_translated = os.path.join(args.output_dir, "ckpt_translated.json")

    # ── Stage 1: media acquisition ──────────────────────────────────────────
    video_file = os.path.join(args.output_dir, "input_video.mp4")
    if args.input.startswith("http"):
        download_vimeo(args.input, video_file)
    else:
        video_file = args.input

    audio_wav = os.path.join(args.output_dir, "original_audio.wav")
    if not os.path.exists(audio_wav):
        extract_audio_to_wav(video_file, audio_wav)

    # ── Stages 2–6: ASR → merge → profile → samples → translate ────────────
    if args.resume and (data := load_checkpoint(ckpt_translated)):
        utterances = data["utterances"]
        print("[RESUME] Loaded translation checkpoint — skipping ASR + translation.")
    else:
        if args.resume and (data := load_checkpoint(ckpt_utterances)):
            utterances = data["utterances"]
            print("[RESUME] Loaded utterances checkpoint — skipping ASR.")
        else:
            if args.resume and (data := load_checkpoint(ckpt_segments)):
                segments = data["segments"]
                print("[RESUME] Loaded segments checkpoint — skipping transcription.")
            else:
                segments = transcribe_with_whisperx(
                    audio_wav,
                    device=device,
                    model_name=args.model,
                    batch_size=args.batch_size,
                    hf_token=_get_secret("HF_TOKEN"),
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                save_checkpoint({"segments": segments}, ckpt_segments)

            utterances = merge_segments_to_utterances(segments, max_duration=args.max_utterance_duration)
            save_checkpoint({"utterances": utterances}, ckpt_utterances)

        print("Translating utterances...")
        translate_utterances(utterances, args.glossary, args.context)
        save_checkpoint({"utterances": utterances}, ckpt_translated)

    # Build speaker profiles (metadata only — no longer drives voice selection).
    profiles = build_speaker_profiles(utterances, audio_wav)

    # ── Stage 5 (new): extract speaker clone samples ─────────────────────────
    print("Extracting speaker voice samples for cloning...")
    speaker_samples = extract_speaker_samples(
        utterances, audio_wav, args.output_dir, min_duration=args.sample_min_duration
    )

    # ── Stage 7: free WhisperX GPU memory, then synthesize with XTTS-V2 ─────
    _free_gpu_memory()

    dub_wav = os.path.join(args.output_dir, "dubbed_audio.wav")
    build_dub_track_xtts(
        utterances,
        speaker_samples,
        lang_code=args.tts_lang,
        original_audio_path=audio_wav,
        output_path=dub_wav,
        device=device,
        temperature=args.tts_temperature,
    )

    # ── Stages 8–10: subtitles, mux, manifest ────────────────────────────────
    write_srt(utterances, os.path.join(args.output_dir, "subtitles.srt"))

    mux_video_with_audio(
        video_file, dub_wav, os.path.join(args.output_dir, "final_dubbed_video.mp4")
    )

    save_manifest(
        {"profiles": profiles, "speaker_samples": speaker_samples, "utterances": utterances},
        os.path.join(args.output_dir, "manifest.json"),
    )

    out = os.path.abspath(args.output_dir)
    print("\nPipeline complete. Output files:")
    for fname in (
        "final_dubbed_video.mp4", "dubbed_audio.wav", "subtitles.srt",
        "manifest.json", os.path.join("speaker_samples", ""),
    ):
        fpath = os.path.join(out, fname)
        if os.path.isdir(fpath):
            files = os.listdir(fpath)
            print(f"  {fpath}  ({len(files)} speaker sample(s))")
        elif os.path.exists(fpath):
            size = f"{os.path.getsize(fpath) / 1024 / 1024:.1f} MB"
            print(f"  {fpath}  ({size})")
        else:
            print(f"  {fpath}  (missing)")


if __name__ == "__main__":
    main()
