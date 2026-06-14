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
  6. Match each diarized speaker to a Google Cloud TTS voice + prosody.
  7. Translate every utterance to French (OQLF standard) with Gemini.
  8. Synthesize dubbed audio and overlay it on a ducked original track.
  9. Write an SRT subtitle file.
 10. Mux the dubbed audio back onto the original video.
 11. Save a JSON manifest describing speakers and utterances.

See docs/USAGE.md for CLI examples and docs/ARCHITECTURE.md for design notes.
"""

import os
import argparse
import json
import subprocess
import tempfile
from datetime import timedelta

import librosa
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from pydub import AudioSegment

import whisperx
from google import genai
from google.api_core.client_options import ClientOptions
from google.cloud import texttospeech

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
            pass  # secret not defined — fall through to env var
        except userdata.NotebookAccessError:
            print(
                f"[WARNING] Colab Secret '{name}' has notebook access disabled. "
                "Open the 🔑 Secrets panel and toggle the switch next to it."
            )
        except Exception:
            # No IPython kernel (e.g. subprocess) — fall through to os.getenv().
            pass
    except ImportError:
        pass  # not running in Colab
    return os.getenv(name, default)


def _make_tts_client():
    """Create a TextToSpeechClient authenticated via API key."""
    api_key = _get_secret("GOOGLE_TTS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_TTS_API_KEY is not set. Add it to Colab Secrets or your .env file."
        )
    return texttospeech.TextToSpeechClient(
        client_options=ClientOptions(api_key=api_key)
    )


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
        Force the diarizer to a known speaker count range ("speaker range
        forcing"). Useful when the automatic estimate over/under-segments.
    """
    print(f"Transcribing {audio_path} with WhisperX ({model_name})...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = whisperx.load_model(model_name, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)

    # Word-level alignment for accurate timestamps.
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False,
    )

    if hf_token:
        print("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )
        diarize_kwargs = {}
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers
        diarize_segments = diarize_model(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("HF_TOKEN missing. Skipping diarization. Assigning 'SPEAKER_00' to all.")
        for seg in result["segments"]:
            seg["speaker"] = "SPEAKER_00"

    # Ensure every segment carries a speaker key (diarization can miss some).
    for seg in result["segments"]:
        seg.setdefault("speaker", "SPEAKER_00")

    return result["segments"]


# --------------------------------------------------------------------------- #
# 3. Utterance merging
# --------------------------------------------------------------------------- #
def merge_segments_to_utterances(segments):
    """Merge consecutive same-speaker segments into longer utterances."""
    utterances = []
    if not segments:
        return utterances

    current = segments[0].copy()
    for next_seg in segments[1:]:
        if next_seg.get("speaker") == current.get("speaker"):
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
            pitches, magnitudes = librosa.piptrack(y=chunk, sr=sr)
            pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            profiles[spk]["pitches"].append(pitch)
            profiles[spk]["rms"].append(np.sqrt(np.mean(chunk ** 2)))
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
# 5. Voice & prosody assignment
# --------------------------------------------------------------------------- #
def assign_voices_and_prosody(profiles, tts_client, language_code="fr-CA"):
    """Map each diarized speaker to a Google TTS voice + speaking rate.

    Strategy: infer a gender preference from the pitch category (high → female,
    otherwise male), then round-robin through the available voices of that
    gender so distinct speakers get distinct voices.
    """
    voices = tts_client.list_voices(language_code=language_code).voices
    if not voices:
        raise RuntimeError(f"No TTS voices available for language '{language_code}'")
    assigned = {}

    for i, spk in enumerate(profiles.keys()):
        pitch_cat = profiles[spk]["pitch_category"]
        ssml_gender = (
            texttospeech.SsmlVoiceGender.FEMALE
            if pitch_cat == "high"
            else texttospeech.SsmlVoiceGender.MALE
        )

        filtered = [v for v in voices if v.ssml_gender == ssml_gender]
        if filtered:
            voice_name = filtered[i % len(filtered)].name
        else:
            voice_name = voices[i % len(voices)].name

        assigned[spk] = {
            "voice_name": voice_name,
            "pitch": 0.0,
            # Normalize rate of speech into TTS' valid [0.25, 4.0] range.
            "speaking_rate": max(0.25, min(4.0, profiles[spk]["rate_of_speech"] / 2.5)),
        }
    return assigned


# --------------------------------------------------------------------------- #
# 6. Translation (Gemini, OQLF French)
# --------------------------------------------------------------------------- #
def translate_batch_with_gemini(texts, glossary_text=""):
    """Translate a list of strings to OQLF French, preserving list order."""
    client = genai.Client(api_key=_get_secret("GEMINI_API_KEY"))
    prompt = (
        "Translate the following strings of video transcript into French "
        "(OQLF standard). Maintain the JSON list format and order exactly. "
        f"Use this glossary where applicable: {glossary_text}\n"
        f"Strings: {json.dumps(texts, ensure_ascii=False)}"
    )

    response = client.models.generate_content(
        model=_get_secret("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )
    result = json.loads(response.text)
    if not isinstance(result, list):
        raise ValueError(f"Gemini returned unexpected type {type(result).__name__}; expected a JSON list")
    return result


def translate_utterances(utterances, glossary_path=None):
    """Attach a ``translated_text`` field to every utterance."""
    glossary = ""
    if glossary_path and os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = f.read()

    texts = [u["text"] for u in utterances]
    translated_texts = translate_batch_with_gemini(texts, glossary)

    for i, utt in enumerate(utterances):
        utt["translated_text"] = translated_texts[i] if i < len(translated_texts) else utt["text"]
    return utterances


# --------------------------------------------------------------------------- #
# 7. Speech synthesis & dub track assembly
# --------------------------------------------------------------------------- #
def synthesize_line(client, text, voice_config, lang_code, output_file):
    """Synthesize a single line to a 16-bit PCM WAV file."""
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code, name=voice_config["voice_name"]
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        pitch=voice_config["pitch"],
        speaking_rate=voice_config["speaking_rate"],
    )
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    with open(output_file, "wb") as out:
        out.write(response.audio_content)


def build_dub_track(utterances, speaker_configs, lang_code, original_audio_path, output_path, tts_client):
    """Synthesize each utterance and overlay it on a ducked original track."""
    final_audio = AudioSegment.from_wav(original_audio_path)
    final_audio = final_audio - 15  # Duck the original audio by 15 dB.

    dub_layer = AudioSegment.silent(duration=len(final_audio))

    print("Synthesizing clips and building dub track...")
    for utt in tqdm(utterances):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            temp_file = tf.name
        try:
            synthesize_line(
                tts_client, utt["translated_text"], speaker_configs[utt["speaker"]],
                lang_code, temp_file,
            )
            dub_segment = AudioSegment.from_wav(temp_file)
            start_ms = utt["start"] * 1000
            dub_layer = dub_layer.overlay(dub_segment, position=start_ms)
        finally:
            os.remove(temp_file)

    combined = final_audio.overlay(dub_layer)
    combined.export(output_path, format="wav")


# --------------------------------------------------------------------------- #
# 8. Subtitles, muxing & manifest
# --------------------------------------------------------------------------- #
def write_srt(utterances, srt_path):
    """Write a SubRip (.srt) subtitle file from translated utterances."""
    def format_ts(seconds):
        td = timedelta(seconds=seconds)
        total_sec = int(td.total_seconds())
        ms = int(td.microseconds / 1000)
        h, m, s = total_sec // 3600, (total_sec % 3600) // 60, total_sec % 60
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, utt in enumerate(utterances):
            f.write(f"{i + 1}\n")
            f.write(f"{format_ts(utt['start'])} --> {format_ts(utt['end'])}\n")
            f.write(f"[{utt['speaker']}] {utt['translated_text']}\n\n")


def mux_video_with_audio(video_path, audio_path, output_path):
    """Replace the audio track of ``video_path`` with ``audio_path``."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def save_manifest(data, path):
    """Persist a JSON manifest of speakers + utterances."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 9. CLI entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="AI Dubbing Pipeline (Colab Optimized)"
    )
    parser.add_argument("--input", required=True, help="Path to MP4 file or a Vimeo URL")
    parser.add_argument("--output_dir", default="outputs", help="Directory for outputs")
    parser.add_argument("--glossary", help="Path to an OQLF glossary text file")
    parser.add_argument("--lang", default=_get_secret("DEFAULT_TTS_LANG", "fr-CA"), help="Target TTS language code")
    parser.add_argument("--model", default=_get_secret("WHISPER_MODEL", "small"), help="WhisperX model size")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=8, help="WhisperX batch size")
    parser.add_argument("--min_speakers", type=int, default=None,
                        help="Force a minimum speaker count (speaker range forcing)")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Force a maximum speaker count (speaker range forcing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_file = os.path.join(args.output_dir, "input_video.mp4")

    if args.input.startswith("http"):
        download_vimeo(args.input, video_file)
    else:
        video_file = args.input

    audio_wav = os.path.join(args.output_dir, "original_audio.wav")
    extract_audio_to_wav(video_file, audio_wav)

    segments = transcribe_with_whisperx(
        audio_wav,
        device=args.device,
        model_name=args.model,
        batch_size=args.batch_size,
        hf_token=_get_secret("HF_TOKEN"),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    utterances = merge_segments_to_utterances(segments)

    profiles = build_speaker_profiles(utterances, audio_wav)
    tts_client = _make_tts_client()
    speaker_configs = assign_voices_and_prosody(profiles, tts_client, args.lang)

    utterances = translate_utterances(utterances, args.glossary)

    dub_wav = os.path.join(args.output_dir, "dubbed_audio.wav")
    build_dub_track(utterances, speaker_configs, args.lang, audio_wav, dub_wav, tts_client)

    write_srt(utterances, os.path.join(args.output_dir, "subtitles.srt"))
    mux_video_with_audio(
        video_file, dub_wav, os.path.join(args.output_dir, "final_dubbed_video.mp4")
    )

    save_manifest(
        {"speakers": speaker_configs, "utterances": utterances},
        os.path.join(args.output_dir, "manifest.json"),
    )
    print("Pipeline Complete.")


if __name__ == "__main__":
    main()
