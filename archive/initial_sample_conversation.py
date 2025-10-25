import os
import io
import base64
import wave
import time
import json
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write as wav_write
import simpleaudio as sa
import openai

# ---------------- Config ----------------
SAMPLE_RATE = 16000     # 16 kHz mono is common for ASR
CHANNELS    = 1
DURATION_S  = 10        # seconds to record; change as you like
REC_PATH    = "input.wav"
TTS_OUT     = "reply.wav"

BOSON_API_KEY = os.getenv("BOSON_API_KEY")

if not BOSON_API_KEY:
    raise RuntimeError("Please set BOSON_API_KEY environment variable.")

client = openai.Client(
    api_key=BOSON_API_KEY,
    base_url="https://hackathon.boson.ai/v1"
)

# --------------- Utils ------------------

import base64, os

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def record_mic_to_wav(path: str, duration_s: int = DURATION_S, sr: int = SAMPLE_RATE, channels: int = CHANNELS):
    print(f"Recording {duration_s}s of audio from mic...")
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=channels, dtype="int16")
    sd.wait()
    wav_write(path, sr, audio)
    print(f"Saved mic recording to {path}")

def encode_audio_to_base64(file_path: str) -> tuple[str, str]:
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = file_path.split(".")[-1].lower()
    return b64, ext

def save_b64_audio_to_wav(b64_data: str, out_path: str):
    # If model returns MP3/other, convert headerless if needed. Here we assume WAV or raw PCM container from API.
    # If "format" is mp3/ogg etc., just write bytes to file with that extension instead.
    audio_bytes = base64.b64decode(b64_data)
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    print(f"Saved generated speech to {out_path}")

def play_wav(path: str):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"(Optional) playback failed: {e}")

# Robust extractors for Boson-style responses
def extract_text_message(resp) -> str:
    """
    Tries multiple shapes:
    - choices[0].message.content as string
    - choices[0].message.content as list of blocks [{'type': 'text', 'text': '...'}]
    """
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        # Find first text block
        for block in content:
            # some SDKs use {"type":"text","text":"..."}
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                txt = block.get("text") or block.get("content") or ""
                if txt:
                    return txt.strip()
        # Fallback: join anything stringy
        joined = " ".join(
            str(b.get("text") or b.get("content") or "")
            for b in content
            if isinstance(b, dict)
        ).strip()
        if joined:
            return joined

    # Some SDKs put it in message["text"]
    txt = getattr(msg, "text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    # Last resort: raw print
    return json.dumps(resp.model_dump())

def extract_audio_b64_and_format(resp) -> tuple[str | None, str | None]:
    """
    Tries common shapes for audio generation responses:
    - choices[0].message.content is a list with a block: {"type":"output_audio","audio":{"data": "...","format":"wav"}}
    - choices[0].message.audio: {"data":"...","format":"wav"}
    """
    try:
        choice = resp.choices[0]
        msg = choice.message

        # Path A: message.audio
        audio_obj = getattr(msg, "audio", None)
        if isinstance(audio_obj, dict):
            data = audio_obj.get("data")
            fmt  = audio_obj.get("format", "wav")
            if data:
                return data, fmt

        # Path B: content list blocks
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") in ("output_audio", "audio", "audio_data"):
                        audio = block.get("audio") or block  # sometimes nested, sometimes flat
                        if isinstance(audio, dict):
                            data = audio.get("data")
                            fmt  = audio.get("format", "wav")
                            if data:
                                return data, fmt
                        # flat shape:
                        data = block.get("data")
                        fmt  = block.get("format", "wav")
                        if data:
                            return data, fmt
    except Exception:
        pass

    return None, None

# ------------- Pipeline -----------------
def transcribe_audio(path: str) -> str:
    audio_b64, fmt = encode_audio_to_base64(path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe this audio verbatim."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": fmt,
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=512,
        temperature=0.0,
    )
    transcript = extract_text_message(resp)
    print("\n--- Transcript ---")
    print(transcript)
    return transcript

def generate_text_reply_from_audio(path: str) -> str:
    audio_b64, fmt = encode_audio_to_base64(path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            # {"role": "system", "content": "You are a hiring manager at an AI startup conducting an interview. Respond succinctly and professionally."},
            {"role": "system", "content": "You are a helpful assistant. You are supposed to pretend that you are a difficult customer complaining about an order that didn't arrive. Generate the customer response, don't output anything else, just the text. You are talking to a customer service manager in the store"},

            # {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "input_audio",
            #             "input_audio": {
            #                 "data": audio_b64,
            #                 "format": fmt,
            #             },
            #         },
            #     ],
            # },
            # {"role": "user", "content": "Please reply to the candidate based on what they said in the audio."},
            {"role": "user", "content": "Generate the first complaint, do not output anything else."},
        ],
        max_completion_tokens=512,
        temperature=0.2,
    )
    reply_text = extract_text_message(resp)
    print("\n--- Manager Reply (text) ---")
    print(reply_text)
    return reply_text

import time
from openai import APIConnectionError, InternalServerError, RateLimitError

# def tts_generate_speech(
#     reply_text: str,
#     out_path: str = TTS_OUT,
#     reference_path: str | None = "./ref-audio/en_woman.wav",
#     reference_transcript: str | None = (
#         "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic. "
#         "And the bond between you and your wand should only grow stronger. Do not be surprised at your new "
#         "wand's ability to perceive your intentions - particularly in a moment of need."
#     ),
#     speaker_tag: str = "SPEAKER0",
# ):
#     """
#     Generate TTS audio using `higgs-audio-generation-Hackathon`, optionally voice-cloned
#     from `reference_path` + `reference_transcript`. Retries on transient 5xx/timeout.

#     Requires:
#       - a global `client = OpenAI(api_key=..., base_url="https://hackathon.boson.ai/v1")`
#       - a helper `b64(path)` that returns base64 string
#     """
#     # Lazy imports to avoid hard dependency if caller doesn't need retries
#     try:
#         from openai import APIConnectionError, InternalServerError, RateLimitError
#     except Exception:
#         APIConnectionError = InternalServerError = RateLimitError = Exception  # fallback

#     # Build the message list following Boson sample schema
#     system = (
#         "You are an AI assistant designed to convert text into speech.\n"
#         "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
#         "If no speaker tag is present, select a suitable voice on your own.\n\n"
#         "<|scene_desc_start|>\nAudio is recorded from a outdoor location.\n<|scene_desc_end|>"
#     )

#     messages = [{"role": "system", "content": system}]

#     # If we have a reference transcript, include it as context for style/voice
#     # if reference_transcript:
#     #     messages.append({"role": "user", "content": reference_transcript})

#     # If we have a reference audio file, attach it as assistant content block (per sample)
#     if reference_path and os.path.exists(reference_path):
#         messages.append({
#             "role": "assistant",
#             "content": [{
#                 "type": "input_audio",
#                 "input_audio": {"data": b64(reference_path), "format": "wav"}
#             }],
#         })

#     # User content with speaker tag (instruct model to use cloned voice)
#     short_text = reply_text.strip()
#     # keep it reasonably short to avoid timeouts
#     if len(short_text) > 1200:
#         short_text = short_text[:1200]
#     messages.append({"role": "user", "content": f"[{speaker_tag}] {short_text}"})

#     backoff = 2.0
#     for attempt in range(5):
#         try:
#             resp = client.chat.completions.create(
#                 model="higgs-audio-generation-Hackathon",
#                 messages=messages,
#                 modalities=["text", "audio"],
#                 # Generous budget; provider may ignore for audio payloads
#                 max_completion_tokens=4096,
#                 temperature=0.8,
#                 top_p=0.95,
#                 stream=False,
#                 stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
#                 extra_body={"top_k": 50},
#             )

#             # Primary extraction path (per sample)
#             audio_obj = resp.choices[0].message.audio
#             audio_b64 = getattr(audio_obj, "data", None) if audio_obj else None
#             fmt = getattr(audio_obj, "format", "wav") if audio_obj else "wav"

#             # Fallbacks in case payload shape differs
#             if not audio_b64:
#                 # Try content blocks
#                 content = getattr(resp.choices[0].message, "content", None)
#                 if isinstance(content, list):
#                     for block in content:
#                         if isinstance(block, dict) and block.get("type") in ("output_audio", "audio"):
#                             audio_dict = block.get("audio") or block
#                             audio_b64 = audio_dict.get("data")
#                             fmt = audio_dict.get("format", "wav")
#                             if audio_b64:
#                                 break

#             if not audio_b64:
#                 raise RuntimeError("No audio payload in TTS response")

#             # Ensure extension matches the returned format
#             ext = (fmt or "wav").lower()
#             if not out_path.lower().endswith(f".{ext}"):
#                 out_path = f"{os.path.splitext(out_path)[0]}.{ext}"

#             with open(out_path, "wb") as f:
#                 f.write(base64.b64decode(audio_b64))

#             return out_path

#         except (InternalServerError, APIConnectionError, RateLimitError) as e:
#             if attempt == 4:
#                 raise
#             time.sleep(backoff)
#             backoff *= 2.0

def tts_generate_speech(
    reply_text: str,
    out_path: str = TTS_OUT,
    reference_path: str | None = "./ref-audio/en_woman.wav",
    reference_transcript: str | None = None,
    speaker_tag: str = "SPEAKER0",
    emotion: str = "neutral",           # <-- NEW: emotion control
    speaking_rate: str | None = None,   # e.g. "slow", "fast" (optional)
    speaking_pitch: str | None = None,  # e.g. "low", "high" (optional)
):
    """
    Generate TTS audio using `higgs-audio-generation-Hackathon`, optionally voice-cloned
    and conditioned on `emotion` (prompt tag). Retries on transient 5xx/timeout.

    Requires:
      - global: client = OpenAI(api_key=..., base_url="https://hackathon.boson.ai/v1")
      - helper: b64(path) -> base64 string  (or swap to your encode_audio_to_base64)
    """
    # Safe import of OpenAI error types
    try:
        from openai import APIConnectionError, InternalServerError, RateLimitError
    except Exception:
        APIConnectionError = InternalServerError = RateLimitError = Exception

    # Normalize/whitelist emotion (optional)
    allowed = {
        "neutral","friendly","confident","empathetic","calm","excited",
        "serious","encouraging","apologetic","assertive","enthusiastic","warm", "angry"
    }
    em = (emotion or "neutral").strip().lower()
    if em not in allowed:
        em = "neutral"

    # Build prosody hints
    prosody_bits = []
    if speaking_rate:
        prosody_bits.append(f"Rate:{speaking_rate}")
    if speaking_pitch:
        prosody_bits.append(f"Pitch:{speaking_pitch}")
    prosody = f" [{' '.join(prosody_bits)}]" if prosody_bits else ""

    # System prompt teaches the model how to use tags
    system = (
        "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag; use that voice.\n"
        # "If the user's message includes an [EMOTION:*] tag, do not read out the tag; use a tone that reflects this emotion.\n"
        "If no tags are present, pick a suitable voice and neutral tone.\n\n"
        "You are extremely sad and disappointed.\n\n"
        "<|scene_desc_start|>\nAudio is recorded from an outdoor location.\n<|scene_desc_end|>"
    )

    messages = [{"role": "system", "content": system}]

    # (Optional) include a style reference transcript if you like
    if reference_transcript:
        messages.append({"role": "user", "content": reference_transcript})

    # (Optional) attach reference audio for cloning
    if reference_path and os.path.exists(reference_path):
        messages.append({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": b64(reference_path), "format": "wav"}  # or encode_audio_to_base64
            }],
        })

    # User content with speaker + emotion + (optional) prosody hints
    short_text = (reply_text or "").strip()
    if len(short_text) > 1200:
        short_text = short_text[:1200]  # keep synthesis snappy

    # Example: "[SPEAKER0] [EMOTION: friendly] [Rate:slow Pitch:low] Hello there..."
    user_line = f"[{speaker_tag}] {short_text}" #[EMOTION: {em}]{prosody} 
    messages.append({"role": "user", "content": user_line})

    backoff = 2.0
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model="higgs-audio-generation-Hackathon",
                messages=messages,
                modalities=["text", "audio"],
                max_completion_tokens=4096,
                temperature=0.8,
                top_p=0.95,
                stream=False,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                extra_body={"top_k": 50},
            )

            # Primary extraction (per Boson sample)
            audio_obj = resp.choices[0].message.audio
            audio_b64 = getattr(audio_obj, "data", None) if audio_obj else None
            fmt = getattr(audio_obj, "format", "wav") if audio_obj else "wav"

            # Fallback: search content blocks
            if not audio_b64:
                content = getattr(resp.choices[0].message, "content", None)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in ("output_audio", "audio"):
                            audio_dict = block.get("audio") or block
                            audio_b64 = audio_dict.get("data")
                            fmt = audio_dict.get("format", "wav")
                            if audio_b64:
                                break

            if not audio_b64:
                raise RuntimeError("No audio payload in TTS response")

            # Save with correct extension
            ext = (fmt or "wav").lower()
            base, _ = os.path.splitext(out_path)
            out_path_final = f"{base}.{ext}"
            with open(out_path_final, "wb") as f:
                f.write(base64.b64decode(audio_b64))

            return out_path_final

        except (InternalServerError, APIConnectionError, RateLimitError) as e:
            if attempt == 4:
                raise
            time.sleep(backoff)
            backoff *= 2.0


def main():
    # EDIT HERE BASED ON HOW YOU WANT TO RUN THE PIPELINE
    # 1) Record from mic
    # record_mic_to_wav(REC_PATH, DURATION_S, SAMPLE_RATE, CHANNELS)
    # REC_PATH = "./output.wav"
    # 2) Transcribe
    # transcript = transcribe_audio(REC_PATH)

    # 3) Generate manager-style text reply from the same audio (you could also use transcript instead)
    reply_text = generate_text_reply_from_audio(REC_PATH)

    # Alternative (if you prefer to base the reply on the transcript only):
    # reply_text = client.chat.completions.create(
    #     model="higgs-audio-understanding-Hackathon",
    #     messages=[
    #         {"role": "system", "content": "You are a hiring manager at an AI startup. Respond succinctly and professionally."},
    #         {"role": "user", "content": f"Candidate said: {transcript}\n\nPlease reply to the candidate."}
    #     ],
    #     max_completion_tokens=512,
    #     temperature=0.2
    # )
    # reply_text = extract_text_message(reply_text)

    # 4) TTS: turn the reply into speech
    # out_audio_path = tts_generate_speech(reply_text, TTS_OUT)

    out_audio_path = tts_generate_speech(
        reply_text,
        TTS_OUT,
        emotion="angry",        # try: confident, empathetic, calm, serious, excited...
        speaking_rate="slow",      # optional
        speaking_pitch="low",      # optional
    )

    print("\nPlaying generated reply...")
    if out_audio_path and os.path.exists(out_audio_path):
        if out_audio_path.lower().endswith(".wav"):
            play_wav(out_audio_path)          # simpleaudio expects WAV
        else:
            print(f"Saved audio at {out_audio_path} (non-WAV format; not playing).")
    else:
        print("No audio file produced.")


if __name__ == "__main__":
    main()
