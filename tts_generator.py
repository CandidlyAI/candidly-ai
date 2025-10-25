import os
import base64
import time
from scipy.io.wavfile import write as wav_write
import openai
import re
from pydub import AudioSegment

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

def b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def chunk_text(text: str, max_chars: int = 1000):
    """
    Split on sentence-ish boundaries and pack into ~max_chars chunks.
    """
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sents:
        if not s:
            continue
        # +1 for the space/newline we'll add
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip())
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    return chunks

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

    # Example: "[SPEAKER0] [EMOTION: friendly] [Rate:slow Pitch:low] Hello there..."
    user_line = f"[{speaker_tag}] {reply_text}" #[EMOTION: {em}]{prosody} 
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

def tts_generate_speech_chunked(
    reply_text: str,
    out_path_base: str,
    reference_path: str | None = "./ref-audio/en_woman.wav",
    reference_transcript: str | None = None,
    speaker_tag: str = "SPEAKER0",
    emotion: str = "neutral",
    speaking_rate: str | None = None,
    speaking_pitch: str | None = None,
    max_chars_per_chunk: int = 1000,
    gap_ms: int = 150
):
    """
    Calls your tts_generate_speech() on chunks, then concatenates them.
    Returns final audio path. Requires pydub for concatenation; if missing, saves sequential files.
    """
    text = (reply_text or "").strip()
    if not text:
        raise ValueError("Empty reply_text for TTS")

    chunks = chunk_text(text, max_chars=max_chars_per_chunk)

    # generate per-chunk files
    part_paths = []
    for i, chunk in enumerate(chunks, 1):
        base_i = f"{out_path_base}_part{i}"
        path_i = tts_generate_speech(
            reply_text=f"[{speaker_tag}] {chunk}",
            out_path=base_i,
            reference_path=reference_path,
            reference_transcript=reference_transcript,
            speaker_tag=speaker_tag,
            emotion=emotion,
            speaking_rate=speaking_rate,
            speaking_pitch=speaking_pitch,
        )
        part_paths.append(path_i)

    # try to stitch with pydub
    try:
        final = AudioSegment.silent(duration=0)
        spacer = AudioSegment.silent(duration=gap_ms)
        for p in part_paths:
            final = final + AudioSegment.from_file(p) + spacer

        # export with same extension as first part
        first_ext = os.path.splitext(part_paths[0])[-1].lstrip(".") or "wav"
        final_path = f"{out_path_base}.{first_ext}"
        final.export(final_path, format=first_ext)
        return final_path

    except Exception as e:
        # fall back: no concat; return list info
        print(f"[warn] pydub unavailable or concat failed: {e}", file=sys.stderr)
        print(f"[info] Parts saved separately: {part_paths}", file=sys.stderr)
        return part_paths[-1]  # last part path as a placeholder
