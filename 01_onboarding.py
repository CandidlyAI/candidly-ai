import os
import base64
import json
import sys
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import openai
from pathlib import Path

# ---- External TTS helper ----
from tts_generator import tts_generate_speech_chunked

# ---- OpenAI (Boson) client ----
client = openai.Client(
    api_key=os.getenv("BOSON_API_KEY"),
    base_url="https://hackathon.boson.ai/v1",
)

# ---- Audio + paths config ----
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION_S = 7
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REC_PATH = DATA_DIR / "input.wav"

# ---- TTS config (override with env if needed) ----
TTS_REF_AUDIO = os.getenv("TTS_REF_AUDIO", "./ref-audio/en_woman.wav")
TTS_REF_TRANSCRIPT = os.getenv("TTS_REF_TRANSCRIPT", None)
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "SPEAKER0")
TTS_EMOTION = os.getenv("TTS_EMOTION", "neutral")
TTS_RATE = os.getenv("TTS_RATE", None)
TTS_PITCH = os.getenv("TTS_PITCH", None)

# ---- System prompt for onboarding ----
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant. "
        "This is the onboarding process for an app that helps users practise difficult conversations with an AI stakeholder. "
        "Your task is to collect the user’s role, the AI’s role (e.g., user is a customer service agent; AI is a difficult customer), "
        "and, if available, a scenario they want to simulate (do not pressure the user if they don't have one). "
        'Respond strictly as JSON with the schema: {'
        '"role": "the user role", '
        '"ai_role": "the AI role", '
        '"is_done": <boolean>, '
        '"scenario": "scenario text or empty string", '
        '"next_prompt": "what to ask next if not done", '
        '"transcription": "transcription of what the user said"'
        '}. Output nothing else.'
    ),
}

# ---------------- Utilities ----------------
def record_mic_to_wav(path: Path, duration_s: int = DURATION_S, sr: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"🎙️ Recording {duration_s}s of audio from mic...")
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=channels, dtype="int16")
    sd.wait()
    wav_write(str(path), sr, audio)
    print(f"💾 Saved mic recording to {path}")

_speak_counter = 0
def speak_text(reply_text: str) -> str | None:
    """Generate + (attempt to) play TTS for the assistant’s reply. Returns audio path or None."""
    global _speak_counter
    try:
        tts_dir = DATA_DIR / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        _speak_counter += 1
        out_base = str(tts_dir / f"turn_{_speak_counter}")
        audio_path = tts_generate_speech_chunked(
            reply_text=reply_text,
            out_path_base=out_base,
            reference_path=TTS_REF_AUDIO,
            reference_transcript=TTS_REF_TRANSCRIPT,
            speaker_tag=TTS_SPEAKER,
            emotion=TTS_EMOTION,
            speaking_rate=TTS_RATE,
            speaking_pitch=TTS_PITCH,
            max_chars_per_chunk=1000,
            gap_ms=150,
        )
        # Try playback via pydub if available
        try:
            from pydub import AudioSegment
            from pydub.playback import play as pydub_play
            seg = AudioSegment.from_file(audio_path)
            pydub_play(seg)
        except Exception as e:
            print(f"[warn] TTS generated but playback failed: {e}. File at: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"[warn] TTS generation failed: {e}")
        return None

def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "done"}
    return False

# ---------------- Managed onboarding loop (inside function) ----------------
def run_onboarding(
    *,
    max_turns: int = 10,
    record_seconds: int = DURATION_S,
    print_result: bool | None = None,
) -> dict:
    """
    Managed onboarding loop. Records audio → sends to model → expects strict JSON → repeats
    until is_done is true or max_turns reached. Returns the final JSON dict.

    Args:
        max_turns: safety cap on rounds.
        record_seconds: mic recording length per turn.
        print_result: if True/False forces printing result; if None, respects ONBOARDING_SUPPRESS_PRINT env.
    """
    messages = [SYSTEM_PROMPT]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # printing control
    if print_result is None:
        suppress = os.getenv("ONBOARDING_SUPPRESS_PRINT")
        print_result = not _to_bool(suppress)

    last_valid = None

    for turn in range(1, max_turns + 1):
        print(f"\n—— Onboarding turn {turn}/{max_turns} ——")
        # 1) Record mic
        record_mic_to_wav(REC_PATH, duration_s=record_seconds, sr=SAMPLE_RATE, channels=CHANNELS)

        # 2) Encode audio and send to model
        with open(REC_PATH, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
        messages.append({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": encoded_string, "format": "wav"}
            }]
        })

        resp = client.chat.completions.create(
            model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        # We keep only one audio object in history for this model
        messages.pop()

        response_text = resp.choices[0].message.content
        print("🧠 Model response (raw):\n", response_text)

        # 3) Parse + manage loop
        try:
            parsed = json.loads(response_text)
            last_valid = parsed
        except json.JSONDecodeError:
            print("⚠️ Model response not valid JSON, retrying this turn...")
            continue

        is_done = _to_bool(parsed.get("is_done"))

        if is_done:
            if print_result:
                print("\n✅ All information collected successfully!")
                print(json.dumps(parsed, indent=2))
            return parsed  # managed loop: return immediately on completion

        # Not done → speak next prompt and seed next round with user+assistant text (not audio)
        next_prompt = parsed.get("next_prompt", "Please continue.")
        transcription = parsed.get("transcription", "")

        print(f"\n🤖 Assistant: {next_prompt}\n")
        try:
            speak_text(next_prompt)
        except Exception as _e:
            print(f"[warn] speak_text failed: {_e}")

        # Keep a light text history for coherence
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": transcription}]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": next_prompt}]
        })

    # If we fall out due to turn cap, return last valid (if any) with is_done coerced
    if last_valid is not None:
        last_valid["is_done"] = _to_bool(last_valid.get("is_done"))
        if print_result:
            print("\n⚠️ Reached max_turns without is_done=true; returning last valid response:")
            print(json.dumps(last_valid, indent=2))
        return last_valid

    raise RuntimeError("Onboarding failed: no valid JSON was produced.")

# ---------------- CLI entrypoint ----------------
if __name__ == "__main__":
    result = run_onboarding()
    # Optional: write for debugging when invoked directly
    out_path = DATA_DIR / "completes.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] Wrote completes JSON to: {out_path}")
