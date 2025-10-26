
import os
import base64
import json
import sys
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import openai
from pathlib import Path
from global_var import state, OnboardingInfo

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

def speak_text(reply_text: str) -> str | None:
    """Generate + (attempt to) play TTS for the assistant’s reply. Returns audio path or None."""
    try:
        state["onboarding"].speak_counter += 1
        turn = f"turn_{state["onboarding"].speak_counter}"
        tts_dir = DATA_DIR / "tts"
        tts_dir.mkdir(parents=True, exist_ok=True)
        out_base = str(tts_dir / turn)
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
        # try:
        #     from pydub import AudioSegment
        #     from pydub.playback import play as pydub_play
        #     seg = AudioSegment.from_file(audio_path)
        #     pydub_play(seg)
        # except Exception as e:
        #     print(f"[warn] TTS generated but playback failed: {e}. File at: {audio_path}")
        return str(f"tts/{turn}.wav")
    except Exception as e:
        print(f"[warn] TTS generation failed: {e}")
        return None

def onboard(message: str):
    with open(REC_PATH, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")

        print(state["onboarding"].messages)

        state["onboarding"].messages.append({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": encoded_string, "format": "wav"}
            }]
        })

        resp = client.chat.completions.create(
            model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
            messages=state["onboarding"].messages,
            max_tokens=2048,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        state["onboarding"].messages.pop()

        response_text = resp.choices[0].message.content

        try:
            parsed = json.loads(response_text)
            last_valid = parsed
        except json.JSONDecodeError:
            print("⚠️ Model response not valid JSON, retrying this turn...")


        next_prompt = parsed.get("next_prompt", "Please continue.")
        transcription = parsed.get("transcription", "")

        state["onboarding"].messages.append({
            "role": "user",
            "content": [{"type": "text", "text": transcription}]
        })
        state["onboarding"].messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": next_prompt}]
        })

        is_done = parsed.get("is_done")

        if is_done:
            state["onboarding"].onboarding_info = OnboardingInfo( parsed.get("ai_role", ""),  parsed.get("role", ""),  parsed.get("scenario", ""))
            return is_done

        audio_path = speak_text(next_prompt)

        state["audio_ready"] = True
        state["last_generated"] = audio_path

        return is_done

