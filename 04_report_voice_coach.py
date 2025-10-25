import os, json, time, base64
from pathlib import Path
from typing import List, Tuple

import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import simpleaudio as sa
import openai

"""
Report QA Voice Coach
---------------------
You (the user) ASK a question (by voice). The AI answers USING ONLY the attached report
(evaluate_sample_input_report.md) as grounding, then SPEAKS the answer back.

Flow per turn:
 1) Record mic (you ask a question)
 2) ASR -> transcript
 3) QA over the report -> answer text (grounded; cites sections when possible)
 4) TTS -> spoken answer + on-screen text

Dependencies (pip):
  pip install sounddevice simpleaudio scipy openai

Env:
  export BOSON_API_KEY=...   # Boson (OpenAI-compatible) key

Run:
  python report_voice_coach.py
"""

# ---------------- Config ----------------
SAMPLE_RATE = 16_000
CHANNELS    = 1
REC_PATH    = "user_question.wav"
ANSWER_WAV  = "answer.wav"
REPORT_PATH = "./evaluate_sample_input_report.md"  # adjust if needed
MAX_TURNS   = 20
LISTEN_SEC  = 12

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
if not BOSON_API_KEY:
    raise RuntimeError("Please set BOSON_API_KEY environment variable.")

client = openai.Client(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

# ---------------- Utils ----------------

def record_mic_to_wav(path: str, duration_s: int = 10, sr: int = SAMPLE_RATE, channels: int = CHANNELS):
    print(f"🎙️  Listening for up to {duration_s}s... (Speak now)")
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=channels, dtype="int16")
    sd.wait()
    wav_write(path, sr, audio)
    print(f"Saved mic recording to {path}")


def play_wav(path: str):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"(Optional) playback failed: {e}")


def encode_audio_to_base64(file_path: str) -> Tuple[str, str]:
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = file_path.split(".")[-1].lower()
    return b64, ext


def extract_text_message(resp) -> str:
    msg = resp.choices[0].message
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                txt = block.get("text") or block.get("content") or ""
                if txt:
                    return txt.strip()
        joined = " ".join(str(b.get("text") or b.get("content") or "") for b in content if isinstance(b, dict)).strip()
        if joined:
            return joined
    txt = getattr(msg, "text", None)
    return (txt or "").strip()


def save_audio_from_response(resp, out_path_base: str = ANSWER_WAV) -> str:
    audio_obj = resp.choices[0].message.audio if hasattr(resp.choices[0].message, "audio") else None
    audio_b64 = None
    fmt = "wav"
    if isinstance(audio_obj, dict):
        audio_b64 = audio_obj.get("data")
        fmt = audio_obj.get("format", "wav") or "wav"

    if not audio_b64:
        content = getattr(resp.choices[0].message, "content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in ("output_audio", "audio", "audio_data"):
                    audio = block.get("audio") or block
                    audio_b64 = audio.get("data") if isinstance(audio, dict) else block.get("data")
                    fmt = (audio.get("format") if isinstance(audio, dict) else block.get("format")) or "wav"
                    if audio_b64:
                        break

    if not audio_b64:
        raise RuntimeError("No audio payload in TTS response")

    out_path = f"{Path(out_path_base).with_suffix('')}.{fmt.lower()}"
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
    return out_path


# ---------------- I/O ----------------

def load_report_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {p}")
    text = p.read_text(encoding="utf-8", errors="ignore")
    # Keep it reasonably sized (model context safety)
    return text[:60_000]


# ---------------- Core: Report-grounded QA ----------------

class ReportQACoach:
    def __init__(self, report_text: str):
        self.report_text = report_text
        self.history: List[dict] = []  # [{'role': 'user'/'assistant', 'content': str}, ...]

    def _system_prompt(self) -> str:
        return (
            "You are a helpful career coach. Answer the user's question USING ONLY the provided report. "
            "Be concise, actionable, and supportive. If the report lacks the info, say so and offer a next-best step."
            "Report (verbatim):<REPORT>" + self.report_text + "</REPORT>"
            "Rules:"
            "- Do not invent facts."
            "- Prefer concrete pointers from the report (sections, bullets, quotes)."
            "- Keep answers under 150 words unless the user asks for more."
        )

    def _messages(self, user_question_text: str) -> list:
        msgs = [{"role": "system", "content": self._system_prompt()}]
        # include brief conversation memory
        msgs.extend(self.history[-10:])
        msgs.append({"role": "user", "content": user_question_text})
        return msgs

    def answer(self, user_question_text: str) -> str:
        resp = client.chat.completions.create(
            model="higgs-audio-understanding-Hackathon",  # capable text model; swap if you prefer another
            messages=self._messages(user_question_text),
            max_completion_tokens=256,
            temperature=0.2,
        )
        ans = extract_text_message(resp)
        if not ans:
            ans = "I couldn't find that in the report. Could you rephrase or ask something more specific?"
        self.history.append({"role": "user", "content": user_question_text})
        self.history.append({"role": "assistant", "content": ans})
        return ans.strip()


# ---------------- TTS ----------------
# Prefer using user's custom /mnt/data/tts_generator.py if present
import importlib.util, sys, inspect

_TTS_MOD = None


def _load_tts_impl(mod_path: str = "./tts_generator.py"):
    global _TTS_MOD
    try:
        p = Path(mod_path)
        if not p.exists():
            return None
        spec = importlib.util.spec_from_file_location("tts_generator", str(p))
        if not spec or not spec.loader:
            return None
        m = importlib.util.module_from_spec(spec)
        sys.modules["tts_generator"] = m
        spec.loader.exec_module(m)
        _TTS_MOD = m
        print(f"✅ Loaded custom TTS from {mod_path}")
        return m
    except Exception as e:
        print(f"[warning] Failed to load custom TTS from {mod_path}: {e}")
        return None


def _call_if_exists(mod, fname: str, **kwargs):
    """Call mod.fname(**kwargs) if present, trimming kwargs to match signature."""
    if not hasattr(mod, fname):
        return None
    fn = getattr(mod, fname)
    if not callable(fn):
        return None
    sig = inspect.signature(fn)
    call_kwargs = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            call_kwargs[k] = v
    return fn(**call_kwargs)


# Attempt to load user's generator at import time
_load_tts_impl()


def tts_answer(text: str, out_path_base: str = ANSWER_WAV, speaker_tag: str = "SPEAKER0", emotion: str = "friendly") -> str:
    """
    Use user's custom tts_generator.py if available.
    Expected to return a path to an audio file. We try common function names/signatures:
      - tts_generate_speech_chunked(reply_text, out_path_base, reference_path, reference_transcript, speaker_tag, emotion, speaking_rate, speaking_pitch, max_chars_per_chunk, gap_ms)
      - tts_generate_speech(reply_text, out_path)
      - generate_tts(text, out_path)
      - synthesize(text, out_path)
    Falls back to built-in Boson chat-completions TTS if custom module not present or fails.
    """
    # 1) Try custom module
    if _TTS_MOD is not None:
        try:
            # Try the most feature-rich first
            path = _call_if_exists(
                _TTS_MOD,
                "tts_generate_speech_chunked",
                reply_text=text,
                out_path_base=Path(out_path_base).with_suffix("").as_posix(),
                reference_path=None,
                reference_transcript=None,
                speaker_tag=speaker_tag,
                emotion=emotion,
                speaking_rate=None,
                speaking_pitch=None,
                max_chars_per_chunk=1000,
                gap_ms=150,
            )
            if isinstance(path, str) and Path(path).exists():
                return path

            # Simpler names
            for fname in [
                "tts_generate_speech",
                "generate_tts",
                "synthesize",
                "tts",
            ]:
                path = _call_if_exists(
                    _TTS_MOD, fname,
                    reply_text=text, text=text,
                    out_path=out_path_base,
                    out_path_base=Path(out_path_base).with_suffix("").as_posix(),
                    speaker_tag=speaker_tag,
                    emotion=emotion,
                )
                if isinstance(path, str) and Path(path).exists():
                    return path

            print("[warning] Custom TTS module loaded but no recognized function succeeded; falling back to built-in TTS.")
        except Exception as e:
            print(f"[warning] Custom TTS failed: {e}; falling back to built-in TTS.")

    # 2) Fallback: built-in Boson TTS via chat.completions
    system = (
        "You convert text to natural speech."
        "If a [SPEAKER*] tag is present, use that voice and DO NOT read the tag."
        "Use a calm, clear tone suitable for professional coaching."
    )
    short_text = text.strip()[:1600]
    user_line  = f"[{speaker_tag}] {short_text}"

    resp = client.chat.completions.create(
        model="higgs-audio-generation-Hackathon",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_line},
        ],
        modalities=["text", "audio"],
        extra_body={"audio": {"voice": speaker_tag, "format": "wav"}},
        max_completion_tokens=1024,
        temperature=0.8,
        top_p=0.95,
        stream=False,
    )

    out_path = save_audio_from_response(resp, out_path_base)
    return out_path


# ---------------- ASR ----------------


def transcribe_audio(path: str) -> str:
    audio_b64, fmt = encode_audio_to_base64(path)
    resp = client.chat.completions.create(
        model="higgs-audio-understanding-Hackathon",
        messages=[
            {"role": "system", "content": "Transcribe this audio verbatim. Return only the words."},
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}}]},
        ],
        max_completion_tokens=512,
        temperature=0.0,
    )
    return extract_text_message(resp)


# ---------------- Main Loop ----------------

def chat_loop(report_path: str = REPORT_PATH, turns: int = MAX_TURNS, listen_seconds: int = LISTEN_SEC):
    report_text = load_report_text(report_path)
    qa = ReportQACoach(report_text)

    print("🔊 Report-grounded Voice Coach ready.Say 'stop' or 'that's all' to end.")

    for i in range(turns):
        # 1) Listen to the user's question
        record_mic_to_wav(REC_PATH, duration_s=listen_seconds)
        user_q = transcribe_audio(REC_PATH)
        print(f"[You] {user_q}")
        if user_q.strip().lower() in {"stop", "that's all", "that is all", "quit", "exit"}:
            print("👋 Ending session.")
            break

        # 2) Answer using the report
        answer = qa.answer(user_q)
        print(f"[Coach] {answer}")

        # 3) Speak the answer
        wav_path = tts_answer(answer, out_path_base=f"answer_{i+1}.wav", emotion="friendly")
        if Path(wav_path).exists() and wav_path.lower().endswith(".wav"):
            play_wav(wav_path)

if __name__ == "__main__":
    print("Launching report-grounded QA voice coach…")
    print(f"Using report at: {REPORT_PATH}")
    chat_loop(REPORT_PATH, turns=MAX_TURNS, listen_seconds=LISTEN_SEC)