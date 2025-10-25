#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_career_conversation.py
"""

import argparse
import os
import sys
import json
from datetime import datetime
import time
import pathlib
import re

# IMPORTANT: import the whole module so we can set its global `client`
import tts_generator as isc  # provides isc.tts_generate_speech, isc.b64

# OpenAI-compatible client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---------- Paths for default outputs ----------
def default_tts_txt_path(input_path: str) -> str:
    p = pathlib.Path(input_path)
    return str(p.with_name(p.stem + "_tts.txt"))

def default_tts_audio_path(input_path: str) -> str:
    p = pathlib.Path(input_path)
    return str(p.with_name(p.stem + "_tts"))


# ---------- Load transcript ----------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Prompts (EVAL + SCRIPT) ----------
SYSTEM_PROMPT = """You are an experienced leadership and career coach.
Your task is to EVALUATE the user's responses in a two-person practice script
where the user is the new manager and the AI is a team member who may soon leave.

Be candid but respectful. Focus on:
1) Managerial effectiveness (clarity, empathy, curiosity, boundaries, next steps).
2) Psychological safety (non-defensive listening, validation, avoiding pressure).
3) Negotiation & retention signals (identifying growth/comp pay/design blockers; proposing options).
4) Tone & delivery (emotional valence, vocal/word-choice cues; include at least 1 sentence on tone).
5) Actionability (give specific, concrete alternative sentences the user could say).

Return a single JSON object with the following keys:
- "summary": 3–4 sentence overview of how the conversation went from the manager's perspective.
- "strengths": array of bullet-point strings describing what the user did well (5–8 items).
- "opportunities": array of bullet-point strings with specific improvements (5–8 items).
- "tone_feedback": object with:
    - "overall_tone": short label (e.g., "calm, supportive", "anxious but respectful")
    - "tone_notes": 2–4 sentences analyzing tone (word choice, pacing, empathy, confidence)
    - "risk_flags": array of short strings if any (e.g., "overpromising timeline", "leading questions")
- "suggested_rewrites": array; each item is an object with:
    - "situation": short label of the moment (e.g., "opening check-in", "transition to action plan")
    - "why_change": 1–2 sentences explaining the improvement
    - "example": a single, polished sentence or short block (1–3 sentences) the manager could say
- "next_steps": array of practical follow-ups the manager should schedule (4–7 items).
Keep the language direct, professional, and oriented to a real manager-employee 1:1.
"""

USER_PROMPT_TEMPLATE = """Below is the full conversation transcript.
Assume lines beginning with "User:" are the manager, and lines beginning with "AI:" are the team member.

Transcript:
----------------
{transcript}
----------------

Please evaluate ONLY the manager's (User) lines.
If the transcript is short or missing parts, still provide best-effort coaching.
"""

COACH_SCRIPT_SYSTEM = """You are a seasoned leadership and career coach.
Write a natural, conversational monologue (not bullets) as if you are speaking directly to the manager.
Sound warm, candid, and practical. Avoid jargon. Keep it human and cohesive, with gentle transitions.
Speak in first person ("I") and address the manager as "you".
Do NOT restate field names like "summary" or "strengths"; weave them into a single flowing narrative.
Conclude with 2–3 crisp next steps the manager can act on this week.
"""

COACH_SCRIPT_USER_TMPL = """Below is structured feedback JSON about a manager's conversation.
Turn it into a single, believable coach script for TTS—no lists, no headings, no markup.
Aim for around {target_words} words. Style: {style}.

JSON:
{feedback_json}
"""


# ---------- Message builders ----------
def build_eval_messages(transcript: str):
    user_prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------- LLM caller (JSON or freeform) ----------
def call_llm_any(messages, model: str, json_mode: bool = True):
    """
    - json_mode=True  -> response_format={"type": "json_object"}
    - json_mode=False -> freeform text
    Uses BOSON_* env if present, else OPENAI_*.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Please `pip install openai`.")

    api_key = (os.environ.get("BOSON_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing BOSON_API_KEY or OPENAI_API_KEY in environment.")

    base = (os.environ.get("BOSON_BASE") or os.environ.get("OPENAI_BASE") or "https://hackathon.boson.ai/v1").strip()
    client = OpenAI(api_key=api_key, base_url=base)

    def _create(_json):
        kwargs = dict(model=model, messages=messages, temperature=0.6)
        if _json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    # Light retry for transient throttling
    for attempt in range(3):
        try:
            resp = _create(json_mode)
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            if json_mode and ("response_format" in msg or "json_object" in msg):
                # endpoint doesn't support JSON mode; retry as freeform
                resp = _create(False)
                return resp.choices[0].message.content
            if "rate" in msg and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise


def render_coach_script_via_llm(feedback: dict, model: str, target_words: int = 350, style: str = "warm"):
    user_prompt = COACH_SCRIPT_USER_TMPL.format(
        target_words=target_words,
        style=style,
        feedback_json=json.dumps(feedback, ensure_ascii=False, indent=2),
    )
    messages = [
        {"role": "system", "content": COACH_SCRIPT_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    script_text = call_llm_any(messages, model, json_mode=False)
    return script_text.strip()

def sanitize_coach_script(text: str) -> str:
    """
    Remove hidden/internal blocks (e.g., <think>...</think>) and tidy whitespace.
    Case-insensitive; handles multiline content.
    """
    if not text:
        return text

    # 1) Drop <think>…</think> blocks (any case)
    cleaned = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)

    # (Optional) also drop other internal blocks if they ever appear:
    cleaned = re.sub(r"(?is)<\s*(analysis|reflection|scratchpad)\s*>.*?<\s*/\s*\1\s*>", "", cleaned)

    # 2) Remove residual XML-ish tags of those names if they appear without closing
    cleaned = re.sub(r"(?is)</?\s*(think|analysis|reflection|scratchpad)\s*>", "", cleaned)

    # 3) Collapse excessive blank lines/spaces
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)     # trim end-of-line spaces
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)     # max 1 blank line
    cleaned = cleaned.strip()

    return cleaned

def default_report_md_path(input_path: str) -> str:
    p = pathlib.Path(input_path)
    return str(p.with_name(p.stem + "_report.md"))

def render_report_md(feedback: dict, source_path: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = []
    parts.append("# Career-Coach Feedback Report\n")
    parts.append(f"_Generated: {ts}_  \n_Source transcript: `{pathlib.Path(source_path).name}`_\n")

    # Summary
    summary = (feedback.get("summary") or "").strip()
    if summary:
        parts.append("## Summary")
        parts.append(summary + "\n")

    # Strengths / Opportunities
    strengths = feedback.get("strengths") or []
    opportunities = feedback.get("opportunities") or []
    parts.append("## Strengths")
    parts.extend([f"- {s}" for s in strengths] or ["- (none)"])
    parts.append("")
    parts.append("## Opportunities")
    parts.extend([f"- {o}" for o in opportunities] or ["- (none)"])
    parts.append("")

    # Tone
    tone = feedback.get("tone_feedback") or {}
    overall_tone = tone.get("overall_tone") or "(n/a)"
    tone_notes = (tone.get("tone_notes") or "").strip()
    risk_flags = tone.get("risk_flags") or []
    parts.append("## Tone Feedback")
    parts.append(f"- **Overall tone:** {overall_tone}")
    if tone_notes:
        parts.append(f"- **Notes:** {tone_notes}")
    if risk_flags:
        parts.append("- **Risk flags:** " + "; ".join(risk_flags))
    parts.append("")

    # Suggested rewrites
    rewrites = feedback.get("suggested_rewrites") or []
    parts.append("## Suggested Rewrites")
    if not rewrites:
        parts.append("- (none)\n")
    else:
        for i, item in enumerate(rewrites, 1):
            parts.append(f"**{i}. {item.get('situation','(situation)')}**")
            why = (item.get("why_change") or "").strip()
            ex = (item.get("example") or "").strip()
            if why:
                parts.append(f"_Why change:_ {why}")
            if ex:
                parts.append("**Example:**")
                parts.append("> " + ex.replace("\n", "\n> "))
            parts.append("")

    # Next steps
    next_steps = feedback.get("next_steps") or []
    parts.append("## Next Steps")
    parts.extend([f"- {n}" for n in next_steps] or ["- (none)"])
    parts.append("")

    return "\n".join(parts)

from contextlib import contextmanager

def make_step_timer(store: dict):
    """
    Returns a context manager `step(label)` that measures elapsed time
    and records it into `store[label]`. Also prints a per-step line.
    """
    @contextmanager
    def step(label: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            store[label] = dt
            print(f"[step] {label}: {dt:.3f} s")
    return step

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a manager/team-member conversation with an LLM career coach and produce a TTS-ready script.")
    parser.add_argument("--input", required=True, help="Path to the transcript .txt file")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name (OpenAI-compatible)")
    parser.add_argument("--output", default="", help="Optional path to save a Markdown summary (not required).")
    parser.add_argument("--script_words", type=int, default=350, help="Approx word count for the coach monologue.")
    parser.add_argument("--script_style", default="warm", help="Tone for the coach script (e.g., warm|calm|direct|upbeat).")

    # TTS voice params
    parser.add_argument("--tts_emotion", default="neutral", help="Emotion for TTS voice (neutral, friendly, calm, etc.)")
    parser.add_argument("--tts_rate", default=None, help="Speaking rate hint (slow, fast)")
    parser.add_argument("--tts_pitch", default=None, help="Speaking pitch hint (low, high)")
    parser.add_argument("--tts_speaker", default="SPEAKER0", help="Speaker tag to pass to the TTS model")
    parser.add_argument("--tts_ref_audio", default="./ref-audio/en_woman.wav", help="Optional reference audio for voice cloning")
    parser.add_argument("--tts_ref_transcript", default=None, help="Optional transcript text for style conditioning")

    args = parser.parse_args()

    # Per-step timings collector + context manager
    step_times = {}
    step = make_step_timer(step_times)

    # Build a client for the TTS module (its function expects a global `client`)
    with step("init: build OpenAI/Boson client for TTS"):
        api_key = (os.environ.get("BOSON_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Missing BOSON_API_KEY or OPENAI_API_KEY in environment.")
        base = (os.environ.get("BOSON_BASE") or os.environ.get("OPENAI_BASE") or "https://hackathon.boson.ai/v1").strip()
        isc.client = OpenAI(api_key=api_key, base_url=base)

    with step("io: load transcript"):
        transcript = load_text(args.input)
        if not transcript.strip():
            print("Input transcript is empty.", file=sys.stderr)
            sys.exit(2)

    # 1) Evaluate -> JSON
    with step("eval: build messages"):
        eval_messages = build_eval_messages(transcript)

    with step("eval: call LLM (JSON)"):
        raw = call_llm_any(eval_messages, args.model, json_mode=True)

    with step("eval: parse JSON"):
        try:
            feedback = json.loads(raw)
        except json.JSONDecodeError:
            # recover from fenced code block etc.
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                raw_clean = raw_clean.strip("`")
                lines = raw_clean.splitlines()
                if lines and lines[0].lstrip().startswith("json"):
                    raw_clean = "\n".join(lines[1:])
            feedback = json.loads(raw_clean)

    # Report
    with step("report: render markdown"):
        report_md_path = default_report_md_path(args.input)
        report_md = render_report_md(feedback, args.input)

    with step("report: write markdown"):
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote evaluation report to: {report_md_path}")

    # 2) Ask same LLM to write a natural coach monologue
    with step("script: generate via LLM"):
        coach_script_raw = render_coach_script_via_llm(
            feedback=feedback,
            model=args.model,
            target_words=args.script_words,
            style=args.script_style,
        )

    with step("script: sanitize"):
        coach_script = sanitize_coach_script(coach_script_raw)

    # 3) Always write TTS text by default (next to input)
    with step("tts: write text file"):
        tts_txt_path = default_tts_txt_path(args.input)
        with open(tts_txt_path, "w", encoding="utf-8") as f:
            f.write(coach_script + ("\n" if not coach_script.endswith("\n") else ""))
        print(f"Wrote TTS (coach script) to: {tts_txt_path}")

    # 4) Synthesize audio via your TTS function
    with step("tts: synthesize audio"):
        try:
            tts_audio_base = default_tts_audio_path(args.input)  # no extension; function will add the correct one
            audio_path = isc.tts_generate_speech_chunked(
                reply_text=coach_script,
                out_path_base=tts_audio_base,
                reference_path=args.tts_ref_audio,
                reference_transcript=args.tts_ref_transcript,
                speaker_tag=args.tts_speaker,
                emotion=args.tts_emotion,
                speaking_rate=args.tts_rate,
                speaking_pitch=args.tts_pitch,
                max_chars_per_chunk=1000,  # adjust if you hit provider limits
                gap_ms=150,
            )
            print(f"Wrote TTS (audio) to: {audio_path}")
        except Exception as e:
            print(f"[warning] TTS audio synthesis failed: {e}", file=sys.stderr)

    # Print summary
    total_steps = sum(step_times.values())
    print("\n[steps summary]")
    for k, v in step_times.items():
        print(f"- {k}: {v:.3f} s")
    print(f"- TOTAL (sum of steps): {total_steps:.3f} s")


if __name__ == "__main__":
    start = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - start
        print(f"[elapsed] {elapsed:.3f} s")
