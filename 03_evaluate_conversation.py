#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_evaluate_conversation.py

Refactor: expose functions that accept the structured `convo_result` (dict) and optional `onboarding`
instead of reading a .txt. Still supports CLI usage by reading a JSON file if provided.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from openai import OpenAI
import time
import random

# ----------------------------- Config -----------------------------

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
DEFAULT_MODEL = os.getenv("EVAL_MODEL", "Qwen3-32B-non-thinking-Hackathon")

# ----------------------------- Data types -----------------------------

@dataclass
class Turn:
    role: str
    content: str

@dataclass
class EvaluationResult:
    rubric_version: str
    overall_score: float
    scores: Dict[str, float]
    strengths: List[str]
    improvements: List[str]
    summary: str
    tokens_hint: Optional[int] = None

# ----------------------------- Helpers -----------------------------
import re

_THINK_BLOCKS = [
    re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", flags=re.S | re.I),
    re.compile(r"<\s*think\s*>.*\Z", flags=re.S | re.I),
]
_THINK_TAGS = re.compile(r"</?\s*think\s*>", flags=re.I)

def strip_think(text: str) -> str:
    if not isinstance(text, str):
        return text
    t = text
    for pat in _THINK_BLOCKS:
        t = pat.sub("", t)
    t = _THINK_TAGS.sub("", t)
    return t.strip()

def _user_only_conversation(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return only USER messages, preserving order."""
    return [
        {"role": "user", "content": strip_think(m.get("content", ""))}
        for m in conversation
        if (m.get("role") or "").lower() == "user"
    ]

def _user_only_emotions(conversation: List[Dict[str, str]], emotions: List[str]) -> List[str]:
    """
    Original `emotions` is aligned to *user* turns in capture order.
    Build a list the same length as the user-only conversation.
    """
    user_count = sum(1 for m in conversation if (m.get("role") or "").lower() == "user")
    return [strip_think(e) for e in (emotions[:user_count] if emotions else [])]

def _format_user_only_for_prompt(conversation: List[Dict[str, str]], emotions: Optional[List[str]] = None) -> str:
    """
    Render only USER turns (01, 02, ...) and optionally attach their tone lines.
    """
    user_msgs = _user_only_conversation(conversation)
    user_emos = _user_only_emotions(conversation, emotions or [])
    lines = []
    for i, msg in enumerate(user_msgs, 1):
        line = f"{i:02d} [USER] {msg['content']}"
        if i-1 < len(user_emos):
            line += f"\n     ↳ **Tone/Emotion:** {user_emos[i-1]}"
        lines.append(line)
    return "\n".join(lines)


def _llm_call(messages: List[Dict[str, Any]], *, model: str = DEFAULT_MODEL, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Thin wrapper with minimal retries to reduce 429s.
    """
    client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")
    delay = 0.7
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            txt = resp.choices[0].message.content
            return txt if isinstance(txt, str) else str(txt)
        except Exception as e:
            if attempt == 4:
                raise
            # backoff with jitter
            time.sleep(delay * (1.5 ** attempt) * (1 + random.random() * 0.3))

# def _build_eval_prompt(conversation: List[Dict[str, str]], emotions: List[str], onboarding: Optional[Dict[str, Any]]) -> str:
#     """
#     Build a rubric-based evaluation prompt that evaluates the USER only.
#     """
#     scenario_bits = ""
#     if onboarding:
#         role = onboarding.get("role") or "user"
#         ai_role = onboarding.get("ai_role") or "stakeholder"
#         sc = onboarding.get("scenario") or ""
#         scenario_bits = f"User role: {role}. AI role (context only): {ai_role}. Scenario: {sc}.\n"

#     # USER-only transcript section
#     convo_block = _format_user_only_for_prompt(conversation, emotions)

#     return (
#         "You are a conversation coach evaluating the **human participant (USER)** in a practice dialog.\n"
#         + scenario_bits +
#         "IMPORTANT:\n"
#         "- Evaluate **USER messages only**. Do **not** score or comment on assistant messages.\n"
#         "- Base your judgment solely on the USER’s wording and (if provided) the tone/emotion cue under each USER turn.\n"
#         "- Ignore any chain-of-thought or meta text such as <think>…</think> if present.\n\n"
#         "Assess the USER on these dimensions (0–5, half-points allowed):\n"
#         "1) Empathy & Tone (respectful, non-escalatory)\n"
#         "2) Clarity & Brevity (clear intent, minimal rambling)\n"
#         "3) Relevance to Scenario (stays on topic, responds to context)\n"
#         "4) Questioning & Guidance (asks clarifying or guiding questions when needed)\n"
#         "5) Professionalism & Non-defensiveness\n\n"
#         "Return STRICT JSON with this schema (no extra text):\n"
#         "{\n"
#         '  "rubric_version": "v1.0-user",\n'
#         '  "overall_score": <0-5 number>,\n'
#         '  "scores": {\n'
#         '    "empathy": <0-5>,\n'
#         '    "clarity": <0-5>,\n'
#         '    "relevance": <0-5>,\n'
#         '    "guidance": <0-5>,\n'
#         '    "professionalism": <0-5>\n'
#         "  },\n"
#         '  "strengths": ["bullet", "..."],\n'
#         '  "improvements": ["bullet", "..."],\n'
#         '  "summary": "2-3 sentence overall feedback focused on the USER"\n'
#         "}\n\n"
#         "USER-only transcript (with optional tone):\n"
#         "------------------------------------------\n"
#         f"{convo_block}\n"
#         "------------------------------------------\n"
#         "Only output JSON."
#     )

def _build_eval_user_md_prompt(conversation: List[Dict[str, str]], emotions: List[str], onboarding: Optional[Dict[str, Any]]) -> str:
    """
    Return a prompt that asks for a USER-only Markdown report.
    """
    scenario_bits = ""
    if onboarding:
        role = onboarding.get("role") or "user"
        ai_role = onboarding.get("ai_role") or "stakeholder"
        sc = onboarding.get("scenario") or ""
        scenario_bits = f"User role: {role}. AI role (context only): {ai_role}. Scenario: {sc}.\n"

    # Reuse your user-only formatter (from the previous step)
    convo_block = _format_user_only_for_prompt(conversation, emotions)

    return (
        "You are a conversation coach evaluating the **human participant (USER)**.\n"
        + scenario_bits +
        "IMPORTANT:\n"
        "- Evaluate **USER messages only**; assistant content is out of scope.\n"
        "- Base your judgment solely on the USER’s text and any provided Tone/Emotion cue.\n"
        "- Do not include chain-of-thought or any <think> blocks in the output.\n\n"
        "Write a concise **Markdown report** with the following sections in this exact order.\n"
        "Do not include JSON, YAML, code fences, or extra pre/post text.\n\n"
        "## Summary\n"
        "- 3–5 sentences summarizing the USER’s performance and key moments.\n\n"
        "## Strengths\n"
        "- 4–6 bullets highlighting specific USER behaviors.\n\n"
        "## Opportunities\n"
        "- 4–6 bullets with concrete, actionable improvement points for the USER.\n\n"
        "## Tone Feedback\n"
        "- **Overall tone:** <one concise phrase>\n"
        "- **Notes:** 2–4 sentences with concrete observations.\n"
        "- **Risk flags:** 1–2 short bullets if any; otherwise “None”.\n\n"
        "## Suggested Rewrites (USER)\n"
        "Pick 2–3 weak USER lines; for each, explain briefly (_Why change: …_) and provide a rewrite as a single blockquote line.\n\n"
        "## Next Steps\n"
        "- 4–6 concrete follow-up actions for the USER.\n\n"
        "Constraints:\n"
        "- Keep the whole report under ~400–600 words.\n"
        "- Focus only on USER messages; do **not** grade or rewrite the assistant.\n\n"
        "USER-only transcript (with optional tone):\n"
        "------------------------------------------\n"
        f"{convo_block}\n"
        "------------------------------------------\n"
        "Output: Markdown only (no code fences)."
    )


# ----------------------------- Public API -----------------------------

# def evaluate_conversation(convo_result: Dict[str, Any], onboarding: Optional[Dict[str, Any]] = None) -> EvaluationResult:
#     """
#     Evaluate the conversation using an LLM. Accepts the exact structure returned by 02_main_loop:
#     { 'conversation': [{'role': 'user'|'assistant', 'content': str}, ...], 'emotions': [...] }
#     """
#     conversation = convo_result.get("conversation") or []
#     emotions = convo_result.get("emotions") or []
#     if not conversation:
#         # Heuristic fallback
#         return EvaluationResult(
#             rubric_version="v1.0",
#             overall_score=0.0,
#             scores={"empathy": 0.0, "clarity": 0.0, "relevance": 0.0, "guidance": 0.0, "professionalism": 0.0},
#             strengths=[],
#             improvements=["No conversation content to evaluate."],
#             summary="No data.",
#             tokens_hint=0,
#         )

#     prompt = _build_eval_prompt(conversation, emotions, onboarding)
#     print(prompt)
#     sys_msg = {"role": "system", "content": "Return STRICT JSON that matches the requested schema. No prose."}
#     user_msg = {"role": "user", "content": prompt}

#     try:
#         txt = _llm_call([sys_msg, user_msg], model=DEFAULT_MODEL, max_tokens=600, temperature=0.2)
#         data = json.loads(txt)
#         # Validate minimal keys
#         ov = float(data.get("overall_score", 0))
#         scores = data.get("scores") or {}
#         strengths = data.get("strengths") or []
#         improvements = data.get("improvements") or []
#         summary = data.get("summary") or ""
#         return EvaluationResult(
#             rubric_version=str(data.get("rubric_version", "v1.0")),
#             overall_score=ov,
#             scores={
#                 "empathy": float(scores.get("empathy", 0)),
#                 "clarity": float(scores.get("clarity", 0)),
#                 "relevance": float(scores.get("relevance", 0)),
#                 "guidance": float(scores.get("guidance", 0)),
#                 "professionalism": float(scores.get("professionalism", 0)),
#             },
#             strengths=[str(s) for s in strengths][:10],
#             improvements=[str(s) for s in improvements][:10],
#             summary=str(summary),
#             tokens_hint=len(prompt.split()),
#         )
#     except Exception as e:
#         # Fallback heuristic if model fails
#         # Simple metrics: length, alternation, assistant share
#         turns = len(conversation)
#         assistant_utts = sum(1 for m in conversation if m.get("role") == "assistant")
#         user_utts = sum(1 for m in conversation if m.get("role") == "user")
#         ratio = assistant_utts / max(1, turns)
#         # naive score guesses
#         overall = round(2.5 + 1.5 * (1 - abs(0.5 - ratio) * 2), 1)  # best near 50/50
#         return EvaluationResult(
#             rubric_version="v1.0-fallback",
#             overall_score=overall,
#             scores={"empathy": overall, "clarity": overall, "relevance": overall, "guidance": overall, "professionalism": overall},
#             strengths=["Fallback heuristic used."],
#             improvements=[f"LLM eval failed: {e}"],
#             summary="Heuristic evaluation due to model failure.",
#             tokens_hint=0,
#         )

def evaluate_conversation_user_markdown(
    convo_result: Dict[str, Any],
    onboarding: Optional[Dict[str, Any]] = None
) -> str:
    """
    Produce a USER-only Markdown evaluation report.
    """
    conversation = convo_result.get("conversation") or []
    emotions = convo_result.get("emotions") or []

    if not conversation:
        return (
            "## Summary\n"
            "No conversation content was provided; unable to evaluate the USER.\n\n"
            "## Strengths\n- None\n\n"
            "## Opportunities\n- Provide a transcript to evaluate the USER.\n\n"
            "## Tone Feedback\n- **Overall tone:** N/A\n- **Notes:** N/A\n- **Risk flags:** None\n\n"
            "## Suggested Rewrites (USER)\n- N/A\n\n"
            "## Next Steps\n- Capture a conversation and re-run the evaluation.\n"
        )

    # (Optional) sanitize USER lines for stray <think> if you added strip_think earlier:
    for m in conversation:
        if (m.get("role") or "").lower() == "user":
            m["content"] = strip_think(m.get("content", ""))

    prompt = _build_eval_user_md_prompt(conversation, emotions, onboarding)
    sys_msg = {
        "role": "system",
        "content": (
            "Return Markdown only with the requested headings. "
            "Evaluate the USER only; assistant content is out of scope. "
            "Do NOT include chain-of-thought or any <think> tags."
        ),
    }
    user_msg = {"role": "user", "content": prompt}

    md = _llm_call([sys_msg, user_msg], model=DEFAULT_MODEL, max_tokens=900, temperature=0.2)
    # Small guard if a model ever adds fences
    md = md.strip()
    if md.startswith("```"):
        md = md.strip("`").strip()
    return md

# def save_evaluation(result: EvaluationResult, out_path: Path) -> None:
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

def save_markdown(md_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text, encoding="utf-8")

# ----------------------------- CLI (optional) -----------------------------

def main_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a conversation JSON file.")
    ap.add_argument("conversation_json", type=Path, help="Path to JSON file with {'conversation': [...], 'emotions': [...]}")
    ap.add_argument("--onboarding_json", type=Path, default=None, help="Optional onboarding JSON with role/ai_role/scenario")
    ap.add_argument("--out", type=Path, default=Path("data/evaluation.json"))
    ap.add_argument("--md", type=Path, default=Path("data/evaluation.md"))  # <-- add this
    args = ap.parse_args()

    convo = json.loads(args.conversation_json.read_text(encoding="utf-8"))
    onboarding = None
    if args.onboarding_json and args.onboarding_json.exists():
        onboarding = json.loads(args.onboarding_json.read_text(encoding="utf-8"))

    # result = evaluate_conversation(convo, onboarding=onboarding)
    # save_evaluation(result, args.out)
    # print(f"[ok] Wrote evaluation to: {args.out}")
    md = evaluate_conversation_user_markdown(convo, onboarding=onboarding)
    save_markdown(md, args.md)
    print(f"[ok] Wrote Markdown report to: {args.md}")

if __name__ == "__main__":
    main_cli()
