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

def _format_conversation_for_prompt(conversation: List[Dict[str, str]]) -> str:
    lines = []
    for i, msg in enumerate(conversation, 1):
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        lines.append(f"{i:02d} [{role}] {content}")
    return "\n".join(lines)

def _build_eval_prompt(conversation: List[Dict[str, str]], onboarding: Optional[Dict[str, Any]]) -> str:
    """
    Build a rubric-based evaluation prompt. The model should return JSON.
    """
    scenario_bits = ""
    if onboarding:
        role = onboarding.get("role") or "user"
        ai_role = onboarding.get("ai_role") or "stakeholder"
        sc = onboarding.get("scenario") or ""
        scenario_bits = f"User role: {role}. AI role: {ai_role}. Scenario: {sc}.\n"
    convo_block = _format_conversation_for_prompt(conversation)
    return (
        "You are a conversation coach evaluating a practice dialog.\n"
        + scenario_bits +
        "Assess the AI assistant on these dimensions (0–5, half-points allowed):\n"
        "1) Empathy & Tone\n"
        "2) Clarity & Brevity\n"
        "3) Relevance to Scenario\n"
        "4) Questioning & Guidance\n"
        "5) Professionalism & Non-defensiveness\n\n"
        "Return STRICT JSON with this schema (no extra text):\n"
        "{\n"
        '  "rubric_version": "v1.0",\n'
        '  "overall_score": <0-5 number>,\n'
        '  "scores": {\n'
        '    "empathy": <0-5>,\n'
        '    "clarity": <0-5>,\n'
        '    "relevance": <0-5>,\n'
        '    "guidance": <0-5>,\n'
        '    "professionalism": <0-5>\n'
        "  },\n"
        '  "strengths": ["bullet", "..."],\n'
        '  "improvements": ["bullet", "..."],\n'
        '  "summary": "2-3 sentence overall feedback"\n'
        "}\n\n"
        "Conversation transcript:\n"
        "------------------------\n"
        f"{convo_block}\n"
        "------------------------\n"
        "Only output JSON."
    )

# ----------------------------- Public API -----------------------------

def evaluate_conversation(convo_result: Dict[str, Any], onboarding: Optional[Dict[str, Any]] = None) -> EvaluationResult:
    """
    Evaluate the conversation using an LLM. Accepts the exact structure returned by 02_main_loop:
    { 'conversation': [{'role': 'user'|'assistant', 'content': str}, ...], 'emotions': [...] }
    """
    conversation = convo_result.get("conversation") or []
    if not conversation:
        # Heuristic fallback
        return EvaluationResult(
            rubric_version="v1.0",
            overall_score=0.0,
            scores={"empathy": 0.0, "clarity": 0.0, "relevance": 0.0, "guidance": 0.0, "professionalism": 0.0},
            strengths=[],
            improvements=["No conversation content to evaluate."],
            summary="No data.",
            tokens_hint=0,
        )

    prompt = _build_eval_prompt(conversation, onboarding)
    sys_msg = {"role": "system", "content": "Return STRICT JSON that matches the requested schema. No prose."}
    user_msg = {"role": "user", "content": prompt}

    try:
        txt = _llm_call([sys_msg, user_msg], model=DEFAULT_MODEL, max_tokens=600, temperature=0.2)
        data = json.loads(txt)
        # Validate minimal keys
        ov = float(data.get("overall_score", 0))
        scores = data.get("scores") or {}
        strengths = data.get("strengths") or []
        improvements = data.get("improvements") or []
        summary = data.get("summary") or ""
        return EvaluationResult(
            rubric_version=str(data.get("rubric_version", "v1.0")),
            overall_score=ov,
            scores={
                "empathy": float(scores.get("empathy", 0)),
                "clarity": float(scores.get("clarity", 0)),
                "relevance": float(scores.get("relevance", 0)),
                "guidance": float(scores.get("guidance", 0)),
                "professionalism": float(scores.get("professionalism", 0)),
            },
            strengths=[str(s) for s in strengths][:10],
            improvements=[str(s) for s in improvements][:10],
            summary=str(summary),
            tokens_hint=len(prompt.split()),
        )
    except Exception as e:
        # Fallback heuristic if model fails
        # Simple metrics: length, alternation, assistant share
        turns = len(conversation)
        assistant_utts = sum(1 for m in conversation if m.get("role") == "assistant")
        user_utts = sum(1 for m in conversation if m.get("role") == "user")
        ratio = assistant_utts / max(1, turns)
        # naive score guesses
        overall = round(2.5 + 1.5 * (1 - abs(0.5 - ratio) * 2), 1)  # best near 50/50
        return EvaluationResult(
            rubric_version="v1.0-fallback",
            overall_score=overall,
            scores={"empathy": overall, "clarity": overall, "relevance": overall, "guidance": overall, "professionalism": overall},
            strengths=["Fallback heuristic used."],
            improvements=[f"LLM eval failed: {e}"],
            summary="Heuristic evaluation due to model failure.",
            tokens_hint=0,
        )

def save_evaluation(result: EvaluationResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------- CLI (optional) -----------------------------

def main_cli():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate a conversation JSON file.")
    ap.add_argument("conversation_json", type=Path, help="Path to JSON file with {'conversation': [...], 'emotions': [...]}")
    ap.add_argument("--onboarding_json", type=Path, default=None, help="Optional onboarding JSON with role/ai_role/scenario")
    ap.add_argument("--out", type=Path, default=Path("data/evaluation.json"))
    args = ap.parse_args()

    convo = json.loads(args.conversation_json.read_text(encoding="utf-8"))
    onboarding = None
    if args.onboarding_json and args.onboarding_json.exists():
        onboarding = json.loads(args.onboarding_json.read_text(encoding="utf-8"))

    result = evaluate_conversation(convo, onboarding=onboarding)
    save_evaluation(result, args.out)
    print(f"[ok] Wrote evaluation to: {args.out}")

if __name__ == "__main__":
    main_cli()
