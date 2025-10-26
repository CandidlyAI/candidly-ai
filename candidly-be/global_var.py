from typing import Any
from dataclasses import dataclass

@dataclass
class OnboardingInfo:
    ai_role: str
    user_role: str
    scenario: str

@dataclass
class OnboardingState:
    messages: list[Any]
    speak_counter: int
    onboarding_info: OnboardingInfo


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

messages = [SYSTEM_PROMPT]

state = {
    "onboarding": OnboardingState(messages=messages, speak_counter=0, onboarding_info=OnboardingInfo("", "", "")),
    "audio_ready": False,
    "last_generated": ""
}

conversation = {
    "messages": [],
    "turn_counter": 0,
    "audio_ready": False,
    "last_generated": None,
    "system_prompt": "",
}