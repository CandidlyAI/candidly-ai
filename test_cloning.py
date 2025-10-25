from openai import OpenAI
import base64
import os

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode("utf-8")

reference_path = "./hackathon-msac-public/ref-audio/hogwarts_wand_seller_v2.wav"
reference_transcript = (
    "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic. "
    "And the bond between you and your wand should only grow stronger. Do not be surprised at your new "
    "wand's ability to perceive your intentions - particularly in a moment of need. [with mystery and awe tone]"
)

resp = client.chat.completions.create(
    model="higgs-audio-generation-Hackathon",
    messages=[
        {"role": "system",   "content": f"""You are to roleplay as a fictional character."""
                f"""Follow the character’s personality, backstory, traits, and description strictly. Stay in character at all times. """
                f"""Character Name: {character['name']} """
                f"""Appearance / Description: {character['description']} """
                f"""Personality: {character['personality']} """
                f"""Backstory: {character['backstory']} """
                f"""Core Traits: {", ".join(character['traits'])} """
                f"""Dialogue Style: Speak with a {character['voice']} tone. Use empathetic and supportive language. """
                f"""Rules: """
                f"""1. Always respond as {character['name']}. """
                f"""2. Never break character or mention that you are an AI. """
                f"""3. Base your answers on {character['name']}'s perspective, knowledge, and worldview. """
                f"""4. When uncertain, improvise in a way consistent with the backstory and traits. """
                f"""Always output exactly ONE string with this format: <user>caption</user><response></response>. """
                f"""Format Rules: (1) The caption is a faithful transcription of the user's audio, in their language. """
                f"""(2) Immediately after the caption, output the literal token </user>. """
                f"""(3) Immediately after </user><response>, output your response. """
                f"""(4) There must be exactly one <user>, </user>, <response>, and </response>. """
                f"""(5) Do not wrap in quotes, code blocks, or add newlines. """
                f"""(6) If audio is unintelligible, caption as [inaudible]. """
                f"""(7) If no speech, caption as [no speech]. """
                f"""Examples: User Input: Hi, how was your day? → """
                f"""Output: <user>Hi, how was your day?</user><response>Hello! I'm great—how about you?</response> """
                f"""User Input: ¿Puedes poner un recordatorio para mañana? → """
                f"""Output: <user>¿Puedes poner un recordatorio para mañana?</user><response>¡Claro! ¿A qué hora te gustaría el recordatorio?</resonse> """
                f"""User Input: [garbled audio] → Output: <user>[inaudible]</user><response>Sorry, I couldn’t catch that. Could you repeat more clearly?</response>"""
            },
        {"role": "user", "content": "Pardon me. Do you just bump into people and walk away? [with furious tone]"},
    ],
    modalities=["text", "audio"],
    max_completion_tokens=4096,
    temperature=1.0,
    top_p=0.95,
    stream=False,
    stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
    extra_body={"top_k": 50},
)

audio_b64 = resp.choices[0].message.audio.data
open("output.wav", "wb").write(base64.b64decode(audio_b64))