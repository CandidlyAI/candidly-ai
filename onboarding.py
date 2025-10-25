import os, openai
import base64
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
import json

client = openai.Client(api_key=os.getenv("BOSON_API_KEY"),
                       base_url="https://hackathon.boson.ai/v1")


SAMPLE_RATE = 16000     # 16 kHz mono is common for ASR
CHANNELS    = 1
DURATION_S  = 7        # seconds to record; change as you like
REC_PATH    = "data/input.wav"
TTS_OUT     = "data/reply.wav"


def record_mic_to_wav(path: str, duration_s: int = DURATION_S, sr: int = SAMPLE_RATE, channels: int = CHANNELS):
    print(f"Recording {duration_s}s of audio from mic...")
    audio = sd.rec(int(duration_s * sr), samplerate=sr, channels=channels, dtype="int16")
    sd.wait()
    wav_write(path, sr, audio)
    print(f"Saved mic recording to {path}")


system_prompt = {
    "role": "system",
    "content": (
        "You are a helpful assistant. " 
        "This part is the onboarding process of an app. The app is to help users navigate difficult conversation. So it helps user practice with an AI chatbot that act as a stakeholder to the user" 
        "Your job is to gather information about the user. And your response is gonna be passed in as a template to another AI agent." "In this case you should gather the role of the user, the role that an AI should play, for example, user could be a customer service agent and AI could be a difficult customer" 
        "You should also try to get a scenario that the user trying to simulate if they have one. However, don't press the user if they don't have any scenario if they don't have one" 
        "Your response should be in the JSON format with { \"role\": \"the role of the users\", \"ai_role\": \"the role of AI\", \"is_done\": <boolean indicating if you are done collecting information>, \"scenario\": \"scenario I want to generate if user has one, empty string otherwise\", \"next_prompt\": \"If not done, next_prompt to collect the information\", \"transcription\": \"transcription of what the user said\" }. Output nothing else"
    )
}

# Conversation history
messages = [system_prompt]

done = False

os.makedirs("data", exist_ok=True)

while not done:
    record_mic_to_wav(REC_PATH, DURATION_S, SAMPLE_RATE, CHANNELS)
    with open(REC_PATH, "rb") as audio_file:
        # Base64 encode the binary data
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
        messages.append({
            "role": "user", 
            "content": [
                    {
                        "type": "input_audio", 
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav",
                        }
                    }
        ]})
        resp = client.chat.completions.create(
            model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        # Pop back the audio because it's too huge, we cannot do >1 audio for this model
        messages.pop()

        response_text = resp.choices[0].message.content
        print("🧠 Model response:\n", response_text)

        try:
            print("Final response", response_text)
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            print("⚠️ Model response not valid JSON, retrying...")
            continue

        done = parsed.get("is_done", False)

        if not done:
            next_prompt = parsed.get("next_prompt", "Please continue.")
            transcription = parsed.get("transcription", "")
            print(f"\n🤖 Assistant: {next_prompt}\n")
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": transcription
                    }
                ]
            })
            messages.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": next_prompt
                    }
                ]
            })
        else:
            print("\n✅ All information collected successfully!")
            print(json.dumps(parsed, indent=2))


