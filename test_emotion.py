import os, openai
import base64
client = openai.Client(api_key=os.getenv("BOSON_API_KEY"),
                       base_url="https://hackathon.boson.ai/v1")

audio_path = "/Users/jianyu-shin/Desktop/candidly-ai/hackathon-msac-public/ref-audio/belinda.wav"
with open(audio_path, "rb") as audio_file:
    # Base64 encode the binary data
    encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
resp = client.chat.completions.create(
    model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "input_audio", 
                "input_audio": {
                    "data": encoded_string,
                    "format": "wav",
                }
            },
            {"type": "text", "text": "Summarize the emotion expressed in the audio clip."}
        ]}
    ],
    max_tokens=1024,
    temperature=0.2,
)
print(resp.choices[0].message.content)

