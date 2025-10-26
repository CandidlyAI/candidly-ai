from openai import OpenAI, RateLimitError
import base64
import os
import sounddevice as sd
import soundfile as sf
import json
import re
import threading
from queue import Queue
import time

BOSON_API_KEY = os.getenv("BOSON_API_KEY")
client = OpenAI(api_key=BOSON_API_KEY, base_url="https://hackathon.boson.ai/v1")

def retry_with_exponential_backoff(
    func,
    max_retries=5,
    initial_delay=1,
    exponential_base=2,
    jitter=True,
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                num_retries += 1
                if num_retries > max_retries:
                    raise Exception(f"Max retries ({max_retries}) exceeded: {e}")
                
                delay *= exponential_base * (1 + jitter * (0.5 - time.time() % 1))
                print(f"⏳ Rate limit hit. Retrying in {delay:.2f} seconds... (attempt {num_retries}/{max_retries})")
                time.sleep(delay)
            except Exception as e:
                raise e
    
    return wrapper

class ConversationalistAgent:
    """Agent 1: Generates conversational responses"""
    
    def __init__(self, client, system_prompt):
        self.client = client
        self.system_prompt = system_prompt or (
            "You are a helpful assistant. You are supposed to pretend that you are a difficult customer "
            "complaining about an order that didn't arrive. Generate the customer response — nothing else, "
            "just the text. You are talking to a customer service manager at the store. "
            "Generate brief, natural customer responses in 1-2 sentences only. Keep it conversational and concise."
        )
        
    @retry_with_exponential_backoff
    def generate_response(self, conversation_history: list) -> str:
        """Given conversation history, generate the next assistant message"""

        
        messages = [{"role": "system", "content": self.system_prompt}] + conversation_history
        
        resp = self.client.chat.completions.create(
            model="Qwen3-32B-non-thinking-Hackathon",  # Using the available model
            messages=messages,
            max_tokens=50,
            temperature=0.7,
        )
        
        response_text = resp.choices[0].message.content
        if isinstance(response_text, str):
            return response_text.strip()
        
        # Handle complex response formats
        if isinstance(response_text, list):
            for block in response_text:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "").strip()
        
        return str(response_text)

    @retry_with_exponential_backoff
    def extract_emotion(self, conversation_history: list) -> str:
        """
        Generate a one-word emotion label describing the AI's tone.
        If base case (one system + one user message saying "Generate the first complaint, do not output anything else"),
        infer emotion from system prompt only.
        Otherwise, predict the next emotional state of the assistant based on the last user-assistant exchange.
        """

        emotions = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

        # --- Otherwise: predict next state based on last user–assistant exchange ---
        user_last = ""
        ai_text = ""

        # Find the last assistant and user messages in order
        for msg in reversed(conversation_history):
            if msg["role"] == "assistant" and not ai_text:
                ai_text = msg["content"]
            elif msg["role"] == "user" and not user_last:
                user_last = msg["content"]
            if ai_text and user_last:
                break

        # If we don’t have both, fallback gracefully
        if not (ai_text and user_last):
            return "neutral"

        system_prompt = (
            "You are an emotion classifier. Given the last exchange between the user and assistant, "
            "predict the *next emotional state* of the assistant. "
            "Output ONE WORD ONLY (lowercase). "
            f"Choose from this list only: {', '.join(emotions)}."
        )

        dialogue_context = f"User: {user_last}\nAssistant: {ai_text}"

        resp = self.client.chat.completions.create(
            model="Qwen3-32B-non-thinking-Hackathon",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dialogue_context},
            ],
            max_tokens=5,
            temperature=0.0,
        )

        emotion = resp.choices[0].message.content.strip().lower()
        emotion = re.findall(r'\b[a-z]+\b', emotion)
        return emotion[0] if emotion else "neutral"

class TTSAgent:
    """Agent 2: Text-to-Speech using higgs-audio-generation-Hackathon"""
    
    def __init__(self, client):
        self.client = client
    
    @retry_with_exponential_backoff
    def synthesize_speech(self, text: str, output_path: str) -> str:
        """Convert text to speech and save to file"""
        system = (
            "You are an AI assistant designed to convert text into speech.\n"
            "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
            "If no speaker tag is present, select a suitable voice on your own.\n\n"
            "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
        )
        
        resp = self.client.chat.completions.create(
            model="higgs-audio-generation-Hackathon",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=4096,
            temperature=1.0,
            top_p=0.95,
            stream=False,
            stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
            extra_body={"top_k": 50},
        )
        
        # Save audio
        audio_b64 = resp.choices[0].message.audio.data
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(audio_b64))
        
        print(f"✓ Saved audio to {output_path}")
        return output_path
    
    def play_audio(self, path: str):
        """Play audio file"""
        try:
            data, samplerate = sf.read(path)
            sd.play(data, samplerate)
            sd.wait()
            print("✓ Playback finished")
        except Exception as e:
            print(f"✗ Playback failed: {e}")


class STTAgent:
    """Agent 3: Speech-to-Text using higgs-audio-understanding-Hackathon"""
    
    def __init__(self, client):
        self.client = client
    
    @retry_with_exponential_backoff
    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        audio_b64, fmt = self._encode_audio(audio_path)
        
        resp = self.client.chat.completions.create(
            model="higgs-audio-understanding-Hackathon",
            messages=[
                {"role": "system", "content": "Transcribe this audio verbatim."},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": fmt,
                            },
                        },
                    ],
                },
            ],
            max_completion_tokens=512,
            temperature=0.0,
        )
        
        transcript = self._extract_text(resp)
        print(f"✓ Transcribed: {transcript}")
        return transcript
    
    def record_audio(self, duration: int, output_path: str) -> str:
        """Record audio from microphone"""
        print(f"🎤 Recording for {duration} seconds...")
        recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
        sd.wait()
        sf.write(output_path, recording, 44100)
        print(f"✓ Saved recording to {output_path}")
        return output_path
    
    @staticmethod
    def _encode_audio(file_path: str) -> tuple[str, str]:
        """Encode audio file to base64"""
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = file_path.split(".")[-1].lower()
        return b64, ext
    
    @staticmethod
    def _extract_text(resp) -> str:
        """Extract text from API response"""
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
        
        txt = getattr(msg, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        
        return json.dumps(resp.model_dump())

class UserToneAnalyzerAgent:
    def __init__(self, client):
        self.client = client

    def encode_audio_to_base64(self, file_path: str) -> str:
        """Encode audio file to base64 format."""
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    @retry_with_exponential_backoff    
    def analyze_tone(self, audio_path):
        audio_base64 = self.encode_audio_to_base64(audio_path)
        file_format = audio_path.split(".")[-1]

        resp = client.chat.completions.create(
            model="Qwen3-Omni-30B-A3B-Thinking-Hackathon",
            messages=[
                {"role": "system", "content": """
                 You are an expert audio analyst. Analyze how the user said what they said in 3-4 concise sentences. Include: overall tone and sentiment (positive/negative/neutral with intensity), speech characteristics (pace, volume, clarity, pitch), emotional indicators (energy level, confidence, stress/frustration/happiness), and communication style (pauses, hesitations, emphasis). 
                 Be specific and direct.

                 Example: "Audio of user speaking" -> "The speaker's tone is neutral with a subdued, slightly hesitant sentiment, suggesting a lack of confidence or urgency in the statement. Their speech is delivered at a slow, deliberate pace with moderate volume and clear articulation, but features a consistently low, flat pitch that lacks energetic inflection. There are no signs of stress or frustration, though the measured delivery conveys a low energy level and cautious, almost robotic, communication style. The slight pause after "We are" and absence of emphasis reinforce the impression of a routine, non-emotional update rather than a passionate or confident response."
                 """
                },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_base64,
                            "format": file_format,
                        },
                    },
                ],
            },
            ],
            max_tokens=1024,
            temperature=0.2,
        )
        processed = re.split(r'</think>\s*', resp.choices[0].message.content, maxsplit=1)[-1]
        processed = processed.strip()
        print(processed)
        return processed
    
def analyze_tone_background(utaa, audio_path, result_queue, turn_num):
    """Background worker function for tone analysis"""
    try:
        print(f"🎭 [Background] Starting tone analysis for turn {turn_num}...")
        tone_sentiment = utaa.analyze_tone(audio_path)
        result_queue.put((turn_num, tone_sentiment))
    except Exception as e:
        print(f"✗ [Background] Tone analysis failed: {e}")
        result_queue.put((turn_num, None))

def run_conversation_cycle(num_turns: int = 3, recording_duration: int = 5):
    """
    Main conversation cycle:
    1. Conversationalist generates response
    2. Append to conversation
    3. TTS converts to speech and plays it
    4. Prompt user for audio input
    5. STT transcribes audio
    6. Append to conversation
    7. Repeat
    """
    # Initialize agents
    conversationalist = ConversationalistAgent(client)
    tts = TTSAgent(client)
    stt = STTAgent(client)
    utaa = UserToneAnalyzerAgent(client)
    
    # Setup
    output_dir = "./test_wav"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize conversation
    conversation = [
        {"role": "user", "content": "Generate the first complaint, do not output anything else."},
    ]
    
    print("\n" + "="*60)
    print("CONVERSATION SYSTEM STARTED")
    print("="*60)

    emotions = []
    tone_threads = []
    tone_results_queue = Queue()

    res = []
    
    for turn in range(num_turns):
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}: AI RESPONSE")
        print(f"{'='*60}")
        
        # 0. extract emotion
        emotion = conversationalist.extract_emotion(conversation)

        # 1. Conversationalist generates response
        print("💭 Generating response...")
        ai_text = conversationalist.generate_response(conversation)
        
        print(f"📝 AI: {ai_text}")
        
        # 2. Append to conversation
        conversation.append({"role": "assistant", "content": ai_text})
        
        # 3. TTS: Convert to speech and play
        print("🔊 Converting to speech...")
        # audio_path = os.path.join(output_dir, f"ai_response_{turn + 1}.wav")
        # tts.synthesize_speech(ai_text, audio_path)
        # tts.play_audio(audio_path)
        
        # Check if we should continue
        if turn == num_turns - 1:
            res.append({
                "ai_text": ai_text,
                "emotion": emotion
            })    
            break
        
        print(f"\n{'='*60}")
        print(f"TURN {turn + 1}: USER INPUT")
        print(f"{'='*60}")
        
        # 4. Prompt user for audio input
        # user_audio_path = os.path.join(output_dir, f"user_input_{turn + 1}.wav")
        # stt.record_audio(recording_duration, user_audio_path)
        
        user_audio_path = f"./test_wav/user_input_{turn + 1}.wav"
        # 5. STT: Transcribe audio
        print("📝 Transcribing...")
        # user_text = stt.transcribe_audio(user_audio_path)
        user_text = "Oh, I'm sorry about that."
        
        # tone_sentiment = utaa.analyze_tone(user_audio_path)
        # emotions.append(tone_sentiment)
        # 6. Start tone analysis in background thread (NON-BLOCKING)
        tone_thread = threading.Thread(
            target=analyze_tone_background,
            args=(utaa, user_audio_path, tone_results_queue, turn + 1),
            daemon=True
        )
        tone_thread.start()
        tone_threads.append(tone_thread)
        
        # 7. Append to conversation
        conversation.append({"role": "user", "content": user_text})
        
        res.append({
            "user_text": user_text,
            "ai_text": ai_text,
            "emotion": emotion
        })



    print("\n" + "="*60)
    print("CONVERSATION COMPLETE")
    print("="*60)



    # Wait for all tone analysis threads to complete
    for thread in tone_threads:
        thread.join()
    
    # Collect results from queue
    while not tone_results_queue.empty():
        turn_num, tone_sentiment = tone_results_queue.get()
        if tone_sentiment:
            emotions.append(tone_sentiment)
    print("\n📋 Full conversation history:")
    # print(conversation)
    # print(emotions)
    for i, msg in enumerate(conversation):
        role_emoji = "🤖" if msg['role'] == "assistant" else "👤"
        print(f"{i+1}. {role_emoji} [{msg['role'].upper()}]: {msg['content']}")

    print(res)

    print(len(res))
    print(len(emotions))

if __name__ == "__main__":
    # Run conversation with 3 turns and 5-second recordings
    run_conversation_cycle(num_turns=3, recording_duration=5)