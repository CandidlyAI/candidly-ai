from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import random
from fastapi.staticfiles import StaticFiles
from global_var import state, conversation
from onboarding import REC_PATH, onboard
from conversation_logic import build_system_prompt_from_onboarding, process_conversation_turn
from openai import OpenAI
import os

app = FastAPI()

# Allow requests from your Next.js frontend (adjust port as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("BOSON_API_KEY"), base_url="https://hackathon.boson.ai/v1")


@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running 🚀"}

@app.post("/onboarding/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Accepts audio from frontend and saves it.
    """
    file_path = REC_PATH

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    is_done = onboard(REC_PATH)

    return {"status": "success", "is_done": is_done, "filename": file.filename, "path": str(file_path)}


# Serve uploaded audio files from /uploads
app.mount("/tts", StaticFiles(directory="data/tts"), name="tts")

@app.get("/download-audio")
def download_audio():
    """
    Returns the public URL of a saved WAV file.
    """
    if not state or not state["audio_ready"]:
        return { "url": "" }
    
    state["audio_ready"] = False
    last_gen = state["last_generated"]

    return {"url": f"http://localhost:8000/{last_gen}"}


@app.get("/scenario")
def get_scenario():
    return {"scenario": state["onboarding"].onboarding_info.scenario }

@app.post("/conversation/reset")
def reset_conversation():
    """
    Resets the conversation state, building a system prompt from onboarding data.
    """
    conversation["messages"].clear()
    conversation["turn_counter"] = 0
    conversation["system_prompt"] = build_system_prompt_from_onboarding()
    conversation["messages"].append({"role": "system", "content": conversation["system_prompt"]})
    conversation["audio_ready"] = False
    conversation["last_generated"] = None
    return {"status": "reset", "system_prompt": conversation["system_prompt"]}


@app.post("/conversation/turn")
async def conversation_turn(file: UploadFile = File(...)):
    """
    Handles a single conversation turn:
    - Receives user audio
    - Transcribes it
    - Generates AI response
    - Synthesizes TTS
    """
    with open(REC_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_conversation_turn(client, REC_PATH)
    return result