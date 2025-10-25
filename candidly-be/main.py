from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import random
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow requests from your Next.js frontend (adjust port as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "FastAPI backend is running 🚀"}

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """
    Accepts audio from frontend and saves it.
    """
    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "filename": file.filename, "path": str(file_path)}


audio_ready = False

# Serve uploaded audio files from /uploads
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/download-audio/{filename}")
def download_audio(filename: str):
    """
    Returns the public URL of a saved WAV file.
    """
    if not audio_ready:
        return { "url": "" }
    
    return {"url": f"http://localhost:8000/uploads/{filename}"}


SCENARIOS = [
    "You're talking to an upset customer who wants a refund.",
    "You're onboarding a new user for your product.",
    "You're resolving a technical issue with a client.",
    "You're assisting a customer who received the wrong order.",
]
@app.get("/scenario")
def get_scenario():
    return {"scenario": random.choice(SCENARIOS)}