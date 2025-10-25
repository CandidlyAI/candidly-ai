from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path

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
