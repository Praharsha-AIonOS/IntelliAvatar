import os
import sys
import uuid
import subprocess
import base64
import shutil

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from sarvamai import SarvamAI

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not SARVAM_API_KEY:
    raise RuntimeError("Missing SARVAM_API_KEY")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

AUDIO_OUTPUT = os.path.join(TEMP_DIR, "tts.wav")
RAW_VIDEO_OUTPUT = os.path.join(OUTPUT_DIR, "invite_raw.mp4")
FINAL_VIDEO_OUTPUT = os.path.join(OUTPUT_DIR, "invite_final.mp4")

WAV2LIP_SCRIPT = os.path.join(BASE_DIR, "inference_onnxModel.py")
WAV2LIP_MODEL = os.path.join(BASE_DIR, "checkpoints", "wav2lip_gan.onnx")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Init services
# -------------------------------------------------
app = FastAPI()

sarvam_client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)

# -------------------------------------------------
# UI
# -------------------------------------------------

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "invitation_index.html"), "r", encoding="utf-8") as f:
        return f.read()
        

@app.get("/download")
def download_video():
    return FileResponse(
        FINAL_VIDEO_OUTPUT,
        media_type="video/mp4",
        filename="lip_synced_avatar.mp4",
        headers={
            "Content-Disposition": "attachment; filename=lip_synced_avatar.mp4",
            "Cache-Control": "no-store"
        }
    )


# -------------------------------------------------
# Video endpoint
# -------------------------------------------------
@app.get("/video")
def get_video():
    return FileResponse(
        FINAL_VIDEO_OUTPUT,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"}
    )

# -------------------------------------------------
# Generation endpoint
# -------------------------------------------------
@app.post("/generate")
def generate(
    video: UploadFile = File(...),
    text: str = Form(...)
):
    # Save uploaded video
    input_video_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.mp4")
    with open(input_video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # -------------------------------------------------
    # EXACT text → Sarvam TTS (no modification)
    # -------------------------------------------------
    tts = sarvam_client.text_to_speech.convert(
        text=text,
        target_language_code="en-IN"
    )

    audio_bytes = base64.b64decode(tts.audios[0])
    with open(AUDIO_OUTPUT, "wb") as f:
        f.write(audio_bytes)

    # -------------------------------------------------
    # Wav2Lip
    # -------------------------------------------------
    subprocess.run([
        sys.executable,
        WAV2LIP_SCRIPT,
        "--checkpoint_path", WAV2LIP_MODEL,
        "--face", input_video_path,
        "--audio", AUDIO_OUTPUT,
        "--outfile", RAW_VIDEO_OUTPUT
    ], check=True)

    # -------------------------------------------------
    # Browser-safe encode
    # -------------------------------------------------
    subprocess.run([
        "ffmpeg", "-y",
        "-i", RAW_VIDEO_OUTPUT,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-c:a", "aac",
        "-movflags", "+faststart",
        FINAL_VIDEO_OUTPUT
    ], check=True)

    return {"status": "done"}

@app.get("/favicon.ico")
def favicon():
    return {}
