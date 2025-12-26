import os
import sys
import uuid
import subprocess
import base64
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from sarvamai import SarvamAI

# -------------------------------------------------
# Load env
# -------------------------------------------------
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
if not SARVAM_API_KEY:
    raise RuntimeError("Missing SARVAM_API_KEY")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
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
# Gender → Speaker mapping
# -------------------------------------------------
GENDER_SPEAKER_MAP = {
    "male": "hitesh",
    "female": "manisha"
}

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI()
app.mount("/inputs", StaticFiles(directory=INPUTS_DIR), name="inputs")

sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def image_to_video(image_path: str, out_path: str, duration: int = 5):
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,"
        "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-r", "25",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        out_path
    ], check=True)

# -------------------------------------------------
# UI
# -------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "invitation_index.html"), "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------
# Video endpoints
# -------------------------------------------------
@app.get("/video")
def get_video():
    return FileResponse(
        FINAL_VIDEO_OUTPUT,
        media_type="video/mp4",
        headers={"Cache-Control": "no-store"}
    )

@app.get("/download")
def download():
    return FileResponse(
        FINAL_VIDEO_OUTPUT,
        media_type="video/mp4",
        filename="lip_synced_avatar.mp4",
        headers={"Content-Disposition": "attachment"}
    )

# -------------------------------------------------
# Generate
# -------------------------------------------------
@app.post("/generate")
def generate(
    text: str = Form(...),
    gender: str = Form(...),
    avatar: str = Form(None),
    video: UploadFile = File(None)
):
    uid = str(uuid.uuid4())

    gender = gender.lower().strip()
    if gender not in GENDER_SPEAKER_MAP:
        raise HTTPException(status_code=400, detail="gender must be male or female")

    speaker = GENDER_SPEAKER_MAP[gender]

    # Input selection
    if video and video.filename:
        ext = os.path.splitext(video.filename)[1].lower()
        input_path = os.path.join(UPLOAD_DIR, f"{uid}{ext}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    elif avatar:
        input_path = os.path.join(INPUTS_DIR, avatar)
        ext = os.path.splitext(avatar)[1].lower()
        if not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail="Avatar not found")
    else:
        raise HTTPException(status_code=400, detail="Avatar or video required")

    # Image → video
    if ext in [".png", ".jpg", ".jpeg"]:
        video_input = os.path.join(UPLOAD_DIR, f"{uid}_img.mp4")
        image_to_video(input_path, video_input)
    else:
        video_input = input_path

    # TTS
    tts = sarvam_client.text_to_speech.convert(
        text=text,
        speaker=speaker,
        target_language_code="en-IN"
    )

    with open(AUDIO_OUTPUT, "wb") as f:
        f.write(base64.b64decode(tts.audios[0]))

    # Wav2Lip
    subprocess.run([
        sys.executable,
        WAV2LIP_SCRIPT,
        "--checkpoint_path", WAV2LIP_MODEL,
        "--face", video_input,
        "--audio", AUDIO_OUTPUT,
        "--outfile", RAW_VIDEO_OUTPUT
    ], check=True)

    # Encode
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
