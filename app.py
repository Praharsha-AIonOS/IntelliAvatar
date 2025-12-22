import os
import sys
import subprocess
import base64

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
from groq import Groq
from sarvamai import SarvamAI

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not GROQ_API_KEY or not SARVAM_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY or SARVAM_API_KEY")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_FACE = os.path.join(BASE_DIR, "inputs", "face_ref.mp4")
VIDEO_WELCOME = os.path.join(BASE_DIR, "inputs", "welcome.mp4")
VIDEO_IDLE = os.path.join(BASE_DIR, "inputs", "idle.mp4")

AUDIO_OUTPUT = os.path.join(BASE_DIR, "temp", "tts.wav")
RAW_VIDEO_OUTPUT = os.path.join(BASE_DIR, "outputs", "result_raw.mp4")
FINAL_VIDEO_OUTPUT = os.path.join(BASE_DIR, "outputs", "result_browser.mp4")

WAV2LIP_SCRIPT = os.path.join(BASE_DIR, "inference_onnxModel.py")
WAV2LIP_MODEL = os.path.join(BASE_DIR, "checkpoints", "wav2lip_gan.onnx")

os.makedirs(os.path.join(BASE_DIR, "temp"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

# -------------------------------------------------
# Init services
# -------------------------------------------------
app = FastAPI()

groq_client = Groq(api_key=GROQ_API_KEY)
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# -------------------------------------------------
# Helper: Make text TTS-safe
# -------------------------------------------------
def make_tts_safe(text: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """
Convert the input into natural spoken English suitable for text-to-speech.
Rules:
- No symbols, emojis, markdown, or code
- Convert math and symbols into words
- Plain conversational English only
"""
            },
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content.strip()

# -------------------------------------------------
# UI
# -------------------------------------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()




# -------------------------------------------------
# Video endpoints
# -------------------------------------------------
@app.get("/video/welcome")
def welcome_video():
    return FileResponse(VIDEO_WELCOME, media_type="video/mp4")

@app.get("/video/idle")
def idle_video():
    return FileResponse(VIDEO_IDLE, media_type="video/mp4")

@app.get("/video/generated")
def generated_video():
    return FileResponse(FINAL_VIDEO_OUTPUT, media_type="video/mp4")

# -------------------------------------------------
# Generation endpoint
# -------------------------------------------------
@app.post("/generate")
def generate_video(query: str = Form(...)):

    # 1. LLM
    llm = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Respond in spoken English only."},
            {"role": "user", "content": query}
        ]
    )
    raw_reply = llm.choices[0].message.content.strip()

    # 2. TTS-safe text
    text_reply = make_tts_safe(raw_reply)

    # 3. Sarvam TTS (female)
    tts = sarvam_client.text_to_speech.convert(
        text=text_reply,
        target_language_code="en-IN"
    )

    audio_bytes = base64.b64decode(tts.audios[0])
    with open(AUDIO_OUTPUT, "wb") as f:
        f.write(audio_bytes)

    # 4. Wav2Lip
    subprocess.run([
        sys.executable,
        WAV2LIP_SCRIPT,
        "--checkpoint_path", WAV2LIP_MODEL,
        "--face", VIDEO_FACE,
        "--audio", AUDIO_OUTPUT,
        "--outfile", RAW_VIDEO_OUTPUT
    ], check=True)

    # 5. Browser-safe encoding
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
