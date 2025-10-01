# app.py






import os
tess_path = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
import pytesseract
pytesseract.pytesseract.tesseract_cmd = tess_path










import os
import io
import tempfile
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import pytesseract

# Detoxify (text toxicity)
from detoxify import Detoxify

# Config
MAX_VIDEO_FRAMES = 5     # max frames to sample from video
IMAGE_OCR_DPI = 300
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", 0.7))
IMAGE_TEXT_WEIGHT = float(os.getenv("IMAGE_TEXT_WEIGHT", 0.3))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", 0.7))

# Path to tesseract if provided via env (Windows typical)
tess_cmd = os.getenv("TESSERACT_CMD")
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

app = FastAPI(title="Toxicity Scoring API - Sample Backend (Python)")


from fastapi.middleware.cors import CORSMiddleware

# DEVELOPMENT: allow all origins + methods so browser preflight passes.
# Do NOT use allow_origins=["*"] with allow_credentials=True in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],       # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # allow content-type, authorization, custom headers
)


executor = ThreadPoolExecutor(max_workers=2)
detox_model = None

@app.on_event("startup")
def load_models():
    global detox_model
    print("Loading Detoxify model — this may take a moment...")
    # 'original' is a commonly used Detoxify model; adjust if you want 'unbiased'
    detox_model = Detoxify('original')
    print("Detoxify model loaded.")

async def score_text_async(text: str) -> Dict[str, float]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _score_text_sync, text)

def _score_text_sync(text: str) -> Dict[str, float]:
    if not text:
        return {"toxicity": 0.0}
    out = detox_model.predict(text)
    if 'toxicity' in out:
        return {"toxicity": float(out['toxicity'])}
    vals = [float(v) for v in out.values() if isinstance(v, (float, int))]
    avg = float(np.mean(vals)) if vals else 0.0
    return {"toxicity": avg}

def ocr_image_pil(pil_img: Image.Image) -> str:
    try:
        gray = pil_img.convert("L")
        w, h = gray.size
        # upscale if small to help OCR
        if max(w, h) < 1000:
            scale = int(IMAGE_OCR_DPI / 72)
            gray = gray.resize((max(1, w*scale), max(1, h*scale)))
        text = pytesseract.image_to_string(gray, lang='eng')
        return text.strip()
    except Exception as e:
        print("OCR failed:", e)
        return ""

async def ocr_image_bytes(image_bytes: bytes) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, _ocr_image_sync, image_bytes)

def _ocr_image_sync(image_bytes: bytes) -> str:
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        return ocr_image_pil(pil)
    except Exception as e:
        print("ocr image sync error:", e)
        return ""

def sample_frames_from_video_bytes(video_bytes: bytes, max_frames: int = MAX_VIDEO_FRAMES) -> list:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        samples = min(max_frames, max(1, frame_count))
        indices = np.linspace(0, frame_count-1, samples, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
        cap.release()
        return frames
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

class ScoreResponse(BaseModel):
    score: float
    label: str
    sources: Dict[str, Any]
    details: Dict[str, Any] = {}

@app.post("/api/toxicity", response_model=ScoreResponse)
async def score_toxicity(
    text: Optional[str] = Form(None),
    screenshot: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    if detox_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    tasks = []
    text_score = 0.0
    image_text_score = 0.0
    image_ocr_text = ""
    frames_sampled = 0

    if text:
        text_task = asyncio.create_task(score_text_async(text))
        tasks.append(("text", text_task))

    if screenshot is not None:
        try:
            data = await screenshot.read()
            if data:
                ocr_task = asyncio.create_task(ocr_image_bytes(data))
                tasks.append(("ocr_image", ocr_task))
        except Exception as e:
            print("Failed reading screenshot:", e)

    if video is not None:
        try:
            v_bytes = await video.read()
            if v_bytes:
                loop = asyncio.get_running_loop()
                frames = await loop.run_in_executor(executor, sample_frames_from_video_bytes, v_bytes, MAX_VIDEO_FRAMES)
                frames_sampled = len(frames)
                for idx, pil_img in enumerate(frames):
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    b = buf.getvalue()
                    ocr_fut = asyncio.create_task(ocr_image_bytes(b))
                    tasks.append((f'ocr_frame_{idx}', ocr_fut))
        except Exception as e:
            print("Video handling failed:", e)

    results = {}
    for name, task in tasks:
        try:
            res = await task
            results[name] = res
        except Exception as e:
            print(f"Task {name} failed: {e}")
            results[name] = ""

    if text:
        t_out = results.get("text")
        if isinstance(t_out, dict) and 'toxicity' in t_out:
            text_score = float(t_out['toxicity'])
        else:
            try:
                t_sync = _score_text_sync(text)
                text_score = float(t_sync.get('toxicity', 0.0))
            except Exception:
                text_score = 0.0

    ocr_texts = []
    for k, v in results.items():
        if k.startswith('ocr_') or k.startswith('ocr_frame_') or k == 'ocr_image':
            if isinstance(v, str) and v.strip():
                ocr_texts.append(v.strip())

    image_ocr_text = "\n".join(ocr_texts).strip()
    if image_ocr_text:
        img_text_out = await score_text_async(image_ocr_text)
        image_text_score = float(img_text_out.get('toxicity', 0.0))

    final_score = 0.0
    weight_sum = 0.0
    if text and (text_score is not None):
        final_score += TEXT_WEIGHT * text_score
        weight_sum += TEXT_WEIGHT
    if image_ocr_text and (image_text_score is not None):
        final_score += IMAGE_TEXT_WEIGHT * image_text_score
        weight_sum += IMAGE_TEXT_WEIGHT

    if weight_sum == 0:
        final_score = 0.0
    else:
        final_score = final_score / weight_sum

    label = "toxic" if final_score >= TOXIC_THRESHOLD else "safe"

    resp = {
        "score": round(float(final_score), 4),
        "label": label,
        "sources": {
            "text_score": round(float(text_score), 4),
            "image_text_score": round(float(image_text_score), 4),
            "ocr_text_snippet": image_ocr_text[:200]
        },
        "details": {
            "model": "detoxify(original)",
            "frames_sampled": frames_sampled
        }
    }
    return JSONResponse(content=resp)
