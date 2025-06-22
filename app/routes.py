# app/routes.py
import io
import os
import torch
from pathlib import Path
from io import BytesIO

from fastapi import APIRouter, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
from facenet_pytorch import MTCNN

from app.model import FaceDetector

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load your detector and move its model onto `device`
detector = FaceDetector(weights_path="models/face_detector.pth")
detector.model.to(device)

# init MTCNN face‐cropper on the same device
mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    post_process=False,
    device=device
)

# prepare uploads folder & static mount
UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
router.mount("/static", StaticFiles(directory="app/static"), name="static")


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request, file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg"):
        raise HTTPException(400, "Invalid file type")

    # save upload
    body = await file.read()
    out_path = UPLOAD_DIR / file.filename
    out_path.write_bytes(body)

    # open PIL image
    img = Image.open(BytesIO(body)).convert("RGB")

    # detect & crop first face
    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        raise HTTPException(400, "No face detected")
    x1, y1, x2, y2 = boxes[0]
    face = img.crop((x1, y1, x2, y2)).convert("RGB")

    # classify
    score = detector.predict(face)
    label = "Real" if score > 0.5 else "AI-generated"
    pct   = f"{score*100:.1f}%"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "uploaded_image": f"/static/uploads/{file.filename}",
        "label": label,
        "score": pct
    })


@router.post("/api/predict", response_class=JSONResponse)
async def predict_api(file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg"):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)

    body = await file.read()
    img = Image.open(BytesIO(body)).convert("RGB")

    boxes, _ = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return JSONResponse({"error": "No face detected"}, status_code=400)
    x1, y1, x2, y2 = boxes[0]
    face = img.crop((x1, y1, x2, y2)).convert("RGB")

    score = detector.predict(face)
    label = "Real" if score > 0.5 else "AI-generated"
    return {"score": score, "label": label}
