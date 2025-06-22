# app/routes.py

from pathlib import Path
from io import BytesIO

from fastapi import APIRouter, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from app.model import FaceDetector

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

detector = FaceDetector(weights_path="models/face_detector.pth")

UPLOAD_DIR = Path("app/static/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/predict", response_class=HTMLResponse)
async def predict_html(request: Request, file: UploadFile = File(...)):
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    body = await file.read()
    out_path = UPLOAD_DIR / file.filename
    out_path.write_bytes(body)

    img = Image.open(BytesIO(body)).convert("RGB")
    score = detector.predict(img)
    label = "Real" if score > 0.5 else "AI-generated"
    pct = f"{score*100:.1f}%"

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
    score = detector.predict(img)
    return {"score": score, "label": "Real" if score > 0.5 else "AI-generated"}
