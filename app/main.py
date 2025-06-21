# main.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.model import FaceDetector

app = FastAPI()
detector = FaceDetector("models/best_cnn_face_classifier.pth")

# serve your React/Vanilla‐JS UI from app/static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("app/templates/index.html") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    score = detector.predict_from_bytes(img)
    return {"real_score": score}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
