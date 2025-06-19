from flask import Blueprint, request, render_template
from .model import FaceDetector

bp = Blueprint("main", __name__)
detector = FaceDetector("../models/face_detector.pth")

@bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@bp.route("/upload", methods=["POST"])
def upload():
    img = request.files["file"].read()
    score = detector.predict(img)
    return {"real_score": score}