import torch
from PIL import Image
from .cnn import AttentionVGG as CNNClassifier
from .utils import inference_transform

class FaceDetector:
    def __init__(self, weights_path="models/face_detector.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = CNNClassifier().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.transform = inference_transform()

    def predict(self, pil_image: Image.Image) -> float:
        x = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)
        return probs[0, 1].item()