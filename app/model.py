import torch

class FaceDetector:
    def __init__(self, weights_path: str):
        self.model = torch.load(weights_path)
        self.model.eval()