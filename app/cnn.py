import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        # load the VGG16 backbone
        vgg = vgg16(weights=weights)
        # replace the final classifier layer
        in_features = vgg.classifier[-1].in_features
        vgg.classifier[-1] = nn.Linear(in_features, num_classes)
        self.model = vgg

    def forward(self, x):
        return self.model(x)