import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
from .attention import AttentionModule

class AttentionVGG(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # 1) load full pretrained VGG
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        vgg   = vgg16(weights=weights)
        feats = vgg.features

        # 2) slice into conv3, conv4, conv5 blocks
        self.block3 = nn.Sequential(*feats[:17])   # up to pool3
        self.block4 = nn.Sequential(*feats[17:24]) # up to pool4
        self.block5 = nn.Sequential(*feats[24:])   # up to pool5

        self.pool3 = nn.AdaptiveAvgPool2d((7,7))
        self.pool4 = nn.AdaptiveAvgPool2d((7,7))

        # 3) the SE‐style attention over all three
        tot_channels = 256 + 512 + 512   # block3 out + block4 out + block5 out
        self.attn = AttentionModule(tot_channels, reduction=16)

        # 4) your fancier classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                  # → [B, tot_channels,1,1]
            nn.LayerNorm([tot_channels, 1, 1]),
            nn.Flatten(),                             # → [B, tot_channels]
            nn.Dropout(0.3),
            nn.Linear(tot_channels, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x3 = self.block3(x)      # [B,256,H3,W3]
        x4 = self.block4(x3)     # [B,512,H4,W4]
        x5 = self.block5(x4)     # [B,512,H5,W5]

        x3 = self.pool3(x3)      # [B,256, 7, 7]
        x4 = self.pool4(x4) 
        # run SE‐attention on the concatenated maps
        x = self.attn((x3, x4, x5))
        # classify
        return self.classifier(x)
