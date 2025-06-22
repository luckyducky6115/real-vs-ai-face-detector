import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    """
    Squeeze-and-Excitation over a concatenation of feature maps.
    """
    def __init__(self, tot_channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Sequential(
            nn.Linear(tot_channels, tot_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(tot_channels // reduction, tot_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feats: tuple[torch.Tensor, ...]):
        # feats is a tuple of feature‐map tensors, e.g. (x3, x4, x5)
        x = torch.cat(feats, dim=1)           # concat along channel axis
        B, C, _, _ = x.shape
        # Squeeze
        y = self.avgpool(x).view(B, C)
        # Excitation
        y = self.fcn(y).view(B, C, 1, 1)
        # Scale
        return x * y
