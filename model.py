"""Small residual U-Net for synthetic low-light restoration (RGB in [0, 1])."""
import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.LeakyReLU(0.1, inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, width=24, ev=-3.0, gamma=2.2):
        super().__init__()
        self.gain = 2.0 ** (-ev / gamma)
        self.enc1 = Block(3, width)
        self.enc2 = Block(width, width * 2)
        self.enc3 = Block(width * 2, width * 4)
        self.bridge = Block(width * 4, width * 8)
        self.dec3 = Block(width * 12, width * 4)
        self.dec2 = Block(width * 6, width * 2)
        self.dec1 = Block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        z = self.bridge(F.avg_pool2d(c, 2))
        z = self.dec3(torch.cat((F.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        z = self.dec2(torch.cat((F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        z = self.dec1(torch.cat((F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        # The network learns a correction to an exposure-compensated image.
        base = (x * self.gain).clamp(0.01, 0.99)
        result = torch.sigmoid(torch.logit(base) + self.head(z))
        return result[..., :h, :w]
