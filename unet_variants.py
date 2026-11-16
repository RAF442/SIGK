"""Controlled U-Net ablations, keeping the original model/checkpoints intact."""
import torch
from torch import nn
from torch.nn import functional as F

from model import UNet


class FeatureBlock(nn.Module):
    def __init__(self, body, inputs, outputs, residual=False, attention=False):
        super().__init__()
        self.body = body
        self.shortcut = (nn.Identity() if inputs == outputs else nn.Conv2d(inputs, outputs, 1)) if residual else None
        hidden = max(4, outputs // 8)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(outputs, hidden, 1),
            nn.ReLU(inplace=True), nn.Conv2d(hidden, outputs, 1), nn.Sigmoid(),
        ) if attention else None

    def forward(self, x):
        y = self.body(x)
        if self.shortcut is not None:
            y = y + self.shortcut(x)
        if self.attention is not None:
            y = y * self.attention(y)
        return y


class ExperimentalUNet(UNet):
    def __init__(self, width=24, ev=-3.0, gamma=2.2, output="logit", activation="leaky_relu",
                 residual=False, attention=False):
        # Construct all original convolutions first, preserving their initialization.
        super().__init__(width, ev, gamma)
        if output not in ("logit", "direct", "rgb_residual"):
            raise ValueError(f"Unknown output: {output}")
        if activation not in ("leaky_relu", "prelu", "silu"):
            raise ValueError(f"Unknown activation: {activation}")
        self.output_mode = output
        blocks = ("enc1", "enc2", "enc3", "bridge", "dec3", "dec2", "dec1")
        for name in blocks:
            body = getattr(self, name)
            inputs, outputs = body[0].in_channels, body[0].out_channels
            if activation != "leaky_relu":
                for index in (1, 3):
                    body[index] = nn.PReLU(outputs, init=0.1) if activation == "prelu" else nn.SiLU(inplace=True)
            if residual or attention:
                setattr(self, name, FeatureBlock(body, inputs, outputs, residual, attention))

    def forward(self, x):
        if self.output_mode == "logit":
            return super().forward(x)
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2))
        c = self.enc3(F.avg_pool2d(b, 2))
        z = self.bridge(F.avg_pool2d(c, 2))
        z = self.dec3(torch.cat((F.interpolate(z, size=c.shape[-2:], mode="bilinear", align_corners=False), c), 1))
        z = self.dec2(torch.cat((F.interpolate(z, size=b.shape[-2:], mode="bilinear", align_corners=False), b), 1))
        z = self.dec1(torch.cat((F.interpolate(z, size=a.shape[-2:], mode="bilinear", align_corners=False), a), 1))
        correction = self.head(z)
        result = torch.sigmoid(correction) if self.output_mode == "direct" else (
            (x * self.gain).clamp(0.01, 0.99) + correction).clamp(0, 1)
        return result[..., :h, :w]


def build_model(args, spec):
    return ExperimentalUNet(args.width, args.ev, args.gamma, spec["output"], spec["activation"],
                            spec["residual"], spec["attention"])
