"""Jedna rodzina bloków residualnych, dwie głowice dla dwóch zadań."""

import math
import torch
from torch import nn

from .common import CONDITIONS


class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(width, width, 3, padding=1),
                                  nn.ReLU(inplace=True), nn.Conv2d(width, width, 3, padding=1))

    def forward(self, x):
        return x + 0.1 * self.body(x)


class Restorer(nn.Module):
    def __init__(self, condition, width=32, blocks=6):
        super().__init__()
        task, value = CONDITIONS[condition]
        self.condition = condition
        self.head = nn.Conv2d(3, width, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(width) for _ in range(blocks)),
                                  nn.Conv2d(width, width, 3, padding=1))
        up = []
        if task == "sr":
            for _ in range(int(math.log2(value))):
                up.extend([nn.Conv2d(width, width * 4, 3, padding=1),
                           nn.PixelShuffle(2), nn.ReLU(inplace=True)])
        self.upsample = nn.Sequential(*up)
        self.tail = nn.Conv2d(width, 3, 3, padding=1)
        # Start od dokładnej metody bazowej SR albo od zaszumionego wejścia.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, image, base):
        features = self.head(image)
        residual = self.tail(self.upsample(features + self.body(features)))
        # Bez clamp w treningu: nie odcinamy gradientu przy 0 i 1.
        return base + residual


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Nieobsługiwany checkpoint: {path}")
    model = Restorer(checkpoint["condition"], **checkpoint["architecture"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
