"""Small restoration backbones adapted to a common direct-RGB sigmoid output.

NAFNet structure: Chen et al., Simple Baselines for Image Restoration, ECCV 2022.
https://github.com/megvii-research/NAFNet
Adaptations: three scales, narrow channels, native PyTorch layer norm,
zero-initialized direct-RGB head instead of the original image residual output.

NAFNet portions are adapted under the MIT License:
Copyright (c) 2022 megvii-model.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MIRNet structure: Zamir et al., Learning Enriched Features for Real Image
Restoration and Enhancement, ECCV 2020. https://github.com/swz30/MIRNet
Authors: Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat,
Fahad Shahbaz Khan, Ming-Hsuan Yang, Ling Shao.
Adaptations: one residual group, one three-scale/two-column MSRB, narrow
channels, independent PReLU per unit, common direct-RGB head.

MIRNet portions are modified under its ACADEMIC PUBLIC LICENSE:
MIRNet is free for use in noncommercial settings: at academic institutions
for teaching and research use, and at non-profit research organizations.
You can use MIRNet in your research, academic work, non-commercial work,
projects and personal work. We only ask you to credit us appropriately.
You have the right to use the software, to distribute copies, to receive
source code, to change the software and distribute your modifications
or the modified software. If you distribute verbatim or modified copies
of this software, they must be distributed under this license.
This license guarantees that you're safe when using MIRNet in your work,
for teaching or research. This license guarantees that MIRNet will remain
available free of charge for nonprofit use. You can modify MIRNet to your
purposes, and you can also share your modifications. Commercial use is not
permitted by this license; contact the original authors for other uses.
License and copyright notices must be retained. No liability or warranty.
https://github.com/swz30/MIRNet/blob/master/LICENSE.md
"""
import torch
from torch import nn
from torch.nn import functional as F

from unet_variants import ExperimentalUNet


def zero_head(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def pad_to(x, factor):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % factor, 0, (-h) % factor), mode="replicate")


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        # Normalization is across channels at each pixel, not across H/W.
        result = F.layer_norm(x.float().permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias, 1e-6)
        return result.permute(0, 3, 1, 2).to(x.dtype)


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = ChannelLayerNorm(channels), ChannelLayerNorm(channels)
        self.expand = nn.Conv2d(channels, channels*2, 1)
        self.depthwise = nn.Conv2d(channels*2, channels*2, 3, padding=1, groups=channels*2)
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.ffn_expand = nn.Conv2d(channels, channels*2, 1)
        self.ffn_project = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.gate(self.depthwise(self.expand(self.norm1(x))))
        y = x + self.beta * self.project(y * self.channel_attention(y))
        return y + self.gamma * self.ffn_project(self.gate(self.ffn_expand(self.norm2(y))))


class SmallNAFNet(nn.Module):
    def __init__(self, width=30, encoder_blocks=(1, 1, 2), middle_blocks=1):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in encoder_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(channels) for _ in range(count)]))
            self.downs.append(nn.Conv2d(channels, channels*2, 2, stride=2))
            channels *= 2
        self.middle = nn.Sequential(*[NAFBlock(channels) for _ in range(middle_blocks)])
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for _ in encoder_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels*2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(NAFBlock(channels))
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        zero_head(self.head)
        self.factor = 2**len(encoder_blocks)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.intro(pad_to(x, self.factor))
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            y = encoder(y)
            skips.append(y)
            y = down(y)
        y = self.middle(y)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            y = decoder(up(y) + skip)
        return torch.sigmoid(self.head(y))[..., :h, :w]


class DualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.PReLU(),
                                  nn.Conv2d(channels, channels, 3, padding=1, bias=False))
        hidden = max(1, channels//8)
        self.channel = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False),
                                     nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1, bias=False), nn.Sigmoid())
        self.spatial = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.fuse = nn.Conv2d(channels*2, channels, 1, bias=False)

    def forward(self, x):
        y = self.body(x)
        descriptor = torch.cat((y.amax(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)), dim=1)
        spatial = y * torch.sigmoid(self.spatial(descriptor))
        return x + self.fuse(torch.cat((spatial, y*self.channel(y)), dim=1))


class SelectiveFusion(nn.Module):
    def __init__(self, channels, branches=3):
        super().__init__()
        hidden = max(4, channels//8)
        self.reduce = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1, bias=False), nn.PReLU())
        self.weights = nn.ModuleList([nn.Conv2d(hidden, channels, 1, bias=False) for _ in range(branches)])

    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        pooled = self.reduce(stacked.sum(dim=1))
        scores = torch.stack([layer(pooled) for layer in self.weights], dim=1).softmax(dim=1)
        return (stacked * scores).sum(dim=1)


class BlurDown(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor([1., 2., 1.])
        kernel = (kernel[:, None]*kernel[None, :])/16
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernel.to(x.dtype), stride=2, groups=self.channels)


class ResidualDown(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.top = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.PReLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.PReLU(),
                                 BlurDown(channels), nn.Conv2d(channels, channels*2, 1, bias=False))
        self.bottom = nn.Sequential(BlurDown(channels), nn.Conv2d(channels, channels*2, 1, bias=False))

    def forward(self, x):
        return self.top(x) + self.bottom(x)


class ResidualUp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.top = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.PReLU(),
                                 nn.ConvTranspose2d(channels, channels, 3, stride=2, padding=1, output_padding=1, bias=False),
                                 nn.PReLU(), nn.Conv2d(channels, channels//2, 1, bias=False))
        self.bottom = nn.Conv2d(channels, channels//2, 1, bias=False)

    def forward(self, x):
        return self.top(x) + self.bottom(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


def scale_transform(width, source, target):
    modules = []
    channels = width*2**source
    for _ in range(abs(target-source)):
        modules.append(ResidualDown(channels) if target > source else ResidualUp(channels))
        channels = channels*2 if target > source else channels//2
    return nn.Sequential(*modules)


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        # The same DAU is reused in both columns of a resolution stream,
        # following the parameter sharing in the original MIRNet implementation.
        self.units = nn.ModuleList([DualAttention(width*2**level) for level in range(3)])
        self.transforms = nn.ModuleDict({f"{source}_{target}": scale_transform(width, source, target)
                                         for source in range(3) for target in range(3) if source != target})
        self.fusions = nn.ModuleList([SelectiveFusion(width*2**level) for level in range(3)])
        self.last_up = nn.ModuleList([scale_transform(width, source, 0) for source in (1, 2)])
        self.output = nn.Conv2d(width, width, 3, padding=1, bias=False)

    def forward(self, x):
        features = [self.units[0](x)]
        for level in (1, 2):
            features.append(self.units[level](self.transforms[f"{level-1}_{level}"](features[-1])))
        mesh = []
        for target in range(3):
            mesh.append(self.fusions[target]([value if source == target else self.transforms[f"{source}_{target}"](value)
                                             for source, value in enumerate(features)]))
        features = [unit(value) for unit, value in zip(self.units, mesh)]
        output = self.fusions[0]([features[0], self.last_up[0](features[1]), self.last_up[1](features[2])])
        return x + self.output(output)


class SmallMIRNet(nn.Module):
    def __init__(self, width=28):
        super().__init__()
        self.intro = nn.Conv2d(3, width, 3, padding=1, bias=False)
        self.group = nn.Sequential(MultiScaleResidualBlock(width), nn.Conv2d(width, width, 3, padding=1, bias=False))
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        zero_head(self.head)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = self.intro(pad_to(x, 4))
        y = y + self.group(y)
        return torch.sigmoid(self.head(y))[..., :h, :w]


def build_architecture(args, spec):
    name = spec["architecture"]
    if name in ("unet", "resunet"):
        return ExperimentalUNet(spec["width"], args.ev, args.gamma, output="direct", residual=name == "resunet")
    if name == "nafnet":
        return SmallNAFNet(spec["width"])
    if name == "mirnet":
        return SmallMIRNet(spec["width"])
    raise ValueError(f"Unknown architecture: {name}")
