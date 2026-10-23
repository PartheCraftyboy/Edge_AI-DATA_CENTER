"""
The backbone zoo — Modules 2 and 3.

Four models classify a 224x224 region crop into five classes. Three are
torchvision architectures the course taught, re-headed and fine-tuned from
ImageNet. The fourth is designed here.

**Why three pretrained backbones rather than one.** The point of Module 2 is
not that a bigger model is better, it is that the most accurate model makes
your pen feel laggy. On a page with eight regions, ResNet-18 costs roughly
350 ms where SqueezeNet costs 40, against a 2000 ms compile budget shared with
a router that is itself slow. Having all three switchable at runtime turns
that trade-off into a measured Pareto frontier instead of a sentence.

**Why a fourth model nobody pretrained.** Module 3 asks for architecture
design, and a profiler that reports somebody else's parameter counts
demonstrates reading, not designing. `PalimpsestNet` is built from the ideas
the module actually taught: depthwise separable convolution, residual skips,
and a 224-to-7 downsampling cascade. It also happens to be the cleanest
available evidence for Module 2's central claim, because it is the only entry
trained from scratch: whatever gap opens between it and the three ImageNet
backbones is the value of transfer learning, measured on this task rather than
asserted from a lecture slide.

Every head emits raw logits. The loss applies its own softmax, and a model
that also normalised would be training on a softmax of a softmax.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from .. import NUM_CLASSES


# ---------------------------------------------------------------------------
# the architecture designed for this project
# ---------------------------------------------------------------------------

class SeparableBlock(nn.Module):
    """Depthwise separable convolution, optionally residual.

    An ordinary 3x3 convolution from C_in to C_out costs `9 * C_in * C_out`
    parameters. Splitting it into a 3x3 convolution applied per channel
    (`groups=C_in`, so `9 * C_in`) followed by a 1x1 mixing convolution
    (`C_in * C_out`) costs roughly `C_in * C_out` instead. For 128 to 192
    channels that is 221k parameters against 25k, for very nearly the same
    representational job. It is the single idea that makes a small network
    possible, and it is why MobileNet exists.

    The residual skip is added only where the shape allows it, which is the
    standard condition: same channel count, stride one. A projection shortcut
    would allow it everywhere at the cost of the parameters this block exists
    to save.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride,
            padding=1, groups=in_channels, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        if self.use_residual:
            x = x + identity
        return self.act(x)


class PalimpsestNet(nn.Module):
    """A small region classifier, designed rather than imported.

    The cascade is 224 -> 112 -> 56 -> 28 -> 14 -> 7, halving spatial extent
    while roughly doubling channels, which is the shape of nearly every
    convolutional classifier and the thing Module 3 asks you to be able to
    account for. What the model has to recognise is stroke density, aspect
    ratio, baseline regularity and the presence of long straight runs, none of
    which need fine detail, so throwing away resolution early is cheap here in
    a way it would not be for, say, character recognition.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, width: int = 32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(3, w, kernel_size=3, stride=2, padding=1, bias=False),  # 112
            nn.BatchNorm2d(w),
            nn.ReLU6(inplace=True),
            SeparableBlock(w, w * 2, stride=2),      # 56
            SeparableBlock(w * 2, w * 2),            # residual
            SeparableBlock(w * 2, w * 4, stride=2),  # 28
            SeparableBlock(w * 4, w * 4),            # residual
            SeparableBlock(w * 4, w * 6, stride=2),  # 14
            SeparableBlock(w * 6, w * 6),            # residual
            SeparableBlock(w * 6, w * 8, stride=2),  # 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(w * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# torchvision backbones, re-headed to five classes
# ---------------------------------------------------------------------------

def _squeezenet(pretrained: bool, num_classes: int) -> nn.Module:
    weights = tvm.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.squeezenet1_1(weights=weights)
    # SqueezeNet classifies with a 1x1 convolution, not a linear layer.
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    model.num_classes = num_classes
    return model


def _mobilenet(pretrained: bool, num_classes: int) -> nn.Module:
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.mobilenet_v2(weights=weights)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def _resnet18(pretrained: bool, num_classes: int) -> nn.Module:
    weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _palimpsest(pretrained: bool, num_classes: int) -> nn.Module:
    # No ImageNet weights exist for an architecture that did not exist before
    # today, and that is the point of including it.
    return PalimpsestNet(num_classes=num_classes)


BACKBONES: dict[str, dict] = {
    "palimpsest_net": {
        "build": _palimpsest,
        "pretrained_available": False,
        "note": "designed for this project; depthwise separable with residual skips",
    },
    "squeezenet1_1": {
        "build": _squeezenet,
        "pretrained_available": True,
        "note": "Fire modules: squeeze 1x1 then expand 1x1 and 3x3",
    },
    "mobilenet_v2": {
        "build": _mobilenet,
        "pretrained_available": True,
        "note": "inverted residuals with linear bottlenecks",
    },
    "resnet18": {
        "build": _resnet18,
        "pretrained_available": True,
        "note": "residual blocks; the accuracy end of the frontier",
    },
}


def build_backbone(
    name: str,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """Construct one backbone, re-headed to `num_classes`."""
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; have {sorted(BACKBONES)}")
    entry = BACKBONES[name]
    want_pretrained = pretrained and entry["pretrained_available"]
    return entry["build"](want_pretrained, num_classes)


def describe(name: str, num_classes: int = NUM_CLASSES) -> dict:
    """Parameter count and provenance for one backbone.

    Shape and timing measurements live in `profiler.py`; this is the cheap
    summary that needs no forward pass.
    """
    model = build_backbone(name, pretrained=False, num_classes=num_classes)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "name": name,
        "parameters": int(total),
        "trainable": int(trainable),
        "pretrained_available": BACKBONES[name]["pretrained_available"],
        "note": BACKBONES[name]["note"],
    }
