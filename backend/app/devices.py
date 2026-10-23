"""
Who gets the GPU when the app is serving.

The laptop has one 6 GB RTX 4050 and two models that want it: the five-class
region classifier and the Qwen vision model that reads each region. Both go on
the card. Measured, the classifier costs at most ~490 MiB there (resnet18:
181 MiB of tensors plus the ~308 MiB every CUDA process pays for its context),
and qwen2.5vl:3b is estimated at 4.0-4.5 GiB with a 4096 context, which leaves
the pair inside the ~5.6 GiB that is free.

The context is most of the classifier's cost, not its weights, so if Qwen ever
starts spilling layers into system RAM, moving the classifier to the CPU is the
first lever: it frees ~0.3-0.5 GiB for ~10 ms per crop. That is one environment
variable, `PALIMPSEST_CLASSIFIER_DEVICE=cpu`, and no code change.

Do not train while serving: a training run takes 1.5-2.5 GiB on its own.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def classifier_device() -> str:
    """CUDA when present, unless `PALIMPSEST_CLASSIFIER_DEVICE` says otherwise."""
    default = "cuda" if torch.cuda.is_available() else "cpu"
    device = os.environ.get("PALIMPSEST_CLASSIFIER_DEVICE", default)
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_serving_classifier(path: str | Path):
    """The trained classifier, placed where the serving app should run it.

    Loaded with `map_location` straight onto the target device, so a
    GPU-trained checkpoint never round-trips through the wrong memory.
    """
    from .evaluate import load_checkpoint

    device = classifier_device()
    if device == "cpu":
        # Leave cores free for Ollama, which runs any layers that spilled out
        # of VRAM on the CPU too. Half the cores is plenty for one crop.
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    return load_checkpoint(path, device=device)
