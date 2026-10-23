"""
Architecture profiler — Module 3.

Parameters, multiply-accumulates, the tensor downsampling cascade, and
measured latency. Enough to account for what each backbone costs and why,
rather than quoting figures from a paper.

**Why MACs and not FLOPs.** One multiply-accumulate is conventionally counted
as a single operation here, so these numbers are roughly half what a tool
reporting FLOPs would print for the same model. Both conventions are in use
and neither is wrong; what matters is saying which one is on the slide.

**Why measured latency as well as MACs.** They disagree, and the disagreement
is the interesting part. MobileNetV2 has far fewer MACs than ResNet-18 but is
nowhere near proportionally faster, because depthwise convolutions have poor
arithmetic intensity: they move a lot of memory per unit of maths and leave
the GPU's compute units idle. A model is not fast because it is small, it is
fast because the hardware likes its shape. That gap is exactly why Module 2's
Pareto frontier is plotted against measured milliseconds.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn

from .zoo import build_backbone


def _mac_hook(store: list[int]):
    def hook(module: nn.Module, inputs, output) -> None:
        if isinstance(module, nn.Conv2d):
            # One MAC per output element per input channel per kernel cell.
            # Dividing by `groups` is what makes a depthwise convolution cheap.
            out_elements = output.numel() // output.shape[0]
            kernel = module.kernel_size[0] * module.kernel_size[1]
            store.append(out_elements * kernel * module.in_channels // module.groups)
        elif isinstance(module, nn.Linear):
            store.append(module.in_features * module.out_features)
    return hook


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Multiply-accumulates for one forward pass at `input_shape`."""
    store: list[int] = []
    handles = [
        module.register_forward_hook(_mac_hook(store))
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(*input_shape))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return int(sum(store))


def layer_cascade(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
) -> list[dict]:
    """Output shape after each spatial stage: the 224-to-7 progression.

    Only modules that change the spatial extent are reported. Listing every
    batch-norm and activation would bury the one thing the cascade is meant to
    show, which is where resolution is thrown away and channels bought with it.
    """
    stages: list[dict] = []
    last_spatial: int | None = None

    def hook(module: nn.Module, inputs, output) -> None:
        nonlocal last_spatial
        if not isinstance(output, torch.Tensor):
            return
        shape = tuple(output.shape)
        spatial = shape[-1] if len(shape) == 4 else None
        if spatial is not None and spatial != last_spatial:
            stages.append({
                "type": module.__class__.__name__,
                "output": list(shape),
                "channels": int(shape[1]),
                "spatial": int(spatial),
            })
            last_spatial = spatial

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d))
    ]
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(*input_shape))
    finally:
        for handle in handles:
            handle.remove()
    return stages


def measure_latency(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    device: str = "cpu",
    runs: int = 30,
    warmup: int = 5,
) -> dict[str, float]:
    """Per-forward latency in milliseconds.

    Warm-up runs are discarded and CUDA is synchronised around each timing,
    because kernel launches are asynchronous: without the synchronise this
    measures how fast Python can queue work, which on a GPU is an impressively
    small and completely meaningless number.
    """
    model = model.to(device).eval()
    sample = torch.zeros(*input_shape, device=device)
    cuda = device.startswith("cuda")

    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if cuda:
            torch.cuda.synchronize()

        timings: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            model(sample)
            if cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    def percentile(p: float) -> float:
        index = min(len(timings) - 1, int(round(p * (len(timings) - 1))))
        return float(timings[index])

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": float(sum(timings) / len(timings)),
    }


def profile(
    name: str,
    batch_size: int = 1,
    device: str | None = None,
    runs: int = 30,
    warmup: int = 5,
) -> dict:
    """Full Module 3 report for one backbone."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(name, pretrained=False)
    shape = (batch_size, 3, 224, 224)

    parameters = sum(p.numel() for p in model.parameters())
    macs = count_macs(model, shape)
    cascade = layer_cascade(model, shape)
    latency = measure_latency(model, shape, device=device, runs=runs, warmup=warmup)

    return {
        "name": name,
        "device": device,
        "batch_size": batch_size,
        "parameters": int(parameters),
        "macs": int(macs),
        "macs_per_parameter": round(macs / max(1, parameters), 1),
        "latency_ms": latency,
        "cascade": cascade,
        "stages": len(cascade),
    }
