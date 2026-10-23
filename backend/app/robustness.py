"""
Robustness under degradation — Modules 4 and 5 together.

The synthetic classes turn out to be near-perfectly separable on clean crops:
every backbone scores above 0.99 macro-F1, and even the deliberately
ambiguous regions barely dent it. That is a real finding about the generator
rather than a flattering one about the model, and it leaves the confusion
matrix with nothing to show.

What still discriminates is **robustness**. A phone photograph is degraded,
and the question that matters for an edge system is how much accuracy each
backbone gives up as the input gets worse, and which degradation costs most.
Both are measured here, over the already-trained checkpoints, with no
retraining:

* a **severity sweep**, macro-F1 against augmentation strength, which is the
  curve for the deck and produces a confusion matrix with real off-diagonal
  mass at the hard end;
* a **per-augmentation ablation**, one kernel at a time at fixed severity,
  which answers "what actually hurts" rather than "augmentation hurts".

The honest framing for the viva: clean accuracy says the task is easy, and
the degradation curve says which model to ship.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data.augment import PHOTOMETRIC_ORDER, AugmentationBank
from .data.dataset import IMAGENET_MEAN, IMAGENET_STD
from .metrics.confusion import MetricCore


def _to_tensor(crops: np.ndarray) -> torch.Tensor:
    x = crops.astype(np.float32) / 255.0
    x = np.repeat(x[:, :, :, None], 3, axis=3)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


@torch.no_grad()
def _score(
    model,
    crops: np.ndarray,
    labels: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> MetricCore:
    core = MetricCore()
    for start in range(0, len(crops), batch_size):
        stop = min(start + batch_size, len(crops))
        x = _to_tensor(crops[start:stop]).to(device)
        probabilities = torch.softmax(model(x), dim=1).cpu().numpy()
        core.update(probabilities, labels[start:stop])
    return core


def _degrade(
    crops: np.ndarray,
    severity: float,
    seed: int,
    only: str | None = None,
) -> np.ndarray:
    """Apply augmentation at a fixed severity, optionally just one kernel."""
    if severity <= 0:
        return crops
    if only is None:
        bank = AugmentationBank(severity_range=(severity, severity))
        return np.stack([
            bank.apply(crop, np.random.default_rng((seed, i)))[0]
            for i, crop in enumerate(crops)
        ])
    kernel = next(a for a in PHOTOMETRIC_ORDER if a.name == only)
    return np.stack([
        kernel.apply(crop, np.random.default_rng((seed, i)), severity)
        for i, crop in enumerate(crops)
    ])


def severity_sweep(
    checkpoint: str | Path,
    data_root: str | Path = "data/dataset",
    split: str = "test",
    severities: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    limit: int | None = None,
    seed: int = 0,
) -> dict:
    """macro-F1 against augmentation severity."""
    from .evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(checkpoint, device=device)
    root = Path(data_root)
    crops = np.load(root / "crops" / f"{split}_crops.npy")
    labels = np.load(root / "crops" / f"{split}_labels.npy")
    if limit:
        crops, labels = crops[:limit], labels[:limit]

    points = []
    matrices = {}
    for severity in severities:
        core = _score(model, _degrade(crops, severity, seed), labels, device)
        points.append({
            "severity": severity,
            "macro_f1": core.macro_f1(),
            "accuracy": core.accuracy(),
            "ece": core.ece(),
            "held_rate": core.held_rate(),
        })
        matrices[str(severity)] = core.confusion_row_normalised().tolist()

    clean = points[0]["macro_f1"]
    worst = points[-1]["macro_f1"]
    return {
        "backbone": model.hparams.backbone,
        "split": split,
        "samples": int(len(labels)),
        "points": points,
        "confusion": matrices,
        "clean_macro_f1": clean,
        "worst_macro_f1": worst,
        "robustness_drop": clean - worst,
    }


def augmentation_ablation(
    checkpoint: str | Path,
    data_root: str | Path = "data/dataset",
    split: str = "test",
    severity: float = 1.0,
    limit: int | None = None,
    seed: int = 0,
) -> dict:
    """One kernel at a time: which degradation actually costs accuracy."""
    from .evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(checkpoint, device=device)
    root = Path(data_root)
    crops = np.load(root / "crops" / f"{split}_crops.npy")
    labels = np.load(root / "crops" / f"{split}_labels.npy")
    if limit:
        crops, labels = crops[:limit], labels[:limit]

    baseline = _score(model, crops, labels, device).macro_f1()
    rows = []
    for kernel in PHOTOMETRIC_ORDER:
        core = _score(model, _degrade(crops, severity, seed, only=kernel.name),
                      labels, device)
        rows.append({
            "augmentation": kernel.name,
            "macro_f1": core.macro_f1(),
            "drop": baseline - core.macro_f1(),
        })
    rows.sort(key=lambda r: -r["drop"])
    return {
        "backbone": model.hparams.backbone,
        "severity": severity,
        "baseline_macro_f1": baseline,
        "rows": rows,
        "worst": rows[0]["augmentation"] if rows else None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Robustness under degradation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sweep = severity_sweep(args.checkpoint, args.data_root, limit=args.limit)
    ablation = augmentation_ablation(args.checkpoint, args.data_root, limit=args.limit)

    print(f"\n{sweep['backbone']}  ({sweep['samples']} crops)")
    print("  severity  macro-F1   ECE")
    for point in sweep["points"]:
        print(f"    {point['severity']:4.2f}    {point['macro_f1']:.4f}  {point['ece']:.4f}")
    print(f"  drop clean->worst: {sweep['robustness_drop']:.4f}")

    print("\n  augmentation ablation at severity 1.0, worst first")
    for row in ablation["rows"]:
        print(f"    {row['augmentation']:14} {row['macro_f1']:.4f}  {-row['drop']:+.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"sweep": sweep, "ablation": ablation}, indent=1))


if __name__ == "__main__":
    main()
