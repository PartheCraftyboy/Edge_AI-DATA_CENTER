"""
Evaluation — Module 5's reporting half.

Runs a trained checkpoint over a split and produces everything the deck and
the live monitor need: confusion matrix, macro-F1, per-class calibration,
held rates, measured per-crop latency, and the same figures again split by
difficulty.

**The difficulty split is the point.** An overall macro-F1 of 0.99 on a set
that is 60% unambiguous regions says almost nothing: it is dominated by the
easy majority. Reporting "0.99 overall, 1.00 on clear regions, 0.97 on
ambiguous ones" says where the pipeline will actually struggle, and it is the
difference between a number and a finding.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import CLASS_NAMES
from .data.dataset import CropDataset
from .metrics.confusion import MetricCore
from .models.classifier import RegionClassifier


def load_checkpoint(path: str | Path, device: str | None = None) -> RegionClassifier:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RegionClassifier.load_from_checkpoint(str(path), map_location=device)
    return model.to(device).eval()


@torch.no_grad()
def run_split(
    model: RegionClassifier,
    data_root: str | Path,
    split: str = "test",
    batch_size: int = 64,
    device: str | None = None,
) -> dict:
    """Infer over one split, timing each batch."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(data_root)
    dataset = CropDataset(
        root / "crops" / f"{split}_crops.npy",
        root / "crops" / f"{split}_labels.npy",
        augment=False,
    )
    hard_path = root / "crops" / f"{split}_hard.npy"
    hard = np.load(hard_path) if hard_path.exists() else np.zeros(len(dataset), bool)

    probabilities: list[np.ndarray] = []
    targets: list[int] = []
    per_crop_ms: list[float] = []

    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        x = torch.stack([item[0] for item in items]).to(device)
        y = [item[1] for item in items]

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        began = time.perf_counter()
        logits = model(x)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - began) * 1000.0

        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.extend(y)
        per_crop_ms.extend([elapsed_ms / len(items)] * len(items))

    return {
        "probabilities": np.concatenate(probabilities),
        "targets": np.asarray(targets),
        "hard": np.asarray(hard[: len(targets)]),
        "latency_ms": np.asarray(per_crop_ms),
        "device": device,
    }


def _core_from(raw: dict, mask: np.ndarray | None = None) -> MetricCore:
    core = MetricCore()
    probabilities, targets, latency = raw["probabilities"], raw["targets"], raw["latency_ms"]
    if mask is not None:
        probabilities, targets, latency = probabilities[mask], targets[mask], latency[mask]
    for p, t, ms in zip(probabilities, targets, latency):
        core.update(p[None], np.array([t]), latency_ms=float(ms))
    return core


def evaluate(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "test",
    batch_size: int = 64,
) -> dict:
    """Full report for one checkpoint on one split."""
    model = load_checkpoint(checkpoint)
    raw = run_split(model, data_root, split=split, batch_size=batch_size)

    overall = _core_from(raw)
    hard_mask = raw["hard"].astype(bool)
    report = {
        "backbone": model.hparams.backbone,
        "split": split,
        "device": raw["device"],
        "overall": overall.report(),
    }
    if hard_mask.any() and (~hard_mask).any():
        report["easy"] = _core_from(raw, ~hard_mask).report()
        report["hard"] = _core_from(raw, hard_mask).report()
        report["difficulty_gap"] = (
            report["easy"]["macro_f1"] - report["hard"]["macro_f1"]
        )
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = evaluate(args.checkpoint, args.data_root, split=args.split)
    summary = {
        "backbone": report["backbone"],
        "macro_f1": round(report["overall"]["macro_f1"], 4),
        "accuracy": round(report["overall"]["accuracy"], 4),
        "ece": round(report["overall"]["ece"], 4),
        "p50_ms": round(report["overall"]["latency_ms"]["p50"], 3),
    }
    if "hard" in report:
        summary["easy_macro_f1"] = round(report["easy"]["macro_f1"], 4)
        summary["hard_macro_f1"] = round(report["hard"]["macro_f1"], 4)
        summary["difficulty_gap"] = round(report["difficulty_gap"], 4)
    print(json.dumps(summary, indent=1))

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))

    matrix = np.asarray(report["overall"]["confusion_row_normalised"])
    header = "      " + "".join(f"{n[:4]:>7}" for n in CLASS_NAMES)
    print("\nconfusion, row-normalised (rows = truth):")
    print(header)
    for name, row in zip(CLASS_NAMES, matrix):
        print(f"{name[:5]:>5} " + "".join(f"{v:7.3f}" for v in row))


if __name__ == "__main__":
    main()
