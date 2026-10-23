"""
Training entry point — Module 5.

Device-agnostic by construction: Lightning picks CUDA when the RTX 4050 is on
the bus and CPU when it is not, and nothing in the model or the data path
changes either way.

One Lightning callback is needed that Lightning does not supply. The crop
dataset seeds its photometric augmentation from `(seed, epoch, index)` rather
than from a shared generator, so that a crop is degraded identically whichever
worker process loads it and differently on the next epoch. Nothing tells the
dataset the epoch has advanced, so `EpochSeed` does, and without it every
epoch would see the same degradations and Module 4 would contribute far less
than it appears to.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .data.dataset import PalimpsestDataModule
from .models.classifier import RegionClassifier


class EpochSeed(Callback):
    """Advance the training dataset's augmentation epoch."""

    def on_train_epoch_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        dataset = getattr(trainer.datamodule, "train_set", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(trainer.current_epoch)


def train(
    backbone: str = "mobilenet_v2",
    data_root: str | Path = "data/dataset",
    out_dir: str | Path = "runs",
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    num_workers: int = 6,
    pretrained: bool = True,
    seed: int = 0,
    extra_callbacks: list[Callback] | None = None,
    precision: str | None = None,
    init_checkpoint: str | Path | None = None,
    disagreement_penalty: float = 1.0,
) -> dict:
    """Fine-tune one backbone and return its validation and test results."""
    L.seed_everything(seed, workers=True)

    out_dir = Path(out_dir) / backbone
    out_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PalimpsestDataModule(
        data_root, batch_size=batch_size, num_workers=num_workers, seed=seed
    )
    model = RegionClassifier(
        backbone=backbone, pretrained=pretrained, lr=lr, max_epochs=epochs,
        disagreement_penalty=disagreement_penalty,
    )
    if init_checkpoint:
        # Fine-tuning: start from weights already trained on synthetic pages
        # rather than from ImageNet, so real labels refine what is known.
        state = torch.load(str(init_checkpoint), map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=False)

    checkpoint = ModelCheckpoint(
        dirpath=out_dir,
        filename="best",
        monitor="val/macro_f1",
        mode="max",
        save_top_k=1,
        # Overwrite `best.ckpt` instead of writing `best-v1.ckpt` beside it.
        # Lightning's default is to add a version counter when the filename
        # already exists, which is sensible for separate experiments and
        # wrong for a loop that trains the same backbone repeatedly: round 2
        # of a distillation run wrote `best-v1.ckpt`, the caller went on
        # reading `best.ckpt`, and so it scored round 1's model twice. The
        # two rounds reported agreement identical to sixteen decimal places,
        # the plateau check believed it, and a round that had improved
        # val macro-F1 from 0.73 to 0.82 was discarded.
        enable_version_counter=False,
    )
    callbacks: list[Callback] = [EpochSeed(), checkpoint, *(extra_callbacks or [])]

    cuda = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision or ("16-mixed" if cuda else "32-true"),
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )

    started = time.time()
    trainer.fit(model, datamodule=datamodule)
    train_seconds = time.time() - started

    # The test split is read exactly once, here, after everything is frozen.
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path="best", verbose=False)

    results = {
        "backbone": backbone,
        "pretrained": pretrained,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "disagreement_penalty": disagreement_penalty,
        "device": "cuda" if cuda else "cpu",
        "train_seconds": round(train_seconds, 1),
        "best_val_macro_f1": float(checkpoint.best_model_score or 0.0),
        "checkpoint": str(checkpoint.best_model_path),
        "test": {k: float(v) for k, v in (test_results[0] if test_results else {}).items()},
        "val_confusion": model.val_confusion.compute().tolist(),
        "test_confusion": model.test_confusion.compute().tolist(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the region classifier.")
    parser.add_argument("--backbone", default="mobilenet_v2")
    parser.add_argument("--data-root", default="data/dataset")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--scratch", action="store_true",
                        help="do not load ImageNet weights")
    args = parser.parse_args()

    results = train(
        backbone=args.backbone,
        data_root=args.data_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        pretrained=not args.scratch,
    )
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("val_confusion", "test_confusion")}, indent=1))


if __name__ == "__main__":
    main()
