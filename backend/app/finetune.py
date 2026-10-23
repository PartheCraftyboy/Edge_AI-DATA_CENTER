"""
Fine-tune a trained backbone on your own labelled pages.

    python -m app.finetune --backbone mobilenet_v2 --epochs 8

The loop the app exposes as its Train button: label pages in the Label view,
press Train, and this rebuilds the real-page dataset, starts from the
synthetic checkpoint in `runs/<backbone>/best.ckpt`, trains, and writes
`runs_real/<backbone>/best.ckpt`. More pages, run it again; it always starts
from the synthetic weights, so a bad batch of labels cannot compound.

**Before and after, on the same real test pages.** Both the synthetic model
and the fine-tuned one are scored on your held-out handwriting, so the
result says exactly what labelling bought. That pair of numbers, synthetic
model on real pages against fine-tuned model on real pages, is the honest
headline the synthetic 0.99 never was.

Progress goes to `runs_real/status.json`, which the app polls. It runs as its
own process so a training run cannot stall the web server.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from lightning.pytorch.callbacks import Callback

STATUS_ROOT = Path(os.environ.get("PALIMPSEST_RUNS_REAL", "runs_real"))
MIN_PAGES = {"train": 3, "val": 1}


def write_status(path: Path, **fields) -> None:
    current = json.loads(path.read_text()) if path.exists() else {}
    current.update(fields, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=1))
    tmp.replace(path)  # atomic, so the app never reads half a file


class StatusCallback(Callback):
    def __init__(self, path: Path, epochs: int):
        self.path, self.epochs = path, epochs

    def on_validation_epoch_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        write_status(self.path, state="training", epoch=trainer.current_epoch + 1,
                     epochs=self.epochs,
                     val_macro_f1=float(metrics.get("val/macro_f1", 0.0)),
                     val_loss=float(metrics.get("val/loss", 0.0)))


def score_on_real_test(checkpoint: Path, data_root: Path) -> dict | None:
    from .evaluate import evaluate

    if not (data_root / "crops" / "test_labels.npy").exists():
        return None
    import numpy as np

    if len(np.load(data_root / "crops" / "test_labels.npy")) == 0:
        return None
    report = evaluate(checkpoint, data_root, split="test")
    overall = report["overall"]
    return {"macro_f1": overall["macro_f1"], "accuracy": overall["accuracy"],
            "samples": overall["samples"], "confusion": overall["confusion"]}


def finetune(backbone: str = "mobilenet_v2", epochs: int = 8, lr: float = 1e-4,
             batch_size: int = 32, synthetic_ratio: float = 1.0) -> dict:
    from .data.real import PageStore, build_real_dataset
    from .train import train

    STATUS_ROOT.mkdir(exist_ok=True)
    status = STATUS_ROOT / "status.json"
    status.write_text("{}")
    write_status(status, state="building", backbone=backbone, started=time.strftime("%H:%M:%S"))

    store = PageStore()
    stats = store.stats()
    by_split = stats["pages_by_split"]
    for split, need in MIN_PAGES.items():
        if by_split[split] < need:
            raise ValueError(f"need at least {need} labelled {split} pages, have {by_split[split]} "
                             f"(splits are fixed per page, about 70/15/15)")

    data_root = Path(os.environ.get("PALIMPSEST_REAL_DATASET", "data/real_dataset"))
    manifest = build_real_dataset(store, data_root, synthetic_ratio=synthetic_ratio)
    base = Path("runs") / backbone / "best.ckpt"
    before = score_on_real_test(base, data_root)
    write_status(status, state="training", dataset=manifest["counts"], before=before,
                 epoch=0, epochs=epochs)

    results = train(backbone=backbone, data_root=data_root, out_dir=STATUS_ROOT,
                    epochs=epochs, batch_size=batch_size, lr=lr, num_workers=2,
                    init_checkpoint=base,
                    extra_callbacks=[StatusCallback(status, epochs)])
    after = score_on_real_test(STATUS_ROOT / backbone / "best.ckpt", data_root)

    # One row on the learning curve. The x axis is regions labelled by hand,
    # so a week of labelling plots as a saturation curve and says when to stop.
    from .history import append_run

    per_class = {name: sum(counts.values()) for name, counts in stats["per_class"].items()}
    append_run(run="finetune", round="", backbone=backbone,
               pages_labelled=stats["labelled_pages"], regions_hand=stats["regions"],
               regions_teacher=0, regions_total=stats["regions"],
               train_crops=manifest["counts"]["train"], val_crops=manifest["counts"]["val"],
               test_crops=manifest["counts"]["test"],
               hand_accuracy=(after or {}).get("accuracy"),
               hand_samples=(after or {}).get("samples"),
               macro_f1=(after or {}).get("macro_f1"),
               accuracy=(after or {}).get("accuracy"),
               epochs=epochs, lr=lr, synthetic_ratio=synthetic_ratio,
               train_seconds=results["train_seconds"],
               note=f"before macro_f1 {(before or {}).get('macro_f1')}",
               **{f"n_{name.lower()}": n for name, n in per_class.items()})

    summary = {"backbone": backbone, "before": before, "after": after,
               "dataset": manifest, "train_seconds": results["train_seconds"]}
    (STATUS_ROOT / backbone / "finetune.json").write_text(json.dumps(summary, indent=1))
    write_status(status, state="done", before=before, after=after,
                 train_seconds=results["train_seconds"])
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", default="mobilenet_v2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    args = parser.parse_args()
    try:
        summary = finetune(args.backbone, args.epochs, args.lr, synthetic_ratio=args.synthetic_ratio)
        print(json.dumps({k: summary[k] for k in ("backbone", "before", "after")}, indent=1))
    except Exception as exc:
        STATUS_ROOT.mkdir(exist_ok=True)
        write_status(STATUS_ROOT / "status.json", state="failed", error=str(exc),
                     trace=traceback.format_exc()[-1500:])
        raise


if __name__ == "__main__":
    main()
