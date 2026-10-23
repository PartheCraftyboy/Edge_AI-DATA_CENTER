"""
The dataset: generated pages on disk, region crops in memory, split by page.

Three decisions are encoded here and each one is a place where the convenient
choice would quietly corrupt the reported numbers.

**The split is by page, never by crop.** Every crop cut from one page shares
that page's paper texture, ruling, ink darkness and camera angle. Splitting
crops at random puts siblings on both sides of the boundary, and the model
scores well on validation by recognising the page rather than the region. The
split is therefore a function of the page seed alone, so a page and all its
crops land together by construction.

**Geometry is baked at generation, photometry at training.** A camera applies
one perspective to a whole sheet. The homography is drawn once per page and
the region boxes are pushed through the same matrix, so the labels stay exact.
The lighting, blur, noise and compression vary per crop per epoch, where they
are cheap and genuinely different each time.

**Pages are written to disk as well as crops.** The crops are what trains, but
a folder of PNGs with JSON sidecars is what can be opened and looked at, and
on this project looking at the output has caught two defects that every
assertion passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .. import CLASS_INDEX, NUM_CLASSES
from .augment import AugmentationBank, camera_homography, warp_page
from .ink import InkProvider, MixedInk, available_providers
from .synth import PageGenerator

CROP_SIZE = 224

# Three splits, not two. Validation fits thresholds and stops training; test is
# read exactly once, at the end. Fitting thresholds on test and then reporting
# test accuracy would make the headline number meaningless.
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# ImageNet statistics. The backbones are ImageNet-initialised, so feeding them
# data centred any other way throws away part of what the pretraining bought.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def split_for_seed(seed: int, total: int) -> str:
    """Which split a page belongs to, decided by its seed alone.

    Deterministic and contiguous: a page cannot migrate between splits when the
    dataset is regenerated, and `test` is the same set of pages every time.
    """
    train_end = int(total * SPLIT_FRACTIONS["train"])
    val_end = train_end + int(total * SPLIT_FRACTIONS["val"])
    index = seed % total
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def jitter_box(
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
    amount: float,
) -> tuple[int, int, int, int]:
    """Perturb a ground-truth box the way an imperfect segmenter would.

    At inference the classifier never sees a perfect box: it sees whatever
    `pipeline/segment.py` proposes, which clips an edge, swallows a strip of
    the neighbouring region, or is offset by a few pixels. Training on exact
    boxes alone teaches the model that region edges are always clean.

    `amount` is the jitter strength as a fraction of the box size, 0 meaning
    no change. Returns `(x, y, w, h)`; clamping to the page is done by the
    caller.
    """
    # TODO(Parth): implement the jitter policy (5-10 lines).
    return box


def build_dataset(
    root: Path,
    n_pages: int = 1200,
    ink: InkProvider | None = None,
    warp_severity: float = 0.7,
    hard_fraction: float = 0.4,
    save_pages: int = 60,
    seed: int = 0,
    crop_size: int = CROP_SIZE,
    realism: bool = False,
    cross_rate: float = 0.0,
    box_jitter: float = 0.0,
) -> dict:
    """Generate pages, warp them as a camera would, and bank their crops.

    `save_pages` PNGs plus sidecars are written for inspection; every page
    contributes its crops regardless.
    """
    root = Path(root)
    (root / "pages").mkdir(parents=True, exist_ok=True)
    (root / "crops").mkdir(parents=True, exist_ok=True)

    if ink is None:
        providers = available_providers()
        ink = providers.get("corpus") or providers["procedural"]

    if cross_rate > 0:
        ink = MixedInk(ink, cross_rate=cross_rate)
    generator = PageGenerator(ink=ink, hard_fraction=hard_fraction, realism=realism)
    buckets: dict[str, dict[str, list]] = {
        name: {"crops": [], "labels": [], "hard": [], "seeds": []}
        for name in SPLIT_FRACTIONS
    }

    for i in range(n_pages):
        page_seed = seed * 1_000_000 + i
        page = generator.page(page_seed)
        split = split_for_seed(i, n_pages)

        # One camera per page, applied to pixels and boxes together.
        rng = np.random.default_rng(page_seed ^ 0x5EED)
        H = camera_homography(page.image.shape, rng, warp_severity)
        image, boxes = warp_page(page.image, H, [r.box for r in page.regions])

        height, width = image.shape[:2]
        for region, box in zip(page.regions, boxes):
            x, y, w, h = jitter_box(box, rng, box_jitter) if box_jitter > 0 else box
            pad_x, pad_y = int(w * 0.04), int(h * 0.04)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = cv2.resize(image[y0:y1, x0:x1], (crop_size, crop_size),
                              interpolation=cv2.INTER_AREA)
            buckets[split]["crops"].append(crop)
            buckets[split]["labels"].append(CLASS_INDEX[region.region_class])
            # Kept per crop so accuracy can be split by difficulty, which
            # turns one number into "97% on clear regions, 74% on ambiguous
            # ones" — a far more useful statement about where this fails.
            buckets[split]["hard"].append(bool(region.hard))

        buckets[split]["seeds"].append(page_seed)

        if i < save_pages:
            stem = f"page_{page_seed:08d}"
            cv2.imwrite(str(root / "pages" / f"{stem}.png"), image)
            sidecar = page.sidecar()
            sidecar["split"] = split
            sidecar["warp_severity"] = warp_severity
            sidecar["regions"] = [
                {"class": r.region_class, "box": [int(v) for v in b]}
                for r, b in zip(page.regions, boxes)
            ]
            (root / "pages" / f"{stem}.json").write_text(json.dumps(sidecar, indent=1))

    paths: dict[str, str] = {}
    counts: dict[str, int] = {}
    page_seeds: dict[str, list[int]] = {}
    for name, bucket in buckets.items():
        crops = np.asarray(bucket["crops"], dtype=np.uint8)
        labels = np.asarray(bucket["labels"], dtype=np.int64)
        hard = np.asarray(bucket["hard"], dtype=bool)
        crop_path = root / "crops" / f"{name}_crops.npy"
        label_path = root / "crops" / f"{name}_labels.npy"
        hard_path = root / "crops" / f"{name}_hard.npy"
        np.save(crop_path, crops)
        np.save(label_path, labels)
        np.save(hard_path, hard)
        paths[f"{name}_crops"] = str(crop_path)
        paths[f"{name}_labels"] = str(label_path)
        paths[f"{name}_hard"] = str(hard_path)
        counts[name] = int(len(labels))
        page_seeds[name] = bucket["seeds"]

    manifest = {
        "root": str(root),
        "n_pages": n_pages,
        "ink_provider": ink.name,
        "warp_severity": warp_severity,
        "hard_fraction": hard_fraction,
        "realism": realism,
        "cross_rate": cross_rate,
        "box_jitter": box_jitter,
        "crop_size": crop_size,
        "counts": counts,
        "hard_counts": {
            name: int(np.load(paths[f"{name}_hard"]).sum()) for name in SPLIT_FRACTIONS
        },
        "paths": paths,
        "class_balance": {
            name: np.bincount(np.load(paths[f"{name}_labels"]), minlength=NUM_CLASSES).tolist()
            for name in SPLIT_FRACTIONS
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return {**manifest, "page_seeds": page_seeds}


class CropDataset(Dataset):
    """Region crops as tensors, with photometric augmentation on train only.

    Augmentation is seeded by `(seed, epoch, index)` rather than by a shared
    generator, so a crop is degraded identically on a given epoch no matter
    which worker process loads it, and differently on the next epoch. A shared
    generator would make results depend on worker scheduling, which is the kind
    of irreproducibility that is very hard to notice and impossible to debug.
    """

    def __init__(
        self,
        crops_path: str | Path,
        labels_path: str | Path,
        augment: bool = False,
        seed: int = 0,
        bank: AugmentationBank | None = None,
    ):
        self.crops = np.load(crops_path, mmap_mode="r")
        self.labels = np.load(labels_path)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.bank = bank or AugmentationBank()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, index: int):
        crop = np.asarray(self.crops[index])
        if self.augment:
            rng = np.random.default_rng((self.seed, self.epoch, index))
            crop, _ = self.bank.apply(crop, rng)

        x = crop.astype(np.float32) / 255.0
        x = np.repeat(x[:, :, None], 3, axis=2)          # grey to 3 channels
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return tensor, int(self.labels[index])


class PalimpsestDataModule(L.LightningDataModule):
    """Lightning data module over a built dataset directory."""

    def __init__(
        self,
        root: str | Path,
        batch_size: int = 64,
        num_workers: int = 4,
        seed: int = 0,
    ):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_set: CropDataset | None = None
        self.val_set: CropDataset | None = None
        self.test_set: CropDataset | None = None

    def _make(self, name: str, augment: bool) -> CropDataset:
        crops = self.root / "crops" / f"{name}_crops.npy"
        labels = self.root / "crops" / f"{name}_labels.npy"
        return CropDataset(crops, labels, augment=augment, seed=self.seed)

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_set = self._make("train", augment=True)
            self.val_set = self._make("val", augment=False)
        if stage in (None, "test"):
            self.test_set = self._make("test", augment=False)

    def _loader(self, dataset: CropDataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate pages and bank their crops.")
    parser.add_argument("--root", default="data/dataset")
    parser.add_argument("--pages", type=int, default=1200)
    parser.add_argument("--hard-fraction", type=float, default=0.4)
    parser.add_argument("--realism", action="store_true",
                        help="draw cross-class confusers (underlines, boxed answers, axis numbers)")
    parser.add_argument("--cross-rate", type=float, default=0.0,
                        help="share of marks drawn from another class's ink, via MixedInk")
    parser.add_argument("--box-jitter", type=float, default=0.0,
                        help="segmenter imprecision, as a fraction of box size")
    args = parser.parse_args()
    manifest = build_dataset(
        Path(args.root), n_pages=args.pages, hard_fraction=args.hard_fraction,
        realism=args.realism, cross_rate=args.cross_rate, box_jitter=args.box_jitter,
    )
    manifest.pop("page_seeds", None)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
