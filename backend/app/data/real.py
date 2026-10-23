"""
Real pages, labelled by hand, and the dataset built from them.

The synthetic generator got the classifier to 0.99 on synthetic pages, which
says the synthetic classes are separable and nothing about your handwriting.
This module is the way out: pages you photograph are stored here, you draw or
correct a box around each region and give it a class, and those labels become
a dataset in exactly the layout `PalimpsestDataModule` already reads, so the
same training code fine-tunes on it.

**The page stored is the preprocessed one**, deskewed and resized to the
working size, and boxes are in its pixels. Compile runs the same
preprocessing, so a box drawn while labelling and a box found while compiling
mean the same thing, and crops are cut by the same recipe the classifier was
trained on (4% context, greyscale, area-resampled to 224).

**Splits are by page, fixed by the page id.** Crops from one page never land
in two splits, so a test score is never a model recognising a page it has
already seen, and adding page 150 never moves pages 1 to 149 between splits.

**Validation and test are real pages only.** Training mixes in synthetic
crops so the model does not forget the classes you have labelled least of,
but the number that decides whether a fine-tune is kept is measured on your
own handwriting and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import CLASS_NAMES

CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
# Overridable so tests and dry runs never touch the pages you labelled.
DEFAULT_ROOT = Path(os.environ.get("PALIMPSEST_REAL_ROOT",
                                   Path(__file__).resolve().parents[2] / "data" / "real"))
CROP_SIZE = 224
PAD_FRAC = 0.04


def split_for_page(page_id: str) -> str:
    """70/15/15 by a stable hash of the id."""
    bucket = int(hashlib.sha1(page_id.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def cut_crop(grey: np.ndarray, box, size: int = CROP_SIZE, pad_frac: float = PAD_FRAC) -> np.ndarray | None:
    """The training crop recipe, shared by labelling, fine-tuning and compile."""
    x, y, w, h = (int(v) for v in box)
    H, W = grey.shape[:2]
    pad_x, pad_y = int(w * pad_frac), int(h * pad_frac)
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(W, x + w + pad_x), min(H, y + h + pad_y)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.resize(grey[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


@dataclass
class PageStore:
    root: Path = DEFAULT_ROOT

    def __post_init__(self):
        self.root = Path(self.root)
        (self.root / "pages").mkdir(parents=True, exist_ok=True)

    def _json(self, page_id: str) -> Path:
        return self.root / "pages" / f"{page_id}.json"

    def image_path(self, page_id: str) -> Path:
        return self.root / "pages" / f"{page_id}.png"

    def add(self, grey: np.ndarray, proposed: list[dict], source_name: str = "") -> dict:
        """Store a preprocessed page with machine-proposed regions, unlabelled."""
        page_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        cv2.imwrite(str(self.image_path(page_id)), grey)
        record = {
            "id": page_id,
            "source_name": source_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "width": int(grey.shape[1]),
            "height": int(grey.shape[0]),
            "split": split_for_page(page_id),
            "labelled": False,
            "regions": proposed,
        }
        self._json(page_id).write_text(json.dumps(record, indent=1))
        return record

    def get(self, page_id: str) -> dict:
        path = self._json(page_id)
        if not path.exists() or "/" in page_id or ".." in page_id:
            raise KeyError(page_id)
        return json.loads(path.read_text())

    def save_labels(self, page_id: str, regions: list[dict]) -> dict:
        record = self.get(page_id)
        clean = []
        for r in regions:
            if r.get("class") not in CLASS_INDEX:
                raise ValueError(f"unknown class {r.get('class')!r}")
            x, y, w, h = (int(round(v)) for v in r["box"])
            x, y = max(0, x), max(0, y)
            w, h = min(w, record["width"] - x), min(h, record["height"] - y)
            if w >= 8 and h >= 8:
                clean.append({"box": [x, y, w, h], "class": r["class"]})
        record["regions"] = clean
        record["labelled"] = True
        record["labelled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._json(page_id).write_text(json.dumps(record, indent=1))
        return record

    def delete(self, page_id: str) -> None:
        self.get(page_id)
        self._json(page_id).unlink()
        self.image_path(page_id).unlink(missing_ok=True)

    def all(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / "pages").glob("*.json"))]

    def stats(self) -> dict:
        pages = self.all()
        labelled = [p for p in pages if p["labelled"]]
        per_class = {name: {"train": 0, "val": 0, "test": 0} for name in CLASS_NAMES}
        for p in labelled:
            for r in p["regions"]:
                per_class[r["class"]][p["split"]] += 1
        return {
            "pages": len(pages),
            "labelled_pages": len(labelled),
            "pages_by_split": {s: sum(1 for p in labelled if p["split"] == s) for s in ("train", "val", "test")},
            "regions": sum(len(p["regions"]) for p in labelled),
            "per_class": per_class,
        }


def jitter(box, rng: np.random.Generator, amount: float = 0.06) -> list[int]:
    """Nudge a hand-drawn box the way the segmenter's boxes differ from it,
    so the model learns to classify a region from an imperfect box too."""
    x, y, w, h = box
    dx, dy = rng.normal(0, amount * w), rng.normal(0, amount * h)
    sw, sh = rng.normal(1, amount), rng.normal(1, amount)
    return [int(x + dx), int(y + dy), max(8, int(w * sw)), max(8, int(h * sh))]


def build_real_dataset(
    store: PageStore,
    out_root: str | Path,
    synthetic_root: str | Path | None = "data/dataset",
    jitter_copies: int = 3,
    synthetic_ratio: float = 1.0,
    seed: int = 0,
) -> dict:
    """Write `crops/{split}_{crops,labels,hard}.npy` from labelled pages.

    Train gets each real crop plus `jitter_copies` jittered versions, and
    `synthetic_ratio` times as many synthetic train crops. Val and test are
    real crops only, unjittered.
    """
    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    (out_root / "crops").mkdir(parents=True, exist_ok=True)
    buckets = {s: ([], []) for s in ("train", "val", "test")}

    for page in store.all():
        if not page["labelled"]:
            continue
        grey = cv2.imread(str(store.image_path(page["id"])), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            continue
        crops, labels = buckets[page["split"]]
        for region in page["regions"]:
            boxes = [region["box"]]
            if page["split"] == "train":
                boxes += [jitter(region["box"], rng) for _ in range(jitter_copies)]
            for box in boxes:
                crop = cut_crop(grey, box)
                if crop is not None:
                    crops.append(crop)
                    labels.append(CLASS_INDEX[region["class"]])

    real_train = len(buckets["train"][0])
    synthetic_used = 0
    if synthetic_root and synthetic_ratio > 0 and real_train:
        syn_crops = np.load(Path(synthetic_root) / "crops" / "train_crops.npy", mmap_mode="r")
        syn_labels = np.load(Path(synthetic_root) / "crops" / "train_labels.npy")
        take = min(len(syn_labels), int(real_train * synthetic_ratio))
        idx = rng.choice(len(syn_labels), size=take, replace=False)
        buckets["train"][0].extend(np.asarray(syn_crops[i]) for i in idx)
        buckets["train"][1].extend(int(syn_labels[i]) for i in idx)
        synthetic_used = take

    counts = {}
    for split, (crops, labels) in buckets.items():
        array = np.asarray(crops, np.uint8).reshape(-1, CROP_SIZE, CROP_SIZE)
        np.save(out_root / "crops" / f"{split}_crops.npy", array)
        np.save(out_root / "crops" / f"{split}_labels.npy", np.asarray(labels, np.int64))
        np.save(out_root / "crops" / f"{split}_hard.npy", np.zeros(len(labels), bool))
        counts[split] = len(labels)

    manifest = {"counts": counts, "real_train": real_train, "synthetic_train": synthetic_used,
                "jitter_copies": jitter_copies, "pages": store.stats()}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
