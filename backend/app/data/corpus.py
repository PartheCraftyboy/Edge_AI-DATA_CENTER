"""
Real handwriting, cleaned and cached.

Two corpora feed the page generator, and both arrive in shapes that are wrong
for compositing:

* **Chars74K** (`dhruvildave/english-handwritten-characters`): 3,410 isolated
  characters, 62 classes, 55 each. Every file is a 900x1200 photograph with one
  small mark somewhere inside it, so the useful pixels are a few percent of the
  file. Clean and uncontaminated, just enormous.
* **Handwriting Recognition** (`landlord/handwriting-recognition`): ~330k crops
  of handwritten names cut from French records forms. Word-shaped, which is what
  `THEORY` needs, but roughly 40% of the crops carry machine-printed field
  labels (`NOM`, `PRENOM:`), form borders, or bleed-through from the line above.

That contamination is the reason this module exists, and it is worth being
precise about the harm. A printed glyph composited into a `THEORY` region
teaches the classifier that printed texture means prose. A form border is a long
straight rule, which is the one structural feature that defines `TABLE`. Both
are shortcut features, both survive every shape-and-range assertion, and both
would surface on submission day as off-diagonal mass with no visible cause.

We can afford to be ruthless, because the corpus holds ~330k usable rows and a
page generator needs a few thousand: compositing resamples them, so a few
thousand words produce an unbounded number of distinct pages. The filter
therefore rejects on suspicion rather than on proof, and throwing away a clean
word costs nothing.

Everything here produces **alpha masks, float32 in [0, 1], 1 is full ink**,
matching `InkProvider`. Paper colour is the compositor's business, never a
provider's.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Canvas heights for the two caches. Both are resampled again at composite time,
# so these only need to be above the largest height a page will ask for.
GLYPH_HEIGHT = 64
WORD_HEIGHT = 40

# Where the built caches live. Outside the package, beside the raw corpora, and
# gitignored: these are rebuildable artefacts, not source.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus" / "cache"


# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------

def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Otsu, returned as a boolean where True is ink.

    Otsu rather than a fixed threshold because scan exposure varies between
    source forms, and rather than Sauvola because these crops are small and
    evenly lit; local thresholding is for the photographed page, later in the
    pipeline, where it earns its cost.
    """
    _, binarised = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binarised > 0


def _alpha_from(gray: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Ink darkness as alpha, with everything Otsu called paper forced to zero.

    Keeping the greyscale ramp inside the strokes rather than taking the binary
    mask preserves the soft pen edge. Compositing a hard binary mask is one of
    the things that makes synthetic ink read as vector art.
    """
    alpha = np.clip((255.0 - gray.astype(np.float32)) / 255.0, 0.0, 1.0)
    alpha[~ink] = 0.0
    return alpha


def clean_glyph(gray: np.ndarray) -> np.ndarray | None:
    """Crop one Chars74K character to its ink. `None` if there is no ink.

    No contamination filtering: this corpus is clean, one mark per file. The
    only real work is discarding the vast empty margin.
    """
    ink = _ink_mask(gray)
    if not ink.any() or ink.mean() > 0.6:
        return None
    ys, xs = np.nonzero(ink)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if (y1 - y0) < 4 or (x1 - x0) < 2:
        return None
    return _alpha_from(gray, ink)[y0:y1, x0:x1]


def clean_symbol(gray: np.ndarray) -> np.ndarray | None:
    """Crop one handwritten mathematics symbol, rejecting scan underlines.

    About one sample in eight carries a stray horizontal rule left over from
    the form it was scanned from. Inside a `FORMULA` crop that is not noise: a
    horizontal run is what a fraction bar looks like, and fraction bars are a
    real structural feature of the class, so a spurious one teaches the
    classifier the wrong thing about the class that matters most.

    The test cannot simply be "contains a long horizontal run", because the
    `sub` class *is* a long horizontal run. So a rule only counts when it is
    **extra**: ink that survives a wide horizontal opening while substantial
    ink also exists outside it.
    """
    ink = _ink_mask(gray)
    if not ink.any() or ink.mean() > 0.6:
        return None
    h, w = ink.shape

    u8 = (ink * 255).astype(np.uint8)
    wide = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, int(w * 0.7)), 1))
    rule = cv2.morphologyEx(u8, cv2.MORPH_OPEN, wide) > 0
    if rule.any():
        remainder = int(ink.sum()) - int((ink & rule).sum())
        if remainder > 0.25 * int(ink.sum()):
            return None  # a symbol plus an underline, not a minus sign

    ys, xs = np.nonzero(ink)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if (y1 - y0) < 3 or (x1 - x0) < 3:
        return None
    return _alpha_from(gray, ink)[y0:y1, x0:x1]


def clean_word(gray: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Crop one word image to its handwriting, or reject it.

    Returns `(alpha, reason)`, where a `None` alpha carries the rejection
    reason so the build can report a breakdown rather than a bare count. If the
    acceptance rate ever climbs towards 100%, the filter has stopped filtering,
    and that is a test rather than a hope.

    The checks run cheapest-first, and each targets one observed contamination.
    """
    ink = _ink_mask(gray)
    h, w = ink.shape
    coverage = float(ink.mean())
    if coverage < 0.01:
        return None, "blank"
    if coverage > 0.35:
        return None, "too-dark"

    # --- form rules ------------------------------------------------------
    # A morphological open survives only where ink runs straight for the whole
    # length of the kernel. Printed rules do that; pen strokes slant and wobble
    # and do not. The vertical kernel spans almost the full image height on
    # purpose: a form border runs edge to edge, while even a tall letter does
    # not, so this cannot be tripped by a capital L.
    u8 = (ink * 255).astype(np.uint8)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 6), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.85))))
    if cv2.morphologyEx(u8, cv2.MORPH_OPEN, horizontal).mean() > 1.0:
        return None, "rule"
    if cv2.morphologyEx(u8, cv2.MORPH_OPEN, vertical).mean() > 1.0:
        return None, "rule"

    count, _, stats, centroids = cv2.connectedComponentsWithStats(u8, 8)
    comps = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 6]
    if len(comps) < 2:
        return None, "too-few-components"

    heights = np.array([stats[i, cv2.CC_STAT_HEIGHT] for i in comps], dtype=float)
    # The 75th percentile, not the median: printed labels are numerous enough in
    # a contaminated crop to drag a median down to their own size, at which
    # point they stop looking small and the filter misses them.
    reference = float(np.percentile(heights, 75))
    if reference < 8:
        return None, "tiny"

    # --- printed labels, by size -----------------------------------------
    # The threshold sits high because a printed field label is often only a
    # little shorter than the hand beside it: `PRENOM: SIVATA` reached a
    # rendered page while the cut was at 0.62.
    keep = [i for i, ch in zip(comps, heights) if ch >= 0.70 * reference]
    if len(keep) < 2:
        return None, "nothing-kept"

    # A *run* of dropped letter-sized marks is a second piece of text; one or
    # two are a letter the scan broke in half. Keying on `any` instead of a
    # count rejected 62% of the corpus, almost all of it broken strokes rather
    # than labels, so the count is what separates the two.
    sizeable = sum(1 for ch in heights if 0.40 * reference <= ch < 0.70 * reference)
    if sizeable >= 3:
        return None, "mixed-sizes"

    # --- labels on their own line, by vertical band ----------------------
    centres = np.array([centroids[i][1] for i in keep])
    band = float(np.median(centres))
    keep = [i for i, y in zip(keep, centres) if abs(y - band) <= 0.9 * reference]
    if len(keep) < 2:
        return None, "nothing-kept"
    if (len(comps) - len(keep)) / len(comps) > 0.45:
        return None, "mixed-sizes"

    # --- bold labels of the same height, by horizontal gap ---------------
    # A label sits apart from the writing with a gap no letter spacing matches.
    # The threshold sits well above normal tracking, because names are often
    # written with generous letter spacing and splitting one of those would
    # discard most of the corpus.
    #
    # More than one group means the crop holds a label as well as a hand, and
    # the crop is dropped rather than resolved. An earlier version kept the
    # widest group, which is a guess: when the printed label is long and the
    # name is short it picks the label, and a printed `PRENOM` duly appeared
    # composited into a rendered table. With 330k rows available and a few
    # thousand needed, discarding every ambiguous crop is close to free.
    keep.sort(key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    for previous, current in zip(keep, keep[1:]):
        prev_right = stats[previous, cv2.CC_STAT_LEFT] + stats[previous, cv2.CC_STAT_WIDTH]
        if stats[current, cv2.CC_STAT_LEFT] - prev_right > 1.6 * reference:
            return None, "ambiguous-groups"

    x0 = min(stats[i, cv2.CC_STAT_LEFT] for i in keep)
    y0 = min(stats[i, cv2.CC_STAT_TOP] for i in keep)
    x1 = max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in keep)
    y1 = max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in keep)

    # --- bleed from the neighbouring line --------------------------------
    # Ink touching the top or bottom edge is a stroke descending from the row
    # above or rising from the row below, cut off by the crop.
    if y0 <= 0 or y1 >= h:
        return None, "touches-edge"
    if (x1 - x0) < 0.12 * w:
        return None, "too-narrow"

    return _alpha_from(gray, ink)[y0:y1, x0:x1], "ok"


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

@dataclass
class CorpusCache:
    """Cleaned ink at a fixed height, stored as a sprite sheet.

    Words have wildly varying widths, and the obvious container for that is an
    array of arrays, which numpy can only store by pickling. Loading pickled
    arrays built from downloaded data is a thing to avoid, and it is slower.

    So entries are resampled to one common height and laid end to end in a
    single strip, with an offsets array marking the boundaries. One contiguous
    `uint8` allocation, `allow_pickle=False` on load, and slicing an entry is a
    view rather than a copy.
    """

    flat: np.ndarray          # uint8, (height, total_width)
    offsets: np.ndarray       # int32, (n + 1,)
    labels: list[str]
    height: int
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def widths(self) -> np.ndarray:
        """Width of every entry, derived from the offsets. No extra storage."""
        return np.diff(self.offsets)

    @property
    def aspects(self) -> np.ndarray:
        """Width over height per entry, which is how a word gets chosen."""
        return self.widths.astype(np.float64) / float(self.height)

    def __getitem__(self, i: int) -> np.ndarray:
        """Entry `i` as float32 alpha in [0, 1]."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.flat[:, a:b].astype(np.float32) / 255.0

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return self[int(rng.integers(0, len(self)))]

    def sample_with_label(self, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        i = int(rng.integers(0, len(self)))
        return self[i], self.labels[i]

    @classmethod
    def from_items(
        cls,
        items: list[np.ndarray],
        labels: list[str],
        height: int,
        meta: dict | None = None,
    ) -> CorpusCache:
        if not items:
            raise ValueError("refusing to build an empty corpus cache")
        widths = [it.shape[1] for it in items]
        offsets = np.zeros(len(items) + 1, dtype=np.int32)
        np.cumsum(widths, out=offsets[1:])
        flat = np.zeros((height, int(offsets[-1])), dtype=np.uint8)
        for i, item in enumerate(items):
            flat[:, offsets[i]:offsets[i + 1]] = np.clip(item * 255.0, 0, 255).astype(np.uint8)
        return cls(flat, offsets, list(labels), height, meta or {})

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            flat=self.flat,
            offsets=self.offsets,
            labels=np.asarray(self.labels, dtype=np.str_),
            height=np.int32(self.height),
            meta=np.str_(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> CorpusCache:
        with np.load(Path(path), allow_pickle=False) as z:
            return cls(
                flat=z["flat"],
                offsets=z["offsets"],
                labels=[str(s) for s in z["labels"]],
                height=int(z["height"]),
                meta=json.loads(str(z["meta"])),
            )


def _to_height(alpha: np.ndarray, height: int) -> np.ndarray:
    """Resample to the cache height, preserving aspect ratio.

    Aspect is load-bearing. Squashing every word to one box would hand the
    classifier a uniform letter width that real writing does not have.
    """
    h, w = alpha.shape
    width = max(2, int(round(w * height / max(1, h))))
    return cv2.resize(alpha, (width, height), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------

def build_glyph_cache(root: Path, out: Path, limit: int | None = None) -> CorpusCache:
    """Ingest Chars74K from `root` (holding `english.csv` and `Img/`)."""
    root = Path(root)
    with (root / "english.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    items: list[np.ndarray] = []
    labels: list[str] = []
    for row in rows:
        gray = cv2.imread(str(root / row["image"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        alpha = clean_glyph(gray)
        if alpha is None:
            continue
        items.append(_to_height(alpha, GLYPH_HEIGHT))
        labels.append(row["label"])

    meta = {
        "source": "dhruvildave/english-handwritten-characters (Chars74K)",
        "considered": len(rows),
        "accepted": len(items),
        "accept_rate": round(len(items) / max(1, len(rows)), 4),
    }
    cache = CorpusCache.from_items(items, labels, GLYPH_HEIGHT, meta)
    cache.save(out)
    return cache


def build_maths_cache(root: Path, out: Path, per_class: int | None = None) -> CorpusCache:
    """Ingest the handwritten mathematics corpus from a directory per class.

    Layout is `root/<class>/*.png|jpg`, where the class names are the 19 the
    corpus ships: ten digits, `add sub mul div eq dec`, and `x y z`. Labels are
    those names, because that is what the formula painter selects on: it needs
    an operator in operator position, not whichever symbol came up.
    """
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())

    items: list[np.ndarray] = []
    labels: list[str] = []
    considered = 0
    for name in classes:
        files = sorted(
            p for p in (root / name).iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if per_class:
            files = files[:per_class]
        for path in files:
            considered += 1
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            alpha = clean_symbol(gray)
            if alpha is None:
                continue
            items.append(_to_height(alpha, GLYPH_HEIGHT))
            labels.append(name)

    meta = {
        "source": "handwritten mathematics symbols",
        "classes": classes,
        "considered": considered,
        "accepted": len(items),
        "accept_rate": round(len(items) / max(1, considered), 4),
    }
    cache = CorpusCache.from_items(items, labels, GLYPH_HEIGHT, meta)
    cache.save(out)
    return cache


def build_word_cache(root: Path, out: Path, limit: int | None = None) -> CorpusCache:
    """Ingest the cleaned word subset from `root` (holding `subset.csv`, `train/`)."""
    root = Path(root)
    with (root / "subset.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    items: list[np.ndarray] = []
    labels: list[str] = []
    reasons: dict[str, int] = {}
    for row in rows:
        gray = cv2.imread(str(root / "train" / row["FILENAME"]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        alpha, reason = clean_word(gray)
        reasons[reason] = reasons.get(reason, 0) + 1
        if alpha is None:
            continue
        items.append(_to_height(alpha, WORD_HEIGHT))
        labels.append(row["IDENTITY"])

    meta = {
        "source": "landlord/handwriting-recognition",
        "considered": len(rows),
        "accepted": len(items),
        "accept_rate": round(len(items) / max(1, len(rows)), 4),
        "rejections": reasons,
    }
    cache = CorpusCache.from_items(items, labels, WORD_HEIGHT, meta)
    cache.save(out)
    return cache
