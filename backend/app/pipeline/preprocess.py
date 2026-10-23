"""
Preprocessing kernels — Module 4, the counter-move.

The augmentation bank teaches the classifier to survive a bad photograph.
These kernels undo some of the damage before it gets there. The Module 4
lesson is that neither is free: every kernel here is timed, and its cost is
charged to the compile budget like any other stage.

    find_page  the surface the page was photographed on
    drop_slabs background that survived the crop, by stroke thickness
    deskew     rotation from a hand-held camera
    CLAHE      illumination gradient and glare
    Sauvola    uneven paper and shadow, where one global threshold fails
    denoise    speckle and JPEG noise, on the binary mask

`find_page` runs first and earns its place on any photograph that is not a
flat scan. Sauvola is a local threshold, so it has no notion of "this part of
the frame is not paper": on a notebook shot on a patterned cloth it marks the
cloth and the floor tiles as ink, they cover more of the frame than the
writing, and every region on the page ends up connected to every other one
through the background. The page then segments as a single whole-frame blob.
Cropping to the paper is what makes the rest of the pipeline meaningful.

`drop_slabs` finishes the job on the ink mask itself, and it is the part that
generalises. Cropping is geometry and a hand-held photograph does not always
have a findable page boundary; what is always true is that **handwriting is
thin and background is solid**. A pen stroke is a few pixels across wherever
the camera was. A patterned cloth or a floor tile that Sauvola called ink is a
slab many times that. Measured as a ratio to the page's own median stroke
thickness the test costs nothing and is scale-free, and on a clean page every
component sits within a third of the median, so it removes nothing at all.

Two images come out, and the split matters. Segmentation reads the
**binary** mask, because finding regions is a question of where the ink is.
The classifier and the reader get the **deskewed greyscale**, not the
binarised one, because that is what the classifier was trained on: binarising
would hand it a texture it has never seen, which is a distribution shift
introduced by the very step meant to reduce one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import threshold_sauvola

# Long side of the working image. A phone photo is 4000 px or more, and every
# kernel below scales with pixel count; the classifier sees 224 px crops and
# the synthetic pages it learnt from are 1024 px wide, so nothing is gained by
# processing more than this.
WORKING_SIDE = 1600


@dataclass
class Preprocessed:
    grey: np.ndarray
    """Deskewed greyscale. What crops are cut from."""
    ink: np.ndarray
    """Boolean mask, True where there is ink. What segmentation reads."""
    skew_deg: float
    scale: float
    """Working size over source size. Boxes are reported in working pixels."""
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timing_ms.values())


def to_grey(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def estimate_skew(grey: np.ndarray, max_deg: float = 6.0, step: float = 0.25) -> float:
    """The rotation that makes text rows sharpest, by projection profile.

    Lines of handwriting are horizontal bands of ink. Rotated to the right
    angle, the row sums alternate between dense and empty, so their variance
    peaks. Searched on a small copy, because only the angle is wanted.
    """
    small = cv2.resize(grey, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = ink.shape
    centre = (w / 2, h / 2)
    best, best_score = 0.0, -1.0
    for angle in np.arange(-max_deg, max_deg + 1e-9, step):
        rotation = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, rotation, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return best


def rotate(grey: np.ndarray, angle: float) -> np.ndarray:
    h, w = grey.shape
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Fill with the paper colour, not black: a black wedge in the corner is
    # the densest "ink" on the page and segmentation would find it first.
    paper = int(np.percentile(grey, 90))
    return cv2.warpAffine(grey, rotation, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=paper)


def find_page(grey: np.ndarray, min_frac: float = 0.20, max_frac: float = 0.98) -> np.ndarray | None:
    """Outline of the sheet of paper in a photograph, or `None` for no crop.

    The paper is the large bright region: brighter than a desk, a cloth or a
    floor, and far larger than any other bright thing in frame. So this takes
    the biggest bright blob, closes the writing back into it so the strokes do
    not punch holes in its outline, and reduces its contour to a quadrilateral.

    Two thresholds are tried, brightest first, because one Otsu split is not
    enough on these photographs. A frame holding dark cloth, a mid floor and
    a bright page is three levels, not two, and the single split lands between
    the cloth and everything else, leaving the page joined to the floor in one
    blob across the whole frame. Splitting the bright class again separates
    the paper from the floor. The plain split is kept as the fallback for the
    ordinary case of a page on a dark desk, and the first threshold whose blob
    is a plausible page wins.

    `None` when the result is not worth acting on, which covers the two cases
    that matter. Below `min_frac` of the frame the detection has latched onto
    something that is not the page and cropping to it would throw writing
    away. Above `max_frac` the whole frame already is the page, which is every
    flat scan and every page drawn on the canvas, and there is nothing to do.

    Searched on a small copy: only the corners are wanted, and the cost is
    charged to the compile budget like every other kernel here.
    """
    scale = 480 / max(grey.shape)
    small = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else grey
    blurred = cv2.GaussianBlur(small, (5, 5), 0)

    level, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds = [level]
    above = blurred[blurred > level]
    if above.size and int(above.max()) - int(above.min()) > 8:
        upper, _ = cv2.threshold(above.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholds.insert(0, upper)

    # Writing is dark, so it cuts holes in the paper blob; close them before
    # the contour is taken or the outline follows the text instead of the edge.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blob = None
    for threshold in thresholds:
        bright = cv2.morphologyEx((blurred > threshold).astype(np.uint8) * 255,
                                  cv2.MORPH_CLOSE, k, iterations=2)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        if count < 2:
            continue
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if min_frac <= stats[biggest, cv2.CC_STAT_AREA] / float(small.size) <= max_frac:
            blob = (labels == biggest).astype(np.uint8)
            break
    if blob is None:
        return None

    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # The convex hull of the blob, and deliberately not a four-corner fit of
    # it. Reducing the outline to a quadrilateral is the textbook step and it
    # is wrong here: on the notebook shots the bright blob is not a clean
    # rectangle, the best four corners cut a diagonal across it, and the fill
    # below then painted out the right-hand third of the writing. A hull
    # cannot cut into the region it was built from, so the worst it can do is
    # keep some background. Keeping background is recoverable; painting over
    # a line of someone's notes is not.
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)
    return hull / scale if scale < 1 else hull


PAGE_MARGIN = 0.04
"""Padding on the crop, as a fraction of the short side. The detected outline
follows the *brightest* part of the paper, and a page photographed at an angle
dims towards its far edge, so the outline sits inside the real page. The pad
buys that back."""


def crop_to_page(grey: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """The page's bounding box, padded, cropped out of the photograph.

    Cropping only, and deliberately not painting the surface outside the
    outline over with paper colour. Painting is the tempting version: it
    removes the wedges of cloth that a tilted page leaves inside its own
    bounding box, which is exactly the ink-coloured texture this is fighting.
    It was tried and it cut the right-hand third off a page of notes, because
    the far edge of a page shot at an angle falls into shadow, the outline
    stops at the bright part, and the fill went straight over the writing
    beyond it.

    The asymmetry decides it. Leaving background in costs some segmentation
    quality on one page. Painting over a line of someone's notes destroys it
    with no sign that anything was lost. So this keeps every pixel inside a
    generous box and accepts the wedges.
    """
    x, y, w, h = cv2.boundingRect(outline.astype(np.int32))
    pad = int(min(grey.shape) * PAGE_MARGIN)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(grey.shape[1], x + w + pad), min(grey.shape[0], y + h + pad)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return grey
    return grey[y0:y1, x0:x1]


SLAB_RATIO = 4.0
"""How many times the page's own median stroke half-thickness a blob must be
before it is called background rather than ink. Measured on the notebook
photographs, handwriting components sit between 0.8 and 1.4 times the median
and the cloth and floor slabs between 2.7 and 5.9, so the gap is wide and the
threshold is not delicate. Set at the top of the gap: keeping a slab costs one
badly segmented page, dropping a genuinely thick stroke loses a word."""


def drop_slabs(ink: np.ndarray, ratio: float = SLAB_RATIO) -> np.ndarray:
    """Ink with the solid background patches removed, by stroke thickness.

    The distance transform gives, at every ink pixel, the distance to the
    nearest paper pixel. Its maximum over a connected component is that
    component's half-thickness: about 2 px for a pen stroke at the working
    resolution, and far more for a filled patch of cloth, whatever the
    component's area or shape. Comparing each component against the median
    over the whole page makes the test independent of pen, camera and scale.
    """
    mask = ink.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return ink
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    thickness = np.zeros(count, np.float32)
    for i in range(1, count):
        x, y, w, h = stats[i, :4]
        window = distance[y:y + h, x:x + w][labels[y:y + h, x:x + w] == i]
        thickness[i] = window.max() if window.size else 0.0
    # Specks are excluded from the median for the same reason they are
    # excluded from the letter-height estimate: there are enough of them on a
    # photograph to drag it down and make real strokes look like slabs.
    real = thickness[1:][stats[1:, cv2.CC_STAT_AREA] >= 8]
    median = float(np.median(real)) if real.size else 0.0
    if median <= 0:
        return ink
    slab = np.zeros(count, bool)
    slab[1:] = thickness[1:] > ratio * median
    if not slab.any():
        return ink
    out = ink.copy()
    out[slab[labels]] = False
    return out


def preprocess(image: np.ndarray, deskew: bool = True, crop_page: bool = True) -> Preprocessed:
    timing: dict[str, float] = {}

    def timed(name: str, fn, *args, **kwargs):
        began = time.perf_counter()
        out = fn(*args, **kwargs)
        timing[name] = timing.get(name, 0.0) + (time.perf_counter() - began) * 1000
        return out

    grey = timed("load", to_grey, image)
    scale = min(1.0, WORKING_SIDE / max(grey.shape))
    if scale < 1.0:
        grey = timed("load", cv2.resize, grey, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_AREA)

    if crop_page:
        outline = timed("find_page", find_page, grey)
        if outline is not None:
            grey = timed("find_page", crop_to_page, grey, outline)

    skew = 0.0
    if deskew:
        skew = timed("deskew", estimate_skew, grey)
        if abs(skew) >= 0.25:
            grey = timed("deskew", rotate, grey, skew)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalised = timed("clahe", clahe.apply, grey)

    def sauvola(img: np.ndarray) -> np.ndarray:
        # Window about one text line tall; must be odd.
        window = max(15, (img.shape[0] // 40) | 1)
        return img < threshold_sauvola(img, window_size=window, k=0.2)

    ink = timed("sauvola", sauvola, equalised)

    def denoise(mask: np.ndarray) -> np.ndarray:
        # A 3x3 median removes isolated specks without eroding a pen stroke,
        # which is at least two pixels wide at this resolution.
        return cv2.medianBlur(mask.astype(np.uint8) * 255, 3) > 0

    ink = timed("denoise", denoise, ink)
    ink = timed("drop_slabs", drop_slabs, ink)
    return Preprocessed(grey=grey, ink=ink, skew_deg=skew, scale=scale, timing_ms=timing)
