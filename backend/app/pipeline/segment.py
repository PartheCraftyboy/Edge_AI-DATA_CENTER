"""
Segmentation: page to region boxes. Classical, with no learned detector.

A deliberate boundary, stated in spec section 6: the project's trained and
measured claim is region *classification*. Detection only has to produce
crops worth classifying, and a learned detector would need box labels and a
second training run. Where this is too coarse, the review panel is the
fallback.

The method is one idea. Within a region, ink sits close together: letters
in a word, words on a line, lines in a paragraph. Between regions there is
more whitespace than that. So the ink mask is **closed** with a rectangle a
little wider than the gap between words and a little taller than the gap
between lines, which fuses each region into one blob and leaves the gaps
between regions open. Each blob's bounding box is a region.

Gap sizes are multiples of the **writing's own letter height**, not fractions
of the page. That distinction is the whole correctness of this file. A page
fraction sounds scale-free and is not: 3% of the height is four letter
heights on a phone photo of a notebook and a fifth of one on a page drawn on
the canvas, so the same setting fused an entire photographed page into one
blob and cut every canvas word into its own region. Measured across the real
pages in `data/real`, a fixed fraction swung over a 20x range in the only
unit that matters. Letter height is estimated from the ink itself, below.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentConfig:
    gap_x: float = 2.0
    """Horizontal closing, in letter heights. Measured across the fixtures and
    the real pages, the widest gap between two words of one line runs to about
    1.9 letter heights, so this bridges words into a line while the much wider
    gap between two side-by-side regions stays open."""
    gap_y: float = 1.6
    """Vertical closing, in letter heights. The gap between two lines of one
    paragraph measures 0.8 to 1.2 letter heights and the gap between two
    stacked regions 3.8 to 5.1, so anything in between separates them. 1.6
    sits in that gap with room on both sides, including the extra half kernel
    the dilation below adds."""
    gap_min_frac: float = 0.003
    gap_max_frac: float = 0.25
    """Floor and ceiling on the kernel, as a fraction of the page's short side.
    These are a guard on a pathological estimate, not a tuning knob: a page of
    nothing but one large diagram gives a letter height that means nothing, and
    the clamp keeps it from producing a kernel that is either a no-op or a
    page-eater. Set wide enough that normal writing never reaches them."""
    min_area: float = 1.2
    """Smallest region kept, in letter areas (letter height squared). A region
    has to hold about one letter's worth of ink to be a region at all; below
    that it is a stray mark, a page number, or a speck the denoiser missed.
    This was a fraction of the page and had the same bug as the kernel: a
    short line of small writing is a perfectly real region that happens to
    cover little of a large photograph, and it was being thrown away."""
    min_area_cap: float = 0.02
    """Never demand more than this fraction of the page, whatever the letter
    estimate says. The guard for the same pathological estimate as above."""
    min_side: int = 16
    margin: float = 0.04
    """Ignore ink this close to the edge: page borders, binder holes, and the
    band of surface that `crop_to_page` leaves around the paper. The crop pads
    outward on purpose, because the far edge of a page shot at an angle falls
    into shadow and a tight box clips the writing there; the cost is a ring of
    background just inside the frame, and this is where it is paid for. Kept
    in step with `preprocess.PAGE_MARGIN`."""


def _merge_overlapping(boxes: list[list[int]]) -> list[list[int]]:
    """Union boxes that overlap until none do. Closing can leave a region in
    two blobs whose boxes still overlap, e.g. a table's header and body."""
    changed = True
    while changed:
        changed = False
        out: list[list[int]] = []
        for x, y, w, h in boxes:
            for o in out:
                ox, oy, ow, oh = o
                if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
                    nx, ny = min(x, ox), min(y, oy)
                    o[:] = [nx, ny, max(x + w, ox + ow) - nx, max(y + h, oy + oh) - ny]
                    changed = True
                    break
            else:
                out.append([x, y, w, h])
        boxes = out
    return boxes


def letter_height(mask: np.ndarray, min_area: int = 8, min_side: int = 3) -> float:
    """The writing's letter height, as the median height of its ink blobs.

    Before any closing, one connected component is roughly one letter, or a
    joined pair of them in cursive. The median is the right summary rather
    than the mean because a page mixes writing with the odd long rule or
    diagram outline, and those are a handful of very tall components that
    would drag a mean upward.

    Components smaller than `min_area` px or thinner than `min_side` are
    dropped: dots, accents, and speckle that survived the denoiser are not
    letters, and there are enough of them on a photograph to halve the median.

    Falls back to 1% of the short side on a page with no usable ink, which
    only reaches `segment` on a blank page where the answer is no regions.
    """
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, count)
               if stats[i, cv2.CC_STAT_AREA] >= min_area
               and min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_side]
    if not heights:
        return max(3.0, min(mask.shape) * 0.01)
    return float(np.median(heights))


def segment(ink: np.ndarray, config: SegmentConfig = SegmentConfig()) -> list[tuple[int, int, int, int]]:
    """Region boxes `(x, y, w, h)` in reading order, top to bottom then left to right."""
    h, w = ink.shape
    mask = ink.astype(np.uint8)
    m = int(min(h, w) * config.margin)
    if m:
        mask[:m, :] = mask[-m:, :] = 0
        mask[:, :m] = mask[:, -m:] = 0

    letter = letter_height(mask)
    lo = max(3, int(min(h, w) * config.gap_min_frac))
    hi = max(lo, int(min(h, w) * config.gap_max_frac))
    kx = min(max(int(letter * config.gap_x), lo), hi) | 1
    ky = min(max(int(letter * config.gap_y), lo), hi) | 1
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    # Closing joins ink across gaps but leaves holes; dilating by the same
    # kernel once more makes each region one solid blob.
    closed = cv2.dilate(closed, cv2.getStructuringElement(cv2.MORPH_RECT, (kx // 2 | 1, ky // 2 | 1)))

    count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    boxes = []
    for i in range(1, count):
        x, y, bw, bh, _ = stats[i]
        # The dilation grew the blob; shrink back to the ink it actually holds.
        inner = mask[y:y + bh, x:x + bw]
        ys, xs = np.nonzero(inner)
        if len(xs) == 0:
            continue
        x0, y0 = x + int(xs.min()), y + int(ys.min())
        x1, y1 = x + int(xs.max()) + 1, y + int(ys.max()) + 1
        boxes.append([x0, y0, x1 - x0, y1 - y0])

    boxes = _merge_overlapping(boxes)
    floor = min(config.min_area * letter * letter, config.min_area_cap * h * w)
    kept = [b for b in boxes
            if b[2] * b[3] >= floor and min(b[2], b[3]) >= config.min_side]
    # Reading order with a tolerance: two regions side by side whose tops
    # differ by a few pixels are one row, left before right.
    row = max(1, int(h * 0.03))
    kept.sort(key=lambda b: (b[1] // row, b[0]))
    return [tuple(b) for b in kept]


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def match_score(predicted, truth, threshold: float = 0.5) -> dict[str, float]:
    """Precision and recall of predicted boxes against ground truth at an IoU
    threshold, matched greedily one to one, best overlap first."""
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(predicted) for j, t in enumerate(truth)),
                   reverse=True)
    used_p, used_t, hits = set(), set(), 0
    for score, i, j in pairs:
        if score < threshold:
            break
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        hits += 1
    return {
        "precision": hits / len(predicted) if predicted else 0.0,
        "recall": hits / len(truth) if truth else 0.0,
        "hits": hits,
    }


def tighten(ink: np.ndarray, box) -> tuple[int, int, int, int]:
    """A box shrunk to the ink inside it. The generator's ground-truth boxes
    are layout slots with padding; scoring tight predictions against loose
    slots would punish a correct segmentation for the generator's margins."""
    x, y, w, h = (int(v) for v in box)
    ys, xs = np.nonzero(ink[y:y + h, x:x + w])
    if len(xs) == 0:
        return x, y, w, h
    return x + int(xs.min()), y + int(ys.min()), int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
