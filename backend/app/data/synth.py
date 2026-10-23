"""
The page generator. Synthetic layout, real-or-procedural ink.

A training image needs two things and they come from different places. **Labels**
come from the layout: if the generator placed a formula at those coordinates then
that region *is* a formula, known by construction, exact, and free. No annotation
pass and no label noise, which is the only way to get region level structural
labels at all, since no public dataset provides them. **Appearance** comes from
the ink provider, which is swappable.

The single most important decision in this file is not how anything is drawn.

**Layout must be randomised, or the classifier learns position instead of
structure.** If `THEORY` always sat at the top of the page and `TABLE` always at
the bottom, a convolutional net would happily reach 95% validation accuracy by
reading the crop's y coordinate, which is information it will not have at
inference and which says nothing about what is in the region. That is shortcut
learning, and a generator is exactly where it gets baked in. So every region's
class, position, size and column arrangement are sampled independently, and the
same class appears at the top of one page and the bottom of the next.

What is left for the model to learn is the structural signature of a region: its
stroke density, aspect ratio, baseline regularity, and whether it contains long
straight runs. That signature is real and transfers to a photographed page. A
y coordinate does not.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import CLASS_NAMES
from .ink import InkProvider, ProceduralInk, StrokePen


@dataclass(frozen=True)
class RegionSpec:
    """One labelled region. The label is known because we placed it."""

    region_class: str
    x: int
    y: int
    w: int
    h: int
    hard: bool = False
    """A deliberately ambiguous instance of this class.

    Recorded per region so accuracy can be reported split by difficulty,
    which turns "the model gets 88%" into "it gets 97% on clear regions and
    74% on the ambiguous ones", a far more useful statement about where the
    pipeline will actually struggle.
    """

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def as_dict(self) -> dict:
        return {
            "class": self.region_class,
            "box": [self.x, self.y, self.w, self.h],
            "hard": bool(self.hard),
        }


@dataclass
class SyntheticPage:
    """A rendered page and its ground truth."""

    image: np.ndarray
    """uint8 grayscale, paper with ink composited on."""
    regions: list[RegionSpec]
    seed: int
    ink_provider: str
    ruled: bool = False
    meta: dict = field(default_factory=dict)

    def crops(self, size: int = 224, pad_frac: float = 0.04) -> Iterator[tuple[np.ndarray, str]]:
        """Yield `(crop, class_name)` for every region.

        A little context is included around each box on purpose. A region
        cropped exactly to its ink loses the whitespace that tells a classifier
        a table's rules run to the margin while a formula's do not.
        """
        H, W = self.image.shape[:2]
        for r in self.regions:
            px, py = int(r.w * pad_frac), int(r.h * pad_frac)
            x0, y0 = max(0, r.x - px), max(0, r.y - py)
            x1, y1 = min(W, r.x + r.w + px), min(H, r.y + r.h + py)
            crop = self.image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            yield cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), r.region_class

    def sidecar(self) -> dict:
        return {
            "seed": self.seed,
            "ink_provider": self.ink_provider,
            "ruled": self.ruled,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "regions": [r.as_dict() for r in self.regions],
            **self.meta,
        }

    def save(self, directory: Path, stem: str | None = None) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        stem = stem or f"page_{self.seed:06d}"
        img_path = directory / f"{stem}.png"
        json_path = directory / f"{stem}.json"
        cv2.imwrite(str(img_path), self.image)
        json_path.write_text(json.dumps(self.sidecar(), indent=1))
        return img_path, json_path


class PageGenerator:
    """Composes labelled pages.

    Page size defaults to 1024x768 rather than the 2048x1536 a phone produces.
    Crops are resized to 224 for the classifier either way, so the larger render
    costs four times the generation time to deliver detail that is immediately
    thrown away. Augmentation simulates camera capture separately.
    """

    def __init__(
        self,
        ink: InkProvider | None = None,
        width: int = 1024,
        height: int = 768,
        hard_fraction: float = 0.4,
        realism: bool = False,
    ):
        self.ink = ink or ProceduralInk()
        # Off by default so every existing benchmark stays reproducible. On, it
        # adds the marks real notes have that cut across class boundaries:
        # underlined prose has long straight runs, a boxed answer is a closed
        # rectangle, a plot carries numbers. Pair it with `MixedInk`.
        self.realism = bool(realism)
        self.width = width
        self.height = height
        # Share of regions drawn as a deliberately ambiguous instance of
        # their class. With this at zero the five classes are caricatures and
        # separable by one feature each, which is measurable: a 154k network
        # trained from scratch reached macro-F1 1.000 on held-out pages. Real
        # notebooks are not like that, and neither is a useful benchmark.
        self.hard_fraction = float(hard_fraction)

    # ---------------------------------------------------------------- layout

    def _layout(self, rng: np.random.Generator) -> list[RegionSpec]:
        """Bands down the page, some split into columns, classes sampled freely.

        Classes are drawn uniformly rather than from the distribution of a real
        page, which is dominated by prose. A balanced training set is what we
        want; the real imbalance is why the headline metric is macro-F1 rather
        than accuracy, and it is handled at evaluation, not here.
        """
        margin = int(min(self.width, self.height) * rng.uniform(0.04, 0.09))
        usable_w = self.width - margin * 2
        usable_h = self.height - margin * 2

        n_bands = int(rng.integers(3, 6))
        # Uneven band heights, so region aspect ratio varies independently of class.
        weights = rng.uniform(0.6, 1.6, size=n_bands)
        weights /= weights.sum()
        gap = int(usable_h * rng.uniform(0.02, 0.045))
        heights = (weights * (usable_h - gap * (n_bands - 1))).astype(int)

        specs: list[RegionSpec] = []
        y = margin
        for bh in heights:
            if bh < 48:
                y += bh + gap
                continue
            # Some bands hold two regions side by side, which is how a real page
            # ends up with a sketch beside a table.
            if rng.random() < 0.3 and usable_w > 320:
                split = rng.uniform(0.38, 0.62)
                col_gap = int(usable_w * 0.035)
                w1 = int((usable_w - col_gap) * split)
                w2 = usable_w - col_gap - w1
                for xoff, w in ((0, w1), (w1 + col_gap, w2)):
                    specs.append(self._spec(rng, margin + xoff, y, w, int(bh)))
            else:
                # Short lines happen: not every region runs to the margin.
                w = int(usable_w * rng.uniform(0.55, 1.0))
                specs.append(self._spec(rng, margin, y, w, int(bh)))
            y += bh + gap

        return specs

    def _spec(self, rng: np.random.Generator, x: int, y: int, w: int, h: int) -> RegionSpec:
        cls = str(rng.choice(np.asarray(CLASS_NAMES)))
        hard = bool(rng.random() < self.hard_fraction)
        return RegionSpec(cls, int(x), int(y), int(w), int(h), hard=hard)

    # ----------------------------------------------------------------- paper

    def _paper(self, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
        """Off-white with grain, sometimes ruled or gridded.

        Ruled paper is included deliberately even though it makes `TABLE` harder,
        because the rules look like a table's own. A model that separates them
        has learned something about structure rather than about horizontal lines,
        and a model that cannot was going to fail on real paper anyway.
        """
        base = rng.uniform(236, 250)
        page = np.full((self.height, self.width), base, dtype=np.float32)
        page += rng.normal(0, 1.6, page.shape).astype(np.float32)

        ruled = False
        roll = rng.random()
        if roll < 0.35:
            ruled = True
            spacing = int(rng.uniform(26, 40))
            shade = base - rng.uniform(10, 20)
            for yy in range(int(rng.uniform(0, spacing)), self.height, spacing):
                cv2.line(page, (0, yy), (self.width, yy), float(shade), 1, cv2.LINE_AA)
        elif roll < 0.5:
            ruled = True
            spacing = int(rng.uniform(22, 34))
            shade = base - rng.uniform(6, 14)
            for yy in range(0, self.height, spacing):
                cv2.line(page, (0, yy), (self.width, yy), float(shade), 1, cv2.LINE_AA)
            for xx in range(0, self.width, spacing):
                cv2.line(page, (xx, 0), (xx, self.height), float(shade), 1, cv2.LINE_AA)
        return page, ruled

    # --------------------------------------------------------------- painters
    # Each painter produces the structural signature of its class. What matters
    # is not that the marks are legible but that the arrangement is right:
    # density, regularity, and the presence or absence of long straight runs.

    def _paint_theory(self, mask, r: RegionSpec, rng) -> None:
        """Baseline-aligned lines of words. Dense, regular, no long straights.

        The hard variant is a bulleted list rather than a paragraph: short
        lines, wide leading, an indent and sometimes a dash. Notes are full of
        these, and once prose stops being a dense block it looks a great deal
        like a column of table values or a stack of short equations.
        """
        line_h = int(r.h * rng.uniform(0.1, 0.17))
        line_h = max(10, min(line_h, 40))
        leading = line_h * (rng.uniform(2.3, 3.1) if r.hard else rng.uniform(1.5, 2.1))
        bullets = r.hard and rng.random() < 0.6
        indent = r.w * rng.uniform(0.04, 0.16) if r.hard else 0.0
        pen = StrokePen(rng, base_width=max(1.2, line_h * 0.1))

        y = r.y + line_h
        while y < r.y + r.h - line_h * 0.3:
            x = r.x + indent
            if bullets:
                # A dash or a dot in the margin, which is most of what makes
                # a list read as a list.
                pen.polyline(mask, [(x, y - line_h * 0.25), (x + line_h * 0.4, y - line_h * 0.25)],
                             wobble=line_h * 0.03, width_scale=0.8)
                x += line_h * 0.8
            if r.hard:
                limit = r.x + r.w * rng.uniform(0.32, 0.62)
            else:
                # A paragraph's last line is short, which is a real cue.
                limit = r.x + r.w * (rng.uniform(0.35, 0.75) if rng.random() < 0.22 else rng.uniform(0.92, 1.0))
            drift = rng.normal(0, line_h * 0.06)
            while x < limit - line_h:
                wlen = int(line_h * rng.uniform(1.2, 4.2))
                if x + wlen > limit:
                    break
                word = self.ink.word(rng, line_h, wlen)
                self._blit(mask, word, int(x), int(y + drift - word.shape[0] * 0.72))
                x += wlen + line_h * rng.uniform(0.3, 0.55)
            y += leading

    def _paint_formula(self, mask, r: RegionSpec, rng) -> None:
        """Sparse, centred, wide spacing, with fraction bars and tall symbols.

        The hard variant is a multi-line derivation: several equations stacked
        down the region, each wrapping most of the width. It stops being
        sparse and centred and starts looking like prose, which is exactly
        what a worked solution on a real page looks like.
        """
        if r.hard:
            self._paint_formula_block(mask, r, rng)
            return
        pen = StrokePen(rng, base_width=max(1.4, r.h * 0.016))
        gh = int(min(r.h * rng.uniform(0.3, 0.5), 54))
        gh = max(12, gh)
        cy = r.y + r.h * rng.uniform(0.42, 0.58)
        n = int(rng.integers(4, 9))
        total = n * gh * rng.uniform(1.5, 2.2)
        x = r.x + max(0.0, (r.w - total) * rng.uniform(0.25, 0.75))

        for i in range(n):
            roll = rng.random()
            if roll < 0.16 and x + gh * 1.6 < r.x + r.w:
                # A fraction: numerator, bar, denominator. The bar is the one
                # genuinely straight run a formula contains.
                bar_w = gh * rng.uniform(1.1, 1.7)
                # The bar stays procedural: it is drawn with a ruler-less hand,
                # not written, so it belongs with the plot axes and table rules.
                pen.polyline(mask, [(x, cy), (x + bar_w, cy)], wobble=gh * 0.02)
                for dy in (-gh * 0.75, gh * 0.75):
                    g = self.ink.math_symbol(rng, int(gh * 0.62), kind="operand")
                    self._blit(mask, g, int(x + bar_w / 2 - g.shape[1] / 2), int(cy + dy - g.shape[0] / 2))
                x += bar_w + gh * 0.5
            elif roll < 0.28:
                # A real handwritten operator where there used to be two
                # procedural strokes approximating a plus sign.
                op = self.ink.math_symbol(rng, gh, kind="operator")
                self._blit(mask, op, int(x), int(cy - op.shape[0] / 2))
                x += op.shape[1] * rng.uniform(1.1, 1.5)
            else:
                g = self.ink.math_symbol(rng, gh, kind="operand")
                self._blit(mask, g, int(x), int(cy - g.shape[0] / 2))
                x += g.shape[1] * rng.uniform(0.9, 1.4)
                # Subscripts break the baseline, which prose does not do.
                if rng.random() < 0.3:
                    s = self.ink.math_symbol(rng, int(gh * 0.5), kind="operand")
                    self._blit(mask, s, int(x - gh * 0.2), int(cy + gh * 0.28))
                    x += s.shape[1] * 0.8
            if x > r.x + r.w - gh:
                break

    def _paint_formula_block(self, mask, r: RegionSpec, rng) -> None:
        """A derivation running over several lines."""
        pen = StrokePen(rng, base_width=max(1.3, r.h * 0.012))
        line_h = int(max(12, min(r.h * rng.uniform(0.16, 0.24), 42)))
        leading = line_h * rng.uniform(1.5, 2.0)
        y = r.y + line_h

        while y < r.y + r.h - line_h * 0.4:
            # Successive lines of a derivation are indented and get shorter,
            # the way a solution converges towards its answer.
            x = r.x + r.w * rng.uniform(0.0, 0.12)
            limit = r.x + r.w * rng.uniform(0.6, 1.0)
            while x < limit - line_h:
                roll = rng.random()
                if roll < 0.14:
                    bar_w = line_h * rng.uniform(1.0, 1.5)
                    pen.polyline(mask, [(x, y), (x + bar_w, y)], wobble=line_h * 0.02)
                    for dy in (-line_h * 0.62, line_h * 0.62):
                        g = self.ink.math_symbol(rng, int(line_h * 0.5), kind="operand")
                        self._blit(mask, g, int(x + bar_w / 2 - g.shape[1] / 2),
                                   int(y + dy - g.shape[0] / 2))
                    x += bar_w + line_h * 0.3
                elif roll < 0.34:
                    op = self.ink.math_symbol(rng, line_h, kind="operator")
                    self._blit(mask, op, int(x), int(y - op.shape[0] / 2))
                    x += op.shape[1] * rng.uniform(1.0, 1.3)
                else:
                    g = self.ink.math_symbol(rng, line_h, kind="operand")
                    self._blit(mask, g, int(x), int(y - g.shape[0] / 2))
                    x += g.shape[1] * rng.uniform(0.85, 1.15)
            y += leading

    def _paint_plot(self, mask, r: RegionSpec, rng) -> None:
        """Two long near-perpendicular runs, curves, and a mostly empty interior."""
        pen = StrokePen(rng, base_width=max(1.4, r.h * 0.014))
        pad = r.h * rng.uniform(0.1, 0.18)
        ox, oy = r.x + pad, r.y + r.h - pad
        ax_h = r.h - pad * 2
        ax_w = (r.w - pad * 2) * rng.uniform(0.7, 1.0)

        pen.polyline(mask, [(ox, oy), (ox, oy - ax_h)], wobble=r.h * 0.012)
        pen.polyline(mask, [(ox, oy), (ox + ax_w, oy)], wobble=r.h * 0.012)

        # Two hard variants, both real things people draw.
        #
        # A bar sketch is the genuinely ambiguous one: upright rectangles
        # standing on an axis share their repeated vertical runs with a ruled
        # table and their closed outlines with a block diagram.
        #
        # Bare axes with nothing plotted yet is the other, but on its own it
        # leaves the crop nearly empty, which is degenerate rather than hard,
        # so it keeps its tick marks and gains an axis label.
        bars = r.hard and rng.random() < 0.6
        if bars:
            n_bars = int(rng.integers(3, 7))
            slot = ax_w / n_bars
            bw = slot * rng.uniform(0.45, 0.72)
            for i in range(n_bars):
                bx = ox + i * slot + (slot - bw) / 2
                bh = ax_h * rng.uniform(0.25, 0.95)
                pen.polyline(mask, [(bx, oy), (bx, oy - bh)], wobble=r.h * 0.01)
                pen.polyline(mask, [(bx, oy - bh), (bx + bw, oy - bh)], wobble=r.h * 0.01)
                pen.polyline(mask, [(bx + bw, oy - bh), (bx + bw, oy)], wobble=r.h * 0.01)

        n_curves = 0 if r.hard else int(rng.integers(1, 4))
        for _ in range(n_curves):
            kind = rng.integers(0, 3)
            t = np.linspace(0, 1, 26)
            if kind == 0:      # rising, accelerating
                ys = t ** rng.uniform(1.6, 3.2)
            elif kind == 1:    # saturating
                ys = 1 - np.exp(-t * rng.uniform(2.5, 5.0))
            else:              # falling
                ys = 1 - t ** rng.uniform(0.7, 1.4)
            pts = np.stack([ox + t * ax_w, oy - ys * ax_h * rng.uniform(0.6, 0.95)], axis=1)
            pen.stroke(mask, pts, wobble=r.h * 0.018, width_scale=rng.uniform(0.8, 1.1))

        n_ticks = int(rng.integers(3, 7)) if (r.hard and not bars) else int(rng.integers(0, 5))
        for _ in range(n_ticks):  # tick marks
            tx = ox + rng.uniform(0.1, 1.0) * ax_w
            pen.polyline(mask, [(tx, oy), (tx, oy + r.h * 0.03)], wobble=0.4, width_scale=0.7)

        if r.hard and not bars:
            # An axis label, so a frame drawn before its curve is still a
            # region with content rather than two bare strokes.
            label_h = int(max(8, min(ax_h * 0.16, 22)))
            word = self.ink.word(rng, label_h, int(ax_w * rng.uniform(0.2, 0.4)))
            self._blit(mask, word, int(ox + ax_w * 0.3), int(oy + r.h * 0.04))

    def _paint_diagram(self, mask, r: RegionSpec, rng) -> None:
        """Closed boxes joined by arrows. Straight runs, but short and enclosing."""
        pen = StrokePen(rng, base_width=max(1.4, r.h * 0.015))
        n = int(rng.integers(2, 5))
        span = r.w * 0.92
        step = span / n
        # Boxes must be narrower than the step, or they sit flush and the whole
        # region reads as a single ruled rectangle: visually a TABLE, which is a
        # neighbouring class. The gap is what the connector lives in, and the
        # connector is most of what distinguishes a diagram from a grid.
        bw = step * rng.uniform(0.5, 0.68)
        bh = min(r.h * rng.uniform(0.3, 0.55), bw * 0.9)
        if r.hard:
            # Taller boxes, because several lines of writing have to fit.
            bh = min(r.h * rng.uniform(0.6, 0.85), bw * 1.4)
        cy = r.y + r.h * rng.uniform(0.4, 0.6)
        x = r.x + r.w * 0.04

        for i in range(n):
            x0, y0 = x, cy - bh / 2
            # Hand-drawn rectangles overshoot their corners.
            o = bh * 0.07
            pen.polyline(mask, [(x0 - o, y0), (x0 + bw, y0)], wobble=bh * 0.04)
            pen.polyline(mask, [(x0 + bw, y0 - o), (x0 + bw, y0 + bh)], wobble=bh * 0.04)
            pen.polyline(mask, [(x0 + bw + o, y0 + bh), (x0, y0 + bh)], wobble=bh * 0.04)
            pen.polyline(mask, [(x0, y0 + bh + o), (x0, y0)], wobble=bh * 0.04)

            if r.hard:
                # Boxes packed with writing. A block diagram annotated this
                # heavily reads as prose at a glance, because stroke density
                # and baseline regularity dominate the thin box outline.
                lines = max(1, int(bh / max(8.0, bh * 0.34)))
                line_h = bh / (lines + 0.6)
                for line in range(lines):
                    lx = x0 + bw * 0.08
                    ly = y0 + line_h * (line + 0.8)
                    while lx < x0 + bw * 0.92:
                        wlen = int(line_h * rng.uniform(1.1, 2.6))
                        if lx + wlen > x0 + bw * 0.92:
                            break
                        word = self.ink.word(rng, int(line_h * 0.72), wlen)
                        self._blit(mask, word, int(lx), int(ly - word.shape[0] * 0.72))
                        lx += wlen + line_h * 0.35
            elif rng.random() < 0.65:  # a label inside
                g = self.ink.word(rng, int(bh * 0.4), int(bw * 0.5))
                self._blit(mask, g, int(x0 + bw * 0.22), int(cy - g.shape[0] * 0.5))

            if i < n - 1:
                gap_a, gap_b = x0 + bw, x0 + step
                pen.polyline(mask, [(gap_a, cy), (gap_b, cy)], wobble=bh * 0.03)
                head = min(bh * 0.22, (gap_b - gap_a) * 0.5)
                pen.polyline(mask, [(gap_b - head, cy - head), (gap_b, cy)], wobble=1.0, width_scale=0.85)
                pen.polyline(mask, [(gap_b - head, cy + head), (gap_b, cy)], wobble=1.0, width_scale=0.85)
            x += step

    def _paint_table(self, mask, r: RegionSpec, rng) -> None:
        """Long parallel rules to the margin, dividers, and cell-aligned marks.

        The hard variant drops the ruling entirely and keeps only the column
        alignment, which is how most people actually jot a small table. With
        no long straight runs left, what remains is text in columns, and the
        only thing separating it from prose is the regularity of the gaps.
        """
        pen = StrokePen(rng, base_width=max(1.3, r.h * 0.012))
        rows = int(rng.integers(3, 6))
        cols = int(rng.integers(2, 5))
        rh = r.h / rows
        cw = r.w / cols

        if not r.hard:
            for i in range(rows + 1):
                y = r.y + i * rh
                pen.polyline(mask, [(r.x, y), (r.x + r.w, y)], wobble=r.h * 0.008)
            # Vertical dividers are often partial, which is a real habit.
            for j in range(1, cols):
                x = r.x + j * cw
                y0 = r.y if rng.random() < 0.8 else r.y + rh
                pen.polyline(mask, [(x, y0), (x, r.y + r.h)], wobble=r.h * 0.008)
        elif rng.random() < 0.45:
            # Sometimes a single underline beneath the header row survives,
            # which is the most common partial ruling of all.
            y = r.y + rh
            pen.polyline(mask, [(r.x, y), (r.x + r.w * rng.uniform(0.5, 1.0), y)],
                         wobble=r.h * 0.008)

        for i in range(rows):
            for j in range(cols):
                if rng.random() < (0.10 if r.hard else 0.28):
                    continue  # empty cells are normal
                gh = int(min(rh * 0.42, 26))
                if gh < 7:
                    continue
                if rng.random() < 0.45:
                    # A table holds columns of values, so a cell is more often
                    # a digit than a letter.
                    m = self.ink.math_symbol(rng, gh, kind="operand")
                else:
                    m = self.ink.word(rng, gh, int(cw * rng.uniform(0.25, 0.55)))
                self._blit(
                    mask, m,
                    int(r.x + j * cw + cw * 0.14),
                    int(r.y + i * rh + rh * 0.5 - m.shape[0] * 0.5),
                )

    # -------------------------------------------------------------- confusers

    def _confusers(self, mask, r: RegionSpec, rng) -> None:
        """Marks that borrow another class's signature, as real notes do.

        Each one gives a class a feature the classifier would otherwise use to
        identify a *neighbouring* class, which is exactly where a shortcut
        lives. They are drawn on top of the painted region, inside its box,
        so the label stays correct by construction.
        """
        pen = StrokePen(rng, base_width=max(1.3, r.h * 0.012))
        wob = r.h * 0.01
        cls = r.region_class

        if cls == "THEORY":
            # Underlining and crossing out: long straight runs inside prose,
            # which until now only tables, plots and diagrams had.
            for _ in range(int(rng.integers(0, 3))):
                y = r.y + r.h * rng.uniform(0.2, 0.9)
                x0 = r.x + r.w * rng.uniform(0.0, 0.4)
                x1 = x0 + r.w * rng.uniform(0.25, 0.6)
                pen.polyline(mask, [(x0, y), (min(x1, r.x + r.w), y)], wobble=wob)
            if rng.random() < 0.25:  # a margin bar marking a paragraph as important
                x = r.x + r.w * 0.005
                pen.polyline(mask, [(x, r.y + r.h * 0.1), (x, r.y + r.h * 0.9)], wobble=wob)

        elif cls == "FORMULA":
            if rng.random() < 0.35:
                # A boxed answer: a closed rectangle, which is a diagram's cue.
                x0, y0 = r.x + r.w * rng.uniform(0.02, 0.2), r.y + r.h * 0.12
                x1, y1 = r.x + r.w * rng.uniform(0.8, 0.98), r.y + r.h * 0.88
                pen.polyline(mask, [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], wobble=wob)
            if rng.random() < 0.3:  # an arrow to the next step, also a diagram cue
                y = r.y + r.h * rng.uniform(0.3, 0.7)
                x0 = r.x + r.w * 0.82
                x1 = r.x + r.w * 0.97
                head = r.h * 0.08
                pen.polyline(mask, [(x0, y), (x1, y)], wobble=wob)
                pen.polyline(mask, [(x1 - head, y - head), (x1, y), (x1 - head, y + head)], wobble=1.0)

        elif cls == "PLOT_SKETCH":
            # Numbers along the axes. A plot then holds handwritten maths
            # symbols, which used to belong to formulas and tables alone.
            gh = int(max(8, min(r.h * 0.08, 20)))
            for _ in range(int(rng.integers(2, 6))):
                g = self.ink.math_symbol(rng, gh, kind="operand")
                self._blit(mask, g, int(r.x + r.w * rng.uniform(0.15, 0.9)),
                           int(r.y + r.h * rng.uniform(0.86, 0.95)))
            if rng.random() < 0.5:  # a curve label up in the plot area
                word = self.ink.word(rng, gh, int(r.w * rng.uniform(0.1, 0.25)))
                self._blit(mask, word, int(r.x + r.w * rng.uniform(0.4, 0.7)),
                           int(r.y + r.h * rng.uniform(0.1, 0.3)))

        elif cls == "DIAGRAM":
            if rng.random() < 0.4:
                # A caption under the boxes: a line of prose inside a diagram.
                lh = int(max(8, min(r.h * 0.1, 24)))
                x = r.x + r.w * 0.05
                y = r.y + r.h * 0.9
                while x < r.x + r.w * rng.uniform(0.5, 0.9):
                    wlen = int(lh * rng.uniform(1.5, 3.5))
                    word = self.ink.word(rng, lh, wlen)
                    self._blit(mask, word, int(x), int(y - word.shape[0]))
                    x += wlen + lh * 0.4

        elif cls == "TABLE":
            if rng.random() < 0.3:
                # One cell crossed out and rewritten. A diagonal stroke in a
                # grid, where until now every stroke was axis-aligned.
                x0 = r.x + r.w * rng.uniform(0.1, 0.6)
                y0 = r.y + r.h * rng.uniform(0.2, 0.7)
                pen.polyline(mask, [(x0, y0), (x0 + r.w * 0.15, y0 + r.h * 0.15)], wobble=wob)

    # ------------------------------------------------------------- compositing

    @staticmethod
    def _blit(mask: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
        """Add a patch to the page mask, clipped at the edges."""
        if patch.size == 0:
            return
        H, W = mask.shape
        ph, pw = patch.shape
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + pw), min(H, y + ph)
        if x1 <= x0 or y1 <= y0:
            return
        sub = patch[y0 - y:y1 - y, x0 - x:x1 - x]
        np.maximum(mask[y0:y1, x0:x1], sub, out=mask[y0:y1, x0:x1])

    def page(self, seed: int) -> SyntheticPage:
        """Render one labelled page. Deterministic in `seed`."""
        rng = np.random.default_rng(seed)
        paper, ruled = self._paper(rng)
        specs = self._layout(rng)

        painters = {
            "THEORY": self._paint_theory,
            "FORMULA": self._paint_formula,
            "PLOT_SKETCH": self._paint_plot,
            "DIAGRAM": self._paint_diagram,
            "TABLE": self._paint_table,
        }

        ink_mask = np.zeros_like(paper)
        for spec in specs:
            painters[spec.region_class](ink_mask, spec, rng)
            if self.realism:
                self._confusers(ink_mask, spec, rng)

        # Ink darkness varies by pen and pressure, never fully black.
        darkness = rng.uniform(0.62, 0.86)
        page = paper * (1.0 - ink_mask * darkness)
        image = np.clip(page, 0, 255).astype(np.uint8)

        return SyntheticPage(
            image=image,
            regions=specs,
            seed=seed,
            ink_provider=self.ink.name,
            ruled=bool(ruled),
            meta={"ink_darkness": round(float(darkness), 3)},
        )

    def pages(self, count: int, start: int = 0) -> Iterator[SyntheticPage]:
        for i in range(start, start + count):
            yield self.page(i)
