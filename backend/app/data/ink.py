"""
Where strokes come from.

The generator needs ink to paint with, and there is more than one place to get
it. Procedural strokes always work and need no download. A corpus of real
handwriting looks far better but has to be fetched, licensed and indexed first.

So ink arrives through an interface. Everything downstream, the page generator,
the augmentation bank, the data module and the training loop, is written against
`InkProvider` rather than against a corpus. Swapping in real handwriting later is
one class, and it becomes a measurable experiment rather than a rewrite: train
twice, compare macro-F1, report whether ink realism mattered.

Masks are float32 in [0, 1], where 1 is full ink. The page compositor decides
what colour that becomes, so providers never think about paper.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np


class InkProvider(ABC):
    """A source of handwritten-looking marks.

    Implementations return alpha masks sized roughly to the request. "Roughly"
    is deliberate: real handwriting does not fill its bounding box exactly, and
    forcing it to would make every word the same width and hand the classifier a
    cue that does not exist on real paper.
    """

    name: str = "abstract"

    @abstractmethod
    def word(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """One handwritten word, for prose and labels."""

    @abstractmethod
    def glyph(self, rng: np.random.Generator, height: int) -> np.ndarray:
        """One character, for formula operands and table cells."""

    def math_symbol(
        self,
        rng: np.random.Generator,
        height: int,
        kind: str = "operand",
    ) -> np.ndarray:
        """One mathematical symbol: `kind` is `"operator"` or `"operand"`.

        Concrete rather than abstract, and defaulting to an ordinary glyph, so
        that a provider without a mathematics corpus keeps working unchanged.
        The formula painter can then always ask for the right kind of mark and
        get the best available answer.
        """
        return self.glyph(rng, height)


class StrokePen:
    """Draws strokes that look handwritten rather than plotted.

    Three things separate ink from a computer line, and all three are cheap:

    * **The line is never straight.** Even a ruled table edge drawn by hand
      bows and overshoots its corner.
    * **Width varies with speed.** Fast segments are thin. This is the single
      biggest cue, and it is why a constant-width polyline reads as a diagram
      even when its path is perfect.
    * **Ends are soft.** A pen leaves a slightly blurred edge, not an aliased
      one.
    """

    def __init__(self, rng: np.random.Generator, base_width: float = 2.0):
        self.rng = rng
        self.base_width = base_width

    def stroke(
        self,
        mask: np.ndarray,
        pts: np.ndarray,
        *,
        wobble: float = 1.0,
        width_scale: float = 1.0,
    ) -> None:
        """Lay one stroke along `pts`, in place."""
        if len(pts) < 2:
            return
        pts = pts.astype(np.float64).copy()

        # Low frequency wander: the hand drifting off its intended path.
        if wobble > 0 and len(pts) > 3:
            drift = np.cumsum(self.rng.normal(0, wobble * 0.45, size=(len(pts), 2)), axis=0)
            drift -= np.linspace(0, 1, len(pts))[:, None] * drift[-1]  # keep the endpoints
            pts += drift

        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        typical = float(np.median(steps)) if len(steps) else 1.0

        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            speed = steps[i - 1] / typical if typical > 0 else 1.0
            w = self.base_width * width_scale * (1.25 - 0.4 * min(1.8, speed))
            w = max(0.7, w)
            cv2.line(
                mask,
                (int(round(a[0])), int(round(a[1]))),
                (int(round(b[0])), int(round(b[1]))),
                color=1.0,
                thickness=max(1, int(round(w))),
                lineType=cv2.LINE_AA,
            )

    def polyline(self, mask: np.ndarray, pts: list[tuple[float, float]], **kw) -> None:
        self.stroke(mask, np.asarray(pts, dtype=np.float64), **kw)

    def soften(self, mask: np.ndarray, amount: float = 0.6) -> np.ndarray:
        """A pen edge is soft. An aliased one reads as vector art."""
        k = 3 if amount < 1.0 else 5
        return cv2.GaussianBlur(mask, (k, k), amount)


class ProceduralInk(InkProvider):
    """Ink synthesised from stroke primitives. Always available, no download.

    This is honestly the weaker option for realism and the stronger one for
    availability, which is exactly why the interface exists. It is good enough
    to train the structural classifier, because what the classifier must learn
    is the shape of a *region*: its stroke density, aspect ratio, baseline
    regularity, and whether it contains long straight runs. None of that depends
    on the glyphs being real letters.
    """

    name = "procedural"

    def word(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        h, w = max(6, height), max(8, width)
        pad = max(2, h // 3)
        mask = np.zeros((h + pad * 2, w + pad * 2), dtype=np.float32)
        pen = StrokePen(rng, base_width=max(1.2, h * 0.11))

        # A cursive word is a baseline with loops over it, plus the occasional
        # ascender or descender breaking the band. The loop count scales with
        # width so a long word is not just a stretched short one.
        x = float(pad)
        baseline = pad + h * 0.72
        amp = h * 0.5
        loops = max(2, int(w / max(4.0, h * 0.62)))
        step = w / loops

        pts: list[tuple[float, float]] = [(x, baseline)]
        for i in range(loops):
            up = rng.uniform(0.55, 1.0) * amp
            mid = x + step * 0.5
            pts.append((mid, baseline - up))
            x += step
            pts.append((x, baseline - rng.uniform(0.0, 0.2) * amp))
        pen.polyline(mask, pts, wobble=h * 0.05)

        # Ascenders and descenders, which is what gives prose its ragged band.
        for _ in range(rng.integers(0, 3)):
            px = pad + rng.uniform(0.1, 0.9) * w
            if rng.random() < 0.5:
                pen.polyline(mask, [(px, baseline), (px + rng.normal(0, h * 0.08), pad + h * 0.05)],
                             wobble=h * 0.03, width_scale=0.85)
            else:
                pen.polyline(mask, [(px, baseline), (px + rng.normal(0, h * 0.08), pad + h * 1.05)],
                             wobble=h * 0.03, width_scale=0.85)

        return np.clip(pen.soften(mask, 0.6), 0.0, 1.0)

    def glyph(self, rng: np.random.Generator, height: int) -> np.ndarray:
        h = max(6, height)
        pad = max(2, h // 4)
        size = h + pad * 2
        mask = np.zeros((size, size), dtype=np.float32)
        pen = StrokePen(rng, base_width=max(1.2, h * 0.13))
        cx, cy = size / 2, size / 2
        r = h * 0.42

        kind = rng.integers(0, 7)
        if kind == 0:  # a loop, like o a e
            t = np.linspace(0, 2 * math.pi, 24)
            pen.stroke(mask, np.stack([cx + r * np.cos(t) * 0.8, cy + r * np.sin(t)], axis=1),
                       wobble=h * 0.04)
        elif kind == 1:  # a vertical with a crossing, like t f
            pen.polyline(mask, [(cx, cy - r), (cx + rng.normal(0, h * 0.05), cy + r)], wobble=h * 0.04)
            pen.polyline(mask, [(cx - r * 0.6, cy - r * 0.1), (cx + r * 0.6, cy - r * 0.15)],
                         wobble=h * 0.03, width_scale=0.8)
        elif kind == 2:  # a diagonal pair, like x v
            pen.polyline(mask, [(cx - r * 0.7, cy - r), (cx + r * 0.7, cy + r)], wobble=h * 0.04)
            pen.polyline(mask, [(cx + r * 0.7, cy - r), (cx - r * 0.7, cy + r)], wobble=h * 0.04)
        elif kind == 3:  # an arc with a tail, like n h r
            t = np.linspace(math.pi, 2 * math.pi, 14)
            pen.stroke(mask, np.stack([cx + r * 0.7 * np.cos(t), cy + r * 0.6 * np.sin(t)], axis=1),
                       wobble=h * 0.04)
            pen.polyline(mask, [(cx + r * 0.7, cy), (cx + r * 0.7, cy + r)], wobble=h * 0.03)
        elif kind == 4:  # a bowl on a stem, like b p d
            pen.polyline(mask, [(cx - r * 0.5, cy - r), (cx - r * 0.5, cy + r)], wobble=h * 0.04)
            t = np.linspace(-math.pi / 2, math.pi / 2, 14)
            pen.stroke(mask, np.stack([cx - r * 0.5 + r * 0.8 * np.cos(t), cy + r * 0.35 * np.sin(t)], axis=1),
                       wobble=h * 0.04)
        elif kind == 5:  # a zigzag, like z or a summation
            pen.polyline(mask, [(cx - r * 0.7, cy - r * 0.8), (cx + r * 0.7, cy - r * 0.8),
                                (cx - r * 0.6, cy + r * 0.8), (cx + r * 0.7, cy + r * 0.8)],
                         wobble=h * 0.04)
        else:  # a stem with a shoulder, like k or a radical
            pen.polyline(mask, [(cx - r * 0.4, cy - r), (cx - r * 0.4, cy + r)], wobble=h * 0.04)
            pen.polyline(mask, [(cx - r * 0.4, cy + r * 0.1), (cx + r * 0.7, cy - r * 0.7)], wobble=h * 0.04)
            pen.polyline(mask, [(cx - r * 0.2, cy - r * 0.1), (cx + r * 0.7, cy + r * 0.9)], wobble=h * 0.04)

        return np.clip(pen.soften(mask, 0.5), 0.0, 1.0)


# The symbol classes that act as operators. Everything else the mathematics
# corpus ships (the ten digits, `dec`, and `x y z`) is an operand.
MATHS_OPERATORS: tuple[str, ...] = ("add", "sub", "mul", "div", "eq")

# How tall each symbol is relative to a digit on the same line.
#
# Not every mark is full height, and assuming otherwise is not a cosmetic
# error. A minus sign is a thin stroke roughly ten times wider than it is
# tall, so scaling one to a digit's height makes it ten digits wide: the first
# render of this produced a 473px black slab lying across a formula. A decimal
# point did the same thing as a blob. Anything absent here is full height,
# which is correct for the digits and for `x y z`.
MATHS_RELATIVE_HEIGHT: dict[str, float] = {
    "sub": 0.10,   # a thin horizontal, on the midline
    "dec": 0.12,   # a speck
    "eq": 0.34,    # two stacked bars
    "mul": 0.52,
    "div": 0.58,
    "add": 0.62,
}

# Nothing may be wider than this many times the line height, whatever its
# natural proportions. A backstop for an unusually elongated sample, so one bad
# crop cannot paint over a region.
MATHS_MAX_ASPECT: float = 2.5


class CorpusInk(InkProvider):
    """Ink taken from real handwriting, cleaned and cached by `corpus.py`.

    Two corpora back this, and they are used for different jobs because they
    have different shapes:

    * **Chars74K** supplies isolated characters, which is exactly what a
      `FORMULA` operand or a `TABLE` cell is.
    * **Handwriting Recognition** supplies whole handwritten words, which is
      what `THEORY` prose needs and what no character corpus can fake.

    Two decisions in here are worth stating, because both are places where the
    obvious implementation would quietly damage the training set.

    **Words are selected by aspect ratio, never stretched to fit.** The painter
    asks for a word of a given height and width. Scaling a four-letter word up
    to a ten-letter width would make every word on the page the same letter
    width, which is a cue real handwriting does not have and which a
    convolutional net will happily learn instead of structure. With 7,496
    cached words spanning three to fifteen characters there is always one whose
    natural proportions are close, so the word is chosen rather than deformed.

    **Pseudo-words are mixed in, built from lowercase glyphs.** The word corpus
    is handwritten *names* from records forms, so it is almost entirely
    uppercase: of 330k rows, seventeen contain a lowercase letter. All-caps
    text has no descenders and a flat upper edge, but `ProceduralInk` modelled
    ascenders and descenders deliberately, because the ragged band they create
    is part of what prose looks like from a distance, which is precisely the
    scale this classifier works at. So a fraction of words are composed from
    Chars74K lowercase characters, scaled per letter class so ascenders rise
    and descenders drop below the baseline.
    """

    name = "corpus"

    # Letter classes, which is what puts the rag into the band.
    ASCENDERS = frozenset("bdfhklt")
    DESCENDERS = frozenset("gjpqy")

    def __init__(
        self,
        cache_dir: Path | None = None,
        compose_probability: float = 0.45,
    ):
        from .corpus import DEFAULT_CACHE_DIR, CorpusCache

        cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.glyphs = CorpusCache.load(cache_dir / "glyphs.npz")
        self.words = CorpusCache.load(cache_dir / "words.npz")
        self.compose_probability = float(compose_probability)

        # The mathematics corpus is optional on top of the other two: without
        # it, `math_symbol` keeps the inherited behaviour and formulas are
        # built from letters, exactly as before it existed.
        self.maths = None
        self._maths_by_kind: dict[str, np.ndarray] = {}
        try:
            self.maths = CorpusCache.load(cache_dir / "maths.npz")
        except (FileNotFoundError, OSError, ValueError):
            pass
        if self.maths is not None:
            operators = set(MATHS_OPERATORS)
            labels = self.maths.labels
            self._maths_by_kind = {
                "operator": np.array([i for i, l in enumerate(labels) if l in operators], dtype=np.int64),
                "operand": np.array([i for i, l in enumerate(labels) if l not in operators], dtype=np.int64),
            }
            if any(v.size == 0 for v in self._maths_by_kind.values()):
                self.maths = None
                self._maths_by_kind = {}

        # Aspect index for word selection: sorted once, searched per word.
        aspects = self.words.aspects
        self._order = np.argsort(aspects)
        self._sorted_aspects = aspects[self._order]

        # Lowercase glyphs only, for composition. Chars74K labels are the
        # characters themselves, so this is a filter rather than a lookup.
        self._lower = np.array(
            [i for i, lab in enumerate(self.glyphs.labels) if lab.isalpha() and lab.islower()],
            dtype=np.int64,
        )
        if self._lower.size == 0:          # a cache built with a small limit
            self._lower = np.arange(len(self.glyphs), dtype=np.int64)

    # ------------------------------------------------------------------ api

    def glyph(self, rng: np.random.Generator, height: int) -> np.ndarray:
        """One real handwritten character, scaled to `height`.

        Natural width is kept rather than forced square: a handwritten `l` and
        a handwritten `m` are not the same width, and the formula painter
        advances by the returned width, so keeping it right keeps the spacing
        right too.
        """
        alpha = self.glyphs.sample(rng)
        return self._scale_to_height(alpha, max(6, int(height)))

    def _maths_index(self, rng: np.random.Generator, kind: str) -> int:
        pool = self._maths_by_kind.get(kind)
        if pool is None or pool.size == 0:
            pool = self._maths_by_kind["operand"]
        return int(pool[rng.integers(0, pool.size)])

    def math_label(self, rng: np.random.Generator, kind: str = "operand") -> str:
        """The class name a `math_symbol` call with this rng would return."""
        if self.maths is None:
            raise RuntimeError("no mathematics corpus is loaded")
        return self.maths.labels[self._maths_index(rng, kind)]

    def math_symbol(
        self,
        rng: np.random.Generator,
        height: int,
        kind: str = "operand",
    ) -> np.ndarray:
        """A real handwritten operator or operand, scaled to `height`.

        Operators are the reason this exists. Every operand on a generated
        formula was already a real handwritten mark, but `+`, `-` and `=` were
        drawn as procedural strokes, so the one class with the highest
        confidence threshold in the policy was also the only one mixing real
        and synthetic ink within a single region.
        """
        if self.maths is None:
            return self.glyph(rng, height)
        height = max(4, int(height))
        index = self._maths_index(rng, kind)
        alpha = self.maths[index]

        label = self.maths.labels[index]
        target = max(2, int(round(height * MATHS_RELATIVE_HEIGHT.get(label, 1.0))))
        mark = self._scale_to_height(alpha, target)

        # Backstop against an unusually elongated crop of any class.
        limit = int(height * MATHS_MAX_ASPECT)
        if mark.shape[1] > limit:
            scaled_h = max(1, int(round(target * limit / mark.shape[1])))
            mark = cv2.resize(mark, (limit, scaled_h), interpolation=cv2.INTER_AREA)
            mark = np.clip(mark, 0.0, 1.0).astype(np.float32)
        return mark

    def word(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """One handwritten word, roughly `height` by `width`."""
        height = max(8, int(height))
        width = max(10, int(width))
        if rng.random() < self.compose_probability:
            return self._compose(rng, height, width)
        return self._select(rng, height, width)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _scale_to_height(alpha: np.ndarray, height: int) -> np.ndarray:
        h, w = alpha.shape
        out_w = max(2, int(round(w * height / max(1, h))))
        scaled = cv2.resize(alpha, (out_w, height), interpolation=cv2.INTER_AREA)
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    def _select(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """Pick the cached word whose proportions best match the request.

        A window of near neighbours is sampled rather than the single closest,
        or every region asking for the same box would receive the same word and
        the page would visibly repeat.
        """
        target = width / height
        pos = int(np.searchsorted(self._sorted_aspects, target))
        lo = max(0, pos - 24)
        hi = min(len(self._sorted_aspects), pos + 24)
        if hi <= lo:
            lo, hi = 0, len(self._sorted_aspects)
        choice = int(self._order[int(rng.integers(lo, hi))])
        return self._scale_to_height(self.words[choice], height)

    def _compose(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """Build a word from lowercase glyphs on a shared baseline.

        Each letter is scaled by its class, so the band gets an upper edge
        broken by ascenders and a lower edge broken by descenders. That ragged
        profile is the structural signature the classifier reads at this scale.
        """
        x_height = height * 0.58
        canvas_h = int(height * 1.3)
        baseline = int(canvas_h * 0.74)
        canvas = np.zeros((canvas_h, width + height), dtype=np.float32)

        x = 0.0
        tracking = height * rng.uniform(0.10, 0.20)
        while x < width:
            index = int(self._lower[rng.integers(0, len(self._lower))])
            label = self.glyphs.labels[index]
            if label in self.DESCENDERS:
                target_h, drop = x_height * 1.45, x_height * 0.42
            elif label in self.ASCENDERS:
                target_h, drop = x_height * 1.62, 0.0
            else:
                target_h, drop = x_height, 0.0

            mark = self._scale_to_height(self.glyphs[index], max(4, int(target_h)))
            mh, mw = mark.shape
            top = int(baseline - mh + drop)
            left = int(x)
            if top < 0 or top + mh > canvas_h or left + mw > canvas.shape[1]:
                break
            np.maximum(canvas[top:top + mh, left:left + mw], mark,
                       out=canvas[top:top + mh, left:left + mw])
            x += mw + tracking

        used = max(2, int(min(x, canvas.shape[1])))
        return np.clip(canvas[:, :used], 0.0, 1.0).astype(np.float32)


class MixedInk(InkProvider):
    """Any provider, with its mark types shuffled across classes.

    The painters ask for different *kinds* of mark: prose asks for words, a
    formula asks for maths symbols, a plot asks for almost nothing but pen
    strokes. With `CorpusInk` behind them each kind comes from a different
    corpus, so the glyph texture inside a crop identifies its class without
    the layout being read at all. Every backbone scoring above 0.99 macro-F1,
    hard variants included, is what that shortcut looks like from outside.

    Real notes do not keep their marks apart like that. Prose says "where
    x = 2", a table column holds `3.14`, a diagram box is labelled `f(x)`, a
    formula's operand is `sin` or `dx`. So this decorator makes each request
    sometimes return the other kind of mark, and the only thing left that
    tells the classes apart is how the marks are *arranged*.

    It is a decorator rather than a flag on `CorpusInk` so it composes with
    any provider and the painters never learn it exists.
    """

    def __init__(self, base: InkProvider, cross_rate: float = 0.35):
        self.base = base
        self.cross_rate = float(cross_rate)
        self.name = f"mixed({base.name})"

    def glyph(self, rng: np.random.Generator, height: int) -> np.ndarray:
        return self.base.glyph(rng, height)

    def word(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        if rng.random() < self.cross_rate:
            return self._math_run(rng, height, width)
        return self.base.word(rng, height, width)

    def math_symbol(self, rng: np.random.Generator, height: int, kind: str = "operand") -> np.ndarray:
        # Operators stay operators: a `=` is a `=` in prose too. It is the
        # operands that are written as letters and short words in real maths.
        if kind == "operand" and rng.random() < self.cross_rate:
            if rng.random() < 0.5:
                return self.base.glyph(rng, height)
            return self.base.word(rng, height, int(height * rng.uniform(1.4, 2.6)))
        return self.base.math_symbol(rng, height, kind=kind)

    def _math_run(self, rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        """A few maths symbols on one baseline, sized like the word it replaces."""
        height = max(8, int(height))
        canvas = np.zeros((int(height * 1.2), max(10, int(width)) + height), dtype=np.float32)
        x = 0
        while x < width:
            kind = "operator" if rng.random() < 0.3 else "operand"
            mark = self.base.math_symbol(rng, int(height * 0.8), kind=kind)
            mh, mw = mark.shape
            top = max(0, (canvas.shape[0] - mh) // 2)
            if x + mw > canvas.shape[1] or top + mh > canvas.shape[0]:
                break
            np.maximum(canvas[top:top + mh, x:x + mw], mark, out=canvas[top:top + mh, x:x + mw])
            x += mw + max(1, int(height * 0.12))
        return canvas[:, :max(2, x)]


def available_providers() -> dict[str, InkProvider]:
    """Every provider that can be used right now.

    A corpus-backed provider registers here once its data is present, so the
    training CLI can offer `--ink procedural` or `--ink corpus` without any
    other code knowing which exists. The corpus is never a hard dependency:
    if the cache has not been built, the pipeline still runs on procedural ink,
    which is the entire reason the interface exists.
    """
    providers: dict[str, InkProvider] = {"procedural": ProceduralInk()}
    try:
        providers["corpus"] = CorpusInk()
    except (FileNotFoundError, OSError, ValueError):
        pass
    return providers
