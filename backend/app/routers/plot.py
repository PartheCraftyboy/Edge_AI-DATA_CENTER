"""
Turning a resolved formula into a curve.

**Two dimensions only, and that is a decision rather than a limitation.**

A formula is plotted when it has exactly one free symbol: that symbol is the
x-axis, the expression is the y-axis. Two independent variables would mean a
surface, and a surface is the wrong answer here for three reasons. It reads
badly on a page meant to be skimmed, it cannot carry the direct value labels
that the design system's contrast relief requires, and nobody sketches a 3D
surface in a lecture and expects it back. Two or more free symbols therefore
produce no figure at all, and the formula simply typesets.

Range selection follows a strict order of authority:

1.  What the page said. A harvested constraint like `0 < rho < 1` is the
    writer's own statement about where their function is meaningful.
2.  The magnitude of the values on the page. Values in the tens get a range of
    ten, values in the hundreds get a hundred, and so on down to fractions.
3.  A flat default, when the page offers nothing to go on.

Whatever is chosen is only a starting point: the range is sent to the frontend
with travel limits so the reader can move it with a slider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .. import PLOT_DEFAULT_SPAN, PLOT_SAMPLES, PLOT_STRICT_EPSILON
from .constraints import Constraint
from .solver import Solved

# One independent variable. See the module docstring.
MAX_FREE_SYMBOLS: int = 1

# How far the slider may travel beyond the chosen range. Generous enough to
# explore, bounded so the control stays usable.
SLIDER_TRAVEL: float = 4.0


@dataclass(frozen=True)
class PlotRange:
    x_min: float
    x_max: float
    source: str
    """Where the range came from, surfaced in the UI so a reader is never
    looking at an axis whose origin is unexplained."""
    limit_min: float
    limit_max: float
    """Travel limits for the slider."""

    @property
    def span(self) -> float:
        return self.x_max - self.x_min


def _nice_span(values: list[float]) -> float:
    """A round span matching the magnitude of the numbers on the page.

    Values in the tens give ten, hundreds give a hundred, tenths give a tenth.
    A page about milliseconds and a page about years should not both get the
    same axis just because the default was written for one of them.
    """
    mags = [abs(v) for v in values if v and math.isfinite(v)]
    if not mags:
        return PLOT_DEFAULT_SPAN
    biggest = max(mags)
    if biggest <= 0:
        return PLOT_DEFAULT_SPAN
    return 10.0 ** math.floor(math.log10(biggest))


def choose_plot_range(
    solved: Solved,
    bindings: dict[str, float],
    constraint: Constraint | None = None,
) -> PlotRange | None:
    """Pick the x-axis range, or None when no figure should be drawn."""
    if not solved.plottable:
        return None

    # A page that contradicts itself gets a note, not an inverted axis.
    if constraint is not None and constraint.empty:
        return None

    # 1. The page's own statement wins outright.
    if constraint is not None and constraint.bounded:
        lo, hi = constraint.lower, constraint.upper
        # A strict bound is approached, never touched. This also steps the
        # sampler off a pole, which very often sits exactly on such a bound.
        pad = (hi - lo) * PLOT_STRICT_EPSILON
        if constraint.lower_strict:
            lo += pad
        if constraint.upper_strict:
            hi -= pad
        return PlotRange(
            lo, hi,
            source=f"page states {constraint.describe()}",
            limit_min=lo - (hi - lo) * SLIDER_TRAVEL,
            limit_max=hi + (hi - lo) * SLIDER_TRAVEL,
        )

    span = _nice_span(list(bindings.values()))

    # 2. Bounded on one side only: start where the page said and extend by a
    #    span that suits the numbers in play.
    if constraint is not None and constraint.lower is not None:
        lo = constraint.lower + span * PLOT_STRICT_EPSILON if constraint.lower_strict else constraint.lower
        return PlotRange(
            lo, lo + span,
            source=f"page states {constraint.describe()}, extended by {span:g}",
            limit_min=lo, limit_max=lo + span * SLIDER_TRAVEL,
        )

    if constraint is not None and constraint.upper is not None:
        hi = constraint.upper - span * PLOT_STRICT_EPSILON if constraint.upper_strict else constraint.upper
        return PlotRange(
            hi - span, hi,
            source=f"page states {constraint.describe()}, extended by {span:g}",
            limit_min=hi - span * SLIDER_TRAVEL, limit_max=hi,
        )

    # 3. Nothing stated. Start at zero, which is almost always the meaningful
    #    origin for a quantity someone wrote in a notebook.
    return PlotRange(
        0.0, span,
        source=f"no stated domain, span {span:g} from the values on the page",
        limit_min=-span * SLIDER_TRAVEL, limit_max=span * SLIDER_TRAVEL,
    )


def plot_series(
    solved: Solved,
    plot_range: PlotRange,
    samples: int = PLOT_SAMPLES,
) -> tuple[list[float], list[float | None]]:
    """Sample the expression across the range.

    Returns `(xs, ys)` where a `y` of None marks a point that should break the
    line rather than be drawn through: a pole, a complex result, or a value
    outside the real domain. Drawing straight through a singularity produces a
    near-vertical stroke that looks like data and is not.
    """
    if not solved.plottable or solved.expr is None:
        return [], []

    symbol = sp.Symbol(solved.free[0])
    try:
        f = sp.lambdify(symbol, solved.expr, modules=["numpy"])
    except Exception:
        return [], []

    xs = np.linspace(plot_range.x_min, plot_range.x_max, samples)
    with np.errstate(all="ignore"):
        try:
            raw = f(xs)
        except Exception:
            return [], []
        values = np.asarray(raw, dtype=complex) * np.ones_like(xs, dtype=complex)

    ys: list[float | None] = []
    for v in values:
        if abs(v.imag) > 1e-9 or not math.isfinite(v.real):
            ys.append(None)
        else:
            ys.append(float(v.real))

    ys = _break_at_poles(ys)
    return [float(x) for x in xs], ys


def _break_at_poles(ys: list[float | None], jump_factor: float = 50.0) -> list[float | None]:
    """Insert breaks where the curve jumps implausibly between samples.

    A pole is rarely sampled exactly, so the values either side are finite but
    wildly apart and the renderer would join them with a vertical line. The
    jump is compared against the typical step so the test adapts to the
    function's own scale rather than using an absolute threshold.
    """
    finite = [y for y in ys if y is not None]
    if len(finite) < 3:
        return ys

    steps = [
        abs(b - a)
        for a, b in zip(finite, finite[1:])
        if a is not None and b is not None
    ]
    if not steps:
        return ys
    typical = float(np.median(steps))
    if typical <= 0:
        return ys

    out = list(ys)
    for i in range(1, len(out)):
        a, b = out[i - 1], out[i]
        if a is None or b is None:
            continue
        if abs(b - a) > typical * jump_factor:
            out[i] = None
    return out


def suggest_y_clip(
    ys: list[float | None],
    percentile: float = 2.0,
    headroom: float = 0.08,
) -> tuple[float, float] | None:
    """Suggested y bounds for the renderer, or None when the data needs none.

    Near an asymptote the true values run to thousands while everything else
    sits between 0 and 5, so an auto-scaled axis flattens the interesting part
    into a line along the bottom. Clipping to a robust percentile band keeps the
    shape readable and lets the steep part run off the top, which is how a
    person would draw it.

    This is the right fix for a roughness noted when sampling was written: the
    pole-break heuristic was stopping a steep curve short of its bound. Clipping
    the axis handles the same problem without discarding samples.
    """
    finite = [y for y in ys if y is not None and math.isfinite(y)]
    if len(finite) < 8:
        return None

    arr = np.asarray(finite, dtype=float)
    lo = float(np.percentile(arr, percentile))
    hi = float(np.percentile(arr, 100.0 - percentile))
    full_lo, full_hi = float(arr.min()), float(arr.max())

    # If the band already covers nearly everything, the data has no outliers
    # worth clipping and the renderer should just autoscale.
    span = hi - lo
    if span <= 0:
        return None
    if (full_hi - full_lo) <= span * 1.5:
        return None

    pad = span * headroom
    return lo - pad, hi + pad


def refuse_reason(solved: Solved) -> str | None:
    """Why no figure was drawn, in words a reader can act on."""
    if solved.expr is None:
        return "the expression could not be parsed"
    n = len(solved.free)
    if n == 0:
        return "every variable is bound, so this is a value rather than a curve"
    if n > MAX_FREE_SYMBOLS:
        free = ", ".join(solved.free)
        return (
            f"{n} free variables ({free}). A curve needs exactly one, and "
            "surfaces are deliberately not drawn"
        )
    return None
