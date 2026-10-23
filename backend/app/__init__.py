"""
Palimpsest — note compiler core.

UCS668 · Edge AI and Robotics: Data Center Vision

A rough handwritten page goes in. A properly typeset engineering notebook comes
out: structured prose, typeset mathematics with the arithmetic already done, and
real plotted figures. Everything runs locally.

The package is laid out along the pipeline:

    app.data        Module 4 — the page generator (synthetic layout, real ink),
                    the augmentation bank, and the Lightning data module.
    app.models      Modules 2 & 3 — the switchable backbone zoo and the
                    architectural profiler (params, MACs, tensor shapes).
    app.pipeline    Modules 1 & 4 — preprocessing kernels, segmentation, and the
                    per-class confidence gate.
    app.routers     Content handlers: transcription specialists, the page symbol
                    table, and the expression solver.
    app.metrics     Module 5 — confusion matrix, macro-F1, calibration error and
                    the Pareto frontier accumulator.
    app.main        The FastAPI ASGI surface.

This module holds only the constants that several of those packages must agree
on. Nothing here imports a third-party library, so it is safe to import from
anywhere, including tests that run without torch installed.
"""

from __future__ import annotations

__version__ = "0.1.0"
__course__ = "UCS668 — Edge AI and Robotics: Data Center Vision"

# ---------------------------------------------------------------------------
# The five region classes
# ---------------------------------------------------------------------------
# Index order is load-bearing. It fixes the row and column order of the 5x5
# confusion matrix, the output layer of every backbone, and the axis labels in
# the frontend. Changing this order invalidates every trained checkpoint.

CLASS_NAMES: tuple[str, ...] = (
    "THEORY",        # 0  prose, sentences, shorthand, bullets
    "FORMULA",       # 1  equations, symbols, fractions, subscripts
    "PLOT_SKETCH",   # 2  hand-drawn axes, curves, bar sketches
    "DIAGRAM",       # 3  boxes, arrows, block diagrams
    "TABLE",         # 4  ruled grids, columns of values
)

NUM_CLASSES: int = len(CLASS_NAMES)

CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Per-class confidence thresholds
# ---------------------------------------------------------------------------
# A single global threshold would be wrong, because the classes do not carry
# equal risk. Each threshold is set by two things together: how harmful a wrong
# reading is, and how likely the reader is to NOTICE it is wrong. A garbled
# sentence is self-evident. A wrong subscript is invisible.
#
# These are an initial guess, not a result. Each is a free parameter to be
# fitted on the validation split against the expected-cost model below, keeping
# these values as the baseline to beat. See spec section 8.

THRESHOLDS: dict[str, float] = {
    "THEORY": 0.60,       # lowest: prose errors are obvious and trivially fixed
    "FORMULA": 0.88,      # highest: a wrong subscript silently corrupts the note
    "PLOT_SKETCH": 0.75,  # a wrong figure misleads, but looks wrong when plotted
    "DIAGRAM": 0.80,      # structure is easy to get subtly wrong
    "TABLE": 0.85,        # transposed cells read as perfectly plausible
}


# ---------------------------------------------------------------------------
# The expected-cost model that the thresholds are fitted against
# ---------------------------------------------------------------------------
# Two things go wrong and they are not equally bad. Weights carry over from the
# previous project's miss_cost scheme, where severity was non-uniform for the
# same reason. Units are arbitrary but relative.

MISS_COST: dict[str, float] = {
    "THEORY": 1.0,
    "FORMULA": 9.0,
    "PLOT_SKETCH": 4.0,
    "DIAGRAM": 5.0,
    "TABLE": 6.0,
}

# Cost of holding a CORRECT region for review. Deliberately low, which pushes
# fitted thresholds up so the compiler holds more than it inserts: safer notes,
# more review friction. One number to revisit against real usage.
REVIEW_COST: float = 0.5


# ---------------------------------------------------------------------------
# Module 1 — the interactive compile budget
# ---------------------------------------------------------------------------
# A page should compile fast enough to feel immediate. With N regions the
# per-region share is COMPILE_BUDGET_MS / N. Service time per region is
# preprocess + infer + route, all measured, and the load factor is total service
# time over budget.
#
# The finding this is expected to produce, stated up front so the measurement is
# a check rather than a reveal: the CNN is not the bottleneck. Classification
# costs 4.5 to 44 ms per region while transcription costs hundreds. Routing
# dominates, which is why transcription uses small specialists rather than one
# large vision-language model.

COMPILE_BUDGET_MS: float = 2000.0


# ---------------------------------------------------------------------------
# Plotting a formula with one free symbol
# ---------------------------------------------------------------------------
# When the page states a domain, like `0 < x < 10`, that statement wins: the
# writer knows where their function is meaningful. These values only cover what
# the page did not say.

# Span to use when the free symbol is bounded on one side only, or not at all.
PLOT_DEFAULT_SPAN: float = 10.0

# A strict bound is approached, not touched. `0 < x` starts sampling just inside
# zero, which also keeps the sampler off a pole sitting exactly on the bound.
PLOT_STRICT_EPSILON: float = 1e-3

# Samples across the range. Enough to look smooth, few enough to stay cheap.
PLOT_SAMPLES: int = 400


# ---------------------------------------------------------------------------
# Human-facing metadata
# ---------------------------------------------------------------------------
# Deliberately carries no colour. The design system in .tastemaker/style-lock.md
# is the single source of truth for colour, and duplicating hex values here is
# how the two drift apart.

CLASS_META: tuple[dict, ...] = (
    {
        "id": 0,
        "key": "THEORY",
        "label": "Theory",
        "blurb": "Prose, sentences, shorthand, bullet lists.",
        "handler": "line split, then handwriting transcription",
    },
    {
        "id": 1,
        "key": "FORMULA",
        "label": "Formula",
        "blurb": "Equations, symbols, fractions, subscripts.",
        "handler": "formula recognition to LaTeX, then the solver",
    },
    {
        "id": 2,
        "key": "PLOT_SKETCH",
        "label": "Plot sketch",
        "blurb": "Hand-drawn axes, curves and bar sketches.",
        "handler": "paired with a plottable equation from the page",
    },
    {
        "id": 3,
        "key": "DIAGRAM",
        "label": "Diagram",
        "blurb": "Boxes, arrows, block diagrams, circuits.",
        "handler": "none in this build, always held for review",
    },
    {
        "id": 4,
        "key": "TABLE",
        "label": "Table",
        "blurb": "Ruled grids and columns of values.",
        "handler": "cell split, then transcription per cell",
    },
)


def class_name(index: int) -> str:
    """Map a model output index to its class name."""
    return CLASS_NAMES[index]


def threshold_for(name: str) -> float:
    """The confidence a region of this class must reach to be written."""
    return THRESHOLDS[name]
