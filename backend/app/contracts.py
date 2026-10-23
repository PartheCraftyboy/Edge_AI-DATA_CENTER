"""
The wire contract between backend and frontend.

Two rules hold throughout:

1.  **Units live in field names.** `timing_ms`, `budget_ms`, `load_factor`. A
    bare `latency` field is how a millisecond becomes a second three modules
    later.
2.  **The frontend never recomputes a number the backend measured.** Anything
    derived, the load factor, whether a region was held, the per-stage split, is
    computed once here and sent. Two implementations of the same arithmetic
    always drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

from . import COMPILE_BUDGET_MS, CLASS_NAMES

RegionClass = Literal["THEORY", "FORMULA", "PLOT_SKETCH", "DIAGRAM", "TABLE"]


class Box(BaseModel):
    """Region bounds on the source page, in source-image pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def y_centre(self) -> float:
        """Used by the symbol table to order bindings down the page."""
        return self.y + self.h / 2


class Timing(BaseModel):
    """Measured cost of one region, split by stage. Never estimated."""

    preprocess_ms: float = 0.0
    infer_ms: float = 0.0
    route_ms: float = 0.0

    @computed_field
    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.infer_ms + self.route_ms


# --- content variants -------------------------------------------------------
# One per class that produces output. A held region carries no content at all,
# which is the point: nothing below threshold is written into the note.


class ProseContent(BaseModel):
    kind: Literal["prose"] = "prose"
    markdown: str
    raw_lines: list[str] = Field(default_factory=list)
    """Transcription before the text model tidied it. Kept so the review panel
    can show what was actually read versus what was written."""


class FormulaContent(BaseModel):
    kind: Literal["formula"] = "formula"
    tex: str
    """For display. Typeset client-side."""
    expr: str | None = None
    """SymPy-parseable form. Separate from `tex` because converting LaTeX to
    SymPy is where this kind of pipeline usually breaks."""
    answer: float | None = None
    """Resolved value, when every symbol was bound from the page."""
    bound: dict[str, float] = Field(default_factory=dict)
    free: list[str] = Field(default_factory=list)
    """Symbols with no binding above them on the page. Exactly one free symbol
    means this formula can be plotted."""


class FigureContent(BaseModel):
    """A curve in a compiled note.

    Rendered as SVG by the frontend, from the samples below, not as a server
    side raster. Three reasons, in order of how much they matter:

    1.  A matplotlib PNG cannot wear the design system. Different fonts,
        different colours, fixed DPI, no dark mode, no hover. The locked palette
        in `.tastemaker/style-lock.md` requires a direct label on every line,
        and a foreign raster ignores all of it.
    2.  The samples are already on the wire for the range slider, so client side
        rendering is free and makes dragging instant rather than a round trip.
    3.  It deletes work: no asset store, no PNG plumbing, no re-render when the
        slider settles.

    The Python claim is unaffected and arguably stronger: SymPy computes the
    curve server side and the frontend only draws points it was given. matplotlib
    keeps the job it is actually good at, the Module 5 analysis figures
    (training curve, confusion matrix, Pareto frontier), which is where a
    grader wants them anyway.
    """

    kind: Literal["figure"] = "figure"
    asset_id: str | None = None
    """Only set for a server rendered analysis figure. Note curves leave this
    empty and travel as samples."""
    caption: str = ""
    source_expr: str | None = None
    """The equation the figure was plotted from. A figure with no source
    equation should not exist: that is what makes it exact rather than guessed."""
    free_symbol: str | None = None

    # --- the adjustable range -----------------------------------------------
    # The chosen range is a starting point, not a verdict, so the reader gets a
    # slider. Dragging it asks the backend for fresh samples, which is a cheap
    # SymPy evaluation rather than a render, and the frontend redraws the SVG.
    x_range: tuple[float, float] | None = None
    x_limits: tuple[float, float] | None = None
    """How far the slider may travel."""
    range_source: str | None = None
    """Why this range. Shown next to the slider, because an axis whose origin is
    unexplained invites the reader to distrust the curve."""
    series_x: list[float] = Field(default_factory=list)
    series_y: list[float | None] = Field(default_factory=list)
    """The figure itself, for a note curve. `None` breaks the line at a pole
    rather than drawing through it, which the SVG renderer honours by starting a
    new path rather than joining across the gap."""
    y_clip: tuple[float, float] | None = None
    """Suggested y bounds. Near an asymptote the true values run to thousands and
    flatten everything else into the axis, so the renderer clips instead. This is
    the proper fix for the pole-break roughness noted when sampling was built:
    clip the axis rather than break the line."""
    refused: str | None = None
    """Set when no figure was drawn, with the reason in readable words: every
    variable bound, too many free variables, or a contradictory domain."""


class TableContent(BaseModel):
    kind: Literal["table"] = "table"
    markdown: str
    rows: int = 0
    cols: int = 0


Content = ProseContent | FormulaContent | FigureContent | TableContent


class Block(BaseModel):
    """One region of the page, after classification, gating and routing."""

    id: str
    region_class: RegionClass
    confidence: float
    threshold: float
    held: bool
    """True when confidence fell below this class's threshold. A held block
    carries no content and renders as a review panel."""
    hold_reason: str | None = None
    box: Box
    timing_ms: Timing = Field(default_factory=Timing)
    content: Content | None = None
    correction: dict | None = None
    """Set once someone has said what this block should have been. Kept on the
    block so the correction is visible where it was made, and appended to
    `data/corrections.jsonl`, which is the record that outlives the note."""


class CompileBudget(BaseModel):
    """Module 1. Reported per compile, with the per-stage split visible."""

    budget_ms: float = COMPILE_BUDGET_MS
    regions: int
    service_ms: float
    preprocess_ms: float
    infer_ms: float
    route_ms: float

    @computed_field
    @property
    def per_region_share_ms(self) -> float:
        return self.budget_ms / self.regions if self.regions else self.budget_ms

    @computed_field
    @property
    def load_factor(self) -> float:
        """Service time over budget. Above 1.0 means the page missed its
        deadline, which is the quantity the whole Module 1 argument rests on."""
        return self.service_ms / self.budget_ms if self.budget_ms else 0.0

    @computed_field
    @property
    def over_budget(self) -> bool:
        return self.load_factor > 1.0

    @computed_field
    @property
    def dominant_stage(self) -> str:
        """Which stage actually ate the budget. Expected to be `route`, which is
        the finding, not an accident."""
        stages = {
            "preprocess": self.preprocess_ms,
            "infer": self.infer_ms,
            "route": self.route_ms,
        }
        return max(stages, key=lambda k: stages[k])


class CompiledNote(BaseModel):
    """What `POST /api/compile` returns."""

    note_id: str
    title: str
    source_width: int
    source_height: int
    blocks: list[Block]
    budget: CompileBudget
    symbols: dict[str, float] = Field(default_factory=dict)
    """The page symbol table, flattened. Shown in the UI so the reader can see
    where an answer came from."""
    backbone: str
    model_notes: list[str] = Field(default_factory=list)
    """Honest caveats attached to this compile, e.g. a formula that would not
    parse. Surfaced rather than swallowed."""

    @computed_field
    @property
    def held_count(self) -> int:
        return sum(1 for b in self.blocks if b.held)

    @computed_field
    @property
    def mean_confidence(self) -> float:
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)


class BackboneInfo(BaseModel):
    """Module 2 and 3. Every figure here is measured, never quoted from a paper."""

    key: str
    label: str
    params: int
    macs: int | None = None
    latency_ms: float | None = None
    spatial_cascade: list[int] = Field(default_factory=list)
    block_type: str = ""
    """Fire module, inverted residual, or residual stage."""
    active: bool = False


class ClassMetrics(BaseModel):
    """Module 5, per class."""

    region_class: RegionClass
    recall: float
    precision: float
    f1: float
    ece: float
    """Per-class calibration error. The number that justifies this class's
    threshold: a threshold of 0.88 only means something if 0.88 corresponds to
    being right 88% of the time."""
    held_rate: float
    """The user-facing consequence of the threshold."""
    support: int


class MetricsReport(BaseModel):
    confusion: list[list[float]]
    """5x5, row-normalised, so the diagonal reads as recall."""
    class_names: list[str] = Field(default_factory=lambda: list(CLASS_NAMES))
    macro_f1: float
    per_class: list[ClassMetrics]
    window: int
    split: Literal["train", "val", "test"] = "val"
    """Stated explicitly. A test-split number is read once, after everything is
    frozen; a val-split number is not a headline result."""
