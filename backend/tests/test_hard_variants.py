"""Hard-variant tests.

The first generator produced a caricature of each class: every TABLE had a
complete ruled grid, every PLOT_SKETCH two long perpendicular axes, every
FORMULA a sparse centred row. One feature separated each class perfectly, and
a 154k-parameter network trained from scratch reached macro-F1 1.000 on held
out pages in 43 seconds. Nothing was leaking; the task was simply trivial, and
a trivial task makes the confusion matrix, the calibration error and the
Pareto frontier all unreportable.

Real notebooks are ambiguous. People write tables with no ruling, diagrams
that are mostly text, formulas that wrap across lines like prose, and axes
with nothing plotted on them yet. Each hard variant below reproduces one of
those, and each one is defined by *losing* the feature that gave its class
away, so the tests assert the absence of that feature rather than the
presence of some new decoration.
"""

import numpy as np
import pytest

import cv2

from app.data.ink import ProceduralInk
from app.data.synth import PageGenerator, RegionSpec


def long_runs(mask: np.ndarray, box, axis: str = "h", min_length: int = 60) -> int:
    """Count pixels belonging to straight runs at least `min_length` long.

    The structural feature that defines a ruled table and a plot axis, and the
    thing a hard variant has to give up.
    """
    x, y, w, h = box
    patch = (mask[y:y + h, x:x + w] > 0.3).astype(np.uint8) * 255
    if patch.size == 0:
        return 0
    size = (min_length, 1) if axis == "h" else (1, min_length)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    return int((cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel) > 0).sum())


def paint(generator, painter_name, region_class, hard, seed=0, w=420, h=260):
    mask = np.zeros((h + 80, w + 80), dtype=np.float32)
    spec = RegionSpec(region_class, 40, 40, w, h, hard=hard)
    getattr(generator, painter_name)(mask, spec, np.random.default_rng(seed))
    return mask, spec.box


@pytest.fixture(scope="module")
def gen():
    return PageGenerator(ink=ProceduralInk())


class TestRegionSpecCarriesTheFlag:
    def test_defaults_to_easy(self):
        assert RegionSpec("TABLE", 0, 0, 10, 10).hard is False

    def test_flag_round_trips_through_the_sidecar(self, gen):
        page = PageGenerator(ink=ProceduralInk(), hard_fraction=1.0).page(1)
        assert all(r["hard"] for r in page.sidecar()["regions"])


class TestHardFraction:
    def test_zero_produces_no_hard_regions(self):
        generator = PageGenerator(ink=ProceduralInk(), hard_fraction=0.0)
        assert not any(r.hard for seed in range(12) for r in generator.page(seed).regions)

    def test_one_produces_only_hard_regions(self):
        generator = PageGenerator(ink=ProceduralInk(), hard_fraction=1.0)
        assert all(r.hard for seed in range(12) for r in generator.page(seed).regions)

    def test_a_fraction_produces_a_mixture(self):
        generator = PageGenerator(ink=ProceduralInk(), hard_fraction=0.4)
        flags = [r.hard for seed in range(40) for r in generator.page(seed).regions]
        assert 0.2 < sum(flags) / len(flags) < 0.6, sum(flags) / len(flags)


class TestTableLosesItsRuling:
    """An unruled table is column-aligned text, which is what makes it
    confusable with prose. Ruling is the one feature that gave it away."""

    def test_hard_tables_have_far_fewer_long_rules(self, gen):
        easy = np.mean([long_runs(*paint(gen, "_paint_table", "TABLE", False, s))
                        for s in range(6)])
        hard = np.mean([long_runs(*paint(gen, "_paint_table", "TABLE", True, s))
                        for s in range(6)])
        assert hard < easy * 0.5, (easy, hard)

    def test_hard_tables_still_contain_ink(self, gen):
        mask, box = paint(gen, "_paint_table", "TABLE", True, 3)
        assert float(mask.max()) > 0.3


def vertical_structures(mask, box) -> int:
    """Distinct long vertical runs. One is just the y-axis; several are bars."""
    x, y, w, h = box
    patch = (mask[y:y + h, x:x + w] > 0.3).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    opened = cv2.morphologyEx(patch, cv2.MORPH_OPEN, kernel)
    count, _ = cv2.connectedComponents((opened > 0).astype(np.uint8))
    return count - 1


class TestPlotLosesItsSmoothCurve:
    """What makes a plot sketch unmistakable is the smooth diagonal curve
    sweeping across an empty interior. Both hard variants drop it: one
    replaces it with upright bars, the other has not drawn it yet."""

    def test_every_hard_plot_is_bars_or_bare_axes(self, gen):
        easy_interior = np.mean([
            float((lambda m, b: (m[b[1] + b[3] // 4: b[1] + b[3] * 3 // 4,
                                   b[0] + b[2] // 4: b[0] + b[2] * 3 // 4] > 0.3).mean())
                  (*paint(gen, "_paint_plot", "PLOT_SKETCH", False, s)))
            for s in range(8)
        ])
        for seed in range(12):
            mask, box = paint(gen, "_paint_plot", "PLOT_SKETCH", True, seed)
            x, y, w, h = box
            interior = float((mask[y + h // 4: y + h * 3 // 4,
                                   x + w // 4: x + w * 3 // 4] > 0.3).mean())
            is_bars = vertical_structures(mask, box) >= 3
            assert is_bars or interior < easy_interior, (seed, interior, easy_interior)


class TestDiagramBecomesTextHeavy:
    """A block diagram whose boxes are full of writing reads as prose."""

    def test_hard_diagrams_carry_more_ink(self, gen):
        def coverage(mask, box):
            x, y, w, h = box
            return float((mask[y:y + h, x:x + w] > 0.3).mean())

        easy = np.mean([coverage(*paint(gen, "_paint_diagram", "DIAGRAM", False, s))
                        for s in range(6)])
        hard = np.mean([coverage(*paint(gen, "_paint_diagram", "DIAGRAM", True, s))
                        for s in range(6)])
        assert hard > easy, (easy, hard)


class TestFormulaBecomesProseLike:
    """A long derivation wraps across several lines and stops being sparse
    and centred."""

    def test_hard_formulas_occupy_more_lines(self, gen):
        def ink_rows(mask, box):
            x, y, w, h = box
            patch = mask[y:y + h, x:x + w] > 0.3
            return int((patch.sum(axis=1) > 0).sum())

        easy = np.mean([ink_rows(*paint(gen, "_paint_formula", "FORMULA", False, s))
                        for s in range(6)])
        hard = np.mean([ink_rows(*paint(gen, "_paint_formula", "FORMULA", True, s))
                        for s in range(6)])
        assert hard > easy * 1.3, (easy, hard)


class TestHardPlotIsAmbiguousNotEmpty:
    """Removing the curve must leave a frame that looks like a diagram, not a
    blank crop. An almost-empty region is degenerate rather than confusable,
    and it would teach the classifier that PLOT_SKETCH means 'nearly nothing'."""

    def test_axes_survive(self, gen):
        runs = [long_runs(*paint(gen, "_paint_plot", "PLOT_SKETCH", True, s), axis="h")
                for s in range(6)]
        assert min(runs) > 0, runs

    def test_some_hard_plots_are_bar_sketches(self, gen):
        """`CLASS_META` already describes this class as 'axes, curves and bar
        sketches', and a bar sketch is the genuinely ambiguous one: upright
        rectangles standing on an axis share their vertical runs with a ruled
        table and their closed outlines with a block diagram. Bare axes alone
        left the crop nearly empty, which is degenerate rather than hard."""
        with_bars = sum(
            vertical_structures(*paint(gen, "_paint_plot", "PLOT_SKETCH", True, s)) >= 3
            for s in range(12)
        )
        assert with_bars >= 3, with_bars

    def test_enough_ink_remains_to_be_a_region(self, gen):
        for seed in range(6):
            mask, (x, y, w, h) = paint(gen, "_paint_plot", "PLOT_SKETCH", True, seed)
            coverage = float((mask[y:y + h, x:x + w] > 0.3).mean())
            assert coverage > 0.015, (seed, coverage)


class TestHardTheoryBecomesSparse:
    """Prose written as a short bulleted list stops being a dense block and
    starts looking like a column of values or a stack of equations."""

    def test_hard_theory_is_less_dense(self, gen):
        def coverage(mask, box):
            x, y, w, h = box
            return float((mask[y:y + h, x:x + w] > 0.3).mean())

        easy = np.mean([coverage(*paint(gen, "_paint_theory", "THEORY", False, s))
                        for s in range(6)])
        hard = np.mean([coverage(*paint(gen, "_paint_theory", "THEORY", True, s))
                        for s in range(6)])
        assert hard < easy * 0.8, (easy, hard)

    def test_hard_theory_lines_are_shorter(self, gen):
        def max_line_width(mask, box):
            x, y, w, h = box
            patch = mask[y:y + h, x:x + w] > 0.3
            cols = patch.any(axis=0)
            return float(cols.mean())

        easy = np.mean([max_line_width(*paint(gen, "_paint_theory", "THEORY", False, s))
                        for s in range(6)])
        hard = np.mean([max_line_width(*paint(gen, "_paint_theory", "THEORY", True, s))
                        for s in range(6)])
        assert hard < easy, (easy, hard)


class TestStillDeterministic:
    def test_same_seed_same_page_with_hard_variants(self):
        generator = PageGenerator(ink=ProceduralInk(), hard_fraction=0.4)
        assert np.array_equal(generator.page(5).image, generator.page(5).image)
        assert generator.page(5).regions == generator.page(5).regions
