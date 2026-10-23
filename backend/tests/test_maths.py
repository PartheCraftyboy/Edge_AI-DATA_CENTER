"""Handwritten mathematics corpus tests.

This corpus closes the gap the spec called out: neither handwriting dataset
contains an operator, so `_paint_formula` drew `+` and `-` as procedural
strokes while every operand was a real letter. That mismatch sat inside the
one class that matters most, because `FORMULA` carries the highest confidence
threshold (0.88) and the highest miss cost (9.0) in the whole policy.

The corpus has its own contamination: roughly one sample in eight carries a
stray underline from the source scan. A spurious horizontal run inside a
formula crop is not noise, it is a fraction bar, which is a real structural
feature of the class. So it has to be rejected rather than tolerated.
"""

from pathlib import Path

import numpy as np
import pytest

from app.data.corpus import DEFAULT_CACHE_DIR, build_maths_cache, clean_symbol
from app.data.ink import MATHS_OPERATORS, CorpusInk, ProceduralInk

MATHS = Path(__file__).resolve().parent.parent / "data" / "corpus" / "maths" / "dataset"
has_maths = MATHS.exists()
has_cache = (DEFAULT_CACHE_DIR / "maths.npz").exists()


def canvas(h=135, w=155):
    return np.full((h, w), 243, np.uint8)


class TestSymbolCleaning:
    def test_rejects_a_stray_underline(self):
        """The artefact actually present in this corpus."""
        img = canvas()
        img[40:90, 60:100] = 30        # the symbol
        img[120:124, 10:145] = 30      # the underline, spanning the frame
        assert clean_symbol(img) is None

    def test_keeps_a_clean_symbol(self):
        img = canvas()
        img[40:90, 60:100] = 30
        assert clean_symbol(img) is not None

    def test_keeps_a_minus_sign(self):
        """A `sub` glyph IS a horizontal run, so the rejection must key on a
        rule being *extra*, not on horizontal ink existing at all."""
        img = canvas()
        img[66:74, 45:110] = 30
        alpha = clean_symbol(img)
        assert alpha is not None, "the minus sign was mistaken for a rule"

    def test_crops_to_the_ink(self):
        img = canvas()
        img[40:90, 60:100] = 30
        alpha = clean_symbol(img)
        assert alpha.shape == (50, 40)

    def test_rejects_blank(self):
        assert clean_symbol(canvas()) is None


@pytest.mark.skipif(not has_maths, reason="maths corpus not extracted")
class TestBuild:
    def test_builds_with_class_names_as_labels(self, tmp_path):
        cache = build_maths_cache(MATHS, tmp_path / "m.npz", per_class=12)
        assert len(cache) > 100
        assert {"add", "sub", "eq", "x"} <= set(cache.labels)

    def test_rejects_some_but_not_most(self, tmp_path):
        cache = build_maths_cache(MATHS, tmp_path / "m.npz", per_class=40)
        assert 0.6 < cache.meta["accept_rate"] < 0.99, cache.meta

    def test_every_entry_has_ink(self, tmp_path):
        cache = build_maths_cache(MATHS, tmp_path / "m.npz", per_class=10)
        assert all(float(cache[i].max()) > 0.3 for i in range(len(cache)))


class RecordingInk(ProceduralInk):
    """A real provider that notes what the painters asked it for.

    A test double rather than a mock: it implements the interface and returns
    genuine ink, so the painters run for real and the assertion is about what
    the generator actually requested.
    """

    def __init__(self):
        self.kinds: list[str] = []

    def math_symbol(self, rng, height, kind="operand"):
        self.kinds.append(kind)
        return super().math_symbol(rng, height, kind)


class TestPaintersRequestMathematics:
    def test_formula_painter_asks_for_operators_and_operands(self):
        """Operators were procedural strokes while operands were real ink, so
        `FORMULA` was the one class mixing synthetic and real marks inside a
        single region."""
        from app.data.synth import PageGenerator

        ink = RecordingInk()
        generator = PageGenerator(ink=ink)
        for seed in range(30):
            generator.page(seed)
        assert "operator" in ink.kinds, "formulas are still drawing operators by hand"
        assert "operand" in ink.kinds

    def test_table_painter_asks_for_operands(self):
        """A table is columns of values, so its cells should be able to hold
        real handwritten digits rather than only letters."""
        from app.data.synth import PageGenerator

        ink = RecordingInk()
        generator = PageGenerator(ink=ink)
        for seed in range(30):
            generator.page(seed)
        assert ink.kinds.count("operand") > 5


class TestProviderFallback:
    """`math_symbol` is on the interface with a working default, so adding it
    does not make the maths corpus a dependency of anything."""

    def test_procedural_falls_back_to_a_glyph(self):
        p = ProceduralInk()
        s = p.math_symbol(np.random.default_rng(0), 30, kind="operator")
        assert s.dtype == np.float32
        assert float(s.max()) > 0.4


@pytest.mark.skipif(not has_cache, reason="maths cache not built")
class TestCorpusMathSymbols:
    @pytest.fixture(scope="class")
    @classmethod
    def ink(cls):
        return CorpusInk()

    def test_operator_returns_only_operators(self, ink):
        rng = np.random.default_rng(0)
        labels = {ink.math_label(rng, kind="operator") for _ in range(60)}
        assert labels <= set(MATHS_OPERATORS), labels
        assert len(labels) > 1

    def test_operand_never_returns_an_operator(self, ink):
        rng = np.random.default_rng(1)
        labels = {ink.math_label(rng, kind="operand") for _ in range(60)}
        assert not (labels & set(MATHS_OPERATORS)), labels

    def test_symbols_never_exceed_the_requested_height(self, ink):
        rng = np.random.default_rng(2)
        for want in (14, 28, 48):
            for kind in ("operator", "operand"):
                got = [ink.math_symbol(rng, want, kind=kind).shape[0] for _ in range(20)]
                assert all(1 <= h <= want for h in got), (want, kind, got)

    def test_operators_are_not_all_full_height(self, ink):
        """A minus sign is a thin stroke on the midline and a decimal point is
        a speck. Scaling every symbol to the nominal glyph height is what put a
        400px black slab across a rendered formula."""
        rng = np.random.default_rng(5)
        heights = {ink.math_symbol(rng, 40, kind="operator").shape[0] for _ in range(80)}
        assert len(heights) > 1, "every operator is the same height"
        assert min(heights) < 20, f"nothing is drawn small: {sorted(heights)}"

    def test_no_symbol_is_absurdly_wide(self, ink):
        """The regression itself: a 10:1 minus sign scaled to 40px tall became
        400px wide and painted a bar across the page."""
        rng = np.random.default_rng(6)
        for kind in ("operator", "operand"):
            widths = [ink.math_symbol(rng, 40, kind=kind).shape[1] for _ in range(120)]
            assert max(widths) <= 40 * 3, (kind, max(widths))

    def test_symbols_are_alpha_in_unit_range(self, ink):
        s = ink.math_symbol(np.random.default_rng(3), 30, kind="operand")
        assert s.dtype == np.float32
        assert 0.0 <= float(s.min()) and float(s.max()) <= 1.0
