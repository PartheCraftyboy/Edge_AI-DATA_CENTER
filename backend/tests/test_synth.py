"""Page generator tests.

The important ones here are not about drawing. They are about the two ways a
synthetic dataset silently ruins a model: leaking a shortcut the classifier can
exploit, and producing classes that are not actually distinguishable.
"""

import numpy as np
import pytest

from app import CLASS_NAMES
from app.data.ink import ProceduralInk
from app.data.synth import PageGenerator, RegionSpec


@pytest.fixture(scope="module")
def gen():
    return PageGenerator()


@pytest.fixture(scope="module")
def pages(gen):
    return [gen.page(s) for s in range(40)]


class TestDeterminism:
    def test_same_seed_same_page(self, gen):
        assert np.array_equal(gen.page(11).image, gen.page(11).image)

    def test_different_seed_different_page(self, gen):
        assert not np.array_equal(gen.page(11).image, gen.page(12).image)

    def test_labels_are_reproducible_too(self, gen):
        assert gen.page(11).regions == gen.page(11).regions


class TestNoShortcutLearning:
    """If a class always sits in the same part of the page, a CNN reaches high
    validation accuracy by reading the crop's position, which it will not have
    at inference and which says nothing about the region's content. A generator
    is exactly where that gets baked in, so it is tested here rather than
    discovered later as an unreasonably good validation score."""

    def test_vertical_position_carries_no_class_information(self, pages):
        ys: dict[str, list[float]] = {}
        for p in pages:
            h = p.image.shape[0]
            for r in p.regions:
                ys.setdefault(r.region_class, []).append((r.y + r.h / 2) / h)

        means = {k: float(np.mean(v)) for k, v in ys.items() if len(v) >= 10}
        assert len(means) >= 4, "not enough classes sampled to judge"
        spread = max(means.values()) - min(means.values())
        assert spread < 0.12, f"class means cluster by position: {means}"

    def test_every_class_spans_the_page(self, pages):
        # Not just equal means: each class must actually appear top and bottom.
        for cls in CLASS_NAMES:
            ys = [
                (r.y + r.h / 2) / p.image.shape[0]
                for p in pages for r in p.regions if r.region_class == cls
            ]
            if len(ys) < 10:
                continue
            assert min(ys) < 0.35 and max(ys) > 0.65, f"{cls} never spans the page"

    def test_region_size_is_not_a_class_cue(self, pages):
        areas: dict[str, list[float]] = {}
        for p in pages:
            total = p.image.shape[0] * p.image.shape[1]
            for r in p.regions:
                areas.setdefault(r.region_class, []).append(r.w * r.h / total)
        means = {k: float(np.mean(v)) for k, v in areas.items() if len(v) >= 10}
        lo, hi = min(means.values()), max(means.values())
        assert hi / lo < 1.8, f"region area separates classes: {means}"


class TestLayout:
    def test_regions_do_not_overlap(self, pages):
        def overlaps(a: RegionSpec, b: RegionSpec) -> bool:
            return not (
                a.x + a.w <= b.x or b.x + b.w <= a.x
                or a.y + a.h <= b.y or b.y + b.h <= a.y
            )

        for p in pages:
            for i, r1 in enumerate(p.regions):
                for r2 in p.regions[i + 1:]:
                    assert not overlaps(r1, r2), f"overlap on seed {p.seed}"

    def test_regions_stay_inside_the_page(self, pages):
        for p in pages:
            h, w = p.image.shape[:2]
            for r in p.regions:
                assert r.x >= 0 and r.y >= 0
                assert r.x + r.w <= w and r.y + r.h <= h

    def test_page_has_several_regions(self, pages):
        counts = [len(p.regions) for p in pages]
        assert min(counts) >= 2
        assert 3 <= float(np.mean(counts)) <= 8

    def test_all_five_classes_are_produced(self, pages):
        seen = {r.region_class for p in pages for r in p.regions}
        assert seen == set(CLASS_NAMES)

    def test_classes_are_roughly_balanced(self, pages):
        from collections import Counter
        c = Counter(r.region_class for p in pages for r in p.regions)
        total = sum(c.values())
        for cls in CLASS_NAMES:
            share = c[cls] / total
            assert 0.10 < share < 0.32, f"{cls} at {share:.2f} of the set"


class TestCrops:
    def test_crops_are_square_and_sized(self, pages):
        crops = list(pages[0].crops(size=224))
        assert crops, "page produced no crops"
        for crop, cls in crops:
            assert crop.shape == (224, 224)
            assert crop.dtype == np.uint8
            assert cls in CLASS_NAMES

    def test_crops_contain_ink(self, pages):
        # A crop of blank paper teaches the model nothing and would be a bug in
        # layout or painting rather than a legitimate empty region.
        for p in pages[:10]:
            for crop, cls in p.crops():
                coverage = float((255 - crop.astype(np.float32)).mean()) / 255
                assert coverage > 0.01, f"{cls} crop is nearly blank"

    def test_crop_count_matches_region_count(self, pages):
        for p in pages[:10]:
            assert len(list(p.crops())) == len(p.regions)


class TestClassesAreStructurallyDistinct:
    """A regression test for a bug the unit tests could not see.

    DIAGRAM boxes were being drawn at exactly the step width, so they sat flush
    against each other with zero gap and the connectors had zero length. The
    region rendered as one wide ruled rectangle: visually a TABLE, a
    neighbouring class. Every numeric test still passed, and only looking at the
    images caught it.

    The structural property that separates the two is long horizontal runs: a
    table's rules cross the full region width, a diagram's box edges do not.
    """

    @staticmethod
    def _full_width_rule_fraction(crop: np.ndarray) -> float:
        ink = (255 - crop.astype(np.float32)) / 255 > 0.25
        rows_covered = ink.mean(axis=1)
        return float((rows_covered > 0.85).mean())

    def _sample(self, gen, cls, n=14):
        out = []
        for s in range(600):
            for crop, c in gen.page(s).crops(size=224):
                if c == cls:
                    out.append(crop)
            if len(out) >= n:
                break
        return out[:n]

    def test_tables_have_more_full_width_rules_than_diagrams(self, gen):
        tables = [self._full_width_rule_fraction(c) for c in self._sample(gen, "TABLE")]
        diagrams = [self._full_width_rule_fraction(c) for c in self._sample(gen, "DIAGRAM")]
        assert np.mean(tables) > np.mean(diagrams) * 1.5, (
            f"TABLE {np.mean(tables):.3f} vs DIAGRAM {np.mean(diagrams):.3f}: "
            "the two classes are not structurally separable"
        )

    def test_prose_is_denser_than_formula(self, gen):
        def coverage(c):
            return float((255 - c.astype(np.float32)).mean()) / 255

        prose = [coverage(c) for c in self._sample(gen, "THEORY")]
        formula = [coverage(c) for c in self._sample(gen, "FORMULA")]
        assert np.mean(prose) > np.mean(formula), (
            "prose should carry more ink than a sparse centred formula"
        )


class TestSidecar:
    def test_sidecar_round_trips(self, pages):
        d = pages[0].sidecar()
        assert d["seed"] == pages[0].seed
        assert d["ink_provider"] == "procedural"
        assert len(d["regions"]) == len(pages[0].regions)
        # `hard` joined the schema when the generator gained ambiguous
        # variants, so that accuracy can be reported split by difficulty.
        assert set(d["regions"][0]) == {"class", "box", "hard"}

    def test_save_writes_both_files(self, pages, tmp_path):
        img, js = pages[0].save(tmp_path)
        assert img.exists() and js.exists()
        assert img.stat().st_size > 0


class TestInkProvider:
    def test_provider_name_is_recorded(self, gen):
        assert gen.page(0).ink_provider == "procedural"

    def test_word_and_glyph_are_masks_in_range(self):
        rng = np.random.default_rng(0)
        ink = ProceduralInk()
        w = ink.word(rng, 20, 60)
        g = ink.glyph(rng, 20)
        for m in (w, g):
            assert m.dtype == np.float32
            assert 0.0 <= m.min() and m.max() <= 1.0
            assert m.max() > 0.2, "mask is effectively blank"

    def test_words_differ_between_draws(self):
        rng = np.random.default_rng(0)
        ink = ProceduralInk()
        a, b = ink.word(rng, 20, 60), ink.word(rng, 20, 60)
        assert not np.array_equal(a, b)
