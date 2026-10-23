"""Ink provider tests.

Everything downstream of the generator is written against `InkProvider`, so the
contract matters more than any one implementation: a provider that returns the
wrong dtype, the wrong polarity or a wildly wrong size does not fail loudly, it
quietly produces pages that composite to nothing and a model that trains to 20%.

`TestProviderContract` therefore runs against every available provider, so a
corpus-backed provider cannot drift away from the procedural one.
"""

import numpy as np
import pytest

from app.data.corpus import DEFAULT_CACHE_DIR
from app.data.ink import CorpusInk, ProceduralInk, available_providers

has_cache = (DEFAULT_CACHE_DIR / "glyphs.npz").exists() and (DEFAULT_CACHE_DIR / "words.npz").exists()


@pytest.fixture(scope="module")
def providers():
    return list(available_providers().values())


class TestProviderContract:
    """Every provider must honour these, or the compositor misbehaves silently."""

    def test_glyph_is_float32_alpha_in_unit_range(self, providers):
        for p in providers:
            g = p.glyph(np.random.default_rng(1), 40)
            assert g.dtype == np.float32, p.name
            assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0, p.name

    def test_glyph_actually_contains_ink(self, providers):
        for p in providers:
            g = p.glyph(np.random.default_rng(2), 40)
            assert float(g.max()) > 0.4, f"{p.name} returned near-empty ink"

    def test_word_is_float32_alpha_in_unit_range(self, providers):
        for p in providers:
            w = p.word(np.random.default_rng(3), 24, 90)
            assert w.dtype == np.float32, p.name
            assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0, p.name

    def test_word_actually_contains_ink(self, providers):
        for p in providers:
            w = p.word(np.random.default_rng(4), 24, 90)
            assert float(w.max()) > 0.4, f"{p.name} returned near-empty ink"

    def test_sizes_are_in_the_right_neighbourhood(self, providers):
        """`roughly` is the documented contract, so this is a loose bound. It
        exists to catch an order-of-magnitude error, not to pin the size."""
        for p in providers:
            w = p.word(np.random.default_rng(5), 24, 90)
            assert 12 <= w.shape[0] <= 72, f"{p.name} height {w.shape}"
            assert 30 <= w.shape[1] <= 260, f"{p.name} width {w.shape}"

    def test_a_word_is_wider_than_a_glyph(self, providers):
        for p in providers:
            w = p.word(np.random.default_rng(6), 24, 90)
            g = p.glyph(np.random.default_rng(6), 24)
            assert w.shape[1] > g.shape[1], p.name

    def test_same_seed_same_ink(self, providers):
        for p in providers:
            a = p.word(np.random.default_rng(9), 24, 90)
            b = p.word(np.random.default_rng(9), 24, 90)
            assert np.array_equal(a, b), p.name

    def test_different_seeds_differ(self, providers):
        for p in providers:
            a = p.word(np.random.default_rng(10), 24, 90)
            b = p.word(np.random.default_rng(11), 24, 90)
            assert a.shape != b.shape or not np.array_equal(a, b), p.name


class TestProviderRegistry:
    def test_procedural_is_always_available(self):
        assert "procedural" in available_providers()

    @pytest.mark.skipif(not has_cache, reason="corpus cache not built")
    def test_corpus_registers_when_its_cache_exists(self):
        assert "corpus" in available_providers()


@pytest.mark.skipif(not has_cache, reason="corpus cache not built")
class TestCorpusInk:
    @pytest.fixture(scope="class")
    @classmethod
    def ink(cls):
        return CorpusInk()

    def test_word_width_tracks_the_requested_width(self, ink):
        """Words are chosen by aspect rather than stretched to fit, so the
        width lands near the request instead of exactly on it."""
        rng = np.random.default_rng(0)
        for want in (60, 120, 200):
            widths = [ink.word(rng, 24, want).shape[1] for _ in range(30)]
            assert 0.55 * want < float(np.median(widths)) < 1.6 * want, (want, widths[:5])

    def test_letterforms_are_not_stretched(self, ink):
        """A short word asked to be wide must not come back as one stretched
        word. Selection by aspect is what prevents that, and if it regressed
        the aspect error would be large and one-sided."""
        rng = np.random.default_rng(1)
        errors = [ink.word(rng, 24, 150).shape[1] / 150 for _ in range(40)]
        assert 0.6 < float(np.median(errors)) < 1.7

    def test_composed_words_reach_below_the_baseline(self, ink):
        """All-caps names carry no descenders, so pseudo-words built from
        lowercase glyphs are mixed in to keep the prose band ragged. Over many
        samples the height variation must be real, not constant."""
        rng = np.random.default_rng(2)
        heights = {ink.word(rng, 24, 100).shape[0] for _ in range(60)}
        assert len(heights) > 1, "every word is the same height; the band is flat"

    def test_mix_is_configurable_and_both_paths_work(self):
        rng = np.random.default_rng(3)
        only_real = CorpusInk(compose_probability=0.0)
        only_composed = CorpusInk(compose_probability=1.0)
        assert float(only_real.word(rng, 24, 90).max()) > 0.4
        assert float(only_composed.word(rng, 24, 90).max()) > 0.4

    def test_glyph_height_is_close_to_the_request(self, ink):
        rng = np.random.default_rng(4)
        for want in (16, 32, 54):
            got = [ink.glyph(rng, want).shape[0] for _ in range(10)]
            assert all(abs(h - want) <= max(3, want * 0.25) for h in got), (want, got)

    def test_reports_itself_as_corpus_backed(self, ink):
        assert ink.name == "corpus"


class TestProceduralStillWorks:
    """The corpus must not become a hard dependency: the whole point of the
    interface is that the pipeline runs with no download at all."""

    def test_procedural_needs_no_cache(self):
        p = ProceduralInk()
        assert float(p.word(np.random.default_rng(0), 20, 70).max()) > 0.4
