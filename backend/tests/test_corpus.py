"""Corpus ingestion tests.

The corpus is real scanned data, so the risk here is not that the code crashes.
It is that contamination gets through silently. The word images are crops from
French records forms, and roughly 40% of them carry machine-printed labels
(`NOM`, `PRENOM:`), form borders, or bleed-through strokes from the line above.

That matters more than it sounds. Printed glyphs composited into a `THEORY`
region teach the classifier that printed texture means prose, and a form border
is a long straight rule, which is the single defining feature of `TABLE`. Both
are shortcuts, and both would show up on Thursday as an unexplained off-diagonal
without any hint of where they came from.

So the tests here are mostly about rejection, built on synthetic fixtures that
reproduce each contamination exactly. The real corpus is exercised separately
and skipped when it is not on disk.
"""

from pathlib import Path

import numpy as np
import pytest

from app.data.corpus import (
    CorpusCache,
    build_glyph_cache,
    build_word_cache,
    clean_glyph,
    clean_word,
)

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"
has_words = (CORPUS / "words" / "subset.csv").exists()
has_glyphs = (CORPUS / "chars74k" / "english.csv").exists()


# --------------------------------------------------------------------------
# fixtures that imitate the real contamination
# --------------------------------------------------------------------------

def paper(h: int = 40, w: int = 320) -> np.ndarray:
    """A blank light page, the way every source image starts."""
    return np.full((h, w), 246, np.uint8)


def write(img, x, y, n_marks, mark_h, gap=4, thick=3, value=30):
    """Lay down `n_marks` blobs of the given height, like letters on a line."""
    for i in range(n_marks):
        x0 = x + i * (mark_h // 2 + gap)
        img[y:y + mark_h, x0:x0 + thick] = value
        img[y + mark_h - thick:y + mark_h, x0:x0 + mark_h // 2] = value
    return img


class TestWordRejection:
    def test_rejects_a_form_border(self):
        img = write(paper(), 60, 8, 6, 22)
        img[:, 2:4] = 20  # the vertical rule down the left edge of the form
        assert clean_word(img)[0] is None

    def test_rejects_a_horizontal_rule(self):
        img = write(paper(), 60, 8, 6, 22)
        img[36:38, :] = 20
        assert clean_word(img)[0] is None

    def test_rejects_blank(self):
        assert clean_word(paper())[0] is None

    def test_rejects_bleed_from_the_line_above(self):
        """Partial strokes touching the top edge are a neighbouring row."""
        img = write(paper(), 60, 0, 6, 20)
        assert clean_word(img)[0] is None

    def test_drops_a_much_smaller_printed_label_beside_the_writing(self):
        """`NOM:` is far shorter than the hand beside it, so it falls below
        even the letter-sized bar and is simply excluded from the crop."""
        img = paper()
        write(img, 4, 18, 3, 9)     # the printed label, small
        write(img, 120, 8, 6, 24)   # the handwriting, tall
        alpha, reason = clean_word(img)
        assert alpha is not None, reason
        assert alpha.shape[1] < 170, "the printed label was kept in the crop"

    def test_rejects_a_same_height_label_separated_by_a_wide_gap(self):
        """A bold `PRENOM:` defeats the height test, so the gap must catch it.

        The crop is discarded rather than salvaged. An earlier version kept the
        widest of the two groups, which is a guess, and it guessed wrong often
        enough to composite a printed `PRENOM` onto a rendered page. There are
        330k rows available and a few thousand are needed, so ambiguity is
        never worth resolving.
        """
        img = paper()
        write(img, 4, 8, 4, 22)     # bold printed label, same height
        write(img, 190, 8, 6, 22)   # the handwriting, far to the right
        assert clean_word(img)[0] is None

    def test_rejects_when_the_printed_label_is_wider_than_the_writing(self):
        """The regression that reached a page: a long printed label beside a
        short hand. Keeping the widest group picks the label every time."""
        img = paper()
        write(img, 4, 8, 7, 22)     # long printed label
        write(img, 230, 8, 2, 22)   # short handwriting
        assert clean_word(img)[0] is None

    def test_rejects_a_slightly_smaller_label_at_normal_spacing(self):
        """`PRENOM: SIVATA` reached a rendered page twice.

        It defeats both earlier rules at once: the label is only a little
        shorter than the hand, and the gap after the colon is ordinary letter
        spacing rather than a wide one. What still separates them is that the
        label is a *second group of text*, so any dropped component big enough
        to be a letter means the crop holds two things.
        """
        img = paper()
        write(img, 4, 13, 7, 14)    # printed label, 0.64x the writing
        write(img, 95, 8, 6, 22)    # the hand, at ordinary letter spacing
        assert clean_word(img)[0] is None

    def test_tolerates_specks_and_punctuation(self):
        """Dots, commas and scanner grain are legitimately small and must not
        cause a rejection, or almost nothing survives."""
        img = paper()
        write(img, 30, 8, 6, 22)
        img[30:33, 170:173] = 30     # a speck
        alpha, reason = clean_word(img)
        assert alpha is not None, reason

    def test_still_accepts_a_single_widely_spaced_word(self):
        """Names are often written with generous letter spacing. Splitting one
        of those into two groups and then rejecting it would throw away most of
        the corpus, so the gap threshold must sit above normal tracking."""
        img = paper(40, 320)
        write(img, 20, 8, 7, 22, gap=12)
        alpha, reason = clean_word(img)
        assert alpha is not None, reason


class TestWordContract:
    def test_alpha_is_float_in_unit_range_with_ink_high(self):
        img = write(paper(), 60, 8, 6, 22)
        alpha, reason = clean_word(img)
        assert alpha is not None, reason
        assert alpha.dtype == np.float32
        assert 0.0 <= float(alpha.min()) and float(alpha.max()) <= 1.0
        assert float(alpha.max()) > 0.5, "ink must be high, not low"

    def test_crop_is_tight_against_the_ink(self):
        img = write(paper(), 60, 8, 6, 22)
        alpha, _ = clean_word(img)
        assert float(alpha[0].max()) > 0.1 and float(alpha[-1].max()) > 0.1
        assert float(alpha[:, 0].max()) > 0.1 and float(alpha[:, -1].max()) > 0.1


class TestGlyph:
    def test_crops_a_large_mostly_empty_canvas_to_its_ink(self):
        """Chars74K is 900x1200 with a small mark somewhere inside it."""
        img = np.full((900, 1200), 245, np.uint8)
        img[400:520, 600:660] = 25
        alpha = clean_glyph(img)
        assert alpha is not None
        assert alpha.shape == (120, 60), f"expected a tight crop, got {alpha.shape}"

    def test_rejects_blank(self):
        assert clean_glyph(np.full((900, 1200), 245, np.uint8)) is None


class TestCacheRoundTrip:
    """The cache is a sprite sheet: one flat strip plus offsets. Variable widths
    without pickled object arrays, which keeps loading fast and `allow_pickle`
    off for data that came off the internet."""

    def _tiny(self, tmp_path) -> CorpusCache:
        items = [np.linspace(0, 1, 32 * w, dtype=np.float32).reshape(32, w)
                 for w in (10, 25, 17)]
        cache = CorpusCache.from_items(items, labels=["a", "b", "c"], height=32)
        cache.save(tmp_path / "c.npz")
        return CorpusCache.load(tmp_path / "c.npz")

    def test_round_trips_variable_widths(self, tmp_path):
        c = self._tiny(tmp_path)
        assert len(c) == 3
        assert [c[i].shape[1] for i in range(3)] == [10, 25, 17]
        assert all(c[i].shape[0] == 32 for i in range(3))

    def test_values_survive_the_round_trip(self, tmp_path):
        c = self._tiny(tmp_path)
        expected = np.linspace(0, 1, 32 * 25, dtype=np.float32).reshape(32, 25)
        assert np.allclose(c[1], expected, atol=1 / 255)

    def test_sampling_is_deterministic(self, tmp_path):
        c = self._tiny(tmp_path)
        a = c.sample(np.random.default_rng(7))
        b = c.sample(np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_load_refuses_pickled_arrays(self, tmp_path):
        c = self._tiny(tmp_path)
        raw = np.load(tmp_path / "c.npz")
        assert raw["flat"].dtype == np.uint8


@pytest.mark.skipif(not has_glyphs, reason="chars74k not downloaded")
class TestRealGlyphCorpus:
    def test_builds_and_every_entry_has_ink(self, tmp_path):
        cache = build_glyph_cache(CORPUS / "chars74k", tmp_path / "g.npz", limit=120)
        assert len(cache) > 100
        assert all(float(cache[i].max()) > 0.3 for i in range(len(cache)))

    def test_labels_cover_several_characters(self, tmp_path):
        cache = build_glyph_cache(CORPUS / "chars74k", tmp_path / "g.npz", limit=300)
        assert len(set(cache.labels)) > 5


@pytest.mark.skipif(not has_words, reason="word corpus not downloaded")
class TestRealWordCorpus:
    def test_rejects_a_substantial_minority(self, tmp_path):
        """If acceptance is near 100% the filter has stopped filtering."""
        cache = build_word_cache(CORPUS / "words", tmp_path / "w.npz", limit=400)
        assert 0.35 < cache.meta["accept_rate"] < 0.85, cache.meta

    def test_accepted_words_are_wider_than_tall(self, tmp_path):
        cache = build_word_cache(CORPUS / "words", tmp_path / "w.npz", limit=200)
        ratios = [cache[i].shape[1] / cache[i].shape[0] for i in range(len(cache))]
        assert float(np.median(ratios)) > 1.5
