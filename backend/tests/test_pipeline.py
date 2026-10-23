"""Page pipeline: preprocess, segment, compile, and the labelled-page store.

The compile tests replace the classifier and the reader with fakes, so they
run without a GPU, Ollama or checkpoints, and test what this module owns:
ordering, gating, the two-pass symbol resolution, and degrading one region
at a time rather than failing the page.
"""

import cv2
import numpy as np
import pytest

from app import CLASS_NAMES
from app.contracts import FormulaContent, ProseContent, TableContent
from app.data.real import PageStore, build_real_dataset, cut_crop, split_for_page
from app.pipeline.compile import Compiler, compose_markdown, to_tensor
from app.pipeline.preprocess import preprocess, rotate
from app.pipeline.segment import SegmentConfig, iou, match_score, segment, tighten


def blank(h=600, w=800):
    return np.full((h, w), 235, np.uint8)


def ink_block(img, x, y, w, h, rows=4):
    """Lines of fake handwriting: letter-height scribbles, word gaps, line pitch
    about twice the letter height, roughly what a page of notes looks like at
    the working resolution. (Thin 2 px lines 27 px apart, the first version of
    this fixture, look nothing like writing and segment as nothing at all.)"""
    rng = np.random.default_rng(x * 31 + y)
    pitch = h // rows
    letter = max(8, min(16, int(pitch * 0.6)))
    for r in range(rows):
        base = y + r * pitch + letter
        xx = x
        while xx < x + w - 10:
            word = int(rng.integers(3, 8))
            for _ in range(word):
                if xx >= x + w - 8:
                    break
                pts = np.array([[xx + int(rng.integers(0, 8)), base - int(rng.integers(0, letter))] for _ in range(4)], np.int32)
                cv2.polylines(img, [pts], False, 40, 2)
                xx += 9
            xx += 12
    return img



def writing(img, x, y, cols, rows, letter):
    """`rows` lines of scribble at a given letter height, with the word gaps and
    line pitch of real notes (word gap about one letter, pitch about twice).

    Letter height is the parameter because it is the thing that varies between
    a phone photo of a notebook and a page drawn on the canvas, and it is what
    the segmenter has to key off. `ink_block` clamps it to 8-16 px, which is
    the narrow band where a fixed page-fraction kernel happens to work."""
    rng = np.random.default_rng(x * 31 + y + letter)
    pitch = int(letter * 2.0)
    for r in range(rows):
        base = y + r * pitch + letter
        xx = x
        for _ in range(cols):
            for _ in range(int(rng.integers(3, 8))):
                pts = np.array([[xx + int(rng.integers(0, letter // 2 + 1)),
                                 base - int(rng.integers(0, letter))] for _ in range(4)], np.int32)
                cv2.polylines(img, [pts], False, 40, max(2, letter // 8))
                xx += int(letter * 0.6)
            xx += letter  # word gap
    return img



def photo_of_a_page(letter=14, bg_seed=7):
    """A page photographed on a patterned surface, like the notebook shots.

    The background is what breaks the real pipeline: a dark, busy cloth that
    Sauvola calls ink, covering more of the frame than the writing does. The
    page itself is a bright quadrilateral with two paragraphs on it."""
    rng = np.random.default_rng(bg_seed)
    frame = rng.integers(20, 90, (1600, 720), dtype=np.uint8)
    frame = cv2.resize(cv2.resize(frame, (90, 200), interpolation=cv2.INTER_AREA),
                       (720, 1600), interpolation=cv2.INTER_NEAREST)
    page = blank(1100, 600)
    writing(page, 60, 80, cols=3, rows=4, letter=letter)
    writing(page, 60, 80 + letter * 12, cols=3, rows=4, letter=letter)
    frame[300:1400, 60:660] = page
    return frame


# ---------------------------------------------------------------- preprocess

class TestPreprocess:
    def test_recovers_rotation(self):
        page = blank()
        for y in range(60, 560, 40):
            ink_block(page, 60, y, 680, 20, rows=1)
        tilted = rotate(page, 3.0)
        assert abs(preprocess(tilted).skew_deg + 3.0) <= 0.5

    def test_every_kernel_is_timed(self):
        out = preprocess(ink_block(blank(), 100, 100, 300, 80))
        assert {"clahe", "sauvola", "denoise"} <= set(out.timing_ms)
        assert out.ink.dtype == bool and out.ink.shape == out.grey.shape

    def test_large_photo_is_downscaled(self):
        out = preprocess(np.full((4000, 3000, 3), 230, np.uint8))
        assert max(out.grey.shape) == 1600 and out.scale < 1

    def test_rotation_fills_with_paper_not_black(self):
        out = rotate(blank(), 5.0)
        assert out[0, 0] > 200


# ---------------------------------------------------------------- segment

class TestSegment:
    def test_two_separate_regions(self):
        page = ink_block(ink_block(blank(), 60, 60, 300, 100), 60, 380, 500, 120)
        boxes = segment(preprocess(page, deskew=False).ink)
        assert len(boxes) == 2
        assert boxes[0][1] < boxes[1][1]  # reading order, top first

    def test_lines_of_one_paragraph_stay_together(self):
        page = ink_block(blank(), 80, 100, 500, 160, rows=6)
        assert len(segment(preprocess(page, deskew=False).ink)) == 1

    @pytest.mark.parametrize("letter", [10, 24, 60, 100])
    def test_two_paragraphs_segment_as_two_at_any_writing_scale(self, letter):
        """The same layout, written small and written large, is still two regions.

        This is the real failure. The closing kernel was a fraction of the
        *page*, so on a phone photo (letter 12 px) it spanned four letter
        heights and fused the whole page into one blob, while on a canvas page
        (letter 190 px) it spanned a fifth of a letter and cut every word into
        its own region. The kernel has to be a multiple of the writing, not of
        the paper."""
        # Sized so both paragraphs fit the page at every letter height: two
        # rows is 4 letters tall, and they are set 7 letters apart.
        page = blank(1200, 1600)
        writing(page, 80, 60, cols=3, rows=2, letter=letter)
        writing(page, 80, 60 + letter * 7, cols=3, rows=2, letter=letter)
        assert len(segment(preprocess(page, deskew=False).ink)) == 2

    def test_background_around_a_photographed_page_is_not_ink(self):
        """The surface the notebook was photographed on must not become ink.

        On the real shots the notebook sits on a patterned cloth with floor
        tiles below it. Sauvola calls both ink, they cover more of the frame
        than the writing does, and once the background is a solid ink field
        every region on the page is connected to every other one through it:
        the page segments as a single whole-frame blob, which the classifier
        then labels DIAGRAM at 0.41."""
        out = preprocess(photo_of_a_page(), deskew=False)
        boxes = segment(out.ink)
        assert len(boxes) == 2, f"expected the two paragraphs, got {len(boxes)} regions"
        for x, y, w, h in boxes:
            assert w * h < 0.5 * out.ink.size, "a region covering half the frame is the background"

    def test_specks_are_dropped(self):
        page = blank()
        cv2.circle(page, (400, 300), 2, 30, -1)
        assert segment(preprocess(page, deskew=False).ink) == []

    def test_side_by_side_read_left_first(self):
        page = ink_block(ink_block(blank(), 420, 100, 300, 100), 40, 104, 300, 100)
        boxes = segment(preprocess(page, deskew=False).ink)
        assert len(boxes) == 2 and boxes[0][0] < boxes[1][0]

    def test_iou_and_matching(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
        assert iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
        score = match_score([(0, 0, 10, 10), (50, 50, 5, 5)], [(0, 0, 10, 10)])
        assert score["recall"] == 1.0 and score["precision"] == 0.5

    def test_tighten_shrinks_to_ink(self):
        ink = np.zeros((100, 100), bool)
        ink[40:50, 30:60] = True
        assert tighten(ink, (0, 0, 100, 100)) == (30, 40, 30, 10)


# ---------------------------------------------------------------- store

class TestPageStore:
    def test_round_trip_and_clamping(self, tmp_path):
        store = PageStore(tmp_path)
        rec = store.add(blank(), [{"box": [1, 2, 30, 40], "class": "THEORY"}])
        assert not rec["labelled"] and store.image_path(rec["id"]).exists()
        saved = store.save_labels(rec["id"], [
            {"box": [790, 10, 100, 50], "class": "TABLE"},   # runs off the right edge
            {"box": [10, 10, 3, 3], "class": "THEORY"},      # too small, dropped
        ])
        assert saved["labelled"] and saved["regions"] == [{"box": [790, 10, 10, 50], "class": "TABLE"}]

    def test_unknown_class_rejected(self, tmp_path):
        store = PageStore(tmp_path)
        rec = store.add(blank(), [])
        with pytest.raises(ValueError):
            store.save_labels(rec["id"], [{"box": [0, 0, 50, 50], "class": "DOODLE"}])

    def test_path_traversal_is_not_a_page(self, tmp_path):
        with pytest.raises(KeyError):
            PageStore(tmp_path).get("../../etc/passwd")

    def test_split_is_stable_and_roughly_70_15_15(self):
        splits = [split_for_page(f"page-{i}") for i in range(2000)]
        assert split_for_page("page-7") == split_for_page("page-7")
        assert 0.65 < splits.count("train") / 2000 < 0.75

    def test_dataset_val_and_test_are_real_only(self, tmp_path):
        store = PageStore(tmp_path / "real")
        for i in range(30):
            page = ink_block(blank(), 50, 50, 300, 120)
            rec = store.add(page, [])
            store.save_labels(rec["id"], [{"box": [50, 50, 300, 120], "class": CLASS_NAMES[i % 5]}])
        syn = tmp_path / "syn" / "crops"
        syn.mkdir(parents=True)
        np.save(syn / "train_crops.npy", np.zeros((500, 224, 224), np.uint8))
        np.save(syn / "train_labels.npy", np.zeros(500, np.int64))
        manifest = build_real_dataset(store, tmp_path / "out", synthetic_root=tmp_path / "syn", jitter_copies=2)
        by_split = store.stats()["pages_by_split"]
        assert manifest["counts"]["val"] == by_split["val"]      # one region per page, no jitter
        assert manifest["counts"]["test"] == by_split["test"]
        assert manifest["real_train"] == by_split["train"] * 3   # original plus two jittered
        assert manifest["synthetic_train"] == manifest["real_train"]
        crops = np.load(tmp_path / "out" / "crops" / "train_crops.npy")
        assert crops.shape[1:] == (224, 224) and crops.dtype == np.uint8


def test_crop_recipe_matches_training():
    crop = cut_crop(blank(), (100, 100, 200, 50))
    assert crop.shape == (224, 224)
    assert cut_crop(blank(), (0, 0, 4, 4)) is None
    x = to_tensor([crop])
    assert x.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------- compile

class FakeReader:
    def __init__(self, prose="", formula=("", None), fail=False):
        self.prose, self.formula, self.fail = prose, formula, fail

    def read_prose(self, crop):
        if self.fail:
            raise RuntimeError("reader exploded")
        return ProseContent(markdown=self.prose, raw_lines=[self.prose])

    def read_formula(self, crop):
        return FormulaContent(tex=self.formula[0], expr=self.formula[1])

    def read_table(self, crop):
        return TableContent(markdown="| a |\n| --- |\n| 1 |", rows=1, cols=1)

    def available(self):
        return True


def compiler_with(tmp_path, reader, classes):
    """A compiler whose classifier returns `classes` in order, one-hot."""
    c = Compiler(vault=tmp_path / "vault", reader=reader)
    c.active = "fake"

    def classify(grey, boxes):
        probs = np.zeros((len(boxes), 5))
        for i, cls in enumerate(classes[:len(boxes)]):
            probs[i, CLASS_NAMES.index(cls)] = 0.99
            probs[i, (CLASS_NAMES.index(cls) + 1) % 5] = 0.01
        return probs, 1.0, [True] * len(boxes)

    c.classify = classify
    return c


class TestCompile:
    BOXES = [(50, 50, 300, 80), (50, 200, 300, 60), (50, 320, 300, 60)]

    def test_formula_resolves_against_prose_above_it(self, tmp_path):
        reader = FakeReader(prose="Arrivals lambda = 25 and service mu = 22.7",
                            formula=("\\rho = \\lambda / \\mu", "rho = lambda / mu"))
        c = compiler_with(tmp_path, reader, ["THEORY", "FORMULA"])
        note = c.compile(blank(), title="queue", boxes=self.BOXES[:2])
        formula = note.blocks[1].content
        assert formula.answer == pytest.approx(25 / 22.7)
        assert note.symbols == {"lambda": 25.0, "mu": 22.7}

    def test_plot_sketch_draws_the_free_formula(self, tmp_path):
        reader = FakeReader(prose="mu = 2", formula=("L = \\rho/(1-\\rho)", "L = rho / (1 - rho)"))
        c = compiler_with(tmp_path, reader, ["THEORY", "FORMULA", "PLOT_SKETCH"])
        note = c.compile(blank(), boxes=self.BOXES)
        fig = note.blocks[2].content
        assert fig.refused is None and fig.free_symbol == "rho" and len(fig.series_x) > 10

    def test_diagram_is_held_and_unread(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(), ["DIAGRAM"])
        block = c.compile(blank(), boxes=self.BOXES[:1]).blocks[0]
        assert block.held and block.content is None

    def test_reader_failure_holds_one_region_not_the_page(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(fail=True), ["THEORY", "TABLE"])
        note = c.compile(blank(), boxes=self.BOXES[:2])
        assert note.blocks[0].held and "reader exploded" in note.blocks[0].hold_reason
        assert not note.blocks[1].held and note.blocks[1].content.rows == 1

    def test_vault_is_written(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(prose="hello"), ["THEORY"])
        note = c.compile(blank(), title="T", boxes=self.BOXES[:1])
        folder = tmp_path / "vault" / note.note_id
        assert {"note.json", "note.md", "page.png"} <= {p.name for p in folder.iterdir()}
        assert "hello" in (folder / "note.md").read_text()

    def test_budget_is_serialised_with_derived_fields(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(prose="x"), ["THEORY"])
        dumped = c.compile(blank(), boxes=self.BOXES[:1]).model_dump()
        assert {"load_factor", "over_budget", "dominant_stage"} <= set(dumped["budget"])
        assert "held_count" in dumped and "mean_confidence" in dumped

    def test_empty_page_compiles_with_a_note(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(), [])
        note = c.compile(blank(), boxes=[])
        assert note.blocks == [] and note.model_notes

    def test_markdown_marks_held_regions(self, tmp_path):
        c = compiler_with(tmp_path, FakeReader(), ["DIAGRAM"])
        assert "Held for review" in compose_markdown(c.compile(blank(), boxes=self.BOXES[:1]))
