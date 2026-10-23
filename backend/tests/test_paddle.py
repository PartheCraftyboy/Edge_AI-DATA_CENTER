"""PaddleOCR-VL reader tests.

The model is swapped for a fake backend, so these run without the 1.8 GB of
weights or a GPU. The OTSL fixture is the model's real reply to an unruled
synthetic table, copied verbatim, not a reconstruction of what it might say.
"""

import numpy as np
import pytest

from app.vlm import VisionReadError
from app.vlm.paddle import PaddleVision, _to_rgb, otsl_to_cells, strip_math

REAL_TABLE_REPLY = "<fcel>GLOR<fcel>5<nl><fcel>SONIA<fcel>9<nl><fcel>OCEANE<fcel>u2xpcysgwoi<nl>"


def fake(reply: str, seen: list | None = None):
    def backend(image, task):
        if seen is not None:
            seen.append((image, task))
        return reply
    return backend


@pytest.fixture
def crop():
    return np.full((60, 120), 255, np.uint8)


class TestOtsl:
    def test_real_reply(self):
        assert otsl_to_cells(REAL_TABLE_REPLY) == [
            ["GLOR", "5"], ["SONIA", "9"], ["OCEANE", "u2xpcysgwoi"]]

    def test_empty_and_merged_cells_keep_the_grid(self):
        assert otsl_to_cells("<fcel>a<ecel><fcel>c<nl><fcel>d<lcel><ucel><nl>") == [
            ["a", "", "c"], ["d", "", ""]]

    def test_no_cells(self):
        assert otsl_to_cells("just some text") == []


class TestStripMath:
    @pytest.mark.parametrize("raw", ["\\[x^2\\]", "$$x^2$$", "\\(x^2\\)", "$x^2$", " x^2 "])
    def test_delimiters(self, raw):
        assert strip_math(raw) == "x^2"

    def test_inner_dollar_is_kept(self):
        assert strip_math("a$b") == "a$b"


class TestReaders:
    def test_table(self, crop):
        out = PaddleVision(backend=fake(REAL_TABLE_REPLY)).read_table(crop)
        assert (out.rows, out.cols) == (2, 2)
        assert "| SONIA | 9 |" in out.markdown

    def test_formula_feeds_the_solver(self, crop):
        """It used to leave `expr` unset, which quietly turned off answers,
        the symbol table and plotted curves. `vlm.latex` converts it."""
        out = PaddleVision(backend=fake("\\[\\rho=\\frac{\\lambda}{\\mu}\\]")).read_formula(crop)
        assert out.tex == "\\rho=\\frac{\\lambda}{\\mu}"
        assert out.expr == "rho=((lambda)/(mu))"

    def test_a_formula_that_will_not_convert_is_left_unset(self, crop):
        """The reader misreads: `m x + c` came back as `m \\times + c` on a real
        page. That converts to `m * + c`, which SymPy reads as `m*c`. An
        expression left unset holds the region; a wrong one answers it."""
        out = PaddleVision(backend=fake("\\[m \\times + c\\]")).read_formula(crop)
        assert out.tex == "m \\times + c"
        assert out.expr is None

    def test_prose(self, crop):
        out = PaddleVision(backend=fake("LINE ONE\n\n  LINE TWO ")).read_prose(crop)
        assert out.raw_lines == ["LINE ONE", "LINE TWO"]

    def test_task_instructions(self, crop):
        seen = []
        PaddleVision(backend=fake("x", seen)).read_formula(crop)
        assert seen[0][1] == "Formula Recognition:"

    def test_empty_reply_is_a_read_error(self, crop):
        with pytest.raises(VisionReadError):
            PaddleVision(backend=fake("  ")).read_prose(crop)

    def test_table_without_cells_is_a_read_error(self, crop):
        with pytest.raises(VisionReadError):
            PaddleVision(backend=fake("GLOR 5\nSONIA 9")).read_table(crop)

    def test_latency_recorded(self, crop):
        v = PaddleVision(backend=fake("x"))
        v.read_prose(crop)
        assert v.last_call_ms is not None and v.last_call_ms >= 0


class TestImages:
    def test_grey_becomes_rgb(self):
        assert _to_rgb(np.zeros((4, 5), np.uint8)).shape == (4, 5, 3)

    def test_bgr_is_reversed(self):
        bgr = np.zeros((1, 1, 3), np.uint8)
        bgr[0, 0] = (255, 0, 0)  # blue, in OpenCV order
        assert tuple(_to_rgb(bgr)[0, 0]) == (0, 0, 255)

    def test_alpha_dropped(self):
        assert _to_rgb(np.zeros((2, 2, 4), np.uint8)).shape == (2, 2, 3)


def test_close_without_load_is_safe():
    PaddleVision(backend=fake("x")).close()


def test_make_reader():
    from app.vlm import OllamaVision, make_reader

    assert isinstance(make_reader("qwen"), OllamaVision)
    assert isinstance(make_reader("paddle"), PaddleVision)
    with pytest.raises(ValueError):
        make_reader("tesseract")
