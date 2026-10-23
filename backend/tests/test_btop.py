"""btop primitives: braille canvas, gradient, meter, chart.

The braille encoding is the one place a silent bug would draw a plausible
but wrong graph, so the dot-to-bit mapping is pinned against the Unicode
table rather than against this module's own constant.
"""

from rich.console import Console

from app.tui.btop import BrailleChart, Canvas, GpuSampler, gradient, meter


def render(renderable, width: int = 40, height: int = 6) -> str:
    console = Console(width=width, record=True, force_terminal=False)
    console.print(renderable, height=height)
    return console.export_text()


class TestCanvas:
    def test_dot_bits_match_unicode(self):
        # U+2801 is dot 1 (top left), U+2880 is dot 8 (bottom right).
        c = Canvas(1, 1)
        c.dot(0, 0)
        assert c.rows_text(lambda r, o: "")[0].plain == "⠁"
        c = Canvas(1, 1)
        c.dot(1, 3)
        assert c.rows_text(lambda r, o: "")[0].plain == "⢀"

    def test_full_cell(self):
        c = Canvas(1, 1)
        for x in range(2):
            for y in range(4):
                c.dot(x, y)
        assert c.rows_text(lambda r, o: "")[0].plain == "⣿"

    def test_out_of_bounds_is_ignored(self):
        c = Canvas(2, 2)
        c.dot(-1, 0)
        c.dot(4, 8)
        assert all(b == 0 for row in c.bits for b in row)

    def test_line_is_connected(self):
        c = Canvas(4, 2)
        c.line(0, 0, 7, 7)
        lit = sum(bin(b).count("1") for row in c.bits for b in row)
        assert lit == 8  # one dot per step on a 45 degree line


class TestGradientAndMeter:
    def test_gradient_endpoints(self):
        assert gradient(0.0) == "#77ca9b"
        assert gradient(1.0) == "#dc4c4c"
        assert gradient(-5) == gradient(0) and gradient(5) == gradient(1)

    def test_meter_width_is_constant(self):
        for f in (0.0, 0.33, 1.0, 1.7):
            assert len(meter(f, 12).plain) == 12


class TestBrailleChart:
    def test_waiting_state(self):
        assert "waiting" in render(BrailleChart({}))

    def test_fills_given_height(self):
        lines = render(BrailleChart({"a": [0.1, 0.9, 0.5]}), height=5).rstrip("\n").split("\n")
        assert len(lines) == 5

    def test_ceiling_caps_axis(self):
        text = render(BrailleChart({"a": [0.9, 1.0]}, ceiling=1.0))
        assert "1.000" in text and "1.01" not in text

    def test_flat_series_does_not_divide_by_zero(self):
        assert "⠀" not in render(BrailleChart({"a": [0.5] * 5}))

    def test_x_labels_take_last_row(self):
        text = render(BrailleChart({"a": [1, 2]}, x_labels=("left", "right")), height=4)
        assert text.rstrip().split("\n")[-1].strip().startswith("left")


def test_gpu_sampler_never_raises():
    sample = GpuSampler().sample()
    assert sample is None or 0 <= sample["util"] <= 100
