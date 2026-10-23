"""
btop-style drawing primitives for palimtop.

Three things make btop look like btop, and this module is those three things:

**Braille graphs.** Each braille character (U+2800 to U+28FF) is a 2x4 grid of
dots, one bit per dot, so a terminal cell holds eight plotted points instead
of one. That is the whole trick behind btop's smooth lines.

**Gradients.** Fill colour is a function of height, green at the bottom to red
at the top, so a graph says "high" by colour before you read its axis.

**Meters.** A row of `■`, each coloured by its position along the gradient,
with the unfilled part dimmed rather than blank so the scale is always shown.

Every chart here implements rich's `__rich_console__`, which hands it the
width and height it actually has at draw time. Graphs therefore fill whatever
box they are put in and follow the terminal when it is resized.
"""

from __future__ import annotations

import time
from collections import deque

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text

# btop's default theme: cool green, through ochre, to red.
GRADIENT = ("#77ca9b", "#cbc06c", "#dc4c4c")
DIM = "#3a3a3a"
LABEL = "#8a8a8a"

# Bit for the dot at [row][column] inside one braille cell.
_DOT = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def _hex(colour: str) -> tuple[int, int, int]:
    colour = colour.lstrip("#")
    return int(colour[0:2], 16), int(colour[2:4], 16), int(colour[4:6], 16)


def gradient(t: float, stops: tuple[str, ...] = GRADIENT) -> str:
    """The colour a fraction `t` of the way along `stops`, as #rrggbb."""
    t = max(0.0, min(1.0, t))
    span = t * (len(stops) - 1)
    i = min(int(span), len(stops) - 2)
    f = span - i
    a, b = _hex(stops[i]), _hex(stops[i + 1])
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def meter(fraction: float, width: int, stops: tuple[str, ...] = GRADIENT) -> Text:
    """btop's bar: filled cells coloured by their own position, the rest dim."""
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    text = Text()
    for i in range(width):
        if i < filled:
            text.append("■", style=gradient(i / max(1, width - 1), stops))
        else:
            text.append("■", style=DIM)
    return text


class Canvas:
    """A grid of braille cells, plotted in dot coordinates.

    Each cell remembers which series last drew in it, so several series can
    share one canvas and still be coloured apart.
    """

    def __init__(self, cols: int, rows: int):
        self.cols, self.rows = max(1, cols), max(1, rows)
        self.bits = [[0] * self.cols for _ in range(self.rows)]
        self.owner: list[list[str | None]] = [[None] * self.cols for _ in range(self.rows)]

    @property
    def width(self) -> int:
        return self.cols * 2

    @property
    def height(self) -> int:
        return self.rows * 4

    def dot(self, x: int, y: int, owner: str | None = None) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            r, c = y // 4, x // 2
            self.bits[r][c] |= _DOT[y % 4][x % 2]
            self.owner[r][c] = owner

    def line(self, x0: int, y0: int, x1: int, y1: int, owner: str | None = None) -> None:
        """Bresenham, so steep segments stay connected rather than dotted."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.dot(x0, y0, owner)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rows_text(self, style_of) -> list[Text]:
        """One Text per cell row; `style_of(row, owner)` picks each colour."""
        out = []
        for r in range(self.rows):
            text = Text()
            for c in range(self.cols):
                bits = self.bits[r][c]
                if bits:
                    text.append(chr(0x2800 + bits), style=style_of(r, self.owner[r][c]))
                else:
                    text.append(" ")
            out.append(text)
        return out


def _bounds(values: list[float], floor: float | None, ceiling: float | None) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        pad = max(abs(hi) * 0.01, 1e-3)
    else:
        pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    if floor is not None:
        lo = max(lo, floor)
    if ceiling is not None:
        hi = min(hi, ceiling)
    return lo, hi


class BrailleChart:
    """Several series as braille lines, or one as a gradient-filled area.

    `lo`/`hi` fix the y-range; otherwise it is zoomed to the data, which is
    what makes four models above 0.99 distinguishable, and the axis labels
    then state the zoomed range so the picture cannot overclaim.
    """

    def __init__(
        self,
        series: dict[str, list[float]],
        colours: dict[str, str] | None = None,
        fill: bool = False,
        lo: float | None = None,
        hi: float | None = None,
        floor: float | None = None,
        ceiling: float | None = None,
        fmt: str = "{:.3f}",
        x_labels: tuple[str, str] | None = None,
        axis: bool = True,
        height: int | None = None,
        align_right: bool = False,
    ):
        self.series = series
        self.colours = colours or {}
        self.fill = fill
        self.lo, self.hi = lo, hi
        self.floor, self.ceiling = floor, ceiling
        self.fmt = fmt
        self.x_labels = x_labels
        self.axis = axis
        self.fixed_height = height
        self.align_right = align_right

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        total_rows = self.fixed_height or options.height or 6
        rows = max(1, total_rows - (1 if self.x_labels else 0))
        values = [v for s in self.series.values() for v in s]
        if not values:
            yield Text("waiting for data…", style=LABEL)
            for _ in range(total_rows - 1):
                yield Text("")
            return

        lo, hi = _bounds(values, self.floor, self.ceiling)
        if self.lo is not None:
            lo = self.lo
        if self.hi is not None:
            hi = self.hi
        span = (hi - lo) or 1.0

        labels = [self.fmt.format(hi), self.fmt.format((hi + lo) / 2), self.fmt.format(lo)]
        label_w = max(map(len, labels)) + 1 if self.axis else 0
        cols = max(2, options.max_width - label_w)
        canvas = Canvas(cols, rows)

        def y_of(v: float) -> int:
            return round((hi - min(max(v, lo), hi)) / span * (canvas.height - 1))

        for name, ys in self.series.items():
            if not ys:
                continue
            n = len(ys)
            if self.align_right:
                # Newest sample at the right edge, scrolling like btop's rather
                # than stretching. Up to three dot columns per sample, so a
                # short history still reads as a graph, not a sliver.
                step = max(1, min(3, canvas.width // max(1, n)))
                ys = ys[-(canvas.width // step):]
                n = len(ys)
                xs = [canvas.width - (n - i) * step for i in range(n)]
            else:
                xs = [0] if n == 1 else [round(i * (canvas.width - 1) / (n - 1)) for i in range(n)]
            pts = [(x, y_of(v)) for x, v in zip(xs, ys)]
            if self.fill:
                for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
                    for x in range(x0, max(x0, x1) + 1):
                        t = 0 if x1 == x0 else (x - x0) / (x1 - x0)
                        top = round(y0 + t * (y1 - y0))
                        for y in range(top, canvas.height):
                            canvas.dot(x, y, name)
            else:
                if len(pts) == 1:
                    canvas.dot(*pts[0], name)
                for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
                    canvas.line(x0, y0, x1, y1, name)

        if self.fill:
            def style_of(r, owner):
                return gradient(1 - r / max(1, rows - 1))
        else:
            def style_of(r, owner):
                return self.colours.get(owner, "white")

        mid = rows // 2
        for r, text in enumerate(canvas.rows_text(style_of)):
            if self.axis:
                label = labels[0] if r == 0 else labels[2] if r == rows - 1 else labels[1] if r == mid else ""
                yield Text(label.rjust(label_w - 1) + " ", style=LABEL) + text
            else:
                yield text
        if self.x_labels:
            left, right = self.x_labels
            gap = max(1, cols - len(left) - len(right))
            yield Text(" " * label_w + left + " " * gap + right, style=LABEL)


class GpuSampler:
    """GPU utilisation, memory, temperature and power through NVML.

    Sampled at most every `interval` seconds, because the screen refreshes
    ten times a second and NVML calls are not free. Returns `None` when NVML
    is missing or there is no NVIDIA device, and the GPU box says so instead
    of crashing a CPU-only run.
    """

    def __init__(self, interval: float = 0.25, history: int = 240):
        self.interval = interval
        self.util: deque[float] = deque(maxlen=history)
        self._last = 0.0
        self._latest: dict | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(self.name, bytes):
                self.name = self.name.decode()
        except Exception:
            self._nvml = None
            self.name = None

    def sample(self) -> dict | None:
        if self._nvml is None:
            return None
        now = time.monotonic()
        if self._latest is not None and now - self._last < self.interval:
            return self._latest
        n, h = self._nvml, self._handle
        try:
            mem = n.nvmlDeviceGetMemoryInfo(h)
            util = float(n.nvmlDeviceGetUtilizationRates(h).gpu)
            self._latest = {
                "name": self.name,
                "util": util,
                "mem_used": mem.used / 2**30,
                "mem_total": mem.total / 2**30,
                "temp": float(n.nvmlDeviceGetTemperature(h, 0)),
                "power": n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                "power_limit": n.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
            }
        except Exception:
            return self._latest
        self.util.append(util)
        self._last = now
        return self._latest
