"""
palimtop compare — all four backbones, side by side, on the same crops.

`palimtop watch` follows one model. This follows the whole zoo at once: every
trained checkpoint is loaded onto the GPU together, each batch of test crops
goes through all of them, and each forward pass is timed on its own. The
screen is laid out like btop:

    header      progress through the split, elapsed time, clock
    gpu         utilisation, VRAM, power and temperature through NVML
    response    per-crop latency of every model, live, as braille lines
    cards       one per backbone: accuracy, macro-F1 and F1 under the worst
                degradation as gradient meters, p50/p95, rate, ECE, and a
                latency graph on a scale shared by all four
    history     validation macro-F1 per epoch, and macro-F1 under degradation

The live numbers are measured here; the two historical charts are read from
what `app.train` and `app.robustness` already wrote, so the monitor never
re-trains or re-sweeps anything.

Two things keep the latency comparison fair. Each model is warmed up before
timing starts, because the first CUDA call pays for cuDNN autotuning and
kernel loading. And the order in which the models see a batch rotates, so
clock boost or thermal throttling drifts onto all four equally instead of
onto whichever happened to run first.
"""

from __future__ import annotations

import csv
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..metrics.confusion import MetricCore
from ..models.zoo import BACKBONES, describe
from .btop import LABEL, BrailleChart, GpuSampler, gradient, meter

# Meters for accuracy and F1 span this floor to 1.0. Every model sits above
# 0.94, so a 0-to-1 bar would be four identical full bars; the footer states
# the span so the zoom is never hidden.
METER_FLOOR = 0.90
CARD_ROWS = 12
ONCE_HEIGHT = 46

COLOURS = {
    "palimpsest_net": "#56b6c2",
    "squeezenet1_1": "#c678dd",
    "mobilenet_v2": "#e5c07b",
    "resnet18": "#61afef",
}


def _box(body, title: str, colour: str, subtitle: str | Text | None = None) -> Panel:
    """A btop box: rounded, coloured border, title set into the top edge."""
    return Panel(body, box=box.ROUNDED, border_style=colour,
                 title=Text(f" {title} ", style=f"bold {colour}"), title_align="left",
                 subtitle=subtitle if isinstance(subtitle, Text) or subtitle is None
                 else Text(f" {subtitle} ", style=LABEL),
                 subtitle_align="right", padding=(0, 1))


def _legend(entrants) -> Text:
    text = Text(" ")
    for e in entrants:
        text.append("⣿ ", style=COLOURS.get(e.name, "white"))
        text.append(f"{e.name}  ", style=LABEL)
    return text


# ---------------------------------------------------------------------------
# reading what training and the sweep already wrote
# ---------------------------------------------------------------------------

def load_history(run_dir: Path, column: str = "val/macro_f1") -> list[float]:
    """One value per epoch from a Lightning `metrics.csv`, empty if absent.

    Lightning writes train and val metrics on separate rows of the same epoch,
    so rows where the column is blank are skipped rather than read as zero.
    The latest `version_*` directory wins, matching what `best.ckpt` came from.
    """
    versions = sorted((run_dir / "logs").glob("version_*/metrics.csv"))
    if not versions:
        return []
    by_epoch: dict[int, float] = {}
    with versions[-1].open() as handle:
        for row in csv.DictReader(handle):
            value = row.get(column, "")
            if value:
                by_epoch[int(row["epoch"])] = float(value)
    return [by_epoch[e] for e in sorted(by_epoch)]


def load_robustness(run_dir: Path) -> list[tuple[float, float]]:
    """(severity, macro-F1) points from the degradation sweep."""
    path = run_dir / "robustness.json"
    if not path.exists():
        return []
    points = json.loads(path.read_text()).get("sweep", {}).get("points", [])
    return [(float(p["severity"]), float(p["macro_f1"])) for p in points]


def discover(runs_root: Path) -> list[tuple[str, Path]]:
    """Every backbone with a `best.ckpt`, in zoo order."""
    return [(name, runs_root / name) for name in BACKBONES
            if (runs_root / name / "best.ckpt").exists()]


# ---------------------------------------------------------------------------
# the arena
# ---------------------------------------------------------------------------

class Entrant:
    """One backbone: its model, its live metrics, and what its run recorded."""

    def __init__(self, name: str, run_dir: Path, model=None):
        self.name = name
        self.run_dir = run_dir
        self.model = model
        self.core = MetricCore()
        self.batch_ms: deque[float] = deque(maxlen=60)
        self.history = load_history(run_dir)
        self.robustness = load_robustness(run_dir)
        self.parameters = describe(name)["parameters"]

    def crops_per_second(self) -> float:
        p50 = self.core.latency()["p50"]
        return 1000.0 / p50 if p50 > 0 else 0.0


def pareto_names(rows: list[dict]) -> set[str]:
    """Backbones not beaten on both macro-F1 (higher) and p50 (lower).

    A model is dominated if another is at least as good on both axes and
    strictly better on one. With accuracy pinned near 1.0, latency usually
    decides, which is itself the finding worth putting on a slide.
    """
    # Judged at the precision the leaderboard shows, so a star never depends
    # on a digit the reader cannot see.
    rows = [{**r, "macro_f1": round(r["macro_f1"], 4), "p50": round(r["p50"], 3)} for r in rows]
    front = set()
    for a in rows:
        dominated = any(
            b["macro_f1"] >= a["macro_f1"] and b["p50"] <= a["p50"]
            and (b["macro_f1"] > a["macro_f1"] or b["p50"] < a["p50"])
            for b in rows if b is not a
        )
        if not dominated:
            front.add(a["name"])
    return front


class Arena:
    def __init__(self, entrants: list[Entrant], device: str, split: str,
                 gpu: GpuSampler | None = None):
        self.entrants = entrants
        self.gpu = gpu
        self.device = device
        self.split = split
        self.seen = 0
        self.total = 0
        self.started = time.time()

    # -------------------------------------------------------------- measure

    def _timed_forward(self, model, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
        if cuda:
            torch.cuda.synchronize()
        return logits, (time.perf_counter() - began) * 1000.0

    def warm_up(self, x: torch.Tensor, passes: int = 3) -> None:
        for entrant in self.entrants:
            for _ in range(passes):
                self._timed_forward(entrant.model, x)

    def step(self, x: torch.Tensor, y: np.ndarray, batch_index: int, full: bool = True) -> None:
        """One batch through every model. `full=False` marks the ragged last
        batch: it still counts towards accuracy, but its per-crop latency is
        kept off the chart, because fixed launch overhead spread over a
        handful of crops reads as a spike that no real batch would show."""
        k = batch_index % len(self.entrants)
        for entrant in self.entrants[k:] + self.entrants[:k]:
            logits, elapsed_ms = self._timed_forward(entrant.model, x)
            per_crop = elapsed_ms / len(y)
            entrant.core.update(torch.softmax(logits, dim=1).cpu().numpy(), y,
                                latency_ms=per_crop)
            if full:
                entrant.batch_ms.append(per_crop)
        self.seen += len(y)

    # --------------------------------------------------------------- render

    def rows(self) -> list[dict]:
        out = []
        for e in self.entrants:
            lat = e.core.latency()
            robust = dict(e.robustness)
            out.append({
                "name": e.name,
                "parameters": e.parameters,
                "accuracy": e.core.accuracy() if len(e.core) else 0.0,
                "macro_f1": e.core.macro_f1() if len(e.core) else 0.0,
                "ece": e.core.ece() if len(e.core) else 0.0,
                "p50": lat["p50"],
                "p95": lat["p95"],
                "rate": e.crops_per_second(),
                "f1_at_1": robust.get(1.0),
            })
        return out

    def bests(self, rows: list[dict]) -> dict[str, str | None]:
        """The best value per column, *as displayed*, so ties on screen are
        ties in colour whatever their sixth decimal says."""
        spec = {"accuracy": (True, "{:.4f}"), "macro_f1": (True, "{:.4f}"),
                "ece": (False, "{:.4f}"), "p50": (False, "{:.3f}"),
                "p95": (False, "{:.3f}"), "rate": (True, "{:,.0f}"),
                "f1_at_1": (True, "{:.4f}")}
        out: dict[str, str | None] = {}
        for key, (high, fmt) in spec.items():
            vals = [r[key] for r in rows if r[key] is not None]
            out[key] = fmt.format(max(vals) if high else min(vals)) if self.seen and vals else None
        return out

    def header(self) -> Table:
        elapsed = time.time() - self.started
        fraction = self.seen / self.total if self.total else 0.0
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        progress = Text.assemble(
            (f"{self.seen:,}/{self.total:,} ", "bold"), ("crops each ", LABEL),
        ) + meter(fraction, 20)
        grid.add_row(
            Text.assemble((" palimtop ", "bold reverse #77ca9b"), ("  compare · ", LABEL),
                          (f"{self.split} split · {self.device}", "bold")),
            progress,
            Text.assemble((f"{elapsed:6.1f} s  ", LABEL), (time.strftime("%H:%M:%S"), "bold"), " "),
        )
        return grid

    def gpu_box(self) -> Panel:
        stats = self.gpu.sample() if self.gpu else None
        if stats is None:
            body = Text("no NVIDIA GPU visible\n(install nvidia-ml-py for stats)", style=LABEL)
            return _box(body, "gpu", "#77ca9b")

        class _Body:
            def __rich_console__(_, console, options):
                w = options.max_width
                bar = max(4, w - 16)
                yield Text("util  ", style=LABEL) + meter(stats["util"] / 100, bar) + Text(f" {stats['util']:4.0f}%")
                yield Text("vram  ", style=LABEL) + meter(stats["mem_used"] / stats["mem_total"], bar) + \
                    Text(f" {stats['mem_used']:4.1f}G")
                yield Text("power ", style=LABEL) + meter(stats["power"] / max(1.0, stats["power_limit"]), bar) + \
                    Text(f" {stats['power']:4.0f}W")
                yield Text.assemble(("temp  ", LABEL), (f"{stats['temp']:.0f}°C", gradient((stats["temp"] - 30) / 60)),
                                    ("   of ", LABEL), (f"{stats['mem_total']:.1f} GiB", ""))
                rest = (options.height or 8) - 4
                if rest > 0:
                    yield BrailleChart({"util": list(self.gpu.util)}, fill=True, lo=0, hi=100,
                                       axis=False, height=rest, align_right=True)

        return _box(_Body(), "gpu", "#77ca9b", subtitle=stats["name"].replace("NVIDIA GeForce ", ""))

    def response_box(self) -> Panel:
        latency = {e.name: list(e.batch_ms) for e in self.entrants}
        chart = BrailleChart(latency, COLOURS, floor=0.0, fmt="{:.2f}", align_right=True)
        return _box(chart, "response time · ms per crop", "#cbc06c", subtitle=_legend(self.entrants))

    def card(self, e: Entrant, row: dict, bests: dict, starred: bool, lat_hi: float) -> Panel:
        colour = COLOURS.get(e.name, "white")

        def value(key: str, fmt: str, suffix: str = "") -> Text:
            if row[key] is None:
                return Text("—", style=LABEL)
            shown = fmt.format(row[key])
            best = bests.get(key) == shown
            return Text(shown + suffix, style="bold #77ca9b" if best else "bold")

        class _Body:
            def __rich_console__(_, console, options):
                w = options.max_width
                bar = max(4, w - 13)
                for label, key in (("acc  ", "accuracy"), ("F1   ", "macro_f1"), ("F1@1 ", "f1_at_1")):
                    v = row[key]
                    fraction = 0.0 if v is None else (v - METER_FLOOR) / (1 - METER_FLOOR)
                    yield Text(label, style=LABEL) + meter(fraction, bar) + Text(" ") + value(key, "{:.4f}")
                yield Text.assemble(("p50  ", LABEL)) + value("p50", "{:.3f}", " ms") + \
                    Text("   p95 ", style=LABEL) + value("p95", "{:.3f}", " ms")
                yield Text.assemble(("rate ", LABEL)) + value("rate", "{:,.0f}", "/s") + \
                    Text("   ECE ", style=LABEL) + value("ece", "{:.4f}")
                rest = (options.height or 10) - 5
                if rest > 0:
                    yield BrailleChart({e.name: list(e.batch_ms)}, fill=True, lo=0.0, hi=lat_hi,
                                       axis=False, height=rest, align_right=True)

        title = ("★ " if starred else "") + e.name
        return _box(_Body(), title, colour, subtitle=f"{e.parameters / 1e6:.2f} M params")

    def training_box(self) -> Panel:
        history = {e.name: e.history for e in self.entrants}
        epochs = max((len(h) for h in history.values()), default=0)
        chart = BrailleChart(history, COLOURS, ceiling=1.0,
                             x_labels=("epoch 0", f"epoch {max(0, epochs - 1)}"))
        return _box(chart, "validation macro-F1 per epoch", "#56b6c2")

    def robustness_box(self) -> Panel:
        sweep = {e.name: [f1 for _, f1 in e.robustness] for e in self.entrants}
        severities = next((e.robustness for e in self.entrants if e.robustness), [])
        labels = (f"clean", f"severity {severities[-1][0]:g}") if severities else None
        chart = BrailleChart(sweep, COLOURS, ceiling=1.0, x_labels=labels)
        return _box(chart, "macro-F1 under degradation", "#dc4c4c")

    def layout(self) -> Layout:
        rows = self.rows()
        bests = self.bests(rows)
        front = pareto_names(rows) if self.seen else set()
        # One latency scale for all four cards, so a redder graph really is a
        # slower model rather than an artefact of per-card zoom.
        lat_hi = max((max(e.batch_ms) for e in self.entrants if e.batch_ms), default=1.0) * 1.1

        root = Layout()
        root.split_column(
            Layout(name="header", size=1),
            Layout(name="top", ratio=3, minimum_size=9),
            Layout(name="cards", size=CARD_ROWS),
            Layout(name="bottom", ratio=3, minimum_size=8),
            Layout(name="footer", size=1),
        )
        root["header"].update(self.header())
        root["top"].split_row(Layout(self.gpu_box(), ratio=1), Layout(self.response_box(), ratio=3))
        root["cards"].split_row(*[
            Layout(self.card(e, r, bests, e.name in front, lat_hi))
            for e, r in zip(self.entrants, rows)
        ])
        root["bottom"].split_row(Layout(self.training_box()), Layout(self.robustness_box()))
        root["footer"].update(Text(
            f" ★ Pareto front, macro-F1 vs p50   green = best in class   "
            f"meters span {METER_FLOOR:.2f} → 1.00   latency per crop, amortised over the batch   ctrl+c quit",
            style=LABEL, no_wrap=True, overflow="ellipsis"))
        return root

    def report(self) -> dict:
        return {
            "split": self.split,
            "device": self.device,
            "crops": self.seen,
            "pareto_front": sorted(pareto_names(self.rows())),
            "backbones": {
                e.name: {
                    "parameters": e.parameters,
                    "accuracy": e.core.accuracy(),
                    "macro_f1": e.core.macro_f1(),
                    "ece": e.core.ece(),
                    "latency_ms": e.core.latency(),
                }
                for e in self.entrants
            },
        }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def compare(
    runs_root: str | Path = "runs",
    data_root: str | Path = "data/dataset",
    split: str = "test",
    batch_size: int = 16,
    loops: int = 1,
    delay: float = 0.03,
    live: bool = True,
    console: Console | None = None,
) -> Arena:
    from ..data.dataset import CropDataset
    from ..evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    found = discover(Path(runs_root))
    if not found:
        raise SystemExit(f"no runs/<backbone>/best.ckpt under {runs_root!r}")
    entrants = [Entrant(name, run_dir, load_checkpoint(run_dir / "best.ckpt", device=device))
                for name, run_dir in found]

    root = Path(data_root)
    dataset = CropDataset(root / "crops" / f"{split}_crops.npy",
                          root / "crops" / f"{split}_labels.npy", augment=False)
    console = console or Console()
    arena = Arena(entrants, device=device, split=split,
                  gpu=GpuSampler() if device == "cuda" else None)
    arena.total = len(dataset) * loops

    def batches():
        index = 0
        for _ in range(loops):
            for start in range(0, len(dataset), batch_size):
                items = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
                x = torch.stack([item[0] for item in items]).to(device)
                yield index, x, np.array([item[1] for item in items]), len(items) == batch_size
                index += 1

    first = torch.stack([dataset[i][0] for i in range(min(batch_size, len(dataset)))]).to(device)
    arena.warm_up(first)
    arena.started = time.time()

    if not live:
        for index, x, y, full in batches():
            arena.step(x, y, index, full)
        console.print(arena.layout(), height=arena_height(arena))
        return arena

    # Full screen, like btop: the alternate buffer keeps the scrollback clean,
    # and the final frame is printed to the normal screen afterwards so the
    # numbers survive the exit.
    try:
        with Live(arena.layout(), console=console, refresh_per_second=10, screen=True) as screen:
            for index, x, y, full in batches():
                arena.step(x, y, index, full)
                screen.update(arena.layout())
                if delay:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    console.print(arena.layout(), height=arena_height(arena))
    return arena


def arena_height(arena: Arena) -> int:
    return ONCE_HEIGHT
