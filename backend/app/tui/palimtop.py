"""
palimtop — a live terminal monitor for the region classifier.

Four modes over one shared metric core:

    palimtop live    the running server, as someone uses it in the browser.
                     Loops until Ctrl-C and shows each region as the page is
                     compiled, which is the mode to leave open in a terminal
    palimtop watch   a trained model classifying region crops, with the
                     compile budget, the queueing model, the live confusion
                     matrix and the held rates updating as it runs
    palimtop train   the same metrics filling in during a Lightning run

Every panel on the screen corresponds to a course module, which is the point
of the layout rather than a coincidence of it: Module 1 is the budget and its
M/D/1 queue, Modules 2 and 3 are the backbone zoo and what each costs,
Module 4 is upstream in the data the crops came from, and Module 5 is the
confusion matrix, macro-F1 and calibration.

`live` works without sharing memory with the server. The compiler appends one
line per region to `runs/activity.jsonl` and this tails it, which is why it
needs no socket, no second port and no cooperation beyond a file. It shows no
confusion matrix, deliberately: a server compiling someone's page has no
ground truth, and a matrix beside live numbers would be the most misleading
thing on the screen.

Installed as a command rather than invoked as a module: `backend/bin/palimtop`
resolves its own symlink to find the checkout, so it can be linked into
`~/.local/bin` and run from anywhere, the way btop is.

Built on `rich`, which is already a dependency for CLI output, so this adds
nothing to install and no licence question for an academic submission.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live

from .. import CLASS_NAMES
from ..metrics.confusion import MetricCore
from ..models.zoo import BACKBONES, describe
from ..pipeline.budget import CompileBudget
from . import panels

# Reference costs for the stages palimtop does not itself run. The classifier
# is measured live; preprocessing and routing are quoted at the documented
# reference cost so the budget panel reflects a whole page rather than only
# the part that happens to be in front of us. Replace with measured values
# once the pipeline runs end to end.
REFERENCE_PREPROCESS_MS = 42.0
REFERENCE_ROUTE_MS = 87.0

TYPICAL_REGIONS_PER_PAGE = 8


class Monitor:
    """Shared state behind both modes."""

    def __init__(self, backbone: str, device: str, window: int = 600):
        self.backbone = backbone
        self.device = device
        self.core = MetricCore(window=window)
        self.throughput_history: deque[float] = deque(maxlen=60)
        # (timestamp, cumulative count) pairs, for a rolling rate. A
        # cumulative average over the whole run would read as a flat line
        # that slowly forgets how fast things are going right now, which is
        # the one thing a live monitor exists to show.
        self._marks: deque[tuple[float, int]] = deque(maxlen=40)
        self.zoo = [describe(name) for name in BACKBONES]
        self.easy_core = MetricCore()
        self.hard_core = MetricCore()
        self.started = time.time()
        self.seen = 0

    def observe(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        latency_ms: float,
        hard: np.ndarray | None = None,
    ) -> None:
        per_crop = latency_ms / max(1, len(targets))
        self.core.update(probabilities, targets, latency_ms=per_crop)
        if hard is not None:
            for p, t, h in zip(probabilities, targets, hard):
                (self.hard_core if h else self.easy_core).update(
                    p[None], np.array([t]), latency_ms=per_crop
                )
        self.seen += len(targets)
        now = time.time()
        self._marks.append((now, self.seen))
        oldest_time, oldest_count = self._marks[0]
        window = now - oldest_time
        if window > 1e-3:
            self.throughput_history.append((self.seen - oldest_count) / window)
        elif not self.throughput_history:
            self.throughput_history.append(0.0)

    # ------------------------------------------------------------- render

    def _vram(self) -> tuple[float, float] | None:
        if not self.device.startswith("cuda") or not torch.cuda.is_available():
            return None
        used = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return used, total

    def budget_report(self) -> dict:
        infer_ms = self.core.latency()["p50"]
        return CompileBudget(
            n_regions=TYPICAL_REGIONS_PER_PAGE,
            stage_ms={
                "preprocess": REFERENCE_PREPROCESS_MS,
                "infer": infer_ms,
                "route": REFERENCE_ROUTE_MS,
            },
        ).report()

    def layout(self) -> Layout:
        budget = self.budget_report()
        for entry in self.zoo:
            if entry["name"] == self.backbone:
                entry["latency_ms"] = self.core.latency()["p50"]
                entry["macro_f1"] = self.core.macro_f1()

        root = Layout()
        root.split_column(
            Layout(name="header", size=5),
            Layout(name="middle", size=11),
            Layout(name="lower", size=9),
            Layout(name="gate", size=5),
        )
        root["header"].update(panels.header_panel(
            self.backbone, self.device,
            throughput=self.throughput_history[-1] if self.throughput_history else 0.0,
            latency=self.core.latency(),
            history=list(self.throughput_history),
            vram=self._vram(),
        ))
        root["middle"].split_row(Layout(name="budget"), Layout(name="confusion"))
        root["middle"]["budget"].update(panels.budget_panel(budget))
        root["middle"]["confusion"].update(panels.confusion_panel(
            self.core.confusion_row_normalised().tolist(),
            self.core.macro_f1(),
            self.core.ece(),
        ))
        root["lower"].split_row(Layout(name="zoo"), Layout(name="stages", ratio=1))
        root["lower"]["zoo"].update(panels.zoo_panel(self.zoo, self.backbone))
        root["lower"]["stages"].update(Group(
            panels.stage_panel(budget["stage_split"], budget["stage_ms"], budget["bottleneck"]),
            panels.difficulty_panel(
                self.easy_core.macro_f1() if len(self.easy_core) else None,
                self.hard_core.macro_f1() if len(self.hard_core) else None,
            ),
        ))
        root["gate"].update(panels.held_panel(self.core.held_rate()))
        return root


# ---------------------------------------------------------------------------
# live
# ---------------------------------------------------------------------------

def live(path: str | Path | None = None, refresh: float = 0.5,
         history: bool = True, console: Console | None = None) -> None:
    """Tail the server's activity and render it until interrupted.

    Loops forever by design. This is the mode meant to sit in a terminal
    beside the browser the way `btop` does, so it has no finish condition:
    an idle server is a valid state and shows as one, not as an exit.
    """
    from ..metrics.activity import ActivityLog

    log = ActivityLog(path) if path else ActivityLog()
    console = console or Console()
    seen: list[dict] = []
    counts: dict[str, int] = {}
    totals: dict[str, float] = {}
    stage = {"preprocess": 0.0, "infer": 0.0, "route": 0.0}
    rate: deque[float] = deque(maxlen=60)
    # Start at the end unless asked otherwise, so opening the monitor does
    # not replay an afternoon of compiles as though they just happened.
    offset = 0 if history else log.end()
    marks: deque[tuple[float, int]] = deque(maxlen=40)

    def screen() -> Layout:
        held = {name: 0 for name in CLASS_NAMES}
        total = {name: 0 for name in CLASS_NAMES}
        for record in seen:
            name = record.get("region_class")
            if name in total:
                total[name] += 1
                held[name] += bool(record.get("held"))
        rates = {name: (held[name] / total[name] if total[name] else 0.0) for name in CLASS_NAMES}
        # Every key `header_panel` reads. The first version supplied p50,
        # p95 and mean and the panel also wants p99, so `palimtop live`
        # died with a KeyError on its very first frame — the one path no
        # test covered, because the panels are tested with a complete dict
        # and this is the only caller that builds one by hand.
        latency = {"p50": stage["infer"], "p95": stage["infer"],
                   "p99": stage["infer"], "mean": stage["infer"]}

        root = Layout()
        root.split_column(
            Layout(name="header", size=5),
            Layout(name="middle", size=11),
            Layout(name="feed", size=11),
            Layout(name="gate", size=5),
        )
        root["header"].update(panels.header_panel(
            seen[-1].get("backbone", "—") if seen else "—",
            seen[-1].get("device", "—") if seen else "—",
            throughput=rate[-1] if rate else 0.0,
            latency=latency, history=list(rate), vram=None,
        ))
        root["middle"].split_row(Layout(name="budget"), Layout(name="mix"))
        root["middle"]["budget"].update(panels.budget_panel(CompileBudget(
            n_regions=TYPICAL_REGIONS_PER_PAGE,
            stage_ms={"preprocess": stage["preprocess"] or REFERENCE_PREPROCESS_MS,
                      "infer": stage["infer"],
                      "route": stage["route"] or REFERENCE_ROUTE_MS},
        ).report()))
        root["middle"]["mix"].update(panels.mix_panel(
            counts, {k: totals[k] / counts[k] for k in counts if counts[k]}))
        root["feed"].update(panels.activity_panel(seen))
        root["gate"].update(panels.held_panel(rates))
        return root

    with Live(screen(), console=console, refresh_per_second=4, screen=True) as display:
        try:
            while True:
                records, offset = log.read(offset)
                for record in records:
                    seen.append(record)
                    name = record.get("region_class")
                    if name:
                        counts[name] = counts.get(name, 0) + 1
                        totals[name] = totals.get(name, 0.0) + float(record.get("confidence", 0.0))
                    for key in stage:
                        value = record.get(f"{key}_ms")
                        if value:
                            # Exponential, so the panel follows the current
                            # page rather than averaging over the whole day.
                            stage[key] = 0.7 * stage[key] + 0.3 * float(value) if stage[key] else float(value)
                del seen[:-400]
                now = time.time()
                marks.append((now, len(seen)))
                span = now - marks[0][0]
                rate.append((len(seen) - marks[0][1]) / span if span > 1e-3 else 0.0)
                display.update(screen())
                time.sleep(refresh)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------

def watch(
    checkpoint: str | Path,
    data_root: str | Path = "data/dataset",
    split: str = "test",
    batch_size: int = 16,
    loops: int = 1,
    delay: float = 0.04,
    console: Console | None = None,
) -> Monitor:
    """Stream a split through a trained model, updating the screen live.

    Deliberately paced: the GPU would finish the whole split in under a second
    and there would be nothing to watch. `delay` is the only artificial thing
    on the screen, and it does not touch the measured latency, which is timed
    around the forward pass alone.
    """
    from ..data.dataset import CropDataset
    from ..evaluate import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_checkpoint(checkpoint, device=device)
    root = Path(data_root)
    dataset = CropDataset(
        root / "crops" / f"{split}_crops.npy",
        root / "crops" / f"{split}_labels.npy",
        augment=False,
    )
    hard_path = root / "crops" / f"{split}_hard.npy"
    hard_flags = np.load(hard_path) if hard_path.exists() else None

    monitor = Monitor(backbone=model.hparams.backbone, device=device)
    console = console or Console()

    with Live(monitor.layout(), console=console, refresh_per_second=12,
              screen=False) as live:
        for _ in range(loops):
            for start in range(0, len(dataset), batch_size):
                stop = min(start + batch_size, len(dataset))
                items = [dataset[i] for i in range(start, stop)]
                x = torch.stack([item[0] for item in items]).to(device)
                y = np.array([item[1] for item in items])

                if device == "cuda":
                    torch.cuda.synchronize()
                began = time.perf_counter()
                with torch.no_grad():
                    logits = model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - began) * 1000.0

                probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                monitor.observe(
                    probabilities, y, elapsed_ms,
                    hard=hard_flags[start:stop] if hard_flags is not None else None,
                )
                live.update(monitor.layout())
                if delay:
                    time.sleep(delay)
    return monitor


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train_callback(console: Console | None = None):
    """A Lightning callback that renders the same panels during a fit."""
    import lightning as L

    class PalimtopCallback(L.Callback):
        def __init__(self):
            self.monitor: Monitor | None = None
            self.live: Live | None = None
            self.console = console or Console()

        def on_fit_start(self, trainer, module):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.monitor = Monitor(backbone=module.hparams.backbone, device=device)
            self.live = Live(self.monitor.layout(), console=self.console,
                             refresh_per_second=4)
            self.live.start()

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if self.monitor is None:
                return
            x, y = batch
            with torch.no_grad():
                probabilities = torch.softmax(module(x), dim=1).cpu().numpy()
            self.monitor.observe(probabilities, y.cpu().numpy(), latency_ms=0.0)
            if self.live:
                self.live.update(self.monitor.layout())

        def on_fit_end(self, trainer, module):
            if self.live:
                self.live.stop()

    return PalimtopCallback()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="palimtop", description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    live_parser = sub.add_parser(
        "live", help="watch the running server, until Ctrl-C")
    live_parser.add_argument("--path", default=None,
                             help="activity file (default runs/activity.jsonl)")
    live_parser.add_argument("--refresh", type=float, default=0.5)
    live_parser.add_argument("--new-only", action="store_true",
                             help="start at the end rather than replaying this session")

    watch_parser = sub.add_parser("watch", help="monitor a trained model")
    watch_parser.add_argument("--checkpoint", required=True)
    watch_parser.add_argument("--data-root", default="data/dataset")
    watch_parser.add_argument("--split", default="test")
    watch_parser.add_argument("--batch-size", type=int, default=16)
    watch_parser.add_argument("--loops", type=int, default=1)
    watch_parser.add_argument("--delay", type=float, default=0.04)

    compare_parser = sub.add_parser("compare", help="all trained backbones side by side")
    compare_parser.add_argument("--runs", default="runs")
    compare_parser.add_argument("--data-root", default="data/dataset")
    compare_parser.add_argument("--split", default="test")
    compare_parser.add_argument("--batch-size", type=int, default=16)
    compare_parser.add_argument("--loops", type=int, default=1)
    compare_parser.add_argument("--delay", type=float, default=0.03)
    compare_parser.add_argument("--once", action="store_true",
                                help="run the split without animating and print the final screen")
    compare_parser.add_argument("--json", help="also write the final numbers here")
    compare_parser.add_argument("--svg", help="also save the final screen as SVG, for the deck")

    train_parser = sub.add_parser("train", help="monitor a training run")
    train_parser.add_argument("--backbone", default="mobilenet_v2")
    train_parser.add_argument("--data-root", default="data/dataset")
    train_parser.add_argument("--epochs", type=int, default=15)

    args = parser.parse_args()
    if args.mode == "live":
        live(args.path, refresh=args.refresh, history=not args.new_only)
    elif args.mode == "watch":
        monitor = watch(args.checkpoint, args.data_root, split=args.split,
                        batch_size=args.batch_size, loops=args.loops, delay=args.delay)
        report = monitor.core.report()
        Console().print(
            f"\n[bold]{monitor.backbone}[/]  "
            f"macro-F1 {report['macro_f1']:.4f}  "
            f"ECE {report['ece']:.4f}  "
            f"p50 {report['latency_ms']['p50']:.2f} ms  "
            f"over {report['samples']} crops"
        )
    elif args.mode == "compare":
        import json

        from .arena import arena_height, compare

        console = Console(record=bool(args.svg))
        arena = compare(args.runs, args.data_root, split=args.split,
                        batch_size=args.batch_size, loops=args.loops,
                        delay=args.delay, live=not args.once, console=console)
        if args.json:
            Path(args.json).write_text(json.dumps(arena.report(), indent=1))
        if args.svg:
            if not args.once:
                # Drop every recorded Live frame; the SVG is the final screen only.
                console.export_text(clear=True)
                console.print(arena.layout(), height=arena_height(arena))
            console.save_svg(args.svg, title="palimtop compare")
    else:
        from ..train import train

        train(backbone=args.backbone, data_root=args.data_root,
              epochs=args.epochs, extra_callbacks=[train_callback()])


if __name__ == "__main__":
    main()
