"""
Panel rendering for palimtop.

Pure functions from numbers to `rich` renderables, kept separate from the loop
that produces the numbers so they can be tested without a model, a GPU, or a
terminal. Every panel corresponds to one course module, which is the point of
the layout: the screen is the submission's argument, live.
"""

from __future__ import annotations

import time

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .. import CLASS_NAMES, THRESHOLDS

# Eight levels of block, for inline sparklines.
BLOCKS = " ▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 24) -> str:
    """A one-line history. Empty string when there is nothing to show."""
    if not values:
        return ""
    recent = values[-width:]
    low, high = min(recent), max(recent)
    span = high - low
    if span <= 0:
        return BLOCKS[4] * len(recent)
    return "".join(BLOCKS[1 + int((v - low) / span * (len(BLOCKS) - 2))] for v in recent)


def bar(fraction: float, width: int = 12, filled: str = "█", empty: str = "░") -> str:
    fraction = max(0.0, min(1.0, float(fraction)))
    n = int(round(fraction * width))
    return filled * n + empty * (width - n)


def header_panel(
    backbone: str,
    device: str,
    throughput: float,
    latency: dict,
    history: list[float],
    vram: tuple[float, float] | None = None,
) -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(justify="right")

    spark = sparkline(history)
    grid.add_row(
        Text.assemble(
            ("throughput  ", "dim"),
            (f"{throughput:6.1f} reg/s  ", "bold"),
            (spark, "cyan"),
        ),
        Text(
            f"VRAM {vram[0]:.1f}/{vram[1]:.1f} GB" if vram else "",
            style="dim",
        ),
    )
    grid.add_row(
        Text.assemble(
            ("latency     ", "dim"),
            (f"p50 {latency['p50']:.1f}ms   ", ""),
            (f"p95 {latency['p95']:.1f}ms   ", ""),
            (f"p99 {latency['p99']:.1f}ms", "dim"),
        ),
        Text(""),
    )
    return Panel(grid, title=f"[bold]PALIMPSEST[/]  {backbone}  ·  {device}",
                 title_align="left", border_style="cyan")


def budget_panel(report: dict) -> Panel:
    """Module 1: the compile budget as an M/D/1 queue."""
    queue = report["queue"]
    rho = queue["rho"]
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left", style="dim")
    table.add_column(justify="right")

    table.add_row("budget", f"{report['budget_ms']:.0f} ms/page")
    table.add_row("regions", f"{report['n_regions']}")
    table.add_row("share", f"{report['share_ms_per_region']:.0f} ms/region")
    table.add_row("service", f"{report['service_ms_per_region']:.1f} ms/region")

    colour = "green" if rho < 0.75 else ("yellow" if rho < 1.0 else "red")
    table.add_row("ρ utilisation", f"[{colour}]{rho:.3f}[/] {bar(rho, 10)}")

    if queue["stable"]:
        table.add_row("M/D/1 Wq", f"{queue['wait_md1_ms']:.0f} ms")
        table.add_row("M/M/1 Wq", f"[dim]{queue['wait_mm1_ms']:.0f} ms (×"
                                  f"{queue['determinism_saving']:.1f})[/]")
    else:
        table.add_row("M/D/1 Wq", "[red]unstable[/]")
        table.add_row("M/M/1 Wq", "[red]unstable[/]")

    verdict = ("[green]✓ WITHIN BUDGET[/]" if report["within_budget"]
               else "[red]✗ OVER BUDGET[/]")
    table.add_row("status", verdict)
    return Panel(table, title="MODULE 1 · compile budget", title_align="left",
                 border_style="blue")


def confusion_panel(matrix: list[list[float]], macro_f1: float, ece: float) -> Panel:
    """Module 5: the 5x5, row-normalised so the diagonal reads as recall."""
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="right", style="dim")
    for _ in CLASS_NAMES:
        table.add_column(justify="right")

    table.add_row("", *[Text(name[:4], style="dim") for name in CLASS_NAMES])
    for i, name in enumerate(CLASS_NAMES):
        cells = []
        for j, value in enumerate(matrix[i]):
            if value == 0:
                cells.append(Text(" ·  ", style="dim"))
            elif i == j:
                style = "bold green" if value > 0.9 else "bold yellow"
                cells.append(Text(f"{value:.2f}", style=style))
            else:
                cells.append(Text(f"{value:.2f}", style="red" if value > 0.05 else "dim"))
        table.add_row(Text(name[:4], style="dim"), *cells)

    footer = Text.assemble(
        ("\nmacro-F1 ", "dim"), (f"{macro_f1:.3f}", "bold"),
        ("    ECE ", "dim"), (f"{ece:.3f}", "bold"),
    )
    grid = Table.grid()
    grid.add_row(table)
    grid.add_row(footer)
    return Panel(grid, title="MODULE 5 · confusion, row-normalised",
                 title_align="left", border_style="magenta")


def zoo_panel(entries: list[dict], active: str) -> Panel:
    """Modules 2 and 3: the switchable backbones and what each costs."""
    table = Table.grid(padding=(0, 1))
    table.add_column(width=2)
    table.add_column(justify="left")
    table.add_column(justify="right", style="dim")
    table.add_column(justify="right")
    table.add_column(justify="right")

    for entry in entries:
        is_active = entry["name"] == active
        marker = "●" if is_active else " "
        style = "bold cyan" if is_active else "dim"
        params = entry["parameters"]
        size = f"{params/1e6:.1f}M" if params >= 1e6 else f"{params/1e3:.0f}k"
        table.add_row(
            Text(marker, style=style),
            Text(entry["name"], style=style),
            Text(size),
            Text(f"{entry.get('latency_ms', 0):.1f}ms"),
            Text(f"{entry.get('macro_f1', 0):.3f}" if entry.get("macro_f1") else "—"),
        )
    return Panel(table, title="MODULES 2+3 · backbone zoo", title_align="left",
                 border_style="green")


def stage_panel(split: dict, stage_ms: dict, bottleneck: str | None) -> Panel:
    """Where the time actually goes."""
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left", style="dim")
    table.add_column()
    table.add_column(justify="right")

    for name, fraction in split.items():
        style = "yellow" if name == bottleneck else "dim"
        table.add_row(
            Text(name, style=style),
            Text(bar(fraction, 12), style=style),
            Text(f"{stage_ms.get(name, 0):.0f}ms {fraction*100:4.0f}%"),
        )
    if bottleneck:
        table.add_row("", "", "")
        table.add_row(Text(""), Text(f"bottleneck: {bottleneck}", style="italic yellow"), Text(""))
    return Panel(table, title="stage split", title_align="left", border_style="yellow")


def held_panel(rates: dict[str, float]) -> Panel:
    """The user-facing consequence of the per-class thresholds."""
    table = Table.grid(padding=(0, 2))
    for _ in CLASS_NAMES:
        table.add_column(justify="left")

    names, bars = [], []
    for name in CLASS_NAMES:
        rate = rates.get(name, 0.0)
        colour = "green" if rate < 0.1 else ("yellow" if rate < 0.3 else "red")
        names.append(Text(f"{name[:7]} τ{THRESHOLDS[name]:.2f}", style="dim"))
        bars.append(Text(f"{bar(rate, 6)} {rate*100:3.0f}%", style=colour))
    table.add_row(*names)
    table.add_row(*bars)
    return Panel(Align.center(table), title="GATE · per-class held rate",
                 title_align="left", border_style="red")


def difficulty_panel(easy_f1: float | None, hard_f1: float | None) -> Panel:
    """Accuracy split by whether the region was drawn ambiguously.

    The single most informative number on the screen: one figure for a mixed
    set hides which half the model is failing on.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="left", style="dim")
    table.add_column(justify="right")
    if easy_f1 is None or hard_f1 is None:
        table.add_row("difficulty split", "[dim]not recorded[/]")
    else:
        table.add_row("clear regions", f"[green]{easy_f1:.3f}[/]")
        table.add_row("ambiguous regions", f"[yellow]{hard_f1:.3f}[/]")
        table.add_row("gap", f"{easy_f1 - hard_f1:+.3f}")
    return Panel(table, title="difficulty", title_align="left", border_style="white")


def mix_panel(counts: dict[str, int], confidence: dict[str, float]) -> Panel:
    """What the server has been classifying, and how sure it was.

    This replaces the confusion matrix in live mode, and the replacement is
    not a compromise: a confusion matrix needs ground truth, and a server
    compiling someone's page has none. Showing an empty or stale matrix
    beside live numbers would be the most misleading thing on the screen.
    What can honestly be shown is the mix and the confidence behind it.
    """
    total = sum(counts.values())
    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left")
    table.add_column(justify="right")
    table.add_column(justify="left")
    table.add_column(justify="right")
    for name in CLASS_NAMES:
        n = counts.get(name, 0)
        share = n / total if total else 0.0
        mean = confidence.get(name)
        table.add_row(
            Text(name[:11], style="dim" if not n else ""),
            Text(str(n)),
            Text(bar(share, 10), style="cyan" if n else "dim"),
            Text("—" if mean is None else f"{mean:.2f}",
                 style="green" if (mean or 0) >= THRESHOLDS[name] else "yellow"),
        )
    return Panel(table, title=f"REGIONS SEEN · {total} · mix and mean confidence",
                 title_align="left", border_style="cyan")


def activity_panel(records: list[dict], rows: int = 7) -> Panel:
    """The last few regions the server classified, newest at the bottom."""
    table = Table.grid(padding=(0, 2))
    for _ in range(5):
        table.add_column(justify="left")
    if not records:
        return Panel(Align.center(Text("waiting for the server to compile a page…",
                                       style="dim")),
                     title="LIVE · from the browser", title_align="left",
                     border_style="magenta")
    for record in records[-rows:]:
        held = record.get("held")
        confidence = float(record.get("confidence", 0.0))
        table.add_row(
            Text(time.strftime("%H:%M:%S", time.localtime(record.get("t", 0))), style="dim"),
            Text(str(record.get("region_class", "?"))[:11],
                 style="yellow" if held else "green"),
            Text(f"{confidence:.2f}"),
            Text("HELD" if held else "written", style="yellow" if held else "dim"),
            Text(f"{record.get('w', 0)}x{record.get('h', 0)}  {record.get('title', '')[:28]}",
                 style="dim"),
        )
    return Panel(table, title="LIVE · from the browser", title_align="left",
                 border_style="magenta")
