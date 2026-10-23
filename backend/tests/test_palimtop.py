"""Dashboard tests.

The panels are pure functions from numbers to renderables, so they are tested
without a model, a GPU or a terminal. What is worth asserting is that the
screen cannot lie: that it renders at all on a narrow terminal, that a value
appearing on it came from the metric core, and that nothing divides by zero
before the first batch arrives, which is the state the dashboard spends its
first second in.
"""

import time
from collections import deque

import numpy as np
import pytest

from rich.console import Console

from app import CLASS_NAMES, NUM_CLASSES
from app.pipeline.budget import CompileBudget
from app.tui import panels
from app.tui.palimtop import Monitor


def render(renderable, width: int = 100) -> str:
    console = Console(width=width, record=True, force_terminal=False)
    console.print(renderable)
    return console.export_text()


def probs(index: int, confidence: float = 0.95) -> np.ndarray:
    rest = (1.0 - confidence) / (NUM_CLASSES - 1)
    row = np.full(NUM_CLASSES, rest)
    row[index] = confidence
    return row[None]


class TestSparkline:
    def test_empty_history_is_empty(self):
        assert panels.sparkline([]) == ""

    def test_flat_history_is_flat(self):
        assert len(set(panels.sparkline([3.0] * 10))) == 1

    def test_rising_history_ends_higher_than_it_starts(self):
        line = panels.sparkline([float(i) for i in range(10)])
        assert panels.BLOCKS.index(line[-1]) > panels.BLOCKS.index(line[0])

    def test_respects_the_width(self):
        assert len(panels.sparkline([float(i) for i in range(100)], width=20)) == 20


class TestBar:
    def test_clamps_out_of_range_values(self):
        assert panels.bar(-5, 10) == "░" * 10
        assert panels.bar(5, 10) == "█" * 10

    def test_half_is_half(self):
        assert panels.bar(0.5, 10).count("█") == 5


class TestPanelsRender:
    def test_budget_panel_shows_both_queueing_models(self):
        report = CompileBudget(n_regions=8, stage_ms={"infer": 18.0, "route": 87.0}).report()
        text = render(panels.budget_panel(report))
        assert "M/D/1" in text and "M/M/1" in text

    def test_budget_panel_survives_an_unstable_queue(self):
        """Over budget is a state the screen must display, not crash on."""
        report = CompileBudget(n_regions=80, stage_ms={"infer": 90.0}).report()
        text = render(panels.budget_panel(report))
        assert "unstable" in text and "OVER BUDGET" in text

    def test_confusion_panel_lists_every_class(self):
        matrix = np.eye(NUM_CLASSES).tolist()
        text = render(panels.confusion_panel(matrix, 1.0, 0.01))
        for name in CLASS_NAMES:
            assert name[:4] in text

    def test_held_panel_shows_each_threshold(self):
        text = render(panels.held_panel({n: 0.1 for n in CLASS_NAMES}))
        assert "0.88" in text, "the FORMULA threshold is not on screen"

    def test_zoo_panel_marks_the_active_backbone(self):
        entries = [{"name": "squeezenet1_1", "parameters": 725_061},
                   {"name": "resnet18", "parameters": 11_179_077}]
        text = render(panels.zoo_panel(entries, active="resnet18"))
        assert "●" in text

    def test_difficulty_panel_handles_missing_data(self):
        assert "not recorded" in render(panels.difficulty_panel(None, None))


class TestMonitor:
    def test_renders_before_any_data_arrives(self):
        """The dashboard's first second. Every rate is a division by a count
        that is still zero."""
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        assert render(monitor.layout(), width=100)

    def test_renders_on_a_narrow_terminal(self):
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        assert render(monitor.layout(), width=80)

    def test_observed_predictions_reach_the_screen(self):
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        for _ in range(20):
            monitor.observe(probs(1), np.array([1]), latency_ms=8.0)
        assert monitor.core.macro_f1() > 0
        assert "1.00" in render(monitor.layout())

    def test_difficulty_split_is_tracked_separately(self):
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        monitor.observe(probs(0), np.array([0]), 5.0, hard=np.array([False]))
        monitor.observe(probs(0), np.array([1]), 5.0, hard=np.array([True]))
        assert len(monitor.easy_core) == 1
        assert len(monitor.hard_core) == 1
        assert monitor.easy_core.macro_f1() > monitor.hard_core.macro_f1()

    def test_budget_uses_the_measured_inference_latency(self):
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        batch = np.repeat(probs(2), 16, axis=0)
        for _ in range(10):
            # 160 ms for a batch of 16 is 10 ms per crop, and it is the
            # per-crop figure that the budget charges against.
            monitor.observe(batch, np.full(16, 2), latency_ms=160.0)
        report = monitor.budget_report()
        assert abs(report["stage_ms"]["infer"] - 10.0) < 1.0

    def test_throughput_needs_two_points_before_it_means_anything(self):
        """A rate is a difference over an interval, so one sample cannot
        define one. It reads zero rather than dividing by a zero interval."""
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        monitor.observe(probs(0), np.array([0]), latency_ms=5.0)
        assert monitor.throughput_history == deque([0.0])

    def test_throughput_becomes_positive_as_work_arrives(self):
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        for _ in range(5):
            monitor.observe(probs(0), np.array([0]), latency_ms=5.0)
            time.sleep(0.002)
        assert monitor.throughput_history[-1] > 0

    def test_throughput_is_a_rolling_rate_not_an_average(self):
        """A cumulative average slowly forgets how fast things are going now,
        which is the one thing a live monitor exists to show."""
        monitor = Monitor(backbone="palimpsest_net", device="cpu")
        assert monitor._marks.maxlen is not None


class TestLiveScreen:
    """`palimtop live` builds its own latency dict by hand rather than taking
    one from MetricCore, so it is the one caller that can be missing a key a
    panel reads. It died with KeyError('p99') on its very first frame."""

    @staticmethod
    def _one_frame(monkeypatch):
        """A Live that records the first screen, then ends the loop from
        inside it, which is where `live` catches KeyboardInterrupt."""
        from app.tui import palimtop

        seen = {}

        class FakeLive:
            def __init__(self, renderable, **kw):
                seen["first"] = renderable

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def update(self, renderable):
                seen["update"] = renderable
                raise KeyboardInterrupt

        monkeypatch.setattr(palimtop, "Live", FakeLive)
        return seen

    def test_renders_a_frame_from_an_empty_feed(self, tmp_path, monkeypatch):
        from app.tui import palimtop

        seen = self._one_frame(monkeypatch)
        palimtop.live(tmp_path / "activity.jsonl", refresh=0.01)
        assert seen["first"] is not None and seen["update"] is not None

    def test_renders_a_frame_with_records(self, tmp_path, monkeypatch):
        from app.metrics.activity import ActivityLog
        from app.tui import palimtop

        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(source="compile", backbone="squeezenet1_1", device="cuda",
                   region_class="FORMULA", confidence=0.99, threshold=0.88,
                   held=False, preprocess_ms=30.0, infer_ms=1.0, w=100, h=40, title="t")

        seen = self._one_frame(monkeypatch)
        palimtop.live(tmp_path / "a.jsonl", refresh=0.01)
        assert seen["update"] is not None

    def test_the_header_gets_every_latency_key_it_reads(self):
        from app.tui import panels

        panels.header_panel("squeezenet1_1", "cuda", throughput=1.0,
                            latency={"p50": 1.0, "p95": 2.0, "p99": 3.0, "mean": 1.5},
                            history=[1.0, 2.0], vram=None)
