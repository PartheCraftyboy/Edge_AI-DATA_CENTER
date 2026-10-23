"""palimtop compare tests.

The arena's render functions are pure, so most of this needs no model and no
GPU. What matters: the charts cannot claim a value above 1.0 for an F1, the
log reader does not mistake Lightning's blank cells for zeros, the Pareto
front is computed the way the footer says, and a tie on screen is a tie in
colour.
"""

import json

import numpy as np
import torch
from rich.console import Console

from app import NUM_CLASSES
from app.tui import arena


def render(renderable, width: int = 150, height: int = arena.ONCE_HEIGHT) -> str:
    console = Console(width=width, record=True, force_terminal=False)
    console.print(renderable, height=height)
    return console.export_text()


class TestRunFiles:
    def test_history_skips_blank_cells(self, tmp_path):
        logs = tmp_path / "logs" / "version_0"
        logs.mkdir(parents=True)
        (logs / "metrics.csv").write_text(
            "epoch,step,train/acc,val/macro_f1\n"
            "0,10,,0.80\n0,10,0.7,\n1,20,,0.95\n1,20,0.9,\n"
        )
        assert arena.load_history(tmp_path) == [0.80, 0.95]

    def test_missing_files_are_empty(self, tmp_path):
        assert arena.load_history(tmp_path) == []
        assert arena.load_robustness(tmp_path) == []

    def test_robustness_points(self, tmp_path):
        (tmp_path / "robustness.json").write_text(json.dumps({"sweep": {"points": [
            {"severity": 0.0, "macro_f1": 1.0}, {"severity": 1.0, "macro_f1": 0.9}]}}))
        assert arena.load_robustness(tmp_path) == [(0.0, 1.0), (1.0, 0.9)]

    def test_discover_keeps_zoo_order(self, tmp_path):
        for name in ("resnet18", "palimpsest_net"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "best.ckpt").write_bytes(b"")
        (tmp_path / "squeezenet1_1").mkdir()  # no checkpoint, skipped
        assert [n for n, _ in arena.discover(tmp_path)] == ["palimpsest_net", "resnet18"]


class TestPareto:
    def test_dominated_model_is_excluded(self):
        rows = [
            {"name": "fast", "macro_f1": 0.99, "p50": 0.1},
            {"name": "accurate", "macro_f1": 1.00, "p50": 0.5},
            {"name": "worse", "macro_f1": 0.99, "p50": 0.6},
        ]
        assert arena.pareto_names(rows) == {"fast", "accurate"}

    def test_invisible_difference_does_not_decide(self):
        rows = [{"name": "a", "macro_f1": 0.998950, "p50": 0.2},
                {"name": "b", "macro_f1": 0.998952, "p50": 0.2}]
        assert arena.pareto_names(rows) == {"a", "b"}

    def test_exact_tie_keeps_both(self):
        rows = [{"name": "a", "macro_f1": 1.0, "p50": 0.2},
                {"name": "b", "macro_f1": 1.0, "p50": 0.2}]
        assert arena.pareto_names(rows) == {"a", "b"}


class _Fixed(torch.nn.Module):
    """Always predicts one class, so the arena can run without a checkpoint."""

    def __init__(self, cls: int):
        super().__init__()
        self.cls = cls

    def forward(self, x):
        logits = torch.zeros(len(x), NUM_CLASSES)
        logits[:, self.cls] = 10.0
        return logits


def _arena(tmp_path):
    entrants = []
    for name, cls in (("palimpsest_net", 0), ("resnet18", 1)):
        e = arena.Entrant(name, tmp_path / name, model=_Fixed(cls))
        entrants.append(e)
    return arena.Arena(entrants, device="cpu", split="test")


class TestArena:
    def test_renders_before_first_batch(self, tmp_path):
        text = render(_arena(tmp_path).layout())
        assert "palimpsest_net" in text and "resnet18" in text

    def test_step_scores_each_model_on_same_batch(self, tmp_path):
        a = _arena(tmp_path)
        x, y = torch.zeros(4, 3, 8, 8), np.zeros(4, dtype=int)
        a.step(x, y, 0)
        a.step(x, y, 1)
        report = a.report()
        assert report["crops"] == 8
        assert report["backbones"]["palimpsest_net"]["accuracy"] == 1.0
        assert report["backbones"]["resnet18"]["accuracy"] == 0.0

    def test_ragged_batch_is_scored_but_not_charted(self, tmp_path):
        a = _arena(tmp_path)
        a.step(torch.zeros(4, 3, 8, 8), np.zeros(4, dtype=int), 0, full=True)
        a.step(torch.zeros(1, 3, 8, 8), np.zeros(1, dtype=int), 1, full=False)
        e = a.entrants[0]
        assert len(e.core) == 5 and len(e.batch_ms) == 1

    def test_display_tie_is_marked_best_for_both(self, tmp_path):
        a = _arena(tmp_path)
        a.seen = 1
        rows = [
            {"name": n, "parameters": 1, "accuracy": acc, "macro_f1": acc, "ece": 0.0,
             "p50": 0.1, "p95": 0.1, "rate": 1.0, "f1_at_1": None}
            for n, acc in (("palimpsest_net", 0.998950), ("resnet18", 0.998952))
        ]
        assert a.bests(rows)["macro_f1"] == "0.9990"
        assert arena.pareto_names(rows) == {"palimpsest_net", "resnet18"}

    def test_cards_show_live_numbers(self, tmp_path):
        a = _arena(tmp_path)
        a.step(torch.zeros(4, 3, 8, 8), np.zeros(4, dtype=int), 0)
        text = render(a.layout())
        assert "1.0000" in text and "0.0000" in text  # perfect and hopeless model
        assert "★ palimpsest_net" in text  # dominates the always-wrong one
        assert "meters span 0.90" in text  # the zoom is stated, not hidden

    def test_no_gpu_box_degrades_gracefully(self, tmp_path):
        assert "no NVIDIA GPU" in render(_arena(tmp_path).layout())
