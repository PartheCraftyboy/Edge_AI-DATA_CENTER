"""Metrics tests (Module 5).

These are the numbers that get reported, so an error here is an error in the
submission rather than in the model. Each metric is checked against a case
whose answer can be worked out by hand.

The two that matter most, and why:

* **Macro-F1, not accuracy.** The synthetic set is balanced by construction,
  but real pages are dominated by prose. A model that never predicts FORMULA
  can still post high accuracy, and macro-F1 is what exposes it.
* **Expected calibration error.** The per-class thresholds are only defensible
  if a reported 0.9 means 0.9. ECE is the measurement that justifies the
  policy, so a confidently wrong model must score badly here.
"""

import numpy as np
import pytest

from app import CLASS_NAMES, NUM_CLASSES
from app.metrics.confusion import MetricCore, ParetoAccumulator


def one_hot(index: int, confidence: float = 1.0) -> np.ndarray:
    """A probability vector peaking at `index` with the given confidence."""
    rest = (1.0 - confidence) / (NUM_CLASSES - 1)
    row = np.full(NUM_CLASSES, rest, dtype=np.float64)
    row[index] = confidence
    return row


class TestConfusion:
    def test_a_perfect_model_is_diagonal(self):
        core = MetricCore()
        for c in range(NUM_CLASSES):
            core.update(np.stack([one_hot(c, 0.95)] * 4), np.full(4, c))
        counts = core.confusion_counts()
        assert int(counts.trace()) == int(counts.sum())

    def test_rows_are_truth_and_columns_are_prediction(self):
        """Orientation is load-bearing: row-normalised, the diagonal reads as
        recall. Transposed, it silently reads as precision instead."""
        core = MetricCore()
        core.update(np.stack([one_hot(1)] * 3), np.zeros(3, dtype=int))
        counts = core.confusion_counts()
        assert counts[0, 1] == 3, counts
        assert counts[1, 0] == 0

    def test_row_normalised_rows_sum_to_one(self):
        core = MetricCore()
        rng = np.random.default_rng(0)
        for _ in range(40):
            c = int(rng.integers(0, NUM_CLASSES))
            core.update(one_hot(int(rng.integers(0, NUM_CLASSES)))[None], np.array([c]))
        rows = core.confusion_row_normalised()
        for i, row in enumerate(rows):
            assert abs(row.sum() - 1.0) < 1e-9 or row.sum() == 0.0, i

    def test_empty_rows_do_not_divide_by_zero(self):
        core = MetricCore()
        core.update(one_hot(0)[None], np.array([0]))
        assert np.isfinite(core.confusion_row_normalised()).all()


class TestF1:
    def test_macro_f1_is_one_for_a_perfect_model(self):
        core = MetricCore()
        for c in range(NUM_CLASSES):
            core.update(np.stack([one_hot(c)] * 5), np.full(5, c))
        assert abs(core.macro_f1() - 1.0) < 1e-9

    def test_macro_f1_punishes_ignoring_a_rare_class(self):
        """The reason macro-F1 is the headline. A model that always says
        THEORY on a set that is 80% THEORY scores 0.8 accuracy and should
        score far worse than that on macro-F1."""
        core = MetricCore()
        core.update(np.stack([one_hot(0)] * 80), np.zeros(80, dtype=int))
        core.update(np.stack([one_hot(0)] * 20), np.ones(20, dtype=int))
        assert abs(core.accuracy() - 0.8) < 1e-9
        assert core.macro_f1() < 0.25, core.macro_f1()

    def test_per_class_f1_has_one_entry_per_class(self):
        core = MetricCore()
        core.update(np.stack([one_hot(2)] * 3), np.full(3, 2))
        scores = core.per_class_f1()
        assert len(scores) == NUM_CLASSES
        assert abs(scores[2] - 1.0) < 1e-9


class TestCalibration:
    def test_a_perfectly_calibrated_model_scores_near_zero(self):
        """70% confident and right 70% of the time."""
        core = MetricCore()
        for i in range(100):
            correct = i < 70
            core.update(one_hot(0, 0.7)[None], np.array([0 if correct else 1]))
        assert core.ece() < 0.05, core.ece()

    def test_confident_and_wrong_scores_badly(self):
        core = MetricCore()
        for _ in range(50):
            core.update(one_hot(0, 0.99)[None], np.array([1]))
        assert core.ece() > 0.8, core.ece()

    def test_per_class_ece_is_reported(self):
        core = MetricCore()
        rng = np.random.default_rng(1)
        for _ in range(200):
            t = int(rng.integers(0, NUM_CLASSES))
            core.update(one_hot(t, 0.8)[None], np.array([t]))
        per_class = core.ece_per_class()
        assert set(per_class) == set(CLASS_NAMES)


class TestHeldRate:
    def test_everything_below_threshold_is_held(self):
        """The user-facing consequence of the threshold policy: FORMULA needs
        0.88, so a run of 0.5-confidence formulas is entirely held."""
        core = MetricCore()
        for _ in range(10):
            core.update(one_hot(1, 0.5)[None], np.array([1]))
        assert core.held_rate()["FORMULA"] == 1.0

    def test_confident_predictions_are_written(self):
        core = MetricCore()
        for _ in range(10):
            core.update(one_hot(1, 0.97)[None], np.array([1]))
        assert core.held_rate()["FORMULA"] == 0.0

    def test_theory_is_held_less_readily_than_formula(self):
        """Different thresholds must actually produce different behaviour at
        the same confidence, or the per-class policy is decorative."""
        core = MetricCore()
        for cls in (0, 1):
            for _ in range(10):
                core.update(one_hot(cls, 0.8)[None], np.array([cls]))
        rates = core.held_rate()
        assert rates["THEORY"] == 0.0
        assert rates["FORMULA"] == 1.0


class TestRollingWindow:
    def test_window_discards_old_samples(self):
        """Switching backbones must visibly change the live matrix rather
        than being averaged away by everything that came before."""
        core = MetricCore(window=20)
        core.update(np.stack([one_hot(0)] * 20), np.zeros(20, dtype=int))
        core.update(np.stack([one_hot(1)] * 20), np.ones(20, dtype=int))
        counts = core.confusion_counts()
        assert counts[0, 0] == 0
        assert counts[1, 1] == 20

    def test_unbounded_by_default(self):
        core = MetricCore()
        core.update(np.stack([one_hot(0)] * 50), np.zeros(50, dtype=int))
        core.update(np.stack([one_hot(1)] * 50), np.ones(50, dtype=int))
        assert int(core.confusion_counts().sum()) == 100


class TestLatency:
    def test_percentiles_are_tracked(self):
        core = MetricCore()
        for ms in range(1, 101):
            core.update(one_hot(0)[None], np.array([0]), latency_ms=float(ms))
        stats = core.latency()
        assert 45 <= stats["p50"] <= 55
        assert 90 <= stats["p95"] <= 100

    def test_latency_is_optional(self):
        core = MetricCore()
        core.update(one_hot(0)[None], np.array([0]))
        assert core.latency()["p50"] == 0.0


class TestPareto:
    def test_dominated_points_are_excluded(self):
        """A model slower AND less accurate than another is never a choice."""
        pareto = ParetoAccumulator()
        pareto.add("fast_bad", accuracy=0.70, latency_ms=5.0)
        pareto.add("slow_good", accuracy=0.92, latency_ms=44.0)
        pareto.add("dominated", accuracy=0.65, latency_ms=50.0)
        names = {p["name"] for p in pareto.frontier()}
        assert names == {"fast_bad", "slow_good"}

    def test_frontier_is_sorted_by_latency(self):
        pareto = ParetoAccumulator()
        pareto.add("c", accuracy=0.95, latency_ms=44.0)
        pareto.add("a", accuracy=0.80, latency_ms=4.0)
        pareto.add("b", accuracy=0.89, latency_ms=18.0)
        assert [p["name"] for p in pareto.frontier()] == ["a", "b", "c"]

    def test_energy_is_carried_when_supplied(self):
        pareto = ParetoAccumulator()
        pareto.add("m", accuracy=0.9, latency_ms=20.0, energy_j=0.42)
        assert pareto.frontier()[0]["energy_j"] == 0.42
