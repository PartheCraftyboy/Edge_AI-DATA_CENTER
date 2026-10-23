"""The gate decides what reaches the reader, so its behaviour is pinned down."""

import pytest

from app import MISS_COST, THRESHOLDS
from app.pipeline.gate import decide, held_rate


class TestThresholds:
    def test_formula_is_the_strictest_and_theory_the_loosest(self):
        # The ordering is the policy. If this inverts, the policy is incoherent.
        assert THRESHOLDS["FORMULA"] > THRESHOLDS["TABLE"]
        assert THRESHOLDS["TABLE"] > THRESHOLDS["DIAGRAM"]
        assert THRESHOLDS["DIAGRAM"] > THRESHOLDS["PLOT_SKETCH"]
        assert THRESHOLDS["PLOT_SKETCH"] > THRESHOLDS["THEORY"]

    def test_threshold_order_matches_cost_order(self):
        # The thresholds are derived from harm, so the two rankings must agree.
        by_threshold = sorted(THRESHOLDS, key=lambda k: -THRESHOLDS[k])
        by_cost = sorted(MISS_COST, key=lambda k: -MISS_COST[k])
        assert by_threshold == by_cost


class TestDecisions:
    def test_confident_theory_is_admitted(self):
        d = decide("THEORY", 0.72)
        assert d.held is False

    def test_the_same_confidence_holds_a_formula(self):
        # 0.72 passes for prose and fails for maths. That asymmetry is the
        # entire reason the thresholds are per class.
        d = decide("FORMULA", 0.72)
        assert d.held is True
        assert "0.88" in d.reason

    def test_exactly_at_threshold_is_admitted(self):
        d = decide("TABLE", THRESHOLDS["TABLE"])
        assert d.held is False

    def test_just_below_threshold_is_held(self):
        d = decide("TABLE", THRESHOLDS["TABLE"] - 0.001)
        assert d.held is True

    def test_diagram_is_held_even_when_certain(self):
        # No handler exists, so label confidence is irrelevant.
        d = decide("DIAGRAM", 0.999)
        assert d.held is True
        assert "no handler" in d.reason

    def test_unknown_class_is_a_programming_error(self):
        with pytest.raises(KeyError):
            decide("MARGINALIA", 0.9)

    def test_reason_is_present_only_when_held(self):
        assert decide("THEORY", 0.95).reason is None
        assert decide("THEORY", 0.10).reason is not None


class TestHeldRate:
    def test_empty_page(self):
        assert held_rate([]) == 0.0

    def test_mixed_page(self):
        ds = [decide("THEORY", 0.9), decide("FORMULA", 0.5)]
        assert held_rate(ds) == pytest.approx(0.5)

    def test_the_designed_demo_page(self):
        # Four admitted, one held: what the mockups show.
        ds = [
            decide("THEORY", 0.96),
            decide("FORMULA", 0.93),
            decide("TABLE", 0.90),
            decide("PLOT_SKETCH", 0.88),
            decide("DIAGRAM", 0.62),
        ]
        assert held_rate(ds) == pytest.approx(0.2)
