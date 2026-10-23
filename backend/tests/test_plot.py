"""Range selection, sampling, and the deliberate refusal to draw surfaces."""

import math

import pytest

from app import PLOT_DEFAULT_SPAN
from app.routers.constraints import Constraint, ConstraintTable
from app.routers.plot import (
    MAX_FREE_SYMBOLS,
    choose_plot_range,
    plot_series,
    suggest_y_clip,
    refuse_reason,
)
from app.routers.solver import solve


class TestMagnitudeAwareSpan:
    """"if the values are in range of 10s and no bound go till 10 by default"."""

    def test_values_in_the_tens_give_a_span_of_ten(self):
        s = solve("y = x*2", {"lambda_": 25.0, "mu": 22.7})
        r = choose_plot_range(s, {"lambda_": 25.0, "mu": 22.7})
        assert (r.x_min, r.x_max) == (0.0, 10.0)

    def test_values_in_the_hundreds_give_a_span_of_a_hundred(self):
        # A page about milliseconds and one about years should not share an axis.
        r = choose_plot_range(solve("y = x*2", {}), {"budget": 450.0})
        assert (r.x_min, r.x_max) == (0.0, 100.0)

    def test_fractional_values_scale_down(self):
        r = choose_plot_range(solve("y = x*2", {}), {"tol": 0.004})
        assert r.x_max == pytest.approx(0.001)

    def test_no_values_at_all_falls_back_to_the_default(self):
        r = choose_plot_range(solve("y = x*2", {}), {})
        assert (r.x_min, r.x_max) == (0.0, PLOT_DEFAULT_SPAN)

    def test_range_explains_itself(self):
        r = choose_plot_range(solve("y = x*2", {}), {"a": 30.0})
        assert "no stated domain" in r.source


class TestThePageWins:
    def test_fully_bounded_constraint_is_honoured(self):
        s = solve("L = rho/(1 - rho)", {})
        c = Constraint("rho", lower=0, upper=1, lower_strict=True, upper_strict=True)
        r = choose_plot_range(s, {}, c)
        # Strict on both ends, so the sampler steps just inside.
        assert 0.0 < r.x_min < 0.01
        assert 0.99 < r.x_max < 1.0
        assert "page states 0 < rho < 1" in r.source

    def test_non_strict_bounds_are_touched_exactly(self):
        s = solve("y = x*2", {})
        c = Constraint("x", lower=0, upper=20, lower_strict=False, upper_strict=False)
        r = choose_plot_range(s, {}, c)
        assert (r.x_min, r.x_max) == (0.0, 20.0)

    def test_lower_bound_only_extends_by_the_magnitude_span(self):
        # The requested behaviour: `0 < x` with values in the tens plots 0 to 10.
        s = solve("y = x*2", {})
        c = Constraint("x", lower=0, lower_strict=True)
        r = choose_plot_range(s, {"a": 25.0}, c)
        assert r.x_min == pytest.approx(0.0, abs=0.02)
        assert r.x_max == pytest.approx(10.0, abs=0.02)
        assert "extended by 10" in r.source

    def test_upper_bound_only_extends_backwards(self):
        s = solve("y = x*2", {})
        c = Constraint("x", upper=10, upper_strict=False)
        r = choose_plot_range(s, {"a": 25.0}, c)
        assert (r.x_min, r.x_max) == (0.0, 10.0)

    def test_contradictory_page_draws_nothing(self):
        t = ConstraintTable()
        t.extend_from_text("x > 5", y=0)
        t.extend_from_text("x < 2", y=1)
        s = solve("y = x*2", {})
        assert choose_plot_range(s, {}, t.for_symbol("x", y=5)) is None


class TestSliderLimits:
    def test_limits_bracket_the_chosen_range(self):
        r = choose_plot_range(solve("y = x*2", {}), {"a": 25.0})
        assert r.limit_min <= r.x_min
        assert r.limit_max >= r.x_max

    def test_a_stated_domain_still_allows_travel(self):
        # The page's statement is a starting point, not a cage.
        c = Constraint("x", lower=0, upper=10)
        r = choose_plot_range(solve("y = x*2", {}), {}, c)
        assert r.limit_max > r.x_max


class TestTwoDimensionsOnly:
    def test_one_free_symbol_is_the_limit(self):
        assert MAX_FREE_SYMBOLS == 1

    def test_two_free_symbols_draw_nothing(self):
        s = solve("z = x*y", {})
        assert choose_plot_range(s, {}) is None

    def test_and_the_refusal_says_why(self):
        s = solve("z = x*y", {})
        reason = refuse_reason(s)
        assert "2 free variables" in reason
        assert "surfaces are deliberately not drawn" in reason

    def test_fully_bound_is_a_value_not_a_curve(self):
        s = solve("rho = lambda/mu", {"lambda_": 25.0, "mu": 22.7})
        assert choose_plot_range(s, {}) is None
        assert "a value rather than a curve" in refuse_reason(s)

    def test_a_plottable_formula_has_no_refusal(self):
        s = solve("L = rho/(1 - rho)", {})
        assert refuse_reason(s) is None


class TestSampling:
    def test_straight_line_samples_cleanly(self):
        s = solve("y = 2*x", {})
        r = choose_plot_range(s, {})
        xs, ys = plot_series(s, r, samples=11)
        assert len(xs) == len(ys) == 11
        assert ys[0] == pytest.approx(0.0)
        assert ys[-1] == pytest.approx(2 * PLOT_DEFAULT_SPAN)
        assert all(y is not None for y in ys)

    def test_pole_is_broken_rather_than_drawn_through(self):
        # `rho/(1 - rho)` blows up at rho = 1. Joining the samples either side
        # would draw a near-vertical line that looks like data.
        s = solve("L = rho/(1 - rho)", {})
        c = Constraint("rho", lower=0, upper=2)
        r = choose_plot_range(s, {}, c)
        xs, ys = plot_series(s, r, samples=201)
        assert any(y is None for y in ys), "expected a break at the pole"

    def test_values_outside_the_real_domain_become_breaks(self):
        # sqrt of a negative is not a real point on the page.
        s = solve("y = sqrt(x)", {})
        c = Constraint("x", lower=-5, upper=5)
        r = choose_plot_range(s, {}, c)
        xs, ys = plot_series(s, r, samples=101)
        assert any(y is None for y in ys)
        assert any(y is not None for y in ys)

    def test_unplottable_returns_empty(self):
        s = solve("z = x*y", {})
        c = Constraint("x", lower=0, upper=1)
        assert plot_series(s, choose_plot_range(s, {}, c) or _dummy(), samples=5) == ([], [])

    def test_every_finite_sample_is_a_real_float(self):
        s = solve("y = sin(x)", {})
        r = choose_plot_range(s, {})
        _, ys = plot_series(s, r, samples=51)
        for y in ys:
            if y is not None:
                assert isinstance(y, float) and math.isfinite(y)


def _dummy():
    from app.routers.plot import PlotRange

    return PlotRange(0.0, 1.0, "test", 0.0, 1.0)


class TestYClip:
    """Clipping the axis is the proper fix for a curve that runs to an
    asymptote: keep the readable part readable, let the steep part run off."""

    def test_asymptote_gets_clipped(self):
        s = solve("L = rho/(1 - rho)", {})
        c = Constraint("rho", lower=0, upper=0.99)
        r = choose_plot_range(s, {}, c)
        _, ys = plot_series(s, r, samples=201)
        clip = suggest_y_clip(ys)
        assert clip is not None
        lo, hi = clip
        # The true maximum is in the dozens; the band should sit well below it.
        finite = [y for y in ys if y is not None]
        assert hi < max(finite)

    def test_straight_line_needs_no_clipping(self):
        s = solve("y = 2*x", {})
        r = choose_plot_range(s, {})
        _, ys = plot_series(s, r, samples=101)
        assert suggest_y_clip(ys) is None

    def test_too_few_points_returns_none(self):
        assert suggest_y_clip([1.0, 2.0, 3.0]) is None

    def test_all_breaks_returns_none(self):
        assert suggest_y_clip([None] * 20) is None

    def test_flat_line_returns_none(self):
        assert suggest_y_clip([3.0] * 40) is None
