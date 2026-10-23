"""Constraint harvesting, including the forms people actually write by hand."""

import pytest

from app.routers.constraints import Constraint, ConstraintTable


class TestOneSidedForms:
    @pytest.mark.parametrize("text", ["0 < x", "x > 0"])
    def test_lower_bound_either_way_round(self, text):
        # `0 < x` and `x > 0` say the same thing. Which side the name fell on is
        # a writing habit, not a difference in meaning.
        t = ConstraintTable()
        assert t.extend_from_text(text, y=0) == 1
        c = t.for_symbol("x", y=10)
        assert c.lower == 0.0
        assert c.lower_strict is True
        assert c.upper is None

    @pytest.mark.parametrize("text", ["x < 10", "10 > x"])
    def test_upper_bound_either_way_round(self, text):
        t = ConstraintTable()
        t.extend_from_text(text, y=0)
        c = t.for_symbol("x", y=10)
        assert c.upper == 10.0
        assert c.upper_strict is True
        assert c.lower is None

    def test_non_strict_is_distinguished(self):
        t = ConstraintTable()
        t.extend_from_text("x >= 0", y=0)
        c = t.for_symbol("x", y=10)
        assert c.lower == 0.0
        assert c.lower_strict is False

    def test_unicode_operators(self):
        t = ConstraintTable()
        t.extend_from_text("t ≥ 0", y=0)
        assert t.for_symbol("t", y=1).lower_strict is False


class TestChainedForms:
    def test_the_requested_case(self):
        # "if it specifically says like var x range 0<x so plot the right part
        # only till let say 10"
        t = ConstraintTable()
        assert t.extend_from_text("0 < x < 10", y=0) == 1
        c = t.for_symbol("x", y=5)
        assert (c.lower, c.upper) == (0.0, 10.0)
        assert c.lower_strict and c.upper_strict

    def test_mixed_strictness(self):
        t = ConstraintTable()
        t.extend_from_text("0 <= t <= 20", y=0)
        c = t.for_symbol("t", y=5)
        assert (c.lower, c.upper) == (0.0, 20.0)
        assert not c.lower_strict and not c.upper_strict

    def test_unicode_chain(self):
        t = ConstraintTable()
        t.extend_from_text("0 ≤ t ≤ 20", y=0)
        c = t.for_symbol("t", y=5)
        assert (c.lower, c.upper) == (0.0, 20.0)

    def test_descending_chain_is_normalised(self):
        # `1 > rho > 0` is the same interval written backwards.
        t = ConstraintTable()
        t.extend_from_text("1 > rho > 0", y=0)
        c = t.for_symbol("rho", y=5)
        assert (c.lower, c.upper) == (0.0, 1.0)

    def test_chain_is_not_double_counted_as_two_halves(self):
        # The single-sided patterns would each match half of a chain. Only one
        # constraint should come out.
        t = ConstraintTable()
        assert t.extend_from_text("0 < x < 10", y=0) == 1

    def test_disagreeing_operators_are_read_conjunctively_not_as_a_range(self):
        # `0 < x > 10` is not an interval, and the chained pattern correctly
        # declines to treat it as one. The one-sided patterns then read it as
        # two claims that both bound x from below, which intersect to x > 10.
        # That is the right answer: x > 10 already implies x > 0.
        t = ConstraintTable()
        t.extend_from_text("0 < x > 10", y=0)
        c = t.for_symbol("x", y=5)
        assert c.lower == 10.0
        assert c.upper is None
        assert c.empty is False


class TestIntervalNotation:
    def test_open_interval(self):
        t = ConstraintTable()
        t.extend_from_text("x in (0, 10)", y=0)
        c = t.for_symbol("x", y=5)
        assert (c.lower, c.upper) == (0.0, 10.0)
        assert c.lower_strict and c.upper_strict

    def test_half_open_with_unicode_member(self):
        t = ConstraintTable()
        t.extend_from_text("rho ∈ [0, 1)", y=0)
        c = t.for_symbol("rho", y=5)
        assert not c.lower_strict
        assert c.upper_strict


class TestAccumulationNotShadowing:
    """Bindings redefine, constraints accumulate. Two halves written apart are
    one statement, so they intersect rather than the later one winning."""

    def test_two_halves_combine(self):
        t = ConstraintTable()
        t.extend_from_text("0 < x", y=10)
        t.extend_from_text("x < 10", y=50)
        c = t.for_symbol("x", y=100)
        assert (c.lower, c.upper) == (0.0, 10.0)

    def test_tightest_bound_wins(self):
        t = ConstraintTable()
        t.extend_from_text("x < 100", y=10)
        t.extend_from_text("x < 5", y=20)
        assert t.for_symbol("x", y=50).upper == 5.0

    def test_looser_later_bound_does_not_widen(self):
        t = ConstraintTable()
        t.extend_from_text("x < 5", y=10)
        t.extend_from_text("x < 100", y=20)
        assert t.for_symbol("x", y=50).upper == 5.0

    def test_tie_keeps_the_stricter(self):
        # Satisfying both `x > 0` and `x >= 0` excludes zero.
        t = ConstraintTable()
        t.extend_from_text("x >= 0", y=10)
        t.extend_from_text("x > 0", y=20)
        assert t.for_symbol("x", y=50).lower_strict is True


class TestPagePosition:
    def test_constraint_below_is_not_visible(self):
        # Same reading direction as bindings: nothing below the formula counts.
        t = ConstraintTable()
        t.extend_from_text("0 < x < 10", y=500)
        assert t.for_symbol("x", y=100) is None

    def test_constraint_at_the_same_row_counts(self):
        t = ConstraintTable()
        t.extend_from_text("0 < x < 10", y=100)
        assert t.for_symbol("x", y=100) is not None


class TestContradictionDetection:
    def test_impossible_range_is_flagged(self):
        t = ConstraintTable()
        t.extend_from_text("x > 5", y=10)
        t.extend_from_text("x < 2", y=20)
        assert t.for_symbol("x", y=50).empty is True

    def test_single_point_with_a_strict_bound_is_empty(self):
        t = ConstraintTable()
        t.extend_from_text("x >= 3", y=10)
        t.extend_from_text("x < 3", y=20)
        assert t.for_symbol("x", y=50).empty is True

    def test_normal_range_is_not_empty(self):
        t = ConstraintTable()
        t.extend_from_text("0 < x < 10", y=0)
        assert t.for_symbol("x", y=5).empty is False


class TestNotConfusedWithBindings:
    def test_a_binding_is_not_a_constraint(self):
        # `lambda = 25` has no comparison in it and must not register here.
        t = ConstraintTable()
        assert t.extend_from_text("lambda = 25", y=0) == 0


class TestGreekNormalisation:
    def test_unicode_greek_matches_the_symbol_table_spelling(self):
        t = ConstraintTable()
        t.extend_from_text("0 < ρ < 1", y=0)
        assert t.for_symbol("rho", y=5) is not None
        assert t.for_symbol("ρ", y=5) is not None


class TestDescribe:
    def test_round_trips_to_something_readable(self):
        c = Constraint("rho", lower=0, upper=1, lower_strict=True, upper_strict=True)
        assert c.describe() == "0 < rho < 1"

    def test_one_sided_reads_correctly(self):
        assert Constraint("x", lower=0, lower_strict=True).describe() == "0 < x"
