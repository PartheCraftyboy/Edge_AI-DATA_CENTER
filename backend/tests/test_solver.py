"""Solver tests, including the security controls.

The model is an untrusted input source, so the rejection cases are tested with
the same seriousness as the arithmetic.
"""

import pytest
import sympy as sp

from app.routers.solver import (
    SolverError,
    parse,
    solve,
    split_relation,
)


class TestSplitRelation:
    def test_plain_assignment(self):
        assert split_relation("rho = lambda/mu") == ("rho", "lambda/mu", [])

    def test_keeps_the_writers_own_working(self):
        # `rho = lambda/mu = 25/22.7 = 1.10` — we evaluate the symbolic part and
        # keep the writer's arithmetic to compare against.
        lhs, rhs, extra = split_relation("rho = lambda/mu = 25/22.7 = 1.10")
        assert lhs == "rho"
        assert rhs == "lambda/mu"
        assert extra == ["25/22.7", "1.10"]

    def test_expression_with_no_assignment(self):
        lhs, rhs, extra = split_relation("lambda/mu")
        assert lhs is None
        assert rhs == "lambda/mu"

    def test_equation_rather_than_definition_has_no_lhs(self):
        # A compound left side is not a variable being defined.
        lhs, _, _ = split_relation("a + b = c")
        assert lhs is None


class TestParsing:
    def test_implicit_multiplication(self):
        # Handwriting almost never includes the multiplication sign.
        expr = parse("2x")
        assert str(expr) == "2*x"

    def test_caret_is_exponentiation(self):
        expr = parse("x^2")
        assert str(expr) == "x**2"

    def test_greek_normalised_to_match_the_symbol_table(self):
        expr = parse("λ/μ")
        assert {str(s) for s in expr.free_symbols} == {"lambda_", "mu"}

    def test_allowed_function(self):
        expr = parse("sqrt(x)")
        assert "sqrt" in str(expr)


class TestSecurity:
    def test_rejects_dunder(self):
        with pytest.raises(SolverError, match="dunder"):
            parse("__import__('os')")

    def test_rejects_attribute_access(self):
        with pytest.raises(SolverError, match="attribute"):
            parse("os.system")

    def test_rejects_illegal_characters(self):
        with pytest.raises(SolverError, match="illegal"):
            parse("x; rm -rf /")

    def test_rejects_absurdly_long_input(self):
        with pytest.raises(SolverError, match="too long"):
            parse("x+" * 400)

    def test_rejects_empty(self):
        with pytest.raises(SolverError, match="empty"):
            parse("   ")

    def test_unlisted_name_becomes_a_symbol_not_a_call(self):
        # The important guarantee: an unknown name is inert. Before the
        # global_dict was restricted this returned Python's real `eval`.
        expr = parse("eval")
        assert expr.is_symbol

    def test_builtins_are_unreachable(self):
        for name in ("open", "exec", "compile", "globals", "getattr"):
            assert parse(name).is_symbol, f"{name} resolved to something callable"


class TestGreekNamesAreNotSympyFunctions:
    """SymPy's namespace defines beta, gamma and zeta as functions, and this
    project normalises handwritten Greek to exactly those names. A page saying
    `gamma = 0.5` must bind a variable, not shadow a special function."""

    @pytest.mark.parametrize("name", ["beta", "gamma", "zeta", "N", "S", "O", "Q", "I"])
    def test_name_parses_as_a_free_symbol(self, name):
        expr = parse(name)
        assert expr.is_symbol, f"{name} parsed as {type(expr).__name__}"

    def test_greek_variable_resolves_against_the_page(self):
        s = solve("y = gamma*2", {"gamma": 0.5})
        assert s.free == []
        assert s.answer == pytest.approx(1.0)


class TestSolving:
    def test_the_worked_example(self):
        # The case from the spec and the design mockups.
        s = solve("rho = lambda/mu", {"lambda_": 25.0, "mu": 22.7})
        assert s.lhs == "rho"
        assert s.free == []
        assert s.answer == pytest.approx(1.1013, abs=1e-4)
        assert s.bound == {"lambda_": 25.0, "mu": 22.7}

    def test_lhs_binding_is_not_used_as_an_input(self):
        # If the page also recorded `rho = 1.10`, resolving against it would be
        # circular: we would report the writer's own answer and prove nothing.
        s = solve("rho = lambda/mu", {"lambda_": 25.0, "mu": 22.7, "rho": 999.0})
        assert s.answer == pytest.approx(1.1013, abs=1e-4)
        assert "rho" not in s.bound

    def test_one_unbound_symbol_is_plottable(self):
        s = solve("L = rho/(1 - rho)", {})
        assert s.free == ["rho"]
        assert s.answer is None
        assert s.plottable is True

    def test_two_unbound_symbols_are_not_plottable(self):
        s = solve("z = x*y", {})
        assert sorted(s.free) == ["x", "y"]
        assert s.plottable is False

    def test_fully_bound_is_not_plottable(self):
        s = solve("rho = lambda/mu", {"lambda_": 25.0, "mu": 22.7})
        assert s.plottable is False

    def test_partial_binding_leaves_the_rest_free(self):
        s = solve("q = a*b/c", {"a": 2.0, "b": 3.0})
        assert s.free == ["c"]
        assert s.bound == {"a": 2.0, "b": 3.0}

    def test_unparseable_degrades_rather_than_raising(self):
        # A solver failure must never fail a compile: the formula still typesets
        # from its LaTeX field, it just carries no answer.
        s = solve("rho = $$$", {})
        assert s.error is not None
        assert s.answer is None
        assert s.expr is None

    def test_division_by_zero_does_not_crash(self):
        s = solve("y = 1/x", {"x": 0.0})
        assert s.answer is None or not _finite(s.answer)

    def test_writers_working_is_retained(self):
        s = solve("rho = lambda/mu = 1.10", {"lambda_": 25.0, "mu": 22.7})
        assert s.writer_working == ["1.10"]
        # Our answer and the writer's disagree slightly, which is exactly the
        # kind of thing worth being able to show.
        assert s.answer == pytest.approx(1.1013, abs=1e-4)


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


class TestMultiLetterNamesSurviveParsing:
    """`implicit_multiplication_application` bundles `split_symbols`, which turns
    a multi-letter name into a product of single letters. SymPy exempts Greek
    letter names, so `mu` and `rho` hid the bug. Engineering notes are full of
    names like `T_service` and `drift`, so splitting is disabled entirely."""

    @pytest.mark.parametrize("name", ["drift", "queue", "budget", "eval", "T_service"])
    def test_name_stays_one_symbol(self, name):
        expr = parse(name)
        assert expr.is_symbol, f"{name} parsed as {expr}"
        assert str(expr) == name

    def test_number_adjacency_still_multiplies(self):
        # The thing splitting was wanted for still works.
        assert str(parse("2x")) == "2*x"

    def test_spaced_names_still_multiply(self):
        expr = parse("a b")
        assert expr == sp.Symbol("a") * sp.Symbol("b")

    def test_named_variable_resolves_end_to_end(self):
        s = solve("total = drift*2", {"drift": 1.5})
        assert s.free == []
        assert s.answer == pytest.approx(3.0)
