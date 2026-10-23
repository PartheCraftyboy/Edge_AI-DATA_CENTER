"""LaTeX to the solver's syntax.

PaddleOCR-VL reads mathematics better than the general model but emits only
LaTeX, so without this the solver gets nothing: no answers, no symbol table,
no plotted curves. What these check, beyond the conversions themselves, is
the half that matters more — that it refuses rather than guesses, because a
formula that converts to the wrong expression is a confident wrong answer in
someone's notes and nothing downstream marks it as suspect.
"""

import pytest

from app.routers.solver import parse_or_none, solve
from app.vlm.latex import latex_to_expr


class TestConverts:
    @pytest.mark.parametrize("tex, expected", [
        (r"y = B_{0} + B_{1} x", "y = B_0 + B_1 x"),
        (r"R = \frac{V}{I}", "R = ((V)/(I))"),
        (r"E = m c^{2}", "E = m c^(2)"),
        (r"\sqrt{x^{2}+y^{2}}", "sqrt(x^(2)+y^(2))"),
        (r"\lambda = \frac{1}{\mu}", "lambda = ((1)/(mu))"),
        (r"F = m \times a", "F = m * a"),
        (r"P = V \cdot I", "P = V * I"),
        (r"\alpha_{1} + \beta_{2}", "alpha_1 + beta_2"),
    ])
    def test_the_subset_handwritten_notes_use(self, tex, expected):
        assert latex_to_expr(tex) == expected

    def test_nested_fractions_recurse(self):
        assert latex_to_expr(r"\frac{\frac{a}{b}}{c}") == "((((a)/(b)))/(c))"

    def test_exponent_is_not_wrapped_repeatedly(self):
        """A repeated-pass expansion re-reads its own output: `x^{2}` became
        `x^(()()()...2)` and stopped parsing. One pass, recursing into the
        arguments, is what handles nesting."""
        out = latex_to_expr(r"x^{2}")
        assert out == "x^(2)" and "()" not in out

    def test_display_delimiters_are_stripped(self):
        assert latex_to_expr(r"\[ y = 2x \]") == "y = 2x"
        assert latex_to_expr(r"$$ y = 2x $$") == "y = 2x"

    def test_only_the_first_line_of_a_derivation_is_taken(self):
        """Several stacked statements are not one equation, and joining them
        would parse as a different one."""
        assert latex_to_expr(r"y = 2x \\ z = 3") == "y = 2x"

    def test_spacing_commands_disappear(self):
        assert latex_to_expr(r"y \, = \; 2 \quad x") == "y = 2 x"

    def test_left_and_right_are_dropped_but_the_brackets_kept(self):
        assert latex_to_expr(r"y = \left( a + b \right)") == "y = ( a + b )"


class TestAccents:
    def test_a_hat_makes_a_distinct_name_not_the_bare_one(self):
        """On a regression page `\\hat{y}` is the prediction and `y` is the
        observation. Collapsing them would bind one to the other and produce
        a confident wrong answer."""
        assert latex_to_expr(r"\hat{y} = m x + c") == "yhat = m x + c"

    def test_subscripts_survive_an_accent(self):
        assert latex_to_expr(r"y_{i} = \hat{y}_{i} + \epsilon_{i}") == \
            "y_i = yhat_i + epsilon_i"

    def test_bar_and_vec_are_distinguished_too(self):
        assert latex_to_expr(r"\bar{x} + \vec{v}") == "xbar + vvec"


class TestRefuses:
    @pytest.mark.parametrize("tex", [
        r"\hat{y} = m \times + c",   # the real misread: `x` read as `\times`
        r"y = 2 + ",
        r"y = * 3",
        r"y = ",
        r"\unknowncmd{x}",
        "",
        "   ",
    ])
    def test_returns_none_rather_than_a_guess(self, tex):
        assert latex_to_expr(tex) is None

    def test_the_misread_would_otherwise_evaluate_to_something(self):
        """`m * + c` is not a typo SymPy rejects — it reads it as `m*(+c)`
        and returns `m*c`. That is why the guard exists rather than relying
        on the parser to complain."""
        assert parse_or_none("m * + c") is not None
        assert latex_to_expr(r"m \times + c") is None

    def test_an_unknown_command_is_not_silently_dropped(self):
        """Dropping `\\unknowncmd` would leave `(x)`, which parses."""
        assert latex_to_expr(r"\unknowncmd{x}") is None

    def test_more_than_one_equals_is_refused(self):
        assert latex_to_expr(r"a = b = c") is None


class TestFeedsTheSolver:
    @pytest.mark.parametrize("tex", [
        r"R = \frac{V}{I}", r"E = m c^{2}", r"\lambda = \frac{1}{\mu}",
        r"y = B_{0} + B_{1} x", r"\sqrt{x^{2}+y^{2}}",
    ])
    def test_everything_it_returns_parses(self, tex):
        """The contract: a non-None result is something the solver reads."""
        assert parse_or_none(latex_to_expr(tex)) is not None

    def test_a_converted_formula_actually_solves(self):
        expr = latex_to_expr(r"R = \frac{V}{I}")
        solved = solve(expr, {"V": 10.0, "I": 2.0})
        assert solved.answer == pytest.approx(5.0)

    def test_an_exponent_survives_into_the_answer(self):
        expr = latex_to_expr(r"E = m c^{2}")
        solved = solve(expr, {"m": 2.0, "c": 3.0})
        assert solved.answer == pytest.approx(18.0)


class TestExplicitMultiplication:
    """`mx` typeset is ambiguous in a way it never was in the notebook: a
    reader cannot tell a product of two variables from one variable whose
    name is two letters long."""

    @pytest.mark.parametrize("tex, expected", [
        (r"m x + c", r"m \cdot x + c"),
        (r"y = m x + c", r"y = m \cdot x + c"),
        (r"2x", r"2 \cdot x"),
        (r"B_{0} + B_{1} x", r"B_{0} + B_{1} \cdot x"),
        (r"E = m c^{2}", r"E = m \cdot c^{2}"),
        (r"\frac{a}{b} x", r"\frac{a}{b} \cdot x"),
        (r"2 \sin x", r"2 \cdot \sin x"),
    ])
    def test_adjacent_operands_get_a_dot(self, tex, expected):
        from app.vlm.latex import explicit_multiplication

        assert explicit_multiplication(tex) == expected

    @pytest.mark.parametrize("tex", [
        r"\sin x",                               # a function, not a product
        r"a \cdot b",                            # already has its operator
        r"a \times b",
        r"x^{2} + y^{2}",                        # a superscript is not a factor
        r"y_{i} = \hat{y}_{i} + \epsilon_{i}",   # nor is a subscript
        r"\rho = \frac{\lambda}{\mu}",
        r"",
    ])
    def test_left_alone(self, tex):
        from app.vlm.latex import explicit_multiplication

        assert explicit_multiplication(tex) == tex

    def test_running_it_twice_changes_nothing(self):
        """It runs on every read, and a formula corrected and recompiled
        goes through it again. Dots must not accumulate."""
        from app.vlm.latex import explicit_multiplication

        once = explicit_multiplication(r"y = m x + c")
        assert explicit_multiplication(once) == once

    def test_a_subscripted_symbol_is_one_factor_not_two(self):
        """`B_1 x` is B-sub-one times x, not B times 1 times x."""
        from app.vlm.latex import explicit_multiplication

        assert explicit_multiplication(r"B_{1} x").count(r"\cdot") == 1

    def test_the_dotted_tex_still_renders_the_same_expression(self):
        """The dot is for the page. `latex_to_expr` must read it back."""
        from app.vlm.latex import explicit_multiplication, latex_to_expr

        assert latex_to_expr(explicit_multiplication(r"y = m x + c")) == "y = m * x + c"


class TestTheWriterLeftTheAnswerBlank:
    """`1 + 2 =` is a question, not a misread.

    The empty-side guard was written against `m \\times + c`, where a misread
    produced an expression that evaluated to something plausible. But it also
    caught the most ordinary thing anyone writes in a notebook: a sum with the
    answer left blank, which is precisely what the solver is for.

    The line between the two is whether the left side evaluates on its own.
    `1 + 2 =` has an answer waiting to be filled in; `y =` defines nothing and
    stays refused, as does every misread the guard was built for.
    """

    @pytest.mark.parametrize("tex, answer", [
        ("1 + 2 =", 3.0),
        ("1+2=", 3.0),
        (r"2 \times 3 =", 6.0),
        (r"\frac{10}{4} =", 2.5),
        ("7 - 2 =", 5.0),
        ("2^{3} =", 8.0),
    ])
    def test_arithmetic_with_a_blank_answer_is_solved(self, tex, answer):
        expr = latex_to_expr(tex)
        assert expr is not None, f"{tex!r} was refused before reaching the solver"
        assert solve(expr, {}).answer == pytest.approx(answer)

    @pytest.mark.parametrize("tex", [
        r"y = ",                     # defines nothing: no answer to fill in
        r"= 5",                      # empty LEFT side is still a misread
        r"\hat{y} = m \times + c",   # the misread the guard exists for
        r"y = 2 + ",
        r"y = * 3",
    ])
    def test_the_guard_still_refuses_everything_it_was_built_for(self, tex):
        assert latex_to_expr(tex) is None

    def test_a_blank_answer_with_page_symbols_still_resolves(self):
        """The same shape, but the operands come from the symbol table."""
        expr = latex_to_expr(r"V \times I =")
        assert expr is not None
        assert solve(expr, {"V": 10.0, "I": 2.0}).answer == pytest.approx(20.0)
