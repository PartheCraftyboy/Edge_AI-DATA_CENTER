"""The resolution rule is the whole feature, so it is tested, not eyeballed."""

from app.routers.symbols import SymbolTable, canonical


class TestCanonical:
    def test_greek_unicode_becomes_a_name(self):
        assert canonical("λ") == "lambda_"
        assert canonical("ρ") == "rho"
        assert canonical("μ") == "mu"

    def test_python_keyword_gets_suffixed(self):
        # `lambda` would break every downstream parser.
        assert canonical("lambda") == "lambda_"

    def test_spellings_converge(self):
        # The same symbol however the writer produced it.
        assert canonical("λ") == canonical("lambda") == canonical("lambda_")

    def test_subscripts_survive(self):
        assert canonical("T_service") == "T_service"


class TestHarvesting:
    def test_finds_bare_number_bindings(self):
        t = SymbolTable()
        found = t.extend_from_text("camera runs at lambda = 25 fps", y=10)
        assert found == 1
        assert t.flat() == {"lambda_": 25.0}

    def test_finds_several_on_one_line(self):
        t = SymbolTable()
        found = t.extend_from_text("lambda = 25, mu = 22.7", y=10)
        assert found == 2
        assert t.flat() == {"lambda_": 25.0, "mu": 22.7}

    def test_ignores_expressions(self):
        # `rho = lambda/mu` is a formula to solve, not a value to record.
        # Recording it would make the solver resolve against itself.
        t = SymbolTable()
        assert t.extend_from_text("rho = lambda/mu", y=10) == 0
        assert t.flat() == {}

    def test_handles_negatives_and_exponents(self):
        t = SymbolTable()
        t.extend_from_text("drift = -0.5 and tol = 1e-3", y=0)
        assert t.flat() == {"drift": -0.5, "tol": 0.001}

    def test_greek_in_source_text(self):
        t = SymbolTable()
        t.extend_from_text("λ = 25", y=5)
        assert t.flat() == {"lambda_": 25.0}


class TestNearestAboveResolution:
    def test_binding_above_is_visible(self):
        t = SymbolTable()
        t.add("lambda", 25, y=10)
        assert t.resolve(y=100) == {"lambda_": 25.0}

    def test_binding_below_is_not_visible(self):
        # The rule exists so an answer never depends on something written later.
        t = SymbolTable()
        t.add("lambda", 25, y=200)
        assert t.resolve(y=100) == {}

    def test_nearest_above_wins_on_redefinition(self):
        t = SymbolTable()
        t.add("mu", 22.7, y=10)
        t.add("mu", 87.0, y=150)
        # A formula between the two sees the first value.
        assert t.resolve(y=100) == {"mu": 22.7}
        # A formula below both sees the second.
        assert t.resolve(y=300) == {"mu": 87.0}

    def test_same_row_counts_as_visible(self):
        # A formula region often defines its own variables on the same line.
        t = SymbolTable()
        t.add("lambda", 25, y=100)
        assert t.resolve(y=100) == {"lambda_": 25.0}

    def test_independent_symbols_resolve_independently(self):
        t = SymbolTable()
        t.add("lambda", 25, y=10)
        t.add("mu", 22.7, y=20)
        t.add("lambda", 30, y=500)
        assert t.resolve(y=100) == {"lambda_": 25.0, "mu": 22.7}


class TestRedefinitionReporting:
    def test_reports_conflicting_values(self):
        t = SymbolTable()
        t.add("mu", 22.7, y=10)
        t.add("mu", 87.0, y=150)
        assert t.redefined() == {"mu": [22.7, 87.0]}

    def test_same_value_twice_is_not_a_conflict(self):
        t = SymbolTable()
        t.add("mu", 22.7, y=10)
        t.add("mu", 22.7, y=150)
        assert t.redefined() == {}
