"""
The expression solver.

Turns a recognised formula into a resolved one: find the variable values
elsewhere on the page, do the arithmetic the writer skipped, and report which
symbol is left free so the plotter can draw it.

Security note, because this is the one genuinely risky surface in the feature.
SymPy's `sympify` calls `eval` internally and must never see a raw model string.
Two controls, applied in order:

1.  `_validate` rejects anything outside a narrow character and token set before
    parsing. No attribute access, no dunders, no calls to names we did not list.
2.  `parse_expr` receives an explicit `local_dict` of permitted symbols and
    functions, so an unlisted name resolves to a plain Symbol rather than to
    anything executable.

The model is an untrusted input source. It is treated like one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from .symbols import canonical

# Handwritten maths omits the multiplication sign (`2x`) and uses `^` for
# exponentiation, so both are read.
#
# Note what is deliberately NOT here: `implicit_multiplication_application`, the
# convenient combined transformation. It bundles `split_symbols`, which breaks a
# multi-letter name into a product of single letters, so a variable named
# `drift` silently becomes `d*r*i*f*t`. SymPy exempts Greek letter names from
# splitting, which is why `mu` and `rho` survived it and the bug nearly went
# unnoticed, but engineering notes are full of names like `T_service` and
# `drift`. Named variables matter far more here than juxtaposed single letters,
# so splitting is off: `2x` still parses as `2*x`, and `xy` stays one symbol.
TRANSFORMS = standard_transformations + (
    implicit_multiplication,
    implicit_application,
    convert_xor,
)

# Functions a formula may legitimately call. Anything else becomes a Symbol.
ALLOWED_FUNCS: dict[str, object] = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "abs": sp.Abs,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "Min": sp.Min, "Max": sp.Max,
    "factorial": sp.factorial,
    "pi": sp.pi, "E": sp.E,
}

# The namespace `parse_expr` evaluates its generated code in.
#
# Two problems with letting it use the default, both found by tests:
#
# 1.  The default is built by `from sympy import *`, which leaves Python's
#     builtins reachable, so `eval` parsed to the real builtin rather than to a
#     symbol. Setting `__builtins__` to an empty mapping closes that.
# 2.  More subtly, the full SymPy namespace contains `beta`, `gamma`, `zeta`,
#     `E`, `I`, `N`, `O`, `Q` and `S` as *functions*. This project normalises
#     handwritten Greek letters to exactly those names, so a page reading
#     `gamma = 0.5` would silently bind nothing and `gamma` would parse as the
#     gamma function. For a maths-notes compiler that is a correctness bug, not
#     a theoretical one.
#
# So the namespace is the minimum the generated code structurally needs. Every
# function a formula may legitimately call comes through `local_dict` instead,
# which keeps Greek variable names as plain symbols.
_GLOBALS: dict[str, object] = {
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    "Add": sp.Add,
    "Mul": sp.Mul,
    "Pow": sp.Pow,
    "__builtins__": {},
}

# Only these characters may appear in an expression. Deliberately tight.
_LEGAL = re.compile(r"^[0-9A-Za-z_Ͱ-Ͽ\s+\-*/^().,=<>!]*$")
# An identifier followed by a dot would be attribute access. A decimal point
# sits between digits, so this stays permissive about numbers and strict about
# names.
_ATTR = re.compile(r"[A-Za-z_]\s*\.")


class SolverError(ValueError):
    """Raised when an expression cannot be parsed safely or at all.

    Callers degrade: the formula still typesets from its LaTeX field and simply
    carries no answer. A solver failure must never fail a compile.
    """


@dataclass
class Solved:
    """The result of resolving one formula against the page."""

    lhs: str | None
    """Left-hand symbol, when the formula was written as an assignment."""
    expr: sp.Expr | None
    expr_str: str = ""
    answer: float | None = None
    bound: dict[str, float] = field(default_factory=dict)
    free: list[str] = field(default_factory=list)
    writer_working: list[str] = field(default_factory=list)
    """Extra `= ...` parts the writer wrote out themselves, e.g. the `25/22.7`
    and `1.10` in `rho = lambda/mu = 25/22.7 = 1.10`. Kept rather than
    discarded: comparing our answer against the writer's own is a cheap and
    genuinely useful check."""
    error: str | None = None

    @property
    def plottable(self) -> bool:
        """Exactly one free symbol means the expression determines a curve."""
        return self.expr is not None and len(self.free) == 1


def _validate(raw: str) -> str:
    """Reject anything that should not reach the parser."""
    s = raw.strip()
    if not s:
        raise SolverError("empty expression")
    if len(s) > 500:
        raise SolverError("expression too long to be a plausible formula")
    if "__" in s:
        raise SolverError("rejected: dunder in expression")
    if _ATTR.search(s):
        raise SolverError("rejected: attribute access in expression")
    if not _LEGAL.match(s):
        bad = sorted({ch for ch in s if not _LEGAL.match(ch)})
        raise SolverError(f"rejected: illegal characters {bad}")
    return s


def _normalise(s: str) -> str:
    """Rewrite symbol names to their canonical ascii form.

    Done on the string before parsing so that `λ` and `lambda` and `lambda_` all
    arrive at the parser as the same symbol, matching the symbol table.
    """
    # Longest names first, so `lambda` is not partially eaten by a shorter match.
    def repl(match: re.Match[str]) -> str:
        return canonical(match.group(0))

    return re.sub(r"[A-Za-zͰ-Ͽ][A-Za-z0-9_Ͱ-Ͽ]*", repl, s)


def split_relation(raw: str) -> tuple[str | None, str, list[str]]:
    """Split `lhs = rhs = extra = extra` into its parts.

    Returns `(lhs, rhs, extra)`. The first part is the symbol being defined, the
    second is the expression we evaluate, and anything further is the writer's
    own arithmetic, which we keep but do not trust.
    """
    parts = [p.strip() for p in raw.split("=")]
    parts = [p for p in parts if p]
    if len(parts) == 1:
        return None, parts[0], []
    # A single bare name on the left is an assignment. Anything else means the
    # writer wrote an equation rather than a definition, so there is no lhs.
    lhs = parts[0] if re.fullmatch(r"[A-Za-z_Ͱ-Ͽ][\wͰ-Ͽ]*", parts[0]) else None
    if lhs is None:
        return None, parts[0], parts[1:]
    return canonical(lhs), parts[1], parts[2:]


def parse(raw: str, known: object = ()) -> sp.Expr:
    """Parse one expression safely. Raises `SolverError`.

    `known` is the names the page has already bound. They are pre-declared as
    symbols so that a page-defined name always wins over anything else the
    parser might make of it.
    """
    s = _normalise(_validate(raw))
    local: dict[str, object] = dict(ALLOWED_FUNCS)
    for name in known:
        local[str(name)] = sp.Symbol(str(name))
    try:
        expr = parse_expr(
            s,
            local_dict=local,
            global_dict=_GLOBALS,
            transformations=TRANSFORMS,
        )
    except SolverError:
        raise
    except Exception as exc:  # sympy raises a wide variety here
        raise SolverError(f"could not parse {raw!r}: {type(exc).__name__}") from exc
    if not isinstance(expr, sp.Basic):
        raise SolverError(f"parsed to a non-expression: {type(expr).__name__}")
    return expr


def parse_or_none(raw: str) -> sp.Expr | None:
    """The right-hand side of `raw`, or `None` if this module cannot read it.

    For callers deciding whether a string is a formula at all, rather than
    callers that have one and want it solved. `vlm.latex` uses it as the
    final check on a converted formula: if the conversion did not produce
    something this module can read, the expression is better left unset and
    the region held than guessed at.

    Relations are split first, exactly as `solve` does, because `parse`
    itself rejects `=` and a formula is usually an equation. Every part has
    to parse: a right-hand side that reads while the left-hand side does not
    is a half-understood formula, which is the kind most likely to produce a
    confident wrong answer.
    """
    try:
        lhs, rhs, extra = split_relation(raw)
        for part in extra:
            parse(part)
        if lhs is None and "=" in raw:
            # An equation whose left side is an expression, not a name:
            # check it too, since `split_relation` hands it back as `rhs`.
            for part in [p for p in (p.strip() for p in raw.split("=")) if p][1:]:
                parse(part)
        return parse(rhs)
    except Exception:
        return None


def solve(raw: str, bindings: dict[str, float]) -> Solved:
    """Resolve one formula against the values visible to it.

    `bindings` comes from `SymbolTable.resolve(y)`, so it already encodes the
    nearest-definition-above rule. The solver does not know about page geometry
    and should not.
    """
    try:
        lhs, rhs, extra = split_relation(raw)
        expr = parse(rhs, known=bindings.keys())
    except SolverError as exc:
        return Solved(lhs=None, expr=None, expr_str=raw, error=str(exc))

    # The lhs is being defined by this formula, so a binding for it is not an
    # input. Without this, `rho = lambda/mu` on a page that also records
    # `rho = 1.10` would resolve to the writer's own answer and prove nothing.
    usable = {k: v for k, v in bindings.items() if k != lhs}

    symbol_names = {str(s) for s in expr.free_symbols}
    bound = {k: v for k, v in usable.items() if k in symbol_names}
    free = sorted(symbol_names - set(bound))

    substituted = expr.subs({sp.Symbol(k): v for k, v in bound.items()})

    answer: float | None = None
    if not free:
        try:
            value = sp.N(substituted)
            if value.is_real and value.is_finite:
                answer = float(value)
        except (TypeError, ValueError, AttributeError):
            answer = None

    return Solved(
        lhs=lhs,
        expr=substituted,
        expr_str=rhs,
        answer=answer,
        bound=bound,
        free=free,
        writer_working=extra,
    )


# Range selection and sampling live in `plot.py`, with the plotter, because
# choosing an axis is a presentation decision rather than an algebraic one. The
# solver's job ends at "here is the expression and here is what is still free".
