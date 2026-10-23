"""
Domain constraints harvested from the page.

If the writer bothered to note `0 < x < 10`, that is the authoritative statement
of where the function is meaningful, and the plotter should respect it rather
than inventing a range. This module reads those statements the way
`symbols.py` reads `name = value` bindings.

Understood forms, in any mixture of ascii and the unicode a person actually
writes by hand:

    0 < x              x > 0             x >= 0
    x < 10             10 > x            x <= 10
    0 < rho < 1        0 <= t <= 20      0 ≤ t ≤ 20
    x in (0, 10)       rho ∈ [0, 1)

**Why the resolution rule differs from bindings, deliberately.**

A binding is a *redefinition*: `mu = 22.7` then later `mu = 87` means the second
replaces the first, so the nearest definition above wins and earlier ones are
shadowed. A constraint is an *accumulation*: `0 < x` on one line and `x < 10`
three lines later are complementary halves of one statement, not competing
claims. Taking only the nearest would throw half the information away.

So constraints at or above the formula are **intersected**, tightest bound
winning, while bindings resolve to the nearest. Both rules read down the page;
they differ because the two things mean different things.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .symbols import canonical

_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_NAME = r"[A-Za-z_Ͱ-Ͽ][A-Za-z0-9_Ͱ-Ͽ]{0,15}"
_OP = r"<=|>=|=<|=>|≤|≥|<|>"

# `0 < x < 10`. Matched first, because the single-sided patterns would each
# match half of it and lose the other half.
_CHAINED = re.compile(
    rf"({_NUM})\s*({_OP})\s*({_NAME})\s*({_OP})\s*({_NUM})"
)
# `x > 0`
_NAME_FIRST = re.compile(rf"(?:^|[\s,;(])({_NAME})\s*({_OP})\s*({_NUM})")
# `0 < x`
_NUM_FIRST = re.compile(rf"(?:^|[\s,;(])({_NUM})\s*({_OP})\s*({_NAME})")
# `x in (0, 10)` or `rho ∈ [0, 1)`
_INTERVAL = re.compile(
    rf"({_NAME})\s*(?:∈|\bin\b)\s*([\[(])\s*({_NUM})\s*,\s*({_NUM})\s*([\])])"
)

_NORMALISE_OP = {"=<": "<=", "=>": ">=", "≤": "<=", "≥": ">="}


def _op(raw: str) -> str:
    return _NORMALISE_OP.get(raw, raw)


@dataclass(frozen=True)
class Constraint:
    """A half-open or closed interval a symbol is restricted to."""

    name: str
    lower: float | None = None
    upper: float | None = None
    lower_strict: bool = False
    upper_strict: bool = False
    y: float = 0.0

    def merged_with(self, other: Constraint) -> Constraint:
        """Intersect two constraints on the same symbol, tightest bound winning.

        Strictness is inherited from whichever bound survives, and a tie on the
        value keeps the stricter of the two: `x > 0` intersected with `x >= 0`
        is `x > 0`, because satisfying both excludes zero.
        """
        lower, lower_strict = self.lower, self.lower_strict
        if other.lower is not None:
            if lower is None or other.lower > lower:
                lower, lower_strict = other.lower, other.lower_strict
            elif other.lower == lower:
                lower_strict = lower_strict or other.lower_strict

        upper, upper_strict = self.upper, self.upper_strict
        if other.upper is not None:
            if upper is None or other.upper < upper:
                upper, upper_strict = other.upper, other.upper_strict
            elif other.upper == upper:
                upper_strict = upper_strict or other.upper_strict

        return Constraint(
            name=self.name,
            lower=lower,
            upper=upper,
            lower_strict=lower_strict,
            upper_strict=upper_strict,
            y=max(self.y, other.y),
        )

    @property
    def bounded(self) -> bool:
        return self.lower is not None and self.upper is not None

    @property
    def empty(self) -> bool:
        """True when the constraints contradict, e.g. `x > 5` and `x < 2`.

        Worth detecting rather than producing an inverted axis: a page that
        says two incompatible things should be reported, not silently plotted.
        """
        if not self.bounded:
            return False
        if self.lower > self.upper:
            return True
        return self.lower == self.upper and (self.lower_strict or self.upper_strict)

    def describe(self) -> str:
        """Human-readable form, for the provenance strip in the UI."""
        lo = "" if self.lower is None else f"{self.lower:g} {'<' if self.lower_strict else '<='} "
        hi = "" if self.upper is None else f" {'<' if self.upper_strict else '<='} {self.upper:g}"
        return f"{lo}{self.name}{hi}"


def _one_sided(name: str, op: str, value: float, *, name_on_left: bool, y: float) -> Constraint:
    """Build a constraint from a single comparison.

    `0 < x` and `x > 0` say the same thing, so the side the name fell on decides
    which way the operator points.
    """
    op = _op(op)
    if not name_on_left:
        op = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}[op]
    if op in ("<", "<="):
        return Constraint(name, upper=value, upper_strict=(op == "<"), y=y)
    return Constraint(name, lower=value, lower_strict=(op == ">"), y=y)


@dataclass
class ConstraintTable:
    """Every domain constraint found on one page."""

    constraints: list[Constraint] = field(default_factory=list)

    def add(self, c: Constraint) -> None:
        self.constraints.append(c)

    def extend_from_text(self, text: str, y: float) -> int:
        """Harvest constraints from one region's transcribed text."""
        found = 0
        consumed: list[tuple[int, int]] = []

        def overlaps(span: tuple[int, int]) -> bool:
            return any(not (span[1] <= s or span[0] >= e) for s, e in consumed)

        for m in _CHAINED.finditer(text):
            lo_raw, op1, name, op2, hi_raw = m.groups()
            o1, o2 = _op(op1), _op(op2)
            # Both operators must point the same way for the chain to be a range.
            if (o1 in ("<", "<=")) != (o2 in ("<", "<=")):
                continue
            lo, hi = float(lo_raw), float(hi_raw)
            if o1 in (">", ">="):
                lo, hi = hi, lo
                o1, o2 = o2, o1
            self.add(Constraint(
                canonical(name),
                lower=lo, upper=hi,
                lower_strict=(o1 == "<"), upper_strict=(o2 == "<"),
                y=y,
            ))
            consumed.append(m.span())
            found += 1

        for m in _INTERVAL.finditer(text):
            name, open_br, lo_raw, hi_raw, close_br = m.groups()
            self.add(Constraint(
                canonical(name),
                lower=float(lo_raw), upper=float(hi_raw),
                lower_strict=(open_br == "("), upper_strict=(close_br == ")"),
                y=y,
            ))
            consumed.append(m.span())
            found += 1

        for pattern, name_on_left in ((_NAME_FIRST, True), (_NUM_FIRST, False)):
            for m in pattern.finditer(text):
                if overlaps(m.span()):
                    continue
                a, op, b = m.groups()
                name, value = (a, float(b)) if name_on_left else (b, float(a))
                self.add(_one_sided(
                    canonical(name), op, value, name_on_left=name_on_left, y=y
                ))
                consumed.append(m.span())
                found += 1

        return found

    def resolve(self, y: float) -> dict[str, Constraint]:
        """Constraints visible to a formula at vertical position `y`.

        Intersected rather than shadowed. See the module docstring for why this
        differs from how bindings resolve.
        """
        out: dict[str, Constraint] = {}
        for c in sorted(self.constraints, key=lambda c: c.y):
            if c.y > y:
                continue
            out[c.name] = c if c.name not in out else out[c.name].merged_with(c)
        return out

    def for_symbol(self, name: str, y: float) -> Constraint | None:
        return self.resolve(y).get(canonical(name))
