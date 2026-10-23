"""
The page symbol table.

This is what makes the compiler *aware of variables*. Every region contributes
any `name = value` bindings it contains, tagged with where on the page they were
written. A formula then resolves each of its symbols against the page.

**Resolution rule: nearest definition above wins.**

A page is read top to bottom, so a formula uses the closest binding above it.
Two consequences, both deliberate:

- Redefining a variable part way down the page works the way the writer meant.
  The formulas above keep the old value, the ones below get the new one.
- A formula's answer can never depend on something written *below* it, which
  would be the surprising, hard-to-debug alternative.

A symbol with no binding above it stays **symbolic**. That is not an error: the
formula typesets correctly and simply carries no numeric answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Greek letters get written by hand far more often than they get typed, so the
# table accepts either the unicode character or the spelled-out name and
# normalises to one canonical ascii form.
GREEK: dict[str, str] = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
    "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon", "φ": "phi",
    "χ": "chi", "ψ": "psi", "ω": "omega",
    "Γ": "Gamma", "Δ": "Delta", "Θ": "Theta", "Λ": "Lambda", "Π": "Pi",
    "Σ": "Sigma", "Φ": "Phi", "Ψ": "Psi", "Ω": "Omega",
}

# `lambda` is a Python keyword, so a symbol named lambda would break any
# downstream parser. Reserved names get a trailing underscore, consistently, in
# both the symbol table and the solver.
RESERVED: frozenset[str] = frozenset(
    {"lambda", "and", "or", "not", "if", "else", "in", "is", "for", "while",
     "def", "class", "return", "import", "from", "as", "pass", "None", "True",
     "False"}
)


def canonical(name: str) -> str:
    """One spelling per symbol, whatever the writer used.

    `λ`, `lambda` and `lambda_` all become `lambda_`.
    """
    name = name.strip()
    name = "".join(GREEK.get(ch, ch) for ch in name)
    name = re.sub(r"[^0-9A-Za-z_]", "_", name)
    name = name.rstrip("_")
    if name in RESERVED:
        name += "_"
    return name


def display(name: str) -> str:
    """The inverse of the reserved-word suffix, for anything a reader sees.
    `lambda_` is the solver's safe spelling; the page said `lambda`."""
    return name[:-1] if name.endswith("_") and name[:-1] in RESERVED else name


# A binding is `name = number`, with optional surrounding junk. Deliberately
# narrow: it must be a bare number, because `rho = lambda/mu` is a formula to
# solve, not a binding to record. Expressions on the right are the solver's job.
_BINDING = re.compile(
    r"""
    (?:^|[\s,;(])                 # start, or a separator
    ([A-Za-zͰ-Ͽ][A-Za-z0-9_Ͱ-Ͽ]{0,15})  # name, maybe greek
    \s*=\s*
    ([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)                   # a bare number
    (?![\w.])                     # not followed by more of a token
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Binding:
    name: str
    value: float
    y: float
    """Vertical position on the page of the region that produced it. Smaller is
    higher up, following image coordinates."""
    source_block: str | None = None


@dataclass
class SymbolTable:
    """Every binding found on one page, resolvable by vertical position."""

    bindings: list[Binding] = field(default_factory=list)

    def add(self, name: str, value: float, y: float, source_block: str | None = None) -> None:
        self.bindings.append(Binding(canonical(name), float(value), y, source_block))

    def extend_from_text(self, text: str, y: float, source_block: str | None = None) -> int:
        """Harvest bindings from one region's transcribed text.

        Returns how many were found, which is worth logging: a page where
        nothing was harvested will resolve nothing, and that is better surfaced
        than silently producing answerless formulas.
        """
        found = 0
        for match in _BINDING.finditer(text):
            name, raw = match.group(1), match.group(2)
            try:
                self.add(name, float(raw), y, source_block)
                found += 1
            except ValueError:
                continue
        return found

    def resolve(self, y: float) -> dict[str, float]:
        """Values visible to a formula at vertical position `y`.

        Bindings at the same `y` are included, because a formula region often
        defines its own variables on the same line it uses them.
        """
        out: dict[str, float] = {}
        best_y: dict[str, float] = {}
        for b in self.bindings:
            if b.y > y:
                continue  # written below this formula, so not visible to it
            if b.name not in best_y or b.y >= best_y[b.name]:
                best_y[b.name] = b.y
                out[b.name] = b.value
        return out

    def flat(self) -> dict[str, float]:
        """The whole page's bindings, last definition per name. For display only,
        never for resolution: resolution is positional."""
        out: dict[str, float] = {}
        for b in sorted(self.bindings, key=lambda b: b.y):
            out[b.name] = b.value
        return out

    def redefined(self) -> dict[str, list[float]]:
        """Names bound more than once with different values.

        Not an error under the nearest-above rule, but worth surfacing in the UI:
        a reader looking at two different answers for the same symbol deserves to
        know the page said both.
        """
        seen: dict[str, list[float]] = {}
        for b in sorted(self.bindings, key=lambda b: b.y):
            seen.setdefault(b.name, []).append(b.value)
        return {k: v for k, v in seen.items() if len(set(v)) > 1}
