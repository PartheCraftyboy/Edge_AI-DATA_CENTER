"""
LaTeX to the solver's syntax.

PaddleOCR-VL reads mathematics far better than the general model does — it
keeps `\\hat{y}` where Qwen drops the hat — but it emits LaTeX and only
LaTeX. `FormulaContent.expr`, the form the solver parses, came back `None`
for every formula, which silently turns the compiler's headline feature off:
no answers, no symbol table, no plotted curves. A better reader that breaks
the solver is not a better reader.

This converts the subset of LaTeX that handwritten engineering notes
actually contain. It is deliberately not a LaTeX parser. SymPy ships one,
`sympy.parsing.latex.parse_latex`, and it is not used here for two reasons:
it needs the `antlr4` runtime pinned to SymPy's grammar version, and it
accepts far more than this needs while still failing on the things it does
not. A small, total, testable transformation over a known subset is the
better trade when the input comes from one model reading one kind of page.

**It returns `None` rather than a guess.** The reader misreads: on a real
page `m x + c` came back as `m \\times + c`, the variable `x` read as the
multiplication sign. Converted literally that is `m * + c`, which is not an
expression. Returning `None` holds the region for review, which is correct;
returning something that happens to parse would put a wrong answer in the
note, which is the failure this project cares most about avoiding.

The output is the solver's syntax, not Python's: `^` is left alone because
`solver.TRANSFORMS` includes `convert_xor`, and implicit multiplication is
left alone because it includes `implicit_multiplication`. See the long note
in `routers/solver.py` about why `split_symbols` is off — it is the reason
`B_1 x` is safe to emit and `B_1*x` is not required.
"""

from __future__ import annotations

import re

# Decorations that name a variable rather than operate on it. `\hat{y}` is a
# distinct quantity from `y` on a page about regression — the predicted value
# against the observed one — so it becomes `yhat`, not `y`. Collapsing them
# would make the solver bind one to the other and quietly produce nonsense.
ACCENTS = {"hat": "hat", "bar": "bar", "tilde": "tilde", "vec": "vec",
           "dot": "dot", "ddot": "ddot"}

# Operators and spacing with a direct equivalent, applied after the
# structural rewrites so nothing below has to cope with them.
SIMPLE = {
    r"\times": "*", r"\cdot": "*", r"\div": "/", r"\ast": "*",
    r"\left": "", r"\right": "", r"\!": "", r"\,": " ", r"\;": " ",
    r"\:": " ", r"\quad": " ", r"\qquad": " ", r"\ ": " ",
    r"\displaystyle": " ", r"\limits": "",
    r"\lbrace": "(", r"\rbrace": ")", r"\lbrack": "(", r"\rbrack": ")",
    r"\%": "%", r"\$": "", r"\&": "", r"\#": "",
}

# Named functions the solver already allows. `\ln` is mapped to `log`, which
# is what `ALLOWED_FUNCS` calls it.
FUNCTIONS = {
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "exp", "log", "ln", "sqrt", "min", "max",
}
FUNCTION_NAMES = {"arcsin": "asin", "arccos": "acos", "arctan": "atan", "ln": "log"}

GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota "
         "kappa lambda mu nu xi pi rho sigma tau upsilon phi varphi chi psi omega "
         "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega").split()


def _matched(text: str, start: int) -> tuple[str, int]:
    """The brace group beginning at `start`, and the index just past it.

    Written out rather than done with a regular expression because these
    nest: `\\frac{\\frac{a}{b}}{c}` is ordinary in a derivation, and a
    non-greedy `\\{.*?\\}` takes the wrong closing brace every time.
    """
    if start >= len(text) or text[start] != "{":
        # A single token group, as in `x^2` or `\frac12`.
        return (text[start], start + 1) if start < len(text) else ("", start)
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return text[start + 1:], len(text)  # unbalanced: take the rest


def _expand(text: str) -> str:
    """Rewrite the constructs that take brace arguments.

    One left-to-right pass, recursing into each argument. Deliberately not a
    repeated pass until nothing changes, which is the obvious shape and is
    wrong here: this rewrites `x^{2}` to `x^(2)`, so a second pass sees the
    `^` again, reads `(` as a one-token group, and wraps it once more. Twelve
    passes turned `x^{2}` into `x^(()()()()()()()()()()()2)`. Nesting is
    already handled by recursing into the arguments, so one pass is enough.
    """
    if True:
        out, i = [], 0
        while i < len(text):
            if text.startswith(r"\frac", i) or text.startswith(r"\dfrac", i) or \
               text.startswith(r"\tfrac", i):
                skip = 5 if text.startswith(r"\frac", i) else 6
                num, j = _matched(text, i + skip)
                den, k = _matched(text, j)
                out.append(f"(({_expand(num)})/({_expand(den)}))")
                i = k
                continue
            if text.startswith(r"\sqrt", i):
                arg, j = _matched(text, i + 5)
                out.append(f"sqrt({_expand(arg)})")
                i = j
                continue
            matched_accent = None
            for name in ACCENTS:
                if text.startswith("\\" + name, i) and not text.startswith("\\" + name + "a", i):
                    matched_accent = name
                    break
            if matched_accent:
                arg, j = _matched(text, i + 1 + len(matched_accent))
                inner = _expand(arg).strip()
                # Only a bare name can carry the suffix; anything else keeps
                # its own shape and loses the decoration.
                out.append(f"{inner}{ACCENTS[matched_accent]}"
                           if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", inner) else inner)
                i = j
                continue
            if text[i] in "^_":
                arg, j = _matched(text, i + 1)
                inner = _expand(arg).strip()
                out.append(f"^({inner})" if text[i] == "^" else
                           (f"_{inner}" if re.fullmatch(r"[A-Za-z0-9]+", inner) else ""))
                i = j
                continue
            out.append(text[i])
            i += 1
        text = "".join(out)
    return text


# A binary operator with nothing to its left, or followed by one that cannot
# start an operand. `a * -b` and `a + -b` are ordinary, so a leading minus is
# allowed after anything; `a * + b` and `a / * b` are not written by anyone.
_BAD_PAIR = re.compile(r"[*/^]\s*[*/^+]|[+\-]\s*[*/^]")
_TRAILING = re.compile(r"[+\-*/^=]\s*$")


def _answer_left_blank(sides: list[str]) -> bool:
    """Is this a complete expression whose answer the writer left blank?

    `sides` is the formula split on `=` and stripped, so `1 + 2 =` arrives as
    `["1 + 2", ""]`. An empty side is normally a misread and `_malformed`
    refuses it. This is the one exception: a notebook full of `1 + 2 =` is
    exactly what the solver was built to finish, and refusing it means the
    arithmetic solver never runs on the most ordinary input there is.

    The exception has to stay narrow, because the guard below it is the only
    thing standing between a misread and a confident wrong answer in someone's
    notes. Two cases it must keep refusing, both already in the test suite:
    `= 5`, where the empty side is on the left and the reader clearly dropped
    something, and `y =`, where the left side is a bare name that defines
    nothing and leaves no arithmetic to do.

    TODO(parth): write this condition.
    """
    raise NotImplementedError


def _malformed(text: str) -> str | None:
    """Why this is not an expression, or `None` if it looks like one.

    SymPy is more permissive than is safe here. The reader turned `m x + c`
    into `m \\times + c` on a real page, reading the variable `x` as the
    multiplication sign; that converts to `m * + c`, which SymPy reads
    happily as `m * (+c)` and evaluates to `m*c`. A wrong answer that parses
    is the single worst output this project can produce, because nothing
    downstream marks it as suspect. Holding the region instead puts it in
    front of someone, which is what the gate is for.
    """
    sides = [side.strip() for side in text.split("=")]
    if _answer_left_blank(sides):
        # The blank is the question, not a misread. Drop it and check the rest
        # exactly as strictly as before.
        sides = sides[:-1]
    for side in sides:
        if not side:
            return "empty side of an equation"
        if _BAD_PAIR.search(side):
            return "two operators in a row"
        if _TRAILING.search(side):
            return "ends on an operator"
    return None


def latex_to_expr(tex: str) -> str | None:
    """The solver's syntax for a LaTeX formula, or `None` if it will not parse.

    `None` is a normal outcome, not a failure of this function: the reader
    misreads, and a region whose expression cannot be recovered is held for
    review rather than answered wrongly.
    """
    if not tex or not tex.strip():
        return None
    text = tex.strip()

    # Display and inline delimiters, in case a caller passes them through.
    for left, right in (("\\[", "\\]"), ("$$", "$$"), ("\\(", "\\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):len(text) - len(right)].strip()

    # An alignment or line break means several statements; the solver takes
    # one, so the first is used and the rest dropped rather than concatenated
    # into something that parses as a different equation.
    for separator in (r"\\", r"\newline"):
        if separator in text:
            text = text.split(separator)[0]
    text = text.replace("&", " ")

    text = _expand(text)

    for name in sorted(FUNCTIONS, key=len, reverse=True):
        text = text.replace("\\" + name, FUNCTION_NAMES.get(name, name))
    for name in sorted(GREEK, key=len, reverse=True):
        text = re.sub(r"\\" + name + r"(?![A-Za-z])", name, text)
    for token, replacement in SIMPLE.items():
        text = text.replace(token, replacement)

    # Anything still carrying a backslash is a command this does not know,
    # and guessing at it is how a wrong answer gets into a note.
    text = text.replace("{", "(").replace("}", ")")
    if "\\" in text:
        return None

    text = re.sub(r"\s+", " ", text).strip().rstrip("., ")
    if not text or not any(c.isalnum() for c in text):
        return None
    # `=` is the solver's business; more than one is not an equation it reads.
    if text.count("=") > 1:
        return None
    if _malformed(text):
        return None

    from ..routers.solver import parse_or_none

    return text if parse_or_none(text) is not None else None


# Commands that take an argument rather than standing for a quantity.
# `\sin x` is a function applied to x, not \sin times x, so no dot goes
# between them. Everything not listed here is treated as an operand.
_FUNCTION_COMMANDS = {
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh",
    "tanh", "exp", "log", "ln", "lg", "sqrt", "frac", "dfrac", "tfrac",
    "min", "max", "lim", "sum", "int", "prod", "hat", "bar", "vec", "tilde",
    "dot", "ddot", "left", "right", "text", "mathrm", "operatorname",
}

# Commands that ARE operators. A dot never goes on either side of one, and
# leaving them out is how `a \cdot b` became `a \cdot \cdot \cdot b`.
_OPERATOR_COMMANDS = {
    "cdot", "times", "div", "pm", "mp", "ast", "star", "circ", "bullet",
    "leq", "geq", "neq", "approx", "equiv", "sim", "propto", "to",
    "rightarrow", "Rightarrow", "quad", "qquad",
}

# One LaTeX token: a command, a brace, a sub/superscript marker, a number,
# a single letter, or any other single character.
_TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[{}]|[_^]|\d+(?:\.\d+)?|[A-Za-z]|\S")


def _operand_end(token: str) -> bool:
    """Could a multiplication sign follow this token?"""
    if token in {")", "}", "]"}:
        return True
    if token[0] == "\\":
        name = token[1:]
        return name not in _FUNCTION_COMMANDS and name not in _OPERATOR_COMMANDS
    return bool(re.fullmatch(r"[A-Za-z]|\d+(?:\.\d+)?", token))


def _operand_start(token: str) -> bool:
    """Could a multiplication sign precede this token?"""
    if token in {"(", "["}:
        return True
    if token[0] == "\\":
        name = token[1:]
        if name in _OPERATOR_COMMANDS:
            return False
        return name not in {"left", "right", "text", "mathrm", "operatorname"}
    return bool(re.fullmatch(r"[A-Za-z]|\d+(?:\.\d+)?", token))


def explicit_multiplication(tex: str) -> str:
    r"""`m x` and `2x` become `m \cdot x` and `2 \cdot x`.

    Handwritten mathematics omits the multiplication sign, and so does the
    LaTeX both readers return. Typeset, `mx` is ambiguous on the page in a
    way it never was in the notebook: a reader cannot tell a product of two
    variables from one variable whose name is two letters long, and the
    point of this project is that the compiled note should be at least as
    readable as the page it came from.

    Only genuinely adjacent operands get a dot. A subscript or superscript
    belongs to the symbol before it, so `B_1 x` is one product and not two.
    A function applied to an argument is not a product, so `\sin x` is left
    alone. An existing operator is not an operand, so `a \cdot b` is left
    alone rather than growing more dots every time this runs.

    The dots are **spliced into the original string** rather than the string
    being rebuilt from its tokens. Rebuilding has to re-invent the spacing
    and gets it wrong — the first version produced `B_{ 0}` and `\hat { y}`
    — and spacing in LaTeX is not always cosmetic.
    """
    if not tex:
        return tex
    matches = list(_TOKEN.finditer(tex))
    insert_at: list[int] = []
    depth = 0
    script_groups: list[int] = []
    pending_script = False
    previous = None

    for match in matches:
        token = match.group()
        if token == "{":
            depth += 1
            if pending_script:
                script_groups.append(depth)
                pending_script = False
        elif token == "}":
            if script_groups and script_groups[-1] == depth:
                script_groups.pop()
            depth -= 1

        in_script = bool(script_groups)
        if (previous is not None and not in_script and not pending_script
                and previous not in {"_", "^"}
                and _operand_end(previous) and _operand_start(token)):
            insert_at.append(match.start())

        if token in {"_", "^"}:
            pending_script = True
        elif pending_script and token != "{":
            pending_script = False     # a one-token script, as in `x^2`
        previous = token

    out, last = [], 0
    for position in insert_at:
        chunk = tex[last:position]
        out.append(chunk)
        out.append(r"\cdot " if chunk.endswith(" ") else r" \cdot ")
        last = position
    out.append(tex[last:])
    return "".join(out)
