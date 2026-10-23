"""
Reading the ink: a local vision-language model served by Ollama.

The classifier says *what kind* of region a crop is. This module says *what it
says*. Each routed class gets its own reader, and each reader returns the
contract object its router already expects (`ProseContent`, `FormulaContent`,
`TableContent`), so nothing downstream knows a language model was involved.

The model is `qwen2.5vl:3b` by default, not the `qwen2.5:3b` that was already
pulled: that one is text-only and cannot see the crop at all. Three billion
parameters is chosen for the 6 GB RTX 4050, where it fits with the classifier
alongside it. Override with `PALIMPSEST_VLM` to compare a larger model.

Two decisions here matter more than the prompts.

**The reply is constrained by a JSON schema, not parsed out of free text.**
Ollama's `format` field accepts a schema and decodes only tokens that keep the
output valid against it. A 3B model asked politely for JSON will still, some
fraction of the time, wrap it in a code fence or add a sentence before it;
constrained decoding removes that whole class of failure instead of writing a
regex to tolerate it. The schema is generated from a pydantic model, so the
shape is declared once and validated on the way back in.

**Temperature is zero.** Transcription has one right answer. Sampling would
make the same page compile differently twice, which is useless for a review
panel and makes latency and accuracy numbers unrepeatable.
"""

from __future__ import annotations

import base64
import os
import re
import time

import cv2
import httpx
import numpy as np
from typing import Literal

from pydantic import BaseModel, ValidationError

from .. import CLASS_NAMES
from ..contracts import FormulaContent, ProseContent, TableContent
from . import residency

DEFAULT_MODEL = os.environ.get("PALIMPSEST_VLM", "qwen2.5vl:3b")
DEFAULT_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class VisionUnavailable(RuntimeError):
    """The server is down or the model is not pulled. Every region will fail."""


class VisionReadError(RuntimeError):
    """The model answered, but not with something usable. Hold this region."""


# ------------------------------------------------------------- reply shapes
# What the model is asked to emit. Deliberately smaller than the contracts:
# fields the model cannot know, such as a formula's resolved answer, are
# computed by the solver afterwards and are never the model's to invent.

class _ProseReply(BaseModel):
    lines: list[str]
    markdown: str


class _FormulaReply(BaseModel):
    tex: str
    expr: str


class _ClassReply(BaseModel):
    """The teacher's vote on what kind of region a crop is.

    **Field order is the whole design.** Constrained decoding fills a schema
    in order, so putting `transcription` first forces the model to write down
    what it can see before it is allowed to name a class, and `mathematics`
    makes it commit to the one distinction the class turns on. Asked for the
    class alone it scored 3 of 6 on crops whose answer is known, every error
    a line of ordinary writing called FORMULA. Made to read first, on the same
    six crops, it scored 6 of 6.

    What makes that work is worth stating, because it looks like it should
    not. The transcriptions are often **wrong**: "Parth" came back as "Booth"
    and "Nice to see you" as "Nice to tell you". The label was right anyway.
    Whether a region is words or mathematics survives a misread word, so a
    model that reads badly can still classify well, as long as it is made to
    read first.

    `Literal` over the class names rather than a free string, so decoding
    cannot return a sixth class, a lowercase variant, or a sentence explaining
    itself.
    """

    transcription: str
    """A sample, not a full reading. Deliberately capped in the prompt: a
    region covering most of a page made the model transcribe the lot, which
    overran the 4096-token context and came back as an empty string that
    failed JSON validation. 414 of 415 regions failed that way on the first
    real corpus. The cap costs nothing, because the class never depended on
    reading all of it — see the note below about labels surviving misreads."""

    mathematics: bool
    region_class: Literal[CLASS_NAMES]  # type: ignore[valid-type]


class _TableReply(BaseModel):
    # Cells, not markdown. Asked for a markdown string, the 3B model read an
    # unruled table as plain lines and the result had no grid at all. With a
    # list of rows in the schema, constrained decoding cannot return
    # anything but a grid, and the markdown is built here where it is exact.
    cells: list[list[str]]


CLASSIFY_PROMPT = (
    "This image is one region cut from a page of handwritten study notes.\n"
    "First, in `transcription`, write out the FIRST FEW WORDS you can read, at "
    "most fifteen words. Do not transcribe the whole region.\n"
    "Then, in `mathematics`, say whether that transcription is a mathematical "
    "statement: an equation, an expression being manipulated, symbols combined "
    "with operators. Ordinary words are NOT mathematics even when short, and a "
    "sentence that merely mentions a variable is NOT mathematics.\n"
    "Then, in `region_class`, give the kind of region:\n"
    "THEORY: words and sentences. Prose, a heading, a bullet, a label, a name, "
    "even a single word or a single line of ordinary writing.\n"
    "FORMULA: mathematics, as you just decided in `mathematics`.\n"
    "PLOT_SKETCH: a hand-drawn graph. Axes, a curve, plotted bars.\n"
    "DIAGRAM: boxes, arrows, a block diagram or a circuit.\n"
    "TABLE: a grid, ruled or unruled, with rows and columns of entries."
)

PROSE_PROMPT = (
    "This image is a region of handwritten study notes. Transcribe it exactly. "
    "Put each handwritten line, as written, in `lines`. Then put a tidied version "
    "in `markdown`: fix line breaks and obvious spelling only, keep every number "
    "and every `name = value` exactly as written, and use markdown bullets if the "
    "writer used bullets. Do not add anything that is not on the page."
)

FORMULA_PROMPT = (
    "This image is a handwritten mathematical expression. Return it twice. "
    "`tex`: LaTeX for display, without surrounding $ signs. "
    "`expr`: the same expression in Python/SymPy syntax, using ** for powers, "
    "explicit * for multiplication, and Greek letters spelled out (lambda, mu, rho). "
    "If it is an equation keep the single `=`. Transcribe only; do not simplify or solve."
)

TABLE_PROMPT = (
    "This image is a handwritten table, possibly without ruled lines. Return it in "
    "`cells` as a list of rows, each row a list of cell strings, first row as the "
    "header. Keep cell contents exactly as written; use an empty string for an "
    "empty cell."
)


class OllamaVision:
    """A client for one vision model on one Ollama server.

    `transport` exists for the tests: they pass an `httpx.MockTransport` and
    never touch the network or need a GPU.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_URL,
        timeout: float = 120.0,
        keep_alive: str = "10m",
        max_side: int = 1280,
        min_side: int = 448,
        num_ctx: int = 4096,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        # Kept loaded between regions: a page has several, and reloading 3 GB
        # into VRAM for each would cost more than reading them.
        self.keep_alive = keep_alive
        self.max_side = max_side
        self.min_side = min_side
        # Fixed rather than left to Ollama's default, because the KV cache is
        # allocated for the whole context up front: 36 KiB per token on this
        # model, so 144 MiB at 4096. That holds a 1280 px crop (~2,090 image
        # tokens) plus prompt and reply, and keeps the budget predictable.
        self.num_ctx = num_ctx
        self.client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self.last_call_ms: float | None = None

    # ------------------------------------------------------------- plumbing

    def available(self) -> bool:
        """Is the server up and the model pulled? Cheap; call it at startup."""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name") for m in response.json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def unload(self) -> bool:
        """Ask Ollama to drop this model from VRAM now.

        `keep_alive: 0` on an empty prompt is the documented way: the request
        loads nothing and sets the model's lifetime to zero, so the runner
        exits. Used to make room for PaddleOCR-VL, which does not fit beside
        it; see `vlm.residency` for the measurements.
        """
        try:
            self.client.post("/api/generate",
                             json={"model": self.model, "keep_alive": 0},
                             timeout=30.0).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def placement(self) -> dict | None:
        """Where the loaded model sits: total size, VRAM share, spill to RAM.

        Ollama offloads whole layers to system RAM when VRAM runs out and says
        nothing about it except here, as `size_vram` below `size`. A spill is
        the difference between seconds and tens of seconds per region, so it
        is worth checking rather than assuming. `None` if not loaded yet.
        """
        try:
            response = self.client.get("/api/ps")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        for m in response.json().get("models", []):
            if m.get("name") == self.model or m.get("model") == self.model:
                size, vram = int(m.get("size", 0)), int(m.get("size_vram", 0))
                return {
                    "size_mib": round(size / 2**20),
                    "vram_mib": round(vram / 2**20),
                    "ram_mib": round((size - vram) / 2**20),
                    "gpu_fraction": round(vram / size, 3) if size else 0.0,
                }
        return None

    def encode(self, image: np.ndarray) -> str:
        """A crop as base64 PNG, resized into the range the model reads well.

        Classifier crops are 224 px, which is too small to read handwriting
        from, so tiny crops are upscaled. Large ones are capped, because the
        model's token count, and with it latency, grows with pixel area.
        """
        h, w = image.shape[:2]
        scale = 1.0
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
        elif max(h, w) < self.min_side:
            scale = self.min_side / max(h, w)
        if scale != 1.0:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=interp)
        ok, png = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("could not encode image as PNG")
        return base64.b64encode(png.tobytes()).decode("ascii")

    def ask(self, image: np.ndarray, prompt: str, reply: type[BaseModel], retries: int = 1) -> BaseModel:
        """Send one image and one instruction, get back a validated reply.

        Claims VRAM first, which evicts PaddleOCR-VL if it is resident in
        this process. The two do not fit together on a 6 GB card.

        A reply that fails validation is retried once. Constrained decoding
        makes malformed JSON rare, but it can still truncate at the context
        limit, and one retry is cheaper than holding a region that would
        have read fine.
        """
        residency.acquire("ollama", self.unload)
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt, "images": [self.encode(image)]}],
            "format": reply.model_json_schema(),
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }

        last_error: Exception | None = None
        for _ in range(retries + 1):
            started = time.perf_counter()
            try:
                response = self.client.post("/api/chat", json=payload)
            except httpx.TransportError as exc:
                raise VisionUnavailable(f"Ollama is not reachable: {exc}") from exc
            self.last_call_ms = (time.perf_counter() - started) * 1000

            if response.status_code == 404:
                raise VisionUnavailable(f"model {self.model!r} is not pulled; run `ollama pull {self.model}`")
            if response.status_code >= 400:
                raise VisionUnavailable(f"Ollama returned {response.status_code}: {response.text[:200]}")

            content = response.json().get("message", {}).get("content", "")
            try:
                return reply.model_validate_json(content)
            except ValidationError as exc:
                last_error = exc
        raise VisionReadError(f"unusable reply after {retries + 1} attempts: {last_error}")

    # -------------------------------------------------------------- readers

    def classify_region(self, image: np.ndarray) -> tuple[str, str]:
        """The teacher's label for one crop, as `(class name, transcription)`.

        The CNN is the classifier this system ships; this is a second opinion
        from a model that has read far more handwriting than any checkpoint
        trained here, used to supervise it on pages nobody has labelled. It is
        slow, about 1.2 s a crop against the CNN's few milliseconds, so it
        belongs in an offline labelling pass and never on the compile path.

        Pass the region at its own aspect ratio, not the classifier's 224px
        square. A squashed crop loses the shape of the thing, and a line of
        prose is mostly recognisable by being long and short.

        The transcription comes back with the label because it is the evidence
        for it: when a pseudo-label looks wrong, the text the model thought it
        was reading says why, which is the difference between a dataset you
        can audit and one you have to trust.
        """
        reply = self.ask(image, CLASSIFY_PROMPT, _ClassReply)
        return reply.region_class, reply.transcription

    def read_prose(self, image: np.ndarray) -> ProseContent:
        r = self.ask(image, PROSE_PROMPT, _ProseReply)
        return ProseContent(markdown=r.markdown.strip(), raw_lines=[l for l in r.lines if l.strip()])

    def read_formula(self, image: np.ndarray) -> FormulaContent:
        # `answer`, `bound` and `free` are left at their defaults on purpose:
        # they belong to the solver, which knows the rest of the page.
        r = self.ask(image, FORMULA_PROMPT, _FormulaReply)
        # The dots are for the page. `expr` keeps the model's own syntax,
        # which the solver already reads.
        from .latex import explicit_multiplication

        return FormulaContent(tex=explicit_multiplication(r.tex.strip().strip("$").strip()),
                              expr=r.expr.strip() or None)

    def read_table(self, image: np.ndarray) -> TableContent:
        r = self.ask(image, TABLE_PROMPT, _TableReply)
        markdown = cells_to_markdown(r.cells)
        rows, cols = table_shape(markdown)
        return TableContent(markdown=markdown, rows=rows, cols=cols)

    def close(self) -> None:
        self.client.close()


def cells_to_markdown(cells: list[list[str]]) -> str:
    """A GitHub markdown table from rows of cells, first row the header.

    Ragged rows are padded to the widest, and `|` inside a cell is escaped so
    it cannot split the cell in two.
    """
    # An all-empty row is kept: it is on the page, and `table_shape` counts it.
    rows = [[c.strip().replace("|", "\\|") for c in row] for row in cells if row]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
    return "\n".join(lines)


def table_shape(markdown: str) -> tuple[int, int]:
    """Data rows and columns of a markdown table, separator row excluded."""
    lines = [l.strip() for l in markdown.strip().splitlines() if l.strip().startswith("|")]
    # The separator is dashes and colons only; an all-empty row is still data.
    body = [l for l in lines if not ("-" in l and set(l.replace("|", "")) <= set("-: "))]
    if not body:
        return 0, 0
    # Split on unescaped pipes only, so an escaped `\|` stays inside its cell.
    cols = max(len(re.split(r"(?<!\\)\|", l.strip("|"))) for l in body)
    return max(0, len(body) - 1), cols
