"""
Reading the ink with a document specialist: PaddleOCR-VL-1.6.

`OllamaVision` uses a general vision-language model and asks it, in prose, to
transcribe. This uses a 0.9B model trained only to read documents, which
takes a fixed task instruction (`OCR:`, `Formula Recognition:`,
`Table Recognition:`) instead of a prompt. Both readers return the same
contract objects, so a router cannot tell which one read its crop.

It runs through Hugging Face `transformers`, not Ollama, because Ollama does
not serve it. That brings three differences worth knowing:

**No constrained decoding.** There is no JSON schema to enforce. Instead the
model has its own fixed output formats, parsed here: tables arrive as OTSL
tokens (a cell grid, not markdown), formulas as LaTeX inside `\\[ ... \\]`.

**Formulas come back as LaTeX, and are converted.** The model emits LaTeX and
only LaTeX, so `FormulaContent.expr` — the form the solver parses — used to
be left `None`, which quietly turned off answers, the symbol table and
plotted curves. `vlm.latex` converts the subset of LaTeX that handwritten
notes contain, and returns `None` rather than a guess when it cannot: a
region whose expression is unknown is held for review, which is right, while
one that parses to the wrong thing is a confident wrong answer in the note,
which is the worst output this project can produce.

**It shares the GPU with Ollama, and they do not fit together.** Measured on
the RTX 4050's 6141 MiB: Qwen's runner takes 4196 MiB and this one then
fails to allocate. That is no longer something to remember — `vlm.residency`
evicts whichever model is resident before this one loads, and Ollama's
runner is asked to unload itself. The model loads on first use and `close()`
gives the memory back.

Measured on the RTX 4050, warm, on synthetic regions: a formula in 0.6 s, a
three-row table in 2.1 s, five lines of prose in 6.8 s. Licence Apache-2.0,
from the LICENSE file shipped with the weights.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable

import numpy as np

from ..contracts import FormulaContent, ProseContent, TableContent
from . import residency
from .latex import explicit_multiplication, latex_to_expr
from .ollama import VisionReadError, VisionUnavailable, cells_to_markdown, table_shape

DEFAULT_MODEL = os.environ.get("PALIMPSEST_PADDLE", "PaddlePaddle/PaddleOCR-VL-1.6")

TASKS = {
    "ocr": "OCR:",
    "formula": "Formula Recognition:",
    "table": "Table Recognition:",
}

# OTSL, the table language the model emits. One token per cell, `<nl>` per row.
# `<fcel>` is followed by its text; the rest are empty or merged cells, which
# a markdown table cannot express, so they become empty strings and the grid
# keeps its shape.
_OTSL_CELL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel)>")

Backend = Callable[[np.ndarray, str], str]


def otsl_to_cells(otsl: str) -> list[list[str]]:
    """Rows of cell strings from OTSL. Text outside any cell is ignored."""
    rows = []
    for line in otsl.split("<nl>"):
        parts = _OTSL_CELL.split(line)
        # split() with one group gives [lead, tag, text, tag, text, ...].
        cells = [text.strip() if tag == "fcel" else ""
                 for tag, text in zip(parts[1::2], parts[2::2])]
        if cells:
            rows.append(cells)
    return rows


def strip_math(text: str) -> str:
    """LaTeX without its display or inline delimiters."""
    text = text.strip()
    for left, right in (("\\[", "\\]"), ("$$", "$$"), ("\\(", "\\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            return text[len(left):len(text) - len(right)].strip()
    return text


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Crops arrive as OpenCV arrays: BGR, or single-channel grey."""
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image[:, :, ::-1].copy()


class PaddleVision:
    """PaddleOCR-VL behind the same three readers as `OllamaVision`.

    `backend` exists for the tests: a callable from (RGB image, task
    instruction) to the model's raw text, so they run without the weights or
    a GPU. Left as `None`, the real model is loaded on first use.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str | None = None,
        max_new_tokens: int = 512,
        max_pixels: int = 1280 * 28 * 28,
        backend: Backend | None = None,
    ):
        self.model_id = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_pixels = max_pixels
        self._backend = backend
        self._model = None
        self._processor = None
        self.last_call_ms: float | None = None

    # ------------------------------------------------------------- plumbing

    def available(self) -> bool:
        """Are the weights on disk? Never downloads; cheap at startup."""
        if self._backend is not None or self._model is not None:
            return True
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(self.model_id, local_files_only=True)
            return True
        except Exception:
            return False

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Evict Ollama's runner first. Loading into the 1576 MiB it leaves
        # behind is the OOM this exists to avoid.
        residency.acquire("paddle", self.close)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device).eval()
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except torch.OutOfMemoryError as exc:
            raise VisionUnavailable(
                "not enough VRAM for PaddleOCR-VL; another model is probably loaded "
                "(try `ollama stop qwen2.5vl:3b`)"
            ) from exc
        except OSError as exc:
            raise VisionUnavailable(
                f"{self.model_id} is not downloaded; run `hf download {self.model_id}`"
            ) from exc
        self._model = model
        self.device = device

    def _generate(self, image: np.ndarray, task: str) -> str:
        import torch
        from PIL import Image

        if self._model is None:
            self._load()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.fromarray(image)},
            {"type": "text", "text": task},
        ]}]
        size = self._processor.image_processor.size
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            images_kwargs={"size": {"shortest_edge": size.shortest_edge,
                                    "longest_edge": self.max_pixels}},
        ).to(self._model.device)
        # Greedy, for the same reason Ollama runs at temperature zero:
        # transcription has one right answer and numbers must repeat.
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                       do_sample=False)
        prompt_len = inputs["input_ids"].shape[-1]
        return self._processor.decode(out[0][prompt_len:], skip_special_tokens=False) \
            .replace("</s>", "").replace("<|end_of_sentence|>", "").strip()

    def ask(self, image: np.ndarray, task: str) -> str:
        rgb = _to_rgb(image)
        started = time.perf_counter()
        text = (self._backend or self._generate)(rgb, task)
        self.last_call_ms = (time.perf_counter() - started) * 1000
        if not text.strip():
            raise VisionReadError(f"empty reply to {task!r}")
        return text

    # -------------------------------------------------------------- readers

    def read_prose(self, image: np.ndarray) -> ProseContent:
        # No tidying pass: this model transcribes and nothing else, so the
        # markdown and the raw lines are the same reading.
        text = self.ask(image, TASKS["ocr"])
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return ProseContent(markdown="\n".join(lines), raw_lines=lines)

    def read_formula(self, image: np.ndarray) -> FormulaContent:
        tex = strip_math(self.ask(image, TASKS["formula"]))
        # `expr` is parsed from the tex as read; the dots are for the page.
        return FormulaContent(tex=explicit_multiplication(tex), expr=latex_to_expr(tex))

    def read_table(self, image: np.ndarray) -> TableContent:
        text = self.ask(image, TASKS["table"])
        cells = otsl_to_cells(text)
        if not cells:
            raise VisionReadError(f"table reply had no OTSL cells: {text[:80]!r}")
        markdown = cells_to_markdown(cells)
        rows, cols = table_shape(markdown)
        return TableContent(markdown=markdown, rows=rows, cols=cols)

    def close(self) -> None:
        """Unload the model and return its VRAM, so Ollama can have it back."""
        if self._model is None:
            return
        import torch

        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        residency.forget("paddle")
