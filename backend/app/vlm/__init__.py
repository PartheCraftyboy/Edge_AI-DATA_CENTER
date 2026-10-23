r"""Readers for routed regions, and the judge that labels them.

Two different jobs, deliberately given to two different models.

`make_reader()` **transcribes**. It defaults to `paddle` (PaddleOCR-VL-1.6,
a 0.9B model trained only to read documents) because on real crops it reads
better: it returned "Nice to see you" where Qwen returned "Nice to tell
you", and kept the hat on `\hat{y}` where Qwen dropped it. Override with
`PALIMPSEST_READER=qwen`.

`make_judge()` **classifies** — which of the five region classes a crop is,
used to supervise the CNN on pages nobody has labelled. That is always Qwen,
and not because of configuration: it is a general vision-language model and
the question is a semantic one, while PaddleOCR-VL is a transcription
specialist with no notion of the classes and no `classify_region` at all.

They are kept apart in code because they cannot be kept together in VRAM:
Qwen takes 4,196 MiB of this card's 6,141 and PaddleOCR-VL then fails to
allocate. See `vlm.residency`, which evicts one before the other loads.
"""

import os

from .ollama import OllamaVision, VisionReadError, VisionUnavailable


class NoReader:
    """A reader that reads nothing, for running the app without a model.

    `PALIMPSEST_READER=none`. Every region is held with the reason in words,
    which the compile path already handles — a reader being unavailable has
    always been a survivable condition, so this needs no special case
    downstream.

    It exists for the case of working on the frontend while a training pass
    has the GPU. Starting the server is harmless on its own, since nothing
    loads until a page is compiled, but one press of Compile would load
    PaddleOCR-VL, and `vlm.residency` would correctly evict the model the
    training run is using. This removes that foot-gun rather than relying on
    remembering not to press the button.
    """

    model = "none"

    def available(self) -> bool:
        return False

    def _refuse(self, *_):
        raise VisionUnavailable(
            "reader disabled with PALIMPSEST_READER=none; regions are classified "
            "and held, not read"
        )

    read_prose = read_formula = read_table = classify_region = _refuse


# What the interface offers, in the order it offers them. Keyed by the name
# `make_reader` takes, so the dropdown and the factory cannot drift apart.
READERS: dict[str, dict] = {
    "paddle": {"label": "PaddleOCR-VL", "where": "local",
               "detail": "0.9B document specialist. Reads best on real crops."},
    "qwen": {"label": "Qwen2.5-VL", "where": "local",
             "detail": "3B general model. Also judges region classes."},
    "none": {"label": "Off", "where": "none",
             "detail": "Classify and hold every region. Leaves the GPU alone."},
}


def make_reader(name: str | None = None):
    """The transcription model. `paddle` unless told otherwise."""
    name = (name or os.environ.get("PALIMPSEST_READER", "paddle")).lower()
    if name == "qwen":
        return OllamaVision()
    if name == "none":
        return NoReader()
    if name == "paddle":
        from .paddle import PaddleVision  # lazy: keeps torch/transformers off the Ollama path

        return PaddleVision()
    raise ValueError(f"unknown reader {name!r}; have {', '.join(map(repr, READERS))}")


def make_judge(model: str | None = None) -> OllamaVision:
    """The model that labels region classes.

    Always Ollama, and not configurable through `PALIMPSEST_READER`. Setting
    that to `paddle` for better transcription must not silently leave the
    distillation pass without a judge: PaddleOCR-VL has no `classify_region`,
    so the failure would be an AttributeError an hour into a training run.
    """
    # A smaller image cap than the reader's. The judge only has to tell
    # words from mathematics, which does not need 1280 px, and a full-page
    # region at that size spends most of the context window on image tokens
    # before the reply has anywhere to go.
    return OllamaVision(model, max_side=896) if model else OllamaVision(max_side=896)


__all__ = ["READERS", "NoReader", "OllamaVision", "VisionReadError",
           "VisionUnavailable", "make_reader", "make_judge"]
