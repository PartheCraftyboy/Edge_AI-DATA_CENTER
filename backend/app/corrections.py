"""
Telling the system it got it wrong.

    python -m app.corrections --report

Someone wrote `x^2 = 49` and the note came back `x^2 = 4`. The classifier was
right, the gate was confident and correct to be, and the reading was still
wrong. Nothing in the pipeline had anywhere to put that fact, so the only
record of it was the person noticing and shrugging.

A correction says what a region should have been. It carries two independent
parts, because two different things can be wrong and they are fixed by
different work:

**the class** — the region was routed to the wrong handler. A line of prose
sent to formula recognition comes back as invented LaTeX. A class correction
is a hand label, and it is written straight back into the page store, so
fixing a block in the note view produces exactly the training data that stops
it happening again.

**the reading** — the class was right and the transcription was not. That
teaches the classifier nothing. It is an evaluation example for the reader,
and it is what `--report` scores: a growing set of regions whose correct text
is known, which is the only way to answer whether a prompt change, a
different model or PaddleOCR-VL instead of Qwen actually reads better, rather
than reading differently.

Append-only, one JSON object per line, because the file is a record of what
was observed and a record that rewrites itself is not one. Each entry keeps
the page and box alongside the text, so the crop can always be cut again and
looked at.

Every function here resolves `path` at call time rather than defaulting to
`CORRECTIONS_PATH` in its signature. A default argument is evaluated once, at
import, so a test that redirects the module constant would still write to the
real file — which is exactly what happened, and put nine fixture rows into
someone's evidence before it was noticed.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

CORRECTIONS_PATH = Path(os.environ.get("PALIMPSEST_CORRECTIONS", "data/corrections.jsonl"))


def record(
    path: str | Path | None = None,
    *,
    note_id: str,
    block_id: str,
    box: list[int],
    region_class: str,
    read_as: str = "",
    should_be: str | None = None,
    correct_class: str | None = None,
    confidence: float | None = None,
    reader: str = "",
    comment: str = "",
) -> dict:
    """Append one correction and return it.

    `should_be` left `None` means the reading was fine and only the class was
    wrong; `correct_class` left `None` means the reverse. Both may be set.
    """
    kinds = []
    if correct_class and correct_class != region_class:
        kinds.append("class")
    if should_be is not None and should_be.strip() != read_as.strip():
        kinds.append("reading")
    entry = {
        "id": uuid.uuid4().hex[:10],
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note_id": note_id,
        "block_id": block_id,
        "box": [int(v) for v in box],
        "region_class": region_class,
        "correct_class": correct_class or region_class,
        "confidence": confidence,
        "reader": reader,
        "read_as": read_as,
        "should_be": should_be,
        "comment": comment,
        "kind": "+".join(kinds) or "confirmed",
    }
    path = Path(path or CORRECTIONS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def load(path: str | Path | None = None) -> list[dict]:
    path = Path(path or CORRECTIONS_PATH)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written line never invalidates the rest
    return out


def normalise(text: str) -> str:
    """Whitespace-insensitive comparison. `x^2 = 49` and `x^2=49` are the same
    reading, and counting them as different would make the score meaningless."""
    return "".join((text or "").split()).lower()


def report(path: str | Path | None = None) -> dict:
    """What the corrections say, split the two ways that need different work."""
    entries = load(path)
    readings = [e for e in entries if "reading" in e["kind"]]
    classes = [e for e in entries if "class" in e["kind"]]
    confirmed = [e for e in entries if e["kind"] == "confirmed"]

    wrong_class = {}
    for entry in classes:
        key = f"{entry['region_class']} -> {entry['correct_class']}"
        wrong_class[key] = wrong_class.get(key, 0) + 1

    seen = len(entries)
    return {
        "entries": seen,
        "confirmed": len(confirmed),
        "reading_errors": len(readings),
        "class_errors": len(classes),
        # Only meaningful once blocks are confirmed as well as corrected: a
        # file of nothing but complaints scores 0% by construction.
        "reading_accuracy": (len(confirmed) / seen) if seen else None,
        "confusions": dict(sorted(wrong_class.items(), key=lambda kv: -kv[1])),
        "by_class": {c: sum(1 for e in entries if e["region_class"] == c)
                     for c in sorted({e["region_class"] for e in entries})},
    }


def eval_set(path: str | Path | None = None) -> list[dict]:
    """Regions whose correct text is known: the reader's growing test set."""
    return [e for e in load(path) if e.get("should_be")]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", default=str(CORRECTIONS_PATH))
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for entry in load(args.path):
            print(f"{entry['at']}  {entry['kind']:14s} {entry['region_class']:11s} "
                  f"{entry['read_as'][:40]!r} -> {str(entry['should_be'])[:40]!r}")
    print(json.dumps(report(args.path), indent=1))


if __name__ == "__main__":
    main()
