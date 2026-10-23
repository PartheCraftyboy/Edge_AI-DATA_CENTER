"""
Notebooks: the shelf the compiled notes sit on.

The design reference (`design/explorations/02-scriptorium.html`) has two
lists in the sidebar and the build only ever had one. The reference shows
**Notebooks** — I. The vision problem, II. First vision model, III.
Architectures — each with a count, and then **This notebook**, the notes
inside the one selected. The build flattened that into a single list of
every note ever compiled, which is the same information in the way that a
pile is the same information as a shelf.

It matters more than it looks. A term's notes are a few hundred pages across
half a dozen subjects, and a flat list ordered by compile time puts last
Tuesday's queueing theory between two pages of linear algebra. The grouping
is the thing that makes the vault usable at the scale it is meant for.

**A notebook is a name and an ordinal, nothing else.** No nesting, no tags,
no colour. Notes carry `notebook_id`; everything else is derived by reading
the vault. That keeps the store a directory of self-describing notes, which
is what makes the vault inspectable with `ls` and recoverable with a text
editor, and it means a note whose notebook has been deleted is still a note.

**Notes that predate this are not lost.** Anything without a `notebook_id`
reads as belonging to the first notebook, so the vault needs no migration
and a half-migrated vault is not a state that exists.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
         "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"]

DEFAULT_NAME = "First notebook"


def numeral(index: int) -> str:
    """`I`, `II`, … and plain numbers past the twentieth, where roman stops
    being a label and starts being a puzzle."""
    return ROMAN[index] if index < len(ROMAN) else str(index + 1)


class NotebookStore:
    """The list of notebooks, in one JSON file beside the notes."""

    def __init__(self, vault: str | Path):
        self.vault = Path(vault)
        self.path = self.vault / "notebooks.json"

    # ------------------------------------------------------------- storage

    def _read(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def _write(self, rows: list[dict]) -> None:
        self.vault.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=1))
        tmp.replace(self.path)  # atomic, so a reader never sees half a file

    # --------------------------------------------------------------- query

    def all(self) -> list[dict]:
        """Every notebook, in order. Creates the first one if there are none,
        so there is always somewhere for a note to go."""
        rows = self._read()
        if not rows:
            rows = [self._new(DEFAULT_NAME)]
            self._write(rows)
        return rows

    def default_id(self) -> str:
        return self.all()[0]["id"]

    def exists(self, notebook_id: str) -> bool:
        return any(n["id"] == notebook_id for n in self.all())

    # -------------------------------------------------------------- change

    def _new(self, name: str) -> dict:
        return {"id": uuid.uuid4().hex[:8], "name": name.strip() or DEFAULT_NAME,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def add(self, name: str) -> dict:
        rows = self.all()
        notebook = self._new(name)
        rows.append(notebook)
        self._write(rows)
        return notebook

    def rename(self, notebook_id: str, name: str) -> dict:
        rows = self.all()
        for row in rows:
            if row["id"] == notebook_id:
                row["name"] = name.strip() or row["name"]
                self._write(rows)
                return row
        raise KeyError(notebook_id)

    def remove(self, notebook_id: str) -> None:
        """Delete a notebook. Its notes are kept and fall back to the first.

        Deleting a shelf is not deleting the books on it. The notes stay in
        the vault with a `notebook_id` that no longer resolves, and `listing`
        files those under the first notebook rather than hiding them.
        """
        rows = [n for n in self.all() if n["id"] != notebook_id]
        if not rows:
            raise ValueError("the last notebook cannot be deleted")
        self._write(rows)


def slug(title: str) -> str:
    """A short label for a note in the sidebar."""
    return re.sub(r"\s+", " ", (title or "").strip())[:60] or "Untitled"


def listing(store: NotebookStore, notes: list[dict]) -> list[dict]:
    """Notebooks with their numeral, their note count, and their notes.

    `notes` is whatever `pipeline.compile.list_vault` returned, so this holds
    no opinion about how a note is stored. A note whose `notebook_id` is
    missing or dangling lands in the first notebook rather than vanishing.
    """
    books = store.all()
    known = {n["id"] for n in books}
    first = books[0]["id"]
    grouped: dict[str, list[dict]] = {n["id"]: [] for n in books}
    for note in notes:
        key = note.get("notebook_id")
        grouped[key if key in known else first].append(note)
    return [
        {**book, "numeral": numeral(i), "count": len(grouped[book["id"]]),
         "notes": grouped[book["id"]]}
        for i, book in enumerate(books)
    ]
