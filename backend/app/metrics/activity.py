"""
What the server is actually doing, written down so something else can watch.

`palimtop watch` replays a dataset through a checkpoint and measures that.
It is a faithful measurement of a thing nobody asked for: the interesting
load is the page someone just compiled in the browser, and the two processes
share no memory, so the monitor could not see it.

This is the smallest thing that fixes it. The compiler appends one line per
region to a JSONL file; anything that wants to watch tails the file. No
socket, no broker, no second port, and the server is unaffected if nothing is
watching, which is the normal case.

**Append-only, with a cap.** A long session would otherwise grow without
bound. When the file passes `MAX_LINES` it is rewritten holding the most
recent half. A reader that is mid-tail sees the file shrink and starts again
from the end, which costs it one frame.

**A failed write is never allowed to fail a compile.** Monitoring is not
worth a 500 on someone's page, so every error here is swallowed. The cost of
that choice is that a missing line is indistinguishable from an idle server,
which is the right way round.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

ACTIVITY_PATH = Path(os.environ.get("PALIMPSEST_ACTIVITY", "runs/activity.jsonl"))
MAX_LINES = 5000


class ActivityLog:
    """One file, appended by the server and tailed by the monitor."""

    def __init__(self, path: str | Path | None = None, max_lines: int = MAX_LINES):
        # Resolved here rather than as a signature default, which would be
        # evaluated once at import and ignore anything that redirects the
        # constant afterwards.
        self.path = Path(path or ACTIVITY_PATH)
        self.max_lines = max_lines

    def append(self, **record) -> None:
        record.setdefault("t", time.time())
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            # Cheap check: only count lines once the file is plausibly large.
            if self.path.stat().st_size > self.max_lines * 160:
                self._trim()
        except Exception:
            pass  # see the module docstring: never fail a compile over this

    def _trim(self) -> None:
        try:
            lines = self.path.read_text().splitlines()
            if len(lines) > self.max_lines:
                keep = lines[-(self.max_lines // 2):]
                self.path.write_text("\n".join(keep) + "\n")
        except Exception:
            pass

    def read(self, offset: int = 0) -> tuple[list[dict], int]:
        """Records from `offset` bytes on, and the new offset.

        Returns `(records, offset)`. If the file has shrunk below `offset`,
        it was trimmed under us, so reading restarts from the beginning.
        """
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], 0
        if size < offset:
            offset = 0
        records = []
        try:
            with self.path.open() as handle:
                handle.seek(offset)
                for line in handle:
                    if not line.endswith("\n"):
                        break  # a line still being written; pick it up next time
                    offset += len(line.encode())
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            return [], offset
        return records, offset

    def end(self) -> int:
        """The current end of the file, for a reader that wants only new work."""
        try:
            return self.path.stat().st_size
        except OSError:
            return 0

    def clear(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass
