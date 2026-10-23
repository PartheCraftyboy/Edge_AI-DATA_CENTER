"""
IAM — the clean counterpart to GNHK.

    python -m app.data.iam --limit 150
    python -m app.distil --images data/raw/iam/pages --rounds 3

GNHK is photographs: shadow, skew, show-through, a phone held over a page.
IAM is flatbed scans of forms, evenly lit and square to the frame. Having
both is the point. A classifier that scores the same on each has learned
something about handwriting; one that scores well on IAM and badly on GNHK
has learned something about scanners, and this project has been caught by
exactly that once already, when synthetic pages gave 0.99 and real crops
gave 0 of 6.

1,539 pages, 657 writers. Each form carries the printed sentence the writer
was asked to copy above their handwritten version of it, which is worth
knowing before reading the teacher's votes: a region of machine print is not
one of the five classes this project has, and the teacher will have to call
it something.

**Licence.** IAM is free for non-commercial research and the maintainers ask
users to register and to cite Marti and Bunke (IJDAR 2002). The Hugging Face
mirror read here is tagged apache-2.0, which does not override the original
terms. For a course submission that is fine; anything else needs the real
registration at fki.tic.heia-fr.ch.

**Streamed, and stopped early.** The form archives are 1.6 GB each and a
bounded ingest should not pull one to get a hundred pages. A gzip stream is
sequential, so `tarfile` in streaming mode reads members in order and this
stops as soon as it has enough, transferring roughly what it keeps.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import numpy as np

REPO = os.environ.get("PALIMPSEST_IAM_REPO", "zikotone/iam-form")
# Alphabetical by writer. One archive is plenty for a bounded ingest, and
# taking them in order keeps a run reproducible.
ARCHIVES = ("formsA-D.tgz", "formsE-H.tgz", "formsI-Z.tgz")
DEFAULT_OUT = Path(os.environ.get("PALIMPSEST_IAM", "data/raw/iam"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def ingest(out_root: str | Path = DEFAULT_OUT, limit: int = 150,
           min_side: int = 400) -> dict:
    """Write up to `limit` IAM form pages as PNGs, and a manifest beside them."""
    import cv2
    from huggingface_hub import HfFileSystem

    out_root = Path(out_root)
    pages = out_root / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    written, skipped, index = 0, 0, []
    for archive in ARCHIVES:
        if written >= limit:
            break
        with fs.open(f"datasets/{REPO}/{archive}", "rb") as handle:
            # "r|gz" is the streaming mode: members come in file order and
            # nothing seeks backwards, so the transfer stops when we stop.
            with tarfile.open(fileobj=handle, mode="r|gz") as tar:
                for member in tar:
                    if written >= limit:
                        break
                    if not member.isfile():
                        continue
                    if Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    grey = cv2.imdecode(np.frombuffer(extracted.read(), np.uint8),
                                        cv2.IMREAD_GRAYSCALE)
                    if grey is None or min(grey.shape) < min_side:
                        skipped += 1
                        continue
                    name = Path(member.name).stem
                    cv2.imwrite(str(pages / f"{name}.png"), grey)
                    index.append({"page": f"{name}.png", "archive": archive,
                                  "h": int(grey.shape[0]), "w": int(grey.shape[1])})
                    written += 1

    manifest = {
        "source": REPO,
        "licence": "IAM: free for non-commercial research; cite Marti & Bunke, IJDAR 2002",
        "pages": written,
        "skipped": skipped,
    }
    (out_root / "manifest.json").write_text(json.dumps({**manifest, "index": index}, indent=1))
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=150)
    args = parser.parse_args()
    manifest = ingest(args.out, args.limit)
    print(json.dumps(manifest, indent=1))
    print(f"\nnext:  python -m app.distil --images {Path(args.out) / 'pages'} --rounds 3")


if __name__ == "__main__":
    main()
