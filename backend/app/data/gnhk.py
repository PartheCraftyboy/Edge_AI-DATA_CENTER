"""
GNHK — real pages of handwriting, photographed rather than scanned.

    python -m app.data.gnhk --limit 200
    python -m app.distil --images data/raw/gnhk/pages --rounds 3

The classifier has never seen a page of real handwriting. It was trained on
synthetic layouts whose `THEORY` regions are always dense multi-line
paragraphs, and measured on crops from real pages it scored 0 of 6. The two
corpora already here cannot fix that: Chars74K is isolated characters and
the handwriting-recognition set is word crops cut from forms. Neither has a
*paragraph*, and neither has a page.

GNHK does, and it is the right corpus for this project for a reason beyond
convenience. It is **camera-captured** — photographs of real handwritten
notes, with the lighting, skew, shadow, perspective and cluttered background
of a phone held over a notebook. IAM is the better-known choice and is
flatbed scans: cleaner than anything this pipeline will ever be given, and
the project's own measurements keep showing that clean data is what hides
the failures. Training robustness on scans would repeat the mistake the
synthetic generator already made.

687 pages, CC-BY-4.0, from GoodNotes, published at ICDAR 2021.

**No annotation is needed from it.** GNHK labels text lines, not the five
region classes this project uses, so none of its labels are read here. The
pages arrive unlabelled and `app.distil` has the vision-language model label
them, which is the whole point: a corpus that needs no hand labelling is a
corpus that can be used tonight.

**Read lazily, over HTTP range requests.** The parquet shards are 730 MB and
a bounded ingest should not pull all of them. pyarrow reads row groups
through `HfFileSystem`, so taking 200 pages downloads roughly 200 pages.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np

REPO = os.environ.get("PALIMPSEST_GNHK_REPO", "bhavya777/GNHK-dataset")
SHARDS = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
DEFAULT_OUT = Path(os.environ.get("PALIMPSEST_GNHK", "data/raw/gnhk"))


def ingest(out_root: str | Path = DEFAULT_OUT, limit: int = 200,
           min_side: int = 400) -> dict:
    """Write up to `limit` GNHK pages as PNGs, and a manifest beside them.

    Pages smaller than `min_side` on their short edge are skipped: the
    pipeline's working size is 1600 px and a thumbnail would be upscaled into
    a blur that teaches the classifier nothing true about handwriting.
    """
    import cv2
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    out_root = Path(out_root)
    pages = out_root / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()

    written, skipped, index = 0, 0, []
    for shard in SHARDS:
        if written >= limit:
            break
        with fs.open(f"datasets/{REPO}/{shard}", "rb") as handle:
            parquet = pq.ParquetFile(handle)
            for batch in parquet.iter_batches(batch_size=16, columns=["image", "text"]):
                rows = batch.to_pydict()
                for image, text in zip(rows["image"], rows["text"]):
                    if written >= limit:
                        break
                    name = Path(str((image or {}).get("path") or f"gnhk-{written:04d}")).stem
                    data = (image or {}).get("bytes")
                    if not data:
                        skipped += 1
                        continue
                    grey = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if grey is None or min(grey.shape) < min_side:
                        skipped += 1
                        continue
                    cv2.imwrite(str(pages / f"{name}.png"), grey)
                    index.append({"page": f"{name}.png", "h": int(grey.shape[0]),
                                  "w": int(grey.shape[1]),
                                  # Kept for reference only; `app.distil` does
                                  # not read it. GNHK marks mathematics with
                                  # %math%, which is a useful sanity check on
                                  # the teacher's FORMULA votes but is a line
                                  # annotation, not a region one.
                                  "text": (text or "")[:2000]})
                    written += 1
                if written >= limit:
                    break

    manifest = {"source": REPO, "licence": "CC-BY-4.0", "pages": written, "skipped": skipped}
    (out_root / "manifest.json").write_text(json.dumps({**manifest, "index": index}, indent=1))
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    manifest = ingest(args.out, args.limit)
    print(json.dumps(manifest, indent=1))
    print(f"\nnext:  python -m app.distil --images {Path(args.out) / 'pages'} --rounds 3")


if __name__ == "__main__":
    main()
