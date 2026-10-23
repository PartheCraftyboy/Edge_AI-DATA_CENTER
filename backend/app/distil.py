"""
Teaching the classifier from the reader.

    python -m app.distil --rounds 3
    python -m app.distil --images data/raw/iam/pages --rounds 3

The classifier is trained on synthetic pages and scores 0.99 on them. On
crops cut from real pages it scored **0 out of 6** against hand labels: every
line of ordinary handwriting came back FORMULA, which is why names and
ordinary words were being typeset as mathematics. The synthetic generator
only ever draws a `THEORY` region as a dense multi-line paragraph, so a
single line or a single word has no class the model recognises and falls into
the nearest thing that is short and symbol-shaped.

Hand labelling fixes that and is slow. This is the other way: the
vision-language model that already reads the regions also *labels* them, and
the classifier is trained on its votes. On the same six crops the teacher
scored 6 of 6. It is not an oracle and this module never pretends it is, but
a teacher that is right most of the time, applied to unlimited unlabelled
pages, moves a student that is currently right none of the time.

**The teacher reads before it labels.** Asked for the class alone, the same
model scored 3 of 6. Made to transcribe first and then classify, 6 of 6. See
`vlm.ollama._ClassReply` for why, and for the detail that makes it work: the
transcription is often wrong and the label is right anyway.

**Disagreements are penalised, and the penalty anneals itself away.** Where
the student contradicts the teacher, that crop's loss is scaled up, so
training concentrates on exactly the regions the two models read differently
and relaxes into ordinary cross-entropy as they converge.

**Rounds stop when agreement stops moving.** Agreement with the teacher on
held-out pages is the number that drives the loop, and it is reported round
by round rather than asserted at the end.

What this does not do is make the teacher right. Every vote is written to
`votes.json` with the transcription that produced it, so a pseudo-label can
be read back and argued with. Hand labels, where they exist, are never
overwritten: they are kept as the held-out truth that says whether any of
this worked.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from . import CLASS_NAMES, CLASS_INDEX
from .data.real import CROP_SIZE, PageStore, cut_crop, split_for_page
from .history import append_run

TEACHER_ROOT = Path(os.environ.get("PALIMPSEST_TEACHER", "data/teacher"))
# The teacher sees the region at its own shape, not the classifier's square.
# A line of prose is largely recognisable by being long and short, and
# squashing it into 224x224 throws that away: it cost three of six crops.
TEACHER_PAD = 0.04

# How many times a training crop is repeated, by how much it is worth
# learning from. Repetition rather than a loss weight because the dataset
# is already built by copying jittered boxes, so this needs no change to
# the batch contract, and `build_real_dataset` sets the same precedent.
#
# The ordering is the point. A crop a person **corrected** is the most
# informative thing in the set: it is ground truth *and* a place the model
# was demonstrably wrong. A crop a person **confirmed** is ground truth the
# model already agrees with, so it is worth more than a guess and less than
# a correction. A teacher vote is a guess by a 3B model that is right about
# three quarters of the time.
#
# Which is which needs no bookkeeping: a hand label that differs from the
# teacher's vote is a correction, and one that matches is a confirmation.
CORRECTED_COPIES = 6
CONFIRMED_COPIES = 3


def native_crop(grey: np.ndarray, box, pad: float = TEACHER_PAD) -> np.ndarray | None:
    """A region at its own aspect ratio, padded, for the teacher to read."""
    x, y, w, h = (int(v) for v in box)
    height, width = grey.shape[:2]
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(width, x + w + px), min(height, y + h + py)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return grey[y0:y1, x0:x1]


# ---------------------------------------------------------------- harvesting

def boxes_for_page(page: dict, grey: np.ndarray) -> list[list[int]]:
    """Regions to label on one page.

    Hand-drawn boxes win where they exist. They are better boxes than the
    segmenter's, and a page that has been labelled is worth more as truth
    than as another handful of pseudo-labels.
    """
    if page.get("labelled") and page.get("regions"):
        return [list(r["box"]) for r in page["regions"]]
    from .pipeline.segment import segment
    from .pipeline.preprocess import preprocess

    return [list(b) for b in segment(preprocess(grey, deskew=False).ink)]


def as_dirs(extra_dir) -> list[Path]:
    """`extra_dir` as a list, whether it arrived as one path or several."""
    if extra_dir is None:
        return []
    if isinstance(extra_dir, (str, Path)):
        return [Path(extra_dir)]
    return [Path(d) for d in extra_dir]


def page_images(store: PageStore, extra_dir=None):
    """`(page_id, split, grey, boxes, hand_labels)` for everything to teach on.

    `extra_dir` is a folder of page images with no labels of any kind, which
    is how a public handwriting corpus joins in: it needs no annotation,
    because the teacher supplies the only labels this module uses.

    **Several folders go in one call, and that is not a convenience.** Run
    as one pass per corpus, each pass starts again from the synthetic
    checkpoint and writes the same `best.ckpt`, so the second silently
    replaces the first rather than building on it: training on GNHK then on
    IAM shipped a model that had seen only IAM. The camera-captured corpus,
    which is the one that looks like what the app is actually given, was the
    one discarded. Passing every folder here builds one dataset from all of
    them and trains once.
    """
    for page in store.all():
        grey = cv2.imread(str(store.image_path(page["id"])), cv2.IMREAD_GRAYSCALE)
        if grey is None:
            continue
        hand = {tuple(r["box"]): r["class"] for r in page["regions"]} if page.get("labelled") else {}
        yield page["id"], page["split"], grey, boxes_for_page(page, grey), hand

    for folder in as_dirs(extra_dir):
        from .pipeline.preprocess import preprocess
        from .pipeline.segment import segment

        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
                continue
            raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            pre = preprocess(raw, deskew=True)
            page_id = f"ext-{path.stem}"
            yield page_id, split_for_page(page_id), pre.grey, [list(b) for b in segment(pre.ink)], {}


def release_teacher(reader) -> bool:
    """Give the teacher's VRAM back before anything else asks for it.

    Measured on the RTX 4050: qwen2.5vl:3b holds 4196 MiB of 6141, and
    Ollama's default `keep_alive` is ten minutes. Training starts seconds
    after the last vote, so without this the first ten minutes of every run
    train a CNN in the 1576 MiB left over — which is slow when it fits and an
    OOM when it does not. The teacher has finished by then and is not needed
    again until the next round's votes, which are nearly all cached.
    """
    for release in (getattr(reader, "unload", None),
                    getattr(reader, "close", None)):
        if callable(release):
            try:
                release()
                return True
            except Exception:
                continue
    return False


def collect_votes(store: PageStore, extra_dir=None, reader=None,
                  cache: Path = TEACHER_ROOT / "votes.json", limit: int | None = None) -> dict:
    """Ask the teacher about every region, and remember what it said.

    Cached by page and box, because the teacher costs about 1.2 s a crop and
    a round that re-asked everything would spend its whole budget re-deriving
    labels it already had.
    """
    cache.parent.mkdir(parents=True, exist_ok=True)
    votes = json.loads(cache.read_text()) if cache.exists() else {}
    if reader is None:
        from .vlm import make_judge

        # The judge, not the reader. `PALIMPSEST_READER=paddle` is the right
        # setting for transcription and PaddleOCR-VL cannot classify at all.
        reader = make_judge()

    # Counting the regions up front costs one segmentation pass over the
    # pages and buys a denominator. Without one the only honest progress line
    # is a rising number with nothing to compare it to, and this pass takes
    # minutes: silence for that long is indistinguishable from a hang, which
    # is exactly how the first version of this was read.
    where = ", ".join(str(d) for d in as_dirs(extra_dir)) or "the page store"
    print(f"segmenting pages in {where}…", flush=True)
    pages = list(page_images(store, extra_dir))
    todo = sum(1 for page_id, _, _, boxes, _ in pages
               for box in boxes
               if f"{page_id}|{','.join(str(int(v)) for v in box)}" not in votes)
    if limit is not None:
        todo = min(todo, limit)
    print(f"teacher: {len(pages)} pages, {todo} regions to label"
          f"{f' ({len(votes)} already cached)' if votes else ''}", flush=True)

    asked = failed = 0
    tally: dict[str, int] = {}
    reasons: dict[str, int] = {}
    began = time.time()
    for page_id, split, grey, boxes, hand in pages:
        for box in boxes:
            key = f"{page_id}|{','.join(str(int(v)) for v in box)}"
            if key in votes:
                continue
            if limit is not None and asked >= limit:
                break
            crop = native_crop(grey, box)
            if crop is None:
                continue
            try:
                region_class, transcription = reader.classify_region(crop)
            except Exception as exc:  # one bad crop must not end the pass
                failed += 1
                # Say why, once per distinct reason. Counting failures and
                # discarding their cause turned a context-window overrun into
                # "414 failed" with nothing to go on, which is a worse bug
                # than the overrun was.
                reason = f"{type(exc).__name__}: {exc}"[:160]
                if reason not in reasons:
                    reasons[reason] = 0
                    print(f"  ! {reason}", flush=True)
                reasons[reason] += 1

                # Reset the model. A vision model can enter a state where it
                # returns nothing to every request — Ollama reports it as
                # "token repeat limit reached", and it survives the next
                # request and the one after. One oversized region did that
                # here and the following 414 regions all failed against the
                # same poisoned runner, which looked like 414 bad crops and
                # was one. A reload costs about four seconds; not reloading
                # costs the rest of the pass.
                release_teacher(reader)
                continue
            asked += 1
            tally[region_class] = tally.get(region_class, 0) + 1
            votes[key] = {
                "page": page_id, "split": split, "box": [int(v) for v in box],
                "class": region_class, "transcription": transcription[:300],
                "hand": hand.get(tuple(box)),
            }
            if asked % 10 == 0:
                cache.write_text(json.dumps(votes, indent=1))

            elapsed = time.time() - began
            rate = asked / elapsed if elapsed > 0 else 0.0
            left = (todo - asked) / rate if rate > 0 and todo > asked else 0.0
            mix = " ".join(f"{name[:4]}={n}" for name, n in sorted(tally.items()))
            print(f"  [{asked:4d}/{todo}] {100 * asked / max(1, todo):5.1f}%  "
                  f"{rate:4.2f}/s  eta {int(left // 60):02d}:{int(left % 60):02d}  "
                  f"{region_class:11s} {mix}", flush=True)
    cache.write_text(json.dumps(votes, indent=1))
    freed = release_teacher(reader)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  ! {count} x {reason}", flush=True)
    return {"votes": votes, "asked": asked, "failed": failed,
            "total": len(votes), "freed_teacher": freed, "reasons": reasons}


# ------------------------------------------------------------------ dataset

def copies_for(record: dict, jitter_copies: int) -> int:
    """How many times this crop goes into the training set.

    A correction outweighs a confirmation outweighs a teacher vote. See
    `CORRECTED_COPIES` for why, and note that nothing has to be looked up
    to tell them apart: disagreeing with the teacher is what makes a hand
    label a correction.
    """
    if not record.get("hand"):
        return jitter_copies
    return CORRECTED_COPIES if record["hand"] != record["class"] else CONFIRMED_COPIES


def build_teacher_dataset(votes: dict, store: PageStore, out_root: str | Path,
                          extra_dir=None, jitter_copies: int = 2, seed: int = 0) -> dict:
    """Write `crops/{split}_{crops,labels,hard}.npy` from the teacher's votes.

    Train gets the teacher's labels. **Validation and test prefer hand labels**,
    because a score against the teacher only says the student learned to copy
    the teacher, and the question worth answering is whether it learned to
    read the page.

    When a split has no hand labels at all, it falls back to teacher labels
    rather than coming back empty. That is a weaker split and the manifest
    says so in `honest_val`, which `history.csv` carries, because a run whose
    validation score is agreement-with-a-teacher must never be read later as
    though it were accuracy. The alternative is worse: an empty validation
    set makes Lightning's checkpoint monitor fail, so the whole corpus of
    unlabelled pages could not be used at all until someone hand-labels a
    page that happens to hash into the validation split.
    """
    from .data.real import jitter

    rng = np.random.default_rng(seed)
    out_root = Path(out_root)
    (out_root / "crops").mkdir(parents=True, exist_ok=True)
    buckets = {s: ([], []) for s in ("train", "val", "test")}
    greys = {pid: grey for pid, _, grey, _, _ in page_images(store, extra_dir)}

    # Which splits have any truth in them at all. Decided before the loop,
    # because whether a teacher-labelled region belongs in validation depends
    # on whether a hand-labelled one exists to take its place.
    hand_in_split = {s: any(v["hand"] for v in votes.values() if v["split"] == s)
                     for s in ("train", "val", "test")}

    taught = handed = 0
    for record in votes.values():
        grey = greys.get(record["page"])
        if grey is None:
            continue
        split = record["split"]
        # Truth beats the teacher wherever we have it.
        label = record["hand"] or record["class"]
        if split != "train" and not record["hand"] and hand_in_split[split]:
            continue
        handed += bool(record["hand"])
        taught += not record["hand"]
        boxes = [record["box"]]
        if split == "train":
            boxes += [jitter(record["box"], rng)
                      for _ in range(copies_for(record, jitter_copies))]
        for box in boxes:
            crop = cut_crop(grey, box)
            if crop is not None:
                buckets[split][0].append(crop)
                buckets[split][1].append(CLASS_INDEX[label])

    counts = {}
    for split, (crops, labels) in buckets.items():
        array = np.asarray(crops, np.uint8).reshape(-1, CROP_SIZE, CROP_SIZE)
        np.save(out_root / "crops" / f"{split}_crops.npy", array)
        np.save(out_root / "crops" / f"{split}_labels.npy", np.asarray(labels, np.int64))
        np.save(out_root / "crops" / f"{split}_hard.npy", np.zeros(len(labels), bool))
        counts[split] = len(labels)

    corrected = sum(1 for v in votes.values() if v.get("hand") and v["hand"] != v["class"])
    confirmed = sum(1 for v in votes.values() if v.get("hand") and v["hand"] == v["class"])
    manifest = {"counts": counts, "teacher_labels": taught, "hand_labels": handed,
                "hand_corrected": corrected, "hand_confirmed": confirmed,
                "honest_val": hand_in_split["val"], "honest_test": hand_in_split["test"],
                "per_class": {name: sum(1 for v in votes.values() if (v["hand"] or v["class"]) == name)
                              for name in CLASS_NAMES}}
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------ scoring

def agreement(checkpoint: Path, votes: dict, store: PageStore, extra_dir=None,
              device: str | None = None) -> dict:
    """How often the student now says what the teacher said, and what a person said.

    Two numbers, deliberately. Agreement with the teacher is what the loss is
    driving at and what says whether the rounds have converged. Accuracy
    against hand labels is the one that says whether converging was worth
    anything, and it is the only one to quote.
    """
    import torch

    from .evaluate import load_checkpoint
    from .pipeline.compile import to_tensor

    greys = {pid: grey for pid, _, grey, _, _ in page_images(store, extra_dir)}
    crops, teacher, hand = [], [], []
    for record in votes.values():
        grey = greys.get(record["page"])
        crop = cut_crop(grey, record["box"]) if grey is not None else None
        if crop is None:
            continue
        crops.append(crop)
        teacher.append(CLASS_INDEX[record["class"]])
        hand.append(CLASS_INDEX[record["hand"]] if record["hand"] else -1)
    if not crops:
        return {"samples": 0}

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(checkpoint, device=device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(crops), 64):
            batch = to_tensor(crops[start:start + 64]).to(device)
            predictions.append(torch.softmax(model(batch), dim=1).argmax(1).cpu().numpy())
    predicted = np.concatenate(predictions)
    teacher, hand = np.asarray(teacher), np.asarray(hand)
    scored = hand >= 0
    return {
        "samples": len(crops),
        "teacher_agreement": float((predicted == teacher).mean()),
        "hand_samples": int(scored.sum()),
        "hand_accuracy": float((predicted[scored] == hand[scored]).mean()) if scored.any() else None,
        "teacher_vs_hand": float((teacher[scored] == hand[scored]).mean()) if scored.any() else None,
    }


# ------------------------------------------------------------------- rounds

def distil(backbone: str = "squeezenet1_1", rounds: int = 3, epochs: int = 6,
           lr: float = 1e-4, penalty: float = 3.0, extra_dir=None,
           plateau: float = 0.01, limit: int | None = None) -> dict:
    """Label with the teacher, train against it, repeat until agreement settles."""
    from .train import train

    store = PageStore()
    page_stats = store.stats()
    harvest = collect_votes(store, extra_dir, limit=limit)
    votes = harvest["votes"]
    print(f"teacher: {harvest['total']} votes ({harvest['asked']} new, {harvest['failed']} failed)"
          + ("; VRAM released for training" if harvest.get("freed_teacher") else ""))

    data_root = Path(os.environ.get("PALIMPSEST_TEACHER_DATASET", "data/teacher_dataset"))
    manifest = build_teacher_dataset(votes, store, data_root, extra_dir)
    print(f"dataset: {manifest['counts']}  "
          f"({manifest['teacher_labels']} taught, {manifest['hand_labels']} hand: "
          f"{manifest['hand_corrected']} corrected x{CORRECTED_COPIES}, "
          f"{manifest['hand_confirmed']} confirmed x{CONFIRMED_COPIES})")
    if not manifest["honest_val"]:
        print("note: validation is teacher-labelled, not hand-labelled. The val score "
              "is agreement with the reader, not accuracy; hand_accuracy below is the "
              "number to read.")

    # Fail here rather than inside Lightning. An empty validation split leaves
    # ModelCheckpoint with nothing to monitor, and it reports that as
    # `.test(ckpt_path="best")` failing several minutes later, which is a long
    # way from the cause. Splits are a stable hash of the page id, so too few
    # pages means every one of them landed in train.
    if not manifest["counts"]["val"]:
        pages = len({v["page"] for v in votes.values()})
        raise SystemExit(
            f"no validation crops: all {pages} page(s) hashed into the train split.\n"
            f"Splits are fixed per page at about 70/15/15, so this means too few pages. "
            f"Add more with --images, or ingest a corpus:\n"
            f"    python -m app.data.gnhk --limit 200\n"
            f"    python -m app.data.iam  --limit 150")

    # The page images are no longer needed as a block: the crops are on disk
    # now, and `agreement` reloads only what it scores. Dropping the
    # reference here keeps the parent small before any fork.
    import gc

    gc.collect()

    out_dir = Path(os.environ.get("PALIMPSEST_RUNS_DISTIL", "runs_distil"))
    base = Path("runs") / backbone / "best.ckpt"
    def record(round_index: int, scores: dict, seconds=None) -> dict:
        """One row in `history.csv`, and one line on stdout."""
        row = {"round": round_index, **scores}
        append_run(run="distil", round=round_index, backbone=backbone,
                   pages_labelled=page_stats["labelled_pages"],
                   regions_hand=manifest["hand_labels"],
                   regions_teacher=manifest["teacher_labels"],
                   regions_total=manifest["hand_labels"] + manifest["teacher_labels"],
                   train_crops=manifest["counts"]["train"],
                   val_crops=manifest["counts"]["val"],
                   test_crops=manifest["counts"]["test"],
                   hand_accuracy=scores.get("hand_accuracy"),
                   hand_samples=scores.get("hand_samples"),
                   teacher_agreement=scores.get("teacher_agreement"),
                   teacher_vs_hand=scores.get("teacher_vs_hand"),
                   epochs=epochs, lr=lr, disagreement_penalty=penalty,
                   train_seconds=seconds,
                   note=("synthetic start; " if round_index == 0 else "")
                        + ("val is hand-labelled" if manifest.get("honest_val")
                           else "val is teacher-labelled, not truth"),
                   **{f"n_{name.lower()}": manifest["per_class"][name] for name in CLASS_NAMES})
        return row

    history = [record(0, agreement(base, votes, store, extra_dir))]
    print(f"round 0 (synthetic): {history[-1]}")

    checkpoint = base
    for round_index in range(1, rounds + 1):
        # num_workers=0 on purpose. A distilled split is a few hundred crops
        # already in memory as one array, so a worker buys no overlap worth
        # having, and each one forks a parent that is holding every page
        # image of the corpus. On a 22 GB machine with a browser open that
        # was enough for the OS to kill both workers mid-epoch, which
        # surfaces only as "DataLoader worker exited unexpectedly" with no
        # traceback of its own.
        results = train(backbone=backbone, data_root=data_root, out_dir=out_dir, epochs=epochs,
                        batch_size=32, lr=lr, num_workers=0, init_checkpoint=checkpoint,
                        disagreement_penalty=penalty)
        checkpoint = out_dir / backbone / "best.ckpt"
        history.append(record(round_index, agreement(checkpoint, votes, store, extra_dir),
                              results.get("train_seconds")))
        print(f"round {round_index}: {history[-1]}")
        moved = history[-1]["teacher_agreement"] - history[-2]["teacher_agreement"]
        if round_index > 1 and moved < plateau:
            print(f"agreement moved {moved:+.3f}, below the {plateau} plateau: stopping")
            break

    # Write eval.json beside the checkpoint, as a training run does. Without
    # it the app can list this model but not show its metrics panel, and
    # /api/metrics answers 404 for the very model it has selected — which is
    # how a distilled checkpoint became the default and immediately broke
    # the page that reports on it.
    try:
        from .evaluate import evaluate as evaluate_checkpoint

        report = evaluate_checkpoint(checkpoint, data_root, split="test")
        (Path(checkpoint).parent / "eval.json").write_text(json.dumps(report, indent=1))
        print(f"wrote {Path(checkpoint).parent / 'eval.json'}", flush=True)
    except Exception as exc:  # a missing report must not fail a good run
        print(f"  ! could not write eval.json: {type(exc).__name__}: {exc}", flush=True)

    summary = {"backbone": backbone, "history": history, "dataset": manifest,
               "penalty": penalty, "checkpoint": str(checkpoint)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "distil.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", default="squeezenet1_1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--penalty", type=float, default=3.0,
                        help="loss multiplier on crops the student and teacher disagree on")
    parser.add_argument("--images", nargs="+", default=None,
                        help="folders of unlabelled page images to teach on as well. "
                             "Give every corpus to one run: a separate run per corpus "
                             "overwrites the previous one's checkpoint instead of "
                             "building on it.")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many new teacher calls, for a quick pass")
    args = parser.parse_args()
    summary = distil(args.backbone, args.rounds, args.epochs, args.lr,
                     penalty=args.penalty, extra_dir=args.images, limit=args.limit)
    print(json.dumps(summary["history"], indent=1))


if __name__ == "__main__":
    main()
