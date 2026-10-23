"""
The learning curve, kept across runs.

    python -m app.history --plot

Every training run appends one row to `history.csv`. Nothing else in the
project remembers anything across runs: `runs/<backbone>/results.json` is
overwritten by the next run of that backbone, and `runs_real/status.json` is
overwritten by the next fine-tune. This file is append-only, so after a week
of labelling it answers the question neither of those can: **is labelling
still buying anything?**

**The x axis is labelled regions, not epochs and not wall-clock.** Epochs
measure how hard the optimiser worked, which is not in question. The decision
this file exists to support is whether to label another fifty regions or
stop, so the curve has to be drawn against the number of regions labelled.
Plotted that way it is a saturation curve, and the point where it flattens is
the point where more labelling stops paying.

**Two accuracies, kept apart.** `hand_accuracy` is the student against labels
a person drew, and is the only number to quote. `teacher_agreement` is the
student against the vision-language model's votes, which says whether the
distillation rounds have converged and says nothing at all about whether they
converged on the truth. `teacher_vs_hand` is the teacher's own score, which is
the ceiling the student is being pulled towards: when it stops rising, more
rounds cannot help and the teacher is the thing to improve.

One row per run, so a run that crashes leaves no row and the file never needs
repairing. Columns are fixed: an unknown field is dropped rather than
silently shifting every column to the right of it in a file something else
is going to plot.
"""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path

from . import CLASS_NAMES

HISTORY_PATH = Path(os.environ.get("PALIMPSEST_HISTORY", "history.csv"))

FIELDS: tuple[str, ...] = (
    "timestamp",
    "run",            # 'finetune' or 'distil'
    "round",          # distillation round; 0 for the starting checkpoint
    "backbone",
    # ---- the x axis: how much labelling has happened
    "pages_labelled",
    "regions_hand",      # regions a person drew and named
    "regions_teacher",   # regions the vision-language model named
    "regions_total",
    "train_crops",       # after jitter copies and any synthetic mix-in
    "val_crops",
    "test_crops",
    # ---- what the student scores
    "hand_accuracy",      # against hand labels on held-out pages. The number.
    "hand_samples",
    "macro_f1",
    "accuracy",
    "ece",
    # ---- what the teacher is doing
    "teacher_agreement",  # student vs teacher: has distillation converged?
    "teacher_vs_hand",    # teacher vs person: the ceiling being pulled towards
    # ---- what was run
    "epochs",
    "lr",
    "disagreement_penalty",
    "synthetic_ratio",
    "train_seconds",
    "device",
    "note",
    # ---- class balance, because a curve that flattens is often a class with
    #      no examples rather than a model that has stopped learning
    *(f"n_{name.lower()}" for name in CLASS_NAMES),
)


def append_run(path: str | Path = HISTORY_PATH, **fields) -> Path:
    """Append one row. Creates the file with a header the first time."""
    path = Path(path)
    fields.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
    row = {key: fields.get(key, "") for key in FIELDS}
    row = {key: ("" if value is None else value) for key, value in row.items()}
    new = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path


def read_history(path: str | Path = HISTORY_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot(path: str | Path = HISTORY_PATH, out: str | Path = "history.png"):
    """The learning curve against labelled regions, plus the class balance.

    Drawn here rather than left as an exercise because a chart nobody can
    produce in one command is a chart nobody looks at.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_history(path)
    if not rows:
        raise SystemExit(f"{path} has no rows yet; run a training pass first")

    def series(key, rows):
        return [float(r[key]) if r.get(key) not in ("", None) else None for r in rows]

    # The x axis is regions labelled, which is the axis the decision lives
    # on. Until someone has labelled a second batch there is only one value
    # of it, and every point lands in a single vertical stack that reads as
    # a broken chart rather than as "no labelling has happened yet". So fall
    # back to run order and say on the chart which axis is being drawn,
    # rather than drawing the right axis uselessly.
    hand = series("regions_hand", rows)
    distinct = {v for v in hand if v is not None}
    by_labels = len(distinct) > 1
    x = hand if by_labels else list(range(1, len(rows) + 1))
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=(2, 1))

    for key, label, style in (
        ("hand_accuracy", "student vs hand labels", "-o"),
        ("teacher_vs_hand", "teacher vs hand labels (ceiling)", "--s"),
        ("teacher_agreement", "student vs teacher (convergence)", ":^"),
        ("macro_f1", "macro-F1", "-.d"),
    ):
        y = series(key, rows)
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if pairs:
            top.plot([p[0] for p in pairs], [p[1] for p in pairs], style, label=label)
    top.set_ylabel("score")
    top.set_ylim(-0.02, 1.05)
    if by_labels:
        top.set_xlabel("regions labelled by hand")
        top.set_title("Does more labelling still buy accuracy?")
    else:
        top.set_xlabel("training run (in order)")
        top.set_title("Scores per run"
                      + f"  \u2014 all at {int(next(iter(distinct))) if distinct else 0} hand labels,"
                        " so this is not a labelling curve yet")
    top.grid(alpha=0.3)
    top.legend(loc="lower right", fontsize=9)

    # Only the latest run's balance. Twenty near-identical grouped bars say
    # nothing that one says, and the question this panel answers — which
    # classes have no real examples at all — is about the data as it stands.
    latest = rows[-1]
    counts = [float(latest.get(f"n_{n.lower()}") or 0) for n in CLASS_NAMES]
    bars = bottom.bar(range(len(CLASS_NAMES)), counts,
                      color=["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c4407a"])
    for rect, value in zip(bars, counts):
        bottom.annotate(str(int(value)), (rect.get_x() + rect.get_width() / 2, value),
                        ha="center", va="bottom", fontsize=9)
    bottom.set_xticks(range(len(CLASS_NAMES)))
    bottom.set_xticklabels(CLASS_NAMES, fontsize=9)
    bottom.set_xlabel("a class with no real examples is carried by synthetic data alone")
    bottom.set_ylabel("regions")
    bottom.set_title("Class balance, latest run")
    bottom.grid(alpha=0.3, axis="y")

    figure.tight_layout()
    figure.savefig(out, dpi=140)
    return Path(out)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--path", default=str(HISTORY_PATH))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", default="history.png")
    args = parser.parse_args()
    rows = read_history(args.path)
    print(f"{len(rows)} runs in {args.path}")
    for row in rows:
        print(f"  {row['timestamp']}  {row['run']:9s} round {row['round'] or '-':>2}  "
              f"hand {row['regions_hand'] or '?':>4} regions  "
              f"hand_acc {row['hand_accuracy'] or '-':>6}  "
              f"teacher_agree {row['teacher_agreement'] or '-':>6}")
    if args.plot:
        print("wrote", plot(args.path, args.out))


if __name__ == "__main__":
    main()
