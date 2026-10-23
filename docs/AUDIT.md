# Code audit — 2026-10-22

What is sound, what is not finished, and what would not survive a question.
Written to be read before the submission, so nothing here is a surprise in
the room.

Scope: `backend/app` (10,743 lines), `frontend` (1,825 lines), 32 test files,
521 tests passing.

---

## A. Findings that change what you should claim

### A1. `jitter_box` is a stub, and the manifest says otherwise

`app/data/dataset.py:71`. The function returns its input unchanged:

```python
def jitter_box(box, rng, amount):
    """Perturb a ground-truth box the way an imperfect segmenter would."""
    # TODO(Parth): implement the jitter policy (5-10 lines).
    return box
```

Three consequences, in increasing order of seriousness:

1. `--box-jitter` is a CLI flag that silently does nothing.
2. No test covers it, so nothing fails.
3. **The manifest records `"box_jitter": 0.08` when you pass it.** A dataset
   built with jitter requested will claim in writing that it was jittered
   when it was not. That is the part to fix first, because a misreported
   dataset is worse than a missing feature.

**Why it matters beyond hygiene.** At inference the classifier never sees a
perfect box — it sees whatever `segment.py` proposes, which clips an edge or
swallows a strip of its neighbour. Training only on exact generator boxes
teaches it that region edges are always clean. That is one plausible
contributor to the synthetic-to-real collapse (0.99 synthetic, 0 of 6 real),
and it is untested either way, which is the honest position.

`app/data/real.py:146` has a working equivalent for hand-drawn boxes, so the
real-page path does jitter and the synthetic path does not. The asymmetry is
accidental, not designed.

**Status: left deliberately unimplemented.** It is 5 to 10 lines and the
reference implementation is in `real.py`. Worth doing yourself — the
decision in it is what distribution to draw from and whether to jitter
position, scale, or both.

### A2. The GNHK pass's result is thrown away — **FIXED 2026-10-22**

`train.sh` runs one `app.distil` pass per corpus. Two things combine:

- `distil()` starts every invocation from `base = runs/<backbone>/best.ckpt`,
  the **synthetic** checkpoint — so the IAM pass does not continue from the
  GNHK-distilled model, it restarts from scratch;
- both passes write to the same `runs_distil/<backbone>/best.ckpt`.

So the second pass overwrites the first and inherits nothing from it. The
shipped checkpoint is **IAM-only**. The GNHK work is discarded entirely.

The run's own output proves it rather than suggesting it. The GNHK pass
finished at `teacher_agreement 0.743`; the IAM pass then opened at
`round 0 … 0.128`, which is the synthetic model's score, not the
GNHK-distilled one.

| Pass | Train crops | Source | Fate |
|---|---|---|---|
| GNHK | 630 | 419 GNHK votes + 5 hand | overwritten |
| IAM | 9,417 | 4,214 IAM votes + 5 hand | shipped |

This is worse than it first looks. GNHK is the camera-captured corpus —
shadow, skew, show-through, a phone held over a page — and IAM is flatbed
scans. The corpus that matches what the pipeline actually meets is the one
being thrown away, and the clean one is the one that ships. A two-corpus
design exists precisely to catch a model that has learned about scanners;
training them in sequence into the same path defeats it.

**Fixed.** `--images` now takes several folders, `page_images` reads them
all, and `train.sh` makes one call instead of looping. Confirmed by the
run's own output:

```
segmenting pages in data/raw/gnhk/pages, data/raw/iam/pages…
teacher: 214 pages, 0 regions to label (4638 already cached)
dataset: {'train': 10032, 'val': 746, 'test': 548}  (4633 taught, 5 hand)
```

Every vote is in one dataset, where before a pass saw 419 or 4,214 of them,
and there is one `round 0` rather than one per corpus. Four tests cover it,
including the symptom itself: a dataset built from the second corpus alone
with the first one's crops missing.

**Caveat on the mechanism, stated because the first draft of this audit got
it wrong:** this is not catastrophic forgetting. Nothing is forgotten,
because nothing is carried over. It is a discarded result.

### A3. Three of the five classes have no real examples

Across 4,638 banked teacher votes:

| Class | Votes |
|---|---|
| `THEORY` | 3,631 |
| `FORMULA` | 1,006 |
| `DIAGRAM` | 1 |
| `PLOT_SKETCH` | 0 |
| `TABLE` | 0 |

GNHK is handwritten notes and IAM is prose forms. Neither contains a plotted
sketch or a ruled table, so `PLOT_SKETCH` and `TABLE` rest **entirely on
synthetic data** after distillation, and `DIAGRAM` effectively does too.

Say this before being asked. The macro-F1 on real pages is not meaningful for
three of five classes, and any real-page number should be quoted as a
two-class result or carry this caveat beside it.

### A4. `hand_samples` was 5 — now 42, and the number changed

Every real-page figure used to rest on five hand-labelled crops. Ninety-six
corrections through the Fix tab took it to **42 regions over 27 pages**, now
spread across train, validation and test rather than all landing in train.

The extra samples changed the answer, which is the argument for having
collected them. On five crops the teacher scored 1.00 and the student 1.00.
On forty-two the teacher scores **0.738** and the student **0.857**: the
teacher is substantially worse than it looked, and the student is beating
it. Neither of those was visible at five.

Still small. Forty-two is enough to rank two models and not enough for a
confidence interval worth printing.

---

## B. Things that are sound

### B1. The solver's injection guard holds

`app/routers/solver.py` is the one genuinely risky surface: SymPy's
`sympify` calls `eval` internally and must never see a raw model string.
Verified by attack rather than by reading:

| Input | Result |
|---|---|
| `__import__("os").system("id")` | rejected |
| `().__class__` | rejected |
| `open("/etc/passwd")` | rejected |
| `x.__dict__` | rejected |
| `eval("1")` | rejected |

Two controls in order: a character and token allowlist before parsing, then
`parse_expr` with an explicit `local_dict`, so an unlisted name resolves to a
plain Symbol rather than to anything executable.

### B2. Calibration is measured, not assumed

The gate's per-class thresholds only mean anything if the probabilities are
honest. ECE is reported per run and is 0.0008 for SqueezeNet. The decision
not to class-weight the loss follows from this and is documented where the
loss is defined, not in a report written afterwards.

### B3. Failures are recoverable and say why

`collect_votes` resets the model on failure and prints each distinct reason
once with a tally. Measured over the real run: **5 failures in 912 regions,
0.55%**, every one recovered. The earlier version counted failures and
discarded their cause, which turned a one-line diagnosis into a long hunt.

### B4. No bare `except:`, no dead TODOs besides A1

One TODO in the codebase, and it is the deliberate one. `print()` appears
only in CLI entry points, which is where it belongs.

---

## C. Known limits, already stated in the code

These are documented at the point of the compromise, not hidden:

- **Segmentation is classical, with no learned detector.** A deliberate
  boundary: the trained claim is region classification, and detection only
  has to produce crops worth classifying. A dense page with an inline formula
  has no whitespace to split on and segments coarsely.
- **`DIAGRAM` has no handler.** Always held, by design.
- **PaddleOCR-VL cannot read a formula into SymPy directly.** `vlm/latex.py`
  converts the subset handwritten notes use and returns `None` rather than a
  guess, so an unconvertible formula holds the region instead of answering it
  wrongly.
- **Validation during distillation is teacher-labelled** when no hand-labelled
  page falls in the validation split. The run says so in its own output.

---

## D. Fix order, if there is time

1. ~~**A2, joint training**~~ — done, 2026-10-22.
2. **A1's manifest lie** — one line, so a dataset stops claiming jitter it
   does not have. Then the jitter itself.
3. **A4, more hand labels** — the only thing that turns a direction into a
   result.
4. A3 needs a corpus that contains tables and plots, which is not a tonight
   problem. State it instead.
