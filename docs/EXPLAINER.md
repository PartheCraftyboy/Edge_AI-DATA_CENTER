# Palimpsest — a project explainer

> **Please read in this order.** Start with **[1. The architecture](#1-the-architecture)**,
> which is one page and explains how a photograph of a notebook becomes a
> typeset note. Then read **[3. The classifier](#3-the-classifier-the-cnn)**,
> which is the convolutional network this project actually trains and
> measures, and **[4. The other models](#4-the-other-models-and-why-a-cnn-is-still-the-one-being-graded)**,
> which explains what the two vision-language models do and why they are
> *not* the thing being assessed. Those three sections are the submission.
> Everything after them is evidence and operating detail.

UCS668 · Edge AI and Robotics: Data Center Vision

---

## 0. What the project does, in one paragraph

A rough handwritten page goes in. A properly typeset engineering notebook
comes out: prose as prose, mathematics as typeset mathematics with the
arithmetic already carried out, and hand-drawn graph sketches replaced by
real plotted figures. Everything runs on one laptop. No page image is
uploaded anywhere, and no note leaves the vault folder on disk.

---

## 1. The architecture

The pipeline is six stages. Each one is timed, and the time is charged
against a 2-second interactive budget so the system can say which stage is
the bottleneck rather than guessing.

```
   photograph or canvas drawing
             |
   [1] preprocess     find the page, deskew, CLAHE, Sauvola threshold,
             |        drop background slabs      -> grey image + ink mask
   [2] segment        close the ink into blobs   -> region boxes
             |
   [3] classify       the CNN, one 224x224 crop  -> 5-way class + confidence
             |        per region                      <- THIS IS THE MODEL
   [4] gate           per-class confidence floor -> admit, or hold for review
             |
   [5] read           a transcription model per   -> text, LaTeX, table cells
             |        admitted class
   [6] resolve        symbol table, solver, plots -> the compiled note
```

**Stage 1, preprocessing**, is the counter-move to a bad photograph. It finds
the sheet of paper in the frame and crops to it, straightens the page, evens
out the lighting, and thresholds locally rather than globally because one
global threshold fails on a page with a shadow across it. It then removes
**background slabs** by stroke thickness: handwriting is thin, a patterned
cloth or a floor tile that the threshold mistook for ink is a solid slab many
times thicker, and the ratio to the page's own median stroke width separates
them without any tuning per photograph.

**Stage 2, segmentation**, is deliberately classical, with no learned
detector. The project's trained and measured claim is region *classification*;
detection only has to produce crops worth classifying. The method is one idea:
within a region, ink sits close together — letters in a word, words on a line,
lines in a paragraph — while between regions there is more whitespace than
that. So the ink is morphologically **closed** with a rectangle a little wider
than a word gap and a little taller than a line gap, which fuses each region
into one blob and leaves the gaps between regions open.

The size of that rectangle is the one thing in this file worth stating twice.
It is a multiple of **the writing's own letter height**, estimated from the
median height of the ink's connected components — never a fraction of the
page. A page fraction sounds scale-free and is not: measured across the real
pages in this repository, the same fixed fraction came out as 4.08 letter
heights on a phone photo of a notebook and 0.17 on a page drawn on the
canvas, a twentyfold swing. At one end it fused an entire page into a single
blob; at the other it cut every individual word into its own region.

**Stage 4, the gate**, is the safety rule, and it is per class rather than
global because the classes do not carry equal risk. A garbled sentence is
self-evident and trivially corrected; a wrong subscript is invisible and
silently corrupts the note. Nothing below its class threshold is written into
the note — it renders as a review panel instead.

| Class | Threshold | Why |
|---|---|---|
| `THEORY` | 0.60 | prose errors are obvious and easy to fix |
| `PLOT_SKETCH` | 0.75 | a wrong figure misleads, but looks wrong when plotted |
| `DIAGRAM` | 0.80 | structure is easy to get subtly wrong |
| `TABLE` | 0.85 | transposed cells read as perfectly plausible |
| `FORMULA` | 0.88 | a wrong subscript silently corrupts the note |

---

## 2. The five region classes

Everything the system does downstream depends on getting this one label
right, because the label chooses the handler.

| Class | What it is | What happens to it |
|---|---|---|
| `THEORY` | prose, sentences, headings, bullets, a name | line split, then handwriting transcription |
| `FORMULA` | equations, symbols, fractions, subscripts | formula recognition to LaTeX, then the solver |
| `PLOT_SKETCH` | hand-drawn axes, curves, bar sketches | paired with a plottable equation from the page |
| `DIAGRAM` | boxes, arrows, block diagrams, circuits | no handler in this build; always held |
| `TABLE` | ruled or unruled grids, columns of values | cell split, then transcription per cell |

---

## 3. The classifier: the CNN

**This is the model the project trains, measures and is assessed on.** It is
a five-way image classifier over 224×224 greyscale crops, implemented as a
PyTorch Lightning module (`app/models/classifier.py`) over a switchable
backbone (`app/models/zoo.py`).

### What a convolutional network is doing here

A convolutional layer slides a small learned filter across the image and
records how strongly each position matches it. Early layers therefore learn
local things — an edge at some orientation, a stroke end, a corner. Stacking
them makes the receptive field grow, so later layers respond to arrangements
of those parts rather than to the parts themselves. Pooling between stages
discards position while keeping presence, which is what makes the answer
depend on *what the region contains* rather than on where in the crop it
happened to sit.

That is exactly the right inductive bias for this problem, and it is worth
saying why rather than asserting it. The five classes are not distinguished
by their content — the system does not need to read the region to classify
it — but by **structure**: stroke density, aspect ratio, baseline regularity,
and whether the region contains long straight runs. A table has ruled lines
and column alignment. A formula breaks the baseline with subscripts and
superscripts. Prose has a regular baseline and even word spacing. A plot
sketch has two long perpendicular strokes with a curve between them. These
are local patterns in a particular arrangement, which is the family of thing
a CNN represents efficiently and a fully connected network represents badly.

### The backbone zoo

Four backbones train from the same data with the same loop, which turns
"which architecture?" into a measured answer rather than a preference. All
scores below are on the held-out **synthetic** test split, 951 crops:

| Backbone | Parameters | Macro-F1 | ECE | Train time | Design |
|---|---|---|---|---|---|
| `palimpsest_net` | 154,373 | 0.9916 | 0.0266 | 43 s | built for this project: depthwise separable convolutions with residual skips |
| `squeezenet1_1` | 725,061 | **0.9990** | **0.0008** | 58 s | Fire modules: squeeze 1×1, then expand 1×1 and 3×3 |
| `mobilenet_v2` | 2,230,277 | 0.9979 | 0.0018 | 128 s | inverted residuals with linear bottlenecks |
| `resnet18` | 11,179,077 | 0.9990 | 0.0015 | 110 s | residual blocks; the accuracy end of the frontier |

The interesting result is not the top score. It is that **SqueezeNet matches
ResNet-18 at one fifteenth of the parameters and better calibration**, which
is the actual finding for an edge-deployment course: on this problem the
accuracy frontier is flat and the sensible operating point is the small model.

### Two training choices that are policy, not convention

**The loss uses uniform class weights.** A misread formula costs nine times a
misread paragraph in this system, and the obvious move is to tell the loss
so. That is wrong here. Weighting the loss biases the predicted
probabilities, and once they are biased a reported confidence of 0.91 no
longer means 0.91 — which makes the calibration error meaningless and
undermines the very threshold policy it is supposed to justify. The cost
asymmetry is applied afterwards, at the gate, where it can be reasoned about
separately. The model's job is to be honest; the gate's job is to be
cautious.

**The headline metric is macro-F1, not accuracy.** The synthetic set is
balanced by construction, but real pages are dominated by prose, and accuracy
would read high for a model that never finds a formula at all. Macro-F1 gives
the rare classes equal say.

### The honest part: 0.99 on synthetic, 0.00 on real

The synthetic page generator produces labels for free — if the generator
placed a formula at those coordinates then that region *is* a formula, known
by construction, with no annotation pass and no label noise. That is the only
way to get region-level structural labels at all, since no public dataset
provides them.

It also made the problem too easy. Every backbone clears 0.99, which says the
synthetic classes are separable and says nothing about real handwriting.
Measured against hand labels on crops cut from **real** pages, the
synthetic-trained classifier scored **0 out of 6**. Not degraded — zero.

The cause is specific and worth stating, because it is a generator design
flaw rather than a training failure: the generator only ever draws a `THEORY`
region as a *dense multi-line paragraph*. A single line, or a single word, is
a shape the model has never seen, so it falls into the nearest class that is
short and symbol-shaped. That is why a handwritten name was being typeset as
mathematics.

---

## 4. The other models, and why a CNN is still the one being graded

Two vision-language models run locally alongside the classifier. **Neither is
the model under assessment**, and the division of labour is deliberate.

| Model | Size | Role | Why this one |
|---|---|---|---|
| **PaddleOCR-VL-1.6** | 0.9 B | **reads** an admitted region | trained only to read documents. On real crops it returned "Nice to see you" where the general model returned "Nice to tell you", and kept the hat on `ŷ` where the other dropped it |
| **qwen2.5vl:3b** | 3 B | **judges** region classes, offline | a general model, and the words-or-mathematics question is semantic. PaddleOCR-VL has no notion of the five classes at all |

They cannot both be resident: measured on the RTX 4050's 6141 MiB, Qwen's
runner takes 4196 MiB and PaddleOCR-VL then fails to allocate. An explicit
residency arbiter evicts one before the other loads.

### Why the CNN is not replaced by the big model

A fair question is why the 3 B model does not simply do the classification at
inference time. Two reasons, and both are the point of the course:

1. **Cost.** The CNN classifies a region in roughly 0.44 ms. The
   vision-language model takes about 1.2 seconds — close to three thousand
   times slower — and a page has eight or more regions. Routing with the
   large model would blow the interactive budget by two orders of magnitude.
2. **Calibration.** The gate policy needs a probability that means something.
   The CNN reports a softmax with a measured expected calibration error of
   0.0008. The vision-language model reported a confidence of 0.95 on every
   single crop it was asked about, which is not a confidence at all.

### What the large model is used for instead: supervising the small one

The reader labels regions on pages nobody has annotated, and the CNN is
trained on its votes. This is knowledge distillation with a
vision-language teacher, and it needs no hand labelling, which is what makes
hundreds of real pages usable in one evening.

One detail decided whether it worked. Asked for the class alone, the teacher
scored 3 of 6 on crops whose answer was known. Made to **transcribe first and
classify second** — by putting the transcription earlier in the
constrained-decoding schema, so the model must write down what it sees before
it is allowed to name a class — it scored 6 of 6. What makes that work is
counter-intuitive and worth recording: the transcriptions are frequently
*wrong* ("Parth" came back as "Booth"), and the labels were right anyway.
Whether a region is words or mathematics survives a misread word.

Where the student and the teacher disagree, that crop's loss is scaled up.
The penalty anneals itself away: each crop the student comes to agree on
drops back to weight one, so the objective slides into ordinary cross-entropy
as the two converge.

### First measured result

Final run, 2026-10-22, SqueezeNet 1.1, both corpora in one pass: 4,676
teacher votes over 214 pages, 10,142 training crops, **42 hand-labelled
regions**.

| | synthetic | round 1 | round 2 | round 3 |
|---|---|---|---|---|
| **accuracy on hand labels** | 0.048 | 0.762 | **0.857** | 0.857 |
| agreement with the teacher | 0.132 | 0.788 | 0.801 | 0.815 |
| the teacher's own accuracy | 0.738 | 0.738 | 0.738 | 0.738 |

Read the last row first. On five samples the teacher scored 1.00 and looked
infallible; on forty-two it scores **0.738**. That is the honest ceiling,
and it means the headline is not "the student reached the teacher" but
something better:

> **The student ends at 0.857 and its teacher is at 0.738. It is more
> accurate than the model that taught it.**

That is a real effect and not an accident of counting. The teacher labels
each crop alone, so its mistakes are scattered and inconsistent; the CNN
has to fit one decision rule across ten thousand crops, which averages that
noise out. The inductive bias is doing the regularising. It is also exactly
why the hand labels are weighted: a crop a person corrected is worth six
copies of a teacher guess, because it is both ground truth and a place the
machine is known to be wrong.

**This must be read with its sample size: five.** Five hand-labelled crops.
`0.00 → 0.60` is a real direction and a magnitude with error bars wide enough
to swallow it. What it establishes solidly is that the synthetic-to-real gap
is total rather than partial, and that the teacher is a sound supervisor on
everything that can currently be checked. Turning it into a defensible number
needs more hand labels, which is what the correction interface exists to
produce. See `docs/history.png`: every run's curve restarts at zero, because
each corpus pass begins again from the synthetic checkpoint — the finding in
`docs/AUDIT.md` A2, visible in the chart.

---

## 5. Two data sources, and why both

| Corpus | What it is | Why it is here |
|---|---|---|
| **GNHK** | 687 **camera-captured** pages of handwritten notes, CC-BY-4.0 | shadow, skew, show-through, a phone held over a page: the conditions the pipeline actually meets |
| **IAM** | 1,539 flatbed **scans** of forms, 657 writers | evenly lit and square to the frame: the clean end of the range |

Having both is the measurement. A classifier scoring the same on each has
learned something about handwriting; one scoring well on IAM and badly on
GNHK has learned something about scanners — which is the mistake the
synthetic pages already made once at 0.99.

---

## 6. Correcting the model, and what a correction becomes

The compiler can be confidently wrong, and a threshold cannot catch that by
definition. The interface therefore shows one region at a time, most
confident first, with the model's reading, and asks a person whether it is
right.

The ordering is the opposite of the obvious one and is the design decision
worth defending. The tempting queue is the *held* regions, since the model has
flagged its own doubt. But a held region is already handled: nothing was
written into the note, and the panel says so. The expensive errors are the
ones the model was **certain** about, because those are written in as fact and
nobody looks again.

A correction has two independent halves, because two different things can be
wrong and they are repaired by different work:

- **a wrong class** is a hand label, written straight back into the page store,
  and becomes training data;
- **a wrong reading** teaches the classifier nothing and becomes an
  evaluation example for the reader — the only way to tell a change that
  reads *better* from one that merely reads *differently*.

`history.csv` records one row per training run, plotted against **regions
labelled** rather than epochs, because the decision it exists to support is
whether to label another fifty or stop.

---

## 7. Running it

```bash
cd backend && uv venv --python 3.12 .venv && uv pip install -r requirements.txt
./run.sh                 # the app, on http://localhost:8000
./train.sh               # teach the classifier from the reader
palimtop live            # a terminal monitor of the running server
```

```bash
cd backend
.venv/bin/python -m app.data.synth      # generate synthetic pages
.venv/bin/python -m app.train           # train one backbone
.venv/bin/python -m app.evaluate        # confusion matrix, macro-F1, ECE
.venv/bin/python -m app.robustness      # degradation sweeps
.venv/bin/python -m app.history --plot  # the learning curve
```

Tests: `cd backend && .venv/bin/python -m pytest` — 521 tests, all passing.

---

## 8. What is not finished, stated plainly

- **Five hand-labelled crops.** The headline real-page number rests on five
  samples. It is a direction, not yet a result.
- **A dense page still segments coarsely.** Whitespace is all the segmenter
  has, and there is no extra whitespace around a formula written inline in a
  paragraph. Hand-drawn boxes remain the answer for those pages.
- **`DIAGRAM` has no handler.** Those regions are always held, by design, and
  the build does not pretend otherwise.
- **Validation during distillation is teacher-labelled** when no hand-labelled
  page falls into the validation split. The run says so in its own output, and
  the figure to read in that case is accuracy against hand labels, which is
  computed separately.
