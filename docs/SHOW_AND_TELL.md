# What to show, what to say, and what to study

A running order for the demo, the sentences that carry each part, and the
questions worth being ready for.

---

## The one-sentence version

> A handwritten page goes in and a typeset engineering notebook comes out,
> entirely on this laptop. The thing we trained and measured is a five-way
> region classifier; the interesting result is that the smallest backbone
> wins, and that a model scoring 0.99 on our own synthetic data scored zero
> on real handwriting until we fixed why.

Lead with the second half. A project that found its own failure is a
stronger submission than one that reports a clean number.

---

## Running order, 8 to 10 minutes

### 1. Compile a real page (90 s)

Open the **Write** tab, write a line of prose and an equation like
`v = u + a t`, press Compile. Show the note appearing on the right with the
prose set as prose and the equation typeset and solved.

Then hover a block. The region lights up on the page and back again.

> "Provenance is a live link, not a footnote. Every typeset block can show
> you the ink it came from, because a compiler you cannot audit is one you
> have to trust."

### 2. The architecture (2 min) — `docs/EXPLAINER.md` §1

Walk the six stages. Spend the time on **stage 2**, because it carries the
best story:

> "The kernel that groups ink into regions used to be a fraction of the
> page. That sounds scale-free and isn't. Measured across our real pages,
> the same fixed fraction came out as four letter-heights on a phone photo
> and a fifth of one on a canvas drawing — a twentyfold swing. At one end it
> fused the whole page into a single region; at the other it cut every word
> into its own. It is now a multiple of the writing's own letter height,
> estimated from the ink."

Have the two overlay images ready: the five one-word FORMULA boxes, and the
same page as two line regions.

### 3. The classifier, and the real finding (3 min) — `docs/EXPLAINER.md` §3

Put the backbone table on screen.

| Backbone | Params | Macro-F1 | ECE |
|---|---|---|---|
| palimpsest_net | 154 K | 0.9916 | 0.0266 |
| **squeezenet1_1** | **725 K** | **0.9990** | **0.0008** |
| mobilenet_v2 | 2.23 M | 0.9979 | 0.0018 |
| resnet18 | 11.18 M | 0.9990 | 0.0015 |

> "SqueezeNet matches ResNet-18 at one fifteenth of the parameters, with
> better calibration. For an edge course that is the result: the accuracy
> frontier is flat here, so the correct operating point is the small model.
> We didn't pick it, we measured it."

Then immediately undercut it, on purpose:

> "All four clear 0.99, which told us the synthetic problem was too easy.
> On crops cut from real pages the same classifier scored zero out of six."

### 4. Why it scored zero, and the fix (2 min)

> "The generator only ever drew a prose region as a dense multi-line
> paragraph. A single line had no class the model recognised, so it fell
> into the nearest thing that was short and symbol-shaped — which is why a
> handwritten name was being typeset as mathematics."

Then the distillation:

> "Rather than hand-label hundreds of pages, the vision-language model that
> already reads the regions also labels them, and the CNN trains on its
> votes. One detail decided whether it worked: asked for the class alone it
> scored three of six, but made to transcribe first and classify second it
> scored six of six. The transcriptions are often wrong — it read 'Parth' as
> 'Booth' — and the labels were right anyway, because whether a region is
> words or mathematics survives a misread word."

### 5. The correction loop (90 s) — the **Fix** tab

Show one card. Press → to confirm, ← to correct.

> "Ordered most-confident-first, which is the opposite of the obvious. The
> held regions are already handled — nothing was written into the note. The
> expensive mistakes are the ones the model was certain about, because those
> go in as fact and nobody looks again. 'Parth' went in at 1.00. No
> threshold catches that; only a person can."

### 6. Close on the honest slide (60 s)

Put `docs/AUDIT.md` §A on screen and say the limits yourself before anyone
asks. Section §8 of the explainer is the short version.

---

## The numbers, with their caveats attached

Never quote the left column without the right.

| Number | The caveat that must travel with it |
|---|---|
| 0.9990 macro-F1 | synthetic data, balanced by construction, too easy |
| 0 of 6 on real crops | six crops; a direction, not a rate |
| 0.00 → 1.00 after distillation | **five hand-labelled samples** |
| teacher agreement 0.822 | agreement with a 3B model, not accuracy |
| 0.55% teacher failure rate | measured over 912 regions, all recovered |
| 0.44 ms/crop | SqueezeNet, CUDA, batched |

Three of the five classes (`PLOT_SKETCH`, `TABLE`, `DIAGRAM`) have
essentially **no real examples** in the distilled data, because GNHK and IAM
are prose. Any real-page figure is effectively a two-class result. Say it
first.

---

## Questions to be ready for

**"Why a CNN and not the 3B model for everything?"**
Cost and calibration. 0.44 ms against 1.2 s, and a page has eight regions —
routing with the big model blows the 2 s budget by two orders of magnitude.
And the gate needs an honest probability: the CNN's ECE is 0.0008, while the
vision-language model reported 0.95 confidence on every crop it ever saw,
which is not a confidence.

**"Isn't training on another model's labels circular?"**
It is distillation, and the ceiling is the teacher. That is why
`teacher_vs_hand` is reported as a separate column: it says how good the
ceiling is. When it stops rising, more rounds cannot help and the teacher is
the thing to improve.

**"Why is validation teacher-labelled?"**
Because all five hand labels hashed into the train split. The run says so in
its own output, and `hand_accuracy` is computed outside the splits for
exactly that reason. It is the number to read.

**"What happens when it's wrong?"**
Nothing is written into the note below the class threshold — it renders as a
review panel. And the Fix tab turns a wrong answer into training data.

**"Why five classes?"**
Because each one routes to a different handler, and the label chooses the
handler. The classes are the routing table, not a taxonomy.

---

## What to study next, in order

Honest about which of these you have done and which you have read about.

1. **Calibration.** ECE is the number that justifies the whole gate policy,
   and the reason the loss is not class-weighted. Read about temperature
   scaling, which is the standard fix when a model is accurate but
   over-confident. You would apply it after training, on the validation
   split.
2. **Distillation.** What this project does is pseudo-labelling with a
   large teacher. Read Hinton's distillation paper for the soft-target
   version, which uses the teacher's full probability distribution rather
   than its argmax — this project only uses the hard label, and soft targets
   are the obvious next step.
3. **Domain shift.** The 0.99-to-0 collapse is the textbook case. The terms
   to look up are covariate shift and domain adaptation; the cheap practical
   lever is stronger augmentation, which is what `jitter_box` was meant to
   be (`docs/AUDIT.md` A1).
4. **Macro-F1 vs accuracy on imbalanced data.** You already chose macro-F1
   for the right reason; be able to say what a confusion matrix shows that a
   single number cannot.
5. **Depthwise separable convolutions.** `palimpsest_net` uses them and gets
   0.99 at 154 K parameters. Being able to explain why they are cheap —
   factorising a k×k×C convolution into a k×k spatial pass and a 1×1
   channel-mixing pass — is the single most useful architecture answer for
   an edge course.

---

## Before you present

- [ ] `cd backend && .venv/bin/python -m pytest` — expect 521 passing
- [ ] `./run.sh`, compile one page, confirm the note renders
- [ ] `.venv/bin/python -m app.history --plot` — the learning curve
- [ ] Have `docs/EXPLAINER.md` and `docs/AUDIT.md` open in tabs
- [x] `AUDIT.md` A2 fixed 2026-10-22: both corpora train in one pass, and
      the joint model (0.822) beats either alone (0.750 / 0.801)
