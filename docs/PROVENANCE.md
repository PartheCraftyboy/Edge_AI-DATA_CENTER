# Dataset provenance

Three public datasets supply the **ink** used by the page generator. None of
them supplies **labels**: no public dataset annotates handwritten pages with
region-level structure, which is why the layout is synthetic and the labels
are exact by construction. See `docs/superpowers/specs/2026-10-20-palimpsest-design.md`
section 6.

> **Licence check outstanding.** The spec requires each dataset's licence to
> be read on its Kaggle page and recorded here before the work is submitted.
> That has not been done yet — the fields below are marked `TO CONFIRM` and
> must be filled in from the source pages, not from secondary sources.
> Chars74K additionally carries academic terms from the originating paper.

| | Chars74K | Handwriting Recognition | Handwritten maths symbols |
|---|---|---|---|
| Kaggle slug | `dhruvildave/english-handwritten-characters` | `landlord/handwriting-recognition` | (see note) |
| Content | 3,410 isolated characters, 62 classes | ~330k handwritten names from records forms | 10,071 symbols, 19 classes |
| Used for | FORMULA operands, TABLE cells, DIAGRAM labels, composed pseudo-words | THEORY prose lines | FORMULA operators and operands, TABLE values |
| Licence | **TO CONFIRM** | **TO CONFIRM** | **TO CONFIRM** |
| Citation | de Campos, Babu & Varma (2009), *Character recognition in natural images* | **TO CONFIRM** | **TO CONFIRM** |

The maths archive arrived as `maths.zip` without its Kaggle slug recorded.
It is kept at `backend/data/raw/maths.zip`.
Its directory layout is `dataset/{0..9,add,dec,div,eq,mul,sub,x,y,z}/`, which
should be enough to identify the source page.

## What was done to each

Ingestion is in `app/data/corpus.py`. Raw archives are **not** in version
control, and neither are the built caches; both are reproducible. Raw files
live under `backend/data/raw/`, which is gitignored:

| Path | What it is |
|---|---|
| `raw/handwriting-recognition/` | the full Kaggle set: `train_v2/`, `validation_v2/`, `test_v2/` and their `written_name_*.csv` (330,961 + 41,370 + 41,370 rows) |
| `raw/maths.zip` | the maths symbols archive |
| `raw/dib-2016-math-symbols/` | a second maths set, **not ingested**; see below |

### Chars74K — 3,410 of 3,410 kept (100%)
Clean corpus, one mark per file on a 900x1200 canvas. Cropped to the ink
bounding box, aspect preserved, resampled to 64px tall.

### Handwriting Recognition — 4,234 of 12,000 kept (35.3%)
These are crops from French records forms, and roughly two in three carry
contamination that would poison training:

| Rejected as | Count | What it is |
|---|---|---|
| `mixed-sizes` | 4,203 | a printed field label (`NOM`, `PRENOM:`) beside the handwriting |
| `ambiguous-groups` | 2,722 | two separate groups of marks, so which one is the hand is a guess |
| `rule` | 274 | a form border, which is a long straight run |
| `touches-edge` | 166 | a stroke bleeding in from the line above or below |
| `blank` / `tiny` / other | 401 | no usable ink |

The harm is specific. A printed glyph composited into a `THEORY` region
teaches the classifier that printed texture means prose, and a form border is
a long straight run, which is the single feature that defines `TABLE`. Both
survive every shape-and-range assertion. Two printed `PRENOM` labels did reach
rendered pages during development and were caught by looking at the output,
not by a test.

The cache was built from the first 12,000 of the 330k rows, so the pool in
use is 4,234. That is ample: compositing resamples them, so a few thousand
words generate unbounded distinct pages. The full set has since been
re-downloaded to `raw/handwriting-recognition/`, so the pool *can* be grown by
re-running ingestion over more rows; nothing trained so far depends on it.

### Handwritten maths — 9,170 of 10,071 kept (91%)
About one sample in eight carries a stray underline from the source scan,
rejected because inside a formula crop a spurious horizontal run reads as a
fraction bar. The test keys on a rule being *extra*, since the `sub` class
is itself a horizontal run.

## Known characteristics, stated rather than hidden

- The word corpus is **almost entirely uppercase**: 17 rows in 330k contain a
  lowercase letter. All-caps text has no descenders, so the ragged prose band
  that `ProceduralInk` modelled deliberately would be lost. `CorpusInk`
  therefore mixes in pseudo-words composed from Chars74K lowercase glyphs,
  scaled per letter class so ascenders rise and descenders drop.
- The word corpus is handwritten **names**, not general prose. For ink that is
  fine, since what is needed is word-shaped marks, not meaning.
- Neither handwriting corpus contains Greek letters or mathematical operators;
  the maths corpus covers the operators but no Greek. λ, ρ, μ, Σ and ∫ are
  still drawn procedurally.

## On disk but not ingested

### Handwritten mathematical symbols, Data in Brief (2016)
Supplementary files of the Data in Brief article with ScienceDirect PII
`S2352340916300828`, submission `DIB-D-15-00413`, authors listed as Belaid
Bouikhalene and Yassine Chajri. The bundled `EXAMPLE/` folder has 17 classes,
including `ALPHA`, `SIGMA`, `PRODUCT`, `INFINITY`, `SQUARE_ROOT`, `SINUS`,
`COSINUNS`, `TANGENT` and `LOGARITHM`.

It matters because it fills the one gap stated above: neither ingested corpus
has Greek letters, so λ, ρ, μ and Σ are still drawn procedurally. **Not used
by any code yet.** Title, licence and citation are **TO CONFIRM** from the
article page; the folder names here are evidence of content, not of terms.

## Models

| Model | Where | Used for | Licence |
|---|---|---|---|
| `qwen2.5vl:3b` | Ollama | reading THEORY, FORMULA and TABLE regions (`app/vlm/ollama.py`) | **TO CONFIRM** |
| `PaddleOCR-VL-1.6` | Hugging Face `PaddlePaddle/PaddleOCR-VL-1.6` | alternative reader (`app/vlm/paddle.py`) | Apache-2.0, from the LICENSE file shipped with the weights |
| torchvision ImageNet weights | torchvision | pretrained SqueezeNet 1.1, MobileNetV2, ResNet-18 | **TO CONFIRM** |
