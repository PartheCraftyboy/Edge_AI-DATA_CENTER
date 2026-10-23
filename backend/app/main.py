"""
The HTTP surface. One process serves the API and the frontend.

    cd backend && .venv/bin/uvicorn app.main:app --port 8000
    open http://localhost:8000

Same origin for page and API, so there is no CORS to configure and nothing to
proxy: the frontend is static files under `/`, the API is under `/api`.

Routes follow spec section 10, plus the two the labelling loop needs: pages
you upload and label, and a training job that fine-tunes on them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import CLASS_NAMES, THRESHOLDS
from .contracts import CompiledNote
from . import corrections as corrections_log
from .data.real import PageStore, cut_crop
from .models.zoo import describe
from .pipeline.compile import Compiler, discover_checkpoints, list_vault

BACKEND = Path(__file__).resolve().parents[1]
FRONTEND = BACKEND.parent / "frontend"
MAX_UPLOAD_BYTES = 25 * 2**20

app = FastAPI(title="Palimpsest", version="1.0")
compiler = Compiler(vault=BACKEND.parent / "vault")
store = PageStore()
_training: subprocess.Popen | None = None


def decode_upload(data: bytes) -> np.ndarray:
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image is larger than 25 MB")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "not a readable image (PNG or JPEG expected)")
    return image


def png(image: np.ndarray) -> Response:
    ok, buf = cv2.imencode(".png", image)
    return Response(buf.tobytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


# ---------------------------------------------------------------- status

def _health() -> dict:
    """The whole status payload, in one place.

    `/api/health` and `/api/reader` both return this, deliberately the same
    shape: the sidebar redraws itself from whichever answered last, and a
    reader swap that came back without `device` left it calling
    `.toUpperCase()` on undefined.
    """
    return {
        "device": compiler.device,
        "active_model": compiler.active,
        "classes": list(CLASS_NAMES),
        "thresholds": THRESHOLDS,
        **compiler.reader_status(),
    }


@app.get("/api/health")
def health():
    return _health()


# ---------------------------------------------------------------- compile

@app.post("/api/compile", response_model=CompiledNote)
async def compile_page(file: UploadFile = File(...), title: str = Form(""), deskew: bool = Form(True)):
    image = decode_upload(await file.read())
    return compiler.compile(image, title=title or Path(file.filename or "").stem, deskew=deskew)


@app.post("/api/pages/{page_id}/compile", response_model=CompiledNote)
def compile_labelled(page_id: str):
    """Compile a stored page using its boxes, so a labelled page shows what
    the pipeline makes of exactly the regions you drew."""
    page = _page(page_id)
    grey = cv2.imread(str(store.image_path(page_id)), cv2.IMREAD_GRAYSCALE)
    return compiler.compile(grey, title=page.get("source_name") or page_id,
                            boxes=[r["box"] for r in page["regions"]])


@app.get("/api/notes")
def notes():
    return list_vault(compiler.vault)


@app.get("/api/notes/{note_id}", response_model=CompiledNote)
def note(note_id: str):
    path = _note_dir(note_id) / "note.json"
    return CompiledNote.model_validate_json(path.read_text())


@app.get("/api/notes/{note_id}/page.png")
def note_page(note_id: str):
    return FileResponse(_note_dir(note_id) / "page.png")


class Correction(BaseModel):
    """What a region should have been. Either part may be left out."""

    should_be: str | None = None
    """The correct transcription, when the reading was wrong."""
    correct_class: str | None = None
    """The correct region class, when it was routed to the wrong handler."""
    comment: str = ""


def _block_text(block) -> str:
    """What the reader produced for this block, as one editable string."""
    content = block.content
    if content is None:
        return ""
    for field in ("tex", "markdown", "caption"):
        value = getattr(content, field, None)
        if value:
            return str(value)
    return ""


@app.post("/api/notes/{note_id}/blocks/{block_id}/correction")
def correct_block(note_id: str, block_id: str, body: Correction):
    """Record that a block was read or classified wrongly.

    A **class** correction is a hand label and is written back into the page
    store, so fixing a block here produces the training data that stops it
    recurring. A **reading** correction teaches the classifier nothing and is
    kept as an evaluation example for the reader, which is the only way to
    tell a prompt or model change that reads better from one that merely
    reads differently.

    Posting neither records a confirmation: the block was right. Those are
    worth keeping, because a file of nothing but complaints cannot be scored.
    """
    folder = _note_dir(note_id)
    note = CompiledNote.model_validate_json((folder / "note.json").read_text())
    block = next((b for b in note.blocks if b.id == block_id), None)
    if block is None:
        raise HTTPException(404, f"no block {block_id!r} in note {note_id!r}")
    if body.correct_class and body.correct_class not in CLASS_NAMES:
        raise HTTPException(422, f"unknown class {body.correct_class!r}")

    box = [block.box.x, block.box.y, block.box.w, block.box.h]
    entry = corrections_log.record(
        note_id=note_id, block_id=block_id, box=box,
        region_class=block.region_class, correct_class=body.correct_class,
        read_as=_block_text(block), should_be=body.should_be,
        confidence=block.confidence,
        reader=str(compiler.reader_status().get("model", "")),
        comment=body.comment,
    )

    # A corrected class is a label. Put the page in the store if it is not
    # already there, then add this region to it.
    labelled = None
    if "class" in entry["kind"]:
        grey = cv2.imread(str(folder / "page.png"), cv2.IMREAD_GRAYSCALE)
        if grey is not None:
            existing = next((p for p in store.all() if p.get("source_name") == f"note:{note_id}"), None)
            if existing is None:
                existing = store.add(grey, [], source_name=f"note:{note_id}")
            regions = [r for r in existing["regions"] if list(r["box"]) != box]
            regions.append({"box": box, "class": entry["correct_class"]})
            labelled = store.save_labels(existing["id"], regions)["id"]

    # Keep it on the note too, so the correction is visible where it was made.
    raw = json.loads((folder / "note.json").read_text())
    for stored in raw["blocks"]:
        if stored["id"] == block_id:
            stored["correction"] = entry
    (folder / "note.json").write_text(json.dumps(raw, indent=1))
    return {"correction": entry, "labelled_page": labelled}


@app.get("/api/corrections")
def list_corrections():
    return {"report": corrections_log.report(), "entries": corrections_log.load()[-200:]}


@app.get("/api/review")
def review_queue(limit: int = 60, only: str = "all"):
    """Regions worth a person's eyes, across every compiled note.

    **Ordered by confidence, highest first**, which is the opposite of the
    obvious. The tempting queue is the held regions: the model has flagged
    its own doubt, so that is where it must be wrong. But a held region is
    already handled — nothing was written into the note and the reader can
    see the panel. The expensive errors are the ones the model was *certain*
    about, because those are written in as fact and nobody looks again.

    This project has the case in front of it: the word "Parth" classified
    FORMULA at 1.00 and typeset as mathematics. No threshold catches that,
    because the confidence was not low. Only a person can.

    Already-corrected regions drop out, so the queue shortens as it is worked
    rather than re-presenting settled decisions.
    """
    rows = []
    for folder in sorted(compiler.vault.iterdir(), reverse=True):
        path = folder / "note.json"
        if not path.is_file():
            continue
        try:
            note = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for block in note.get("blocks", []):
            if block.get("correction"):
                continue
            if only == "held" and not block.get("held"):
                continue
            if only == "confident" and block.get("confidence", 0) < 0.9:
                continue
            content = block.get("content") or {}
            rows.append({
                "note_id": note["note_id"],
                "title": note.get("title", ""),
                "block_id": block["id"],
                "region_class": block["region_class"],
                "confidence": block.get("confidence", 0.0),
                "held": block.get("held", False),
                "box": block["box"],
                "read_as": content.get("tex") or content.get("markdown")
                           or content.get("caption") or "",
            })
    rows.sort(key=lambda r: (-r["confidence"], r["note_id"]))
    return {"pending": len(rows), "regions": rows[:limit],
            "report": corrections_log.report()}


@app.get("/api/notes/{note_id}/markdown")
def note_markdown(note_id: str):
    return Response((_note_dir(note_id) / "note.md").read_text(), media_type="text/markdown")


def _note_dir(note_id: str) -> Path:
    folder = compiler.vault / note_id
    if "/" in note_id or ".." in note_id or not (folder / "note.json").exists():
        raise HTTPException(404, "no such note")
    return folder


# ---------------------------------------------------------------- models

class SetModel(BaseModel):
    key: str


@app.get("/api/models")
def models():
    out = []
    for key, ckpt in discover_checkpoints().items():
        origin, backbone = key.split("/", 1)
        evaluation = _read_json(ckpt.parent / "eval.json")
        tuned = _read_json(ckpt.parent / "finetune.json")
        overall = (evaluation or {}).get("overall", {})
        out.append({
            "key": key, "origin": origin, "backbone": backbone,
            "params": _parameters(backbone), "active": key == compiler.active,
            "synthetic_macro_f1": overall.get("macro_f1"),
            "latency_ms": (overall.get("latency_ms") or {}).get("p50"),
            "ece": overall.get("ece"),
            "real_before": (tuned or {}).get("before"),
            "real_after": (tuned or {}).get("after"),
        })
    return out


@lru_cache(maxsize=None)
def _parameters(backbone: str) -> int:
    # describe() builds the network to count it; ResNet-18 alone takes a
    # noticeable fraction of a second, and the count never changes.
    return describe(backbone)["parameters"]


@app.post("/api/models")
def set_model(body: SetModel):
    try:
        compiler.set_active(body.key)
    except KeyError:
        raise HTTPException(404, f"no checkpoint {body.key!r}")
    return {"active": compiler.active}


# ---------------------------------------------------------------- reader

class SetReader(BaseModel):
    name: str


@app.post("/api/reader")
def set_reader(body: SetReader):
    """Choose which model transcribes the next page.

    Returns the same shape `/api/health` reports the reader with, including
    `available` and, when it is not, `why` — so the interface can show that
    the cloud reader was selected but has no key, which is a different and
    more fixable state than it being unreachable.

    Selecting a reader does not load it. Nothing loads until a page is
    compiled, which keeps an idle switch free and, on the local readers,
    keeps `vlm.residency` out of the request path.
    """
    try:
        compiler.set_reader(body.name)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    return _health()


@app.get("/api/metrics")
def metrics():
    """Module 5 for the active model, from its saved evaluation."""
    key = compiler.active or ""
    ckpt = discover_checkpoints().get(key)
    report = _read_json(ckpt.parent / "eval.json") if ckpt else None
    if report is None:
        raise HTTPException(404, "the active model has no saved evaluation")
    overall = report["overall"]
    return {"key": key, "split": report.get("split"), "macro_f1": overall["macro_f1"],
            "accuracy": overall["accuracy"], "ece": overall["ece"],
            "per_class_f1": overall["per_class_f1"], "held_rate": overall["held_rate"],
            "confusion": overall["confusion_row_normalised"], "classes": list(CLASS_NAMES)}


@app.get("/api/pareto")
def pareto():
    return [{"key": m["key"], "macro_f1": m["synthetic_macro_f1"], "latency_ms": m["latency_ms"]}
            for m in models() if m["latency_ms"] is not None]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- labelling

class Region(BaseModel):
    box: list[float]
    region_class: str


class Labels(BaseModel):
    regions: list[Region]


@app.post("/api/pages")
async def upload_page(file: UploadFile = File(...)):
    """Store a page for labelling, with machine-proposed boxes as a draft."""
    image = decode_upload(await file.read())
    grey, regions, info = compiler.propose(image)
    record = store.add(grey, regions, source_name=file.filename or "")
    return {**record, **info}


@app.get("/api/pages")
def pages():
    return [{k: p[k] for k in ("id", "source_name", "created", "split", "labelled")}
            | {"regions": len(p["regions"])} for p in reversed(store.all())]


@app.get("/api/pages/stats")
def page_stats():
    return store.stats()


@app.get("/api/pages/{page_id}")
def page(page_id: str):
    return _page(page_id)


@app.get("/api/pages/{page_id}/image.png")
def page_image(page_id: str):
    _page(page_id)
    return FileResponse(store.image_path(page_id))


@app.get("/api/pages/{page_id}/crop/{index}.png")
def page_crop(page_id: str, index: int):
    page = _page(page_id)
    if not 0 <= index < len(page["regions"]):
        raise HTTPException(404, "no such region")
    grey = cv2.imread(str(store.image_path(page_id)), cv2.IMREAD_GRAYSCALE)
    return png(cut_crop(grey, page["regions"][index]["box"]))


@app.put("/api/pages/{page_id}/labels")
def save_labels(page_id: str, labels: Labels):
    _page(page_id)
    try:
        return store.save_labels(page_id, [{"box": r.box, "class": r.region_class}
                                           for r in labels.regions])
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.delete("/api/pages/{page_id}")
def delete_page(page_id: str):
    _page(page_id)
    store.delete(page_id)
    return {"deleted": page_id}


def _page(page_id: str) -> dict:
    try:
        return store.get(page_id)
    except KeyError:
        raise HTTPException(404, "no such page")


# ---------------------------------------------------------------- training

class TrainRequest(BaseModel):
    backbone: str = "squeezenet1_1"
    epochs: int = 8


@app.post("/api/train")
def start_training(body: TrainRequest):
    global _training
    if _training is not None and _training.poll() is None:
        raise HTTPException(409, "a training run is already going")
    if body.backbone not in {k.split("/", 1)[1] for k in discover_checkpoints() if k.startswith("synthetic/")}:
        raise HTTPException(404, f"no synthetic checkpoint for {body.backbone!r} to start from")
    log = open(BACKEND / "runs_real.log", "w")
    _training = subprocess.Popen(
        [sys.executable, "-m", "app.finetune", "--backbone", body.backbone,
         "--epochs", str(max(1, min(50, body.epochs)))],
        cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT,
    )
    return {"started": True, "pid": _training.pid}


@app.get("/api/train/status")
def training_status():
    status = _read_json(BACKEND / "runs_real" / "status.json") or {"state": "idle"}
    running = _training is not None and _training.poll() is None
    if not running and status.get("state") in ("building", "training"):
        status["state"] = "failed"
        status.setdefault("error", "the training process exited early; see backend/runs_real.log")
    if status.get("state") == "done":
        compiler.forget("real/")  # pick up the new weights on next use
    return {**status, "running": running}


# ---------------------------------------------------------------- frontend

if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
