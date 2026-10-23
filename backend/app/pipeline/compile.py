"""
The compile pipeline: page in, `CompiledNote` out.

    preprocess -> segment -> classify -> gate -> read -> resolve -> compose

Every stage is timed and charged to the Module 1 budget. The page-level costs
(preprocessing, one batched forward pass for every region) are shared out
evenly across regions; reading is measured per region, because it is the
stage that dominates and the one worth seeing region by region.

**Reading happens before resolving, in two passes.** A formula's answer
depends on `name = value` bindings that may sit in prose above it, so every
admitted region is read first, the symbol table is built from all of it, and
only then is each formula solved against what was written above it.

**A failure in one region never fails the page.** If the reader is offline,
or a reply is unusable, that region is held with the reason in words, and the
rest of the page still compiles. A compile that fails whole is a demo that
fails whole.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from .. import CLASS_NAMES
from ..contracts import (Block, Box, CompileBudget, CompiledNote, FigureContent,
                         FormulaContent, ProseContent, TableContent, Timing)
from ..data.dataset import IMAGENET_MEAN, IMAGENET_STD
from ..data.real import cut_crop
from ..metrics.activity import ActivityLog
from ..routers.plot import choose_plot_range, plot_series, refuse_reason, suggest_y_clip
from ..routers.solver import solve
from ..routers.symbols import SymbolTable, display
from .gate import decide
from .preprocess import preprocess
from .segment import segment

DEFAULT_BACKBONE = "squeezenet1_1"
BACKEND = Path(__file__).resolve().parents[2]
# Absolute, so the server finds its checkpoints whatever directory it starts in.
# Three origins, and the third was missing: a distillation run wrote
# runs_distil/<backbone>/best.ckpt and nothing could load it, so a night of
# training produced a model the app could not select. The key prefix is what
# the UI shows, so the origin of a checkpoint is always visible beside it.
RUN_ROOTS = {
    "synthetic": BACKEND / "runs",        # trained on generated pages
    "real": BACKEND / "runs_real",        # fine-tuned on hand-labelled pages
    "taught": BACKEND / "runs_distil",    # distilled from the reader's votes
}


def discover_checkpoints() -> dict[str, Path]:
    """`synthetic/<backbone>` and `real/<backbone>` keys for every best.ckpt."""
    out = {}
    for origin, root in RUN_ROOTS.items():
        for ckpt in sorted(root.glob("*/best.ckpt")):
            out[f"{origin}/{ckpt.parent.name}"] = ckpt
    return out


def to_tensor(crops: list[np.ndarray]) -> torch.Tensor:
    """The exact transform `CropDataset` applies: grey to three channels,
    ImageNet normalisation. Anything else is a train/serve skew."""
    x = np.stack(crops).astype(np.float32) / 255.0
    x = np.repeat(x[:, :, :, None], 3, axis=3)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


class Compiler:
    def __init__(self, vault: str | Path = "../vault", reader=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vault = Path(vault)
        self.vault.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, torch.nn.Module] = {}
        self._reader = reader
        self._reader_name: str | None = None
        self._reader_error: str | None = None
        # One line per region, for `palimtop live` in another terminal. The
        # server never reads it back and never waits on it.
        self.activity = ActivityLog()
        self._lock = threading.Lock()  # one page at a time on one GPU
        checkpoints = discover_checkpoints()
        # Hand labels first, then the reader's votes, then synthetic. The
        # order is by how much the weights have seen of real handwriting.
        preferred = [f"real/{DEFAULT_BACKBONE}", f"taught/{DEFAULT_BACKBONE}",
                     f"synthetic/{DEFAULT_BACKBONE}"]
        self.active = next((k for k in preferred if k in checkpoints), next(iter(checkpoints), None))

    # ------------------------------------------------------------- models

    def model(self, key: str | None = None) -> torch.nn.Module:
        from ..evaluate import load_checkpoint

        key = key or self.active
        if key is None:
            raise RuntimeError("no trained checkpoint found under runs/ or runs_real/")
        if key not in self._models:
            checkpoints = discover_checkpoints()
            if key not in checkpoints:
                raise KeyError(key)
            self._models[key] = load_checkpoint(checkpoints[key], device=self.device)
        return self._models[key]

    def set_active(self, key: str) -> None:
        self.model(key)
        self.active = key

    def forget(self, prefix: str = "real/") -> None:
        """Drop cached models so a fresh fine-tune is picked up."""
        for key in [k for k in self._models if k.startswith(prefix)]:
            del self._models[key]

    def classify(self, grey: np.ndarray, boxes) -> tuple[np.ndarray, float, list[bool]]:
        """Softmax rows for each box, the measured forward-pass time, and
        which boxes produced a crop at all."""
        crops, ok = [], []
        for box in boxes:
            crop = cut_crop(grey, box)
            ok.append(crop is not None)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return np.zeros((0, len(CLASS_NAMES))), 0.0, ok
        model = self.model()
        x = to_tensor(crops).to(self.device)
        if self.device == "cuda":
            torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()
        if self.device == "cuda":
            torch.cuda.synchronize()
        return probs, (time.perf_counter() - began) * 1000, ok

    # ------------------------------------------------------------- reader

    def reader(self):
        if self._reader is None:
            from ..vlm import make_reader

            self._reader = make_reader(self._reader_name)
        return self._reader

    def set_reader(self, name: str) -> None:
        """Swap the transcription model for the next compile.

        The old reader is closed before the new one is built, not after.
        Local readers hold VRAM and the card does not fit two, so building
        first would make `vlm.residency` evict the outgoing model during the
        handover — correct, but it would reload it again on the next page.
        Closing first keeps the swap to one unload.

        Raises `ValueError` for a name `make_reader` does not know, before
        anything is torn down, so a bad request leaves the current reader
        working.
        """
        from ..vlm import READERS

        name = name.lower()
        if name not in READERS:
            raise ValueError(f"unknown reader {name!r}")
        if name == self.reader_name():
            return
        old, self._reader = self._reader, None
        self._reader_name, self._reader_error = name, None
        if old is not None:
            try:
                old.close()
            except Exception:
                # A reader that will not close is still a reader we are done
                # with. The next one must not be blocked by its exit.
                pass

    def reader_name(self) -> str:
        """The key the interface uses, not the class name."""
        import os

        return (self._reader_name or os.environ.get("PALIMPSEST_READER", "paddle")).lower()

    def reader_status(self) -> dict:
        from ..vlm import READERS

        reader = self.reader()
        name = type(reader).__name__
        model = getattr(reader, "model", None) or getattr(reader, "model_id", None)
        try:
            ok = reader.available()
        except Exception as exc:  # a status probe must never raise
            ok, self._reader_error = False, str(exc)
        why = None
        if not ok:
            asked = getattr(reader, "unavailable_because", None)
            why = asked() if asked else self._reader_error
        return {"reader": name, "model": str(model), "available": bool(ok),
                "name": self.reader_name(), "why": why,
                "where": READERS.get(self.reader_name(), {}).get("where", "local"),
                "choices": [{"name": k, **v} for k, v in READERS.items()]}

    # ------------------------------------------------------------- propose

    def propose(self, image: np.ndarray) -> tuple[np.ndarray, list[dict], dict]:
        """Preprocess and segment a page, with the model's guess per box.
        Used by the Label view, so labelling starts from a draft, not blank."""
        pre = preprocess(image)
        boxes = segment(pre.ink)
        probs, _, ok = self.classify(pre.grey, boxes)
        regions, k = [], 0
        for box, good in zip(boxes, ok):
            if not good:
                continue
            p = probs[k]
            k += 1
            regions.append({"box": [int(v) for v in box], "class": CLASS_NAMES[int(p.argmax())],
                            "confidence": round(float(p.max()), 4), "proposed": True})
        return pre.grey, regions, {"skew_deg": pre.skew_deg, "timing_ms": pre.timing_ms}

    # ------------------------------------------------------------- compile

    def compile(self, image: np.ndarray, title: str = "", boxes=None, deskew: bool = True) -> CompiledNote:
        """`boxes`, when given, replaces segmentation: compiling a labelled page
        from the Label view uses the boxes you drew. `deskew=False` is for pages
        drawn on the canvas, which are straight by construction; skipping it
        keeps the returned boxes in exactly the canvas's pixel coordinates."""
        with self._lock:
            return self._compile(image, title, boxes, deskew)

    def _compile(self, image: np.ndarray, title: str, given_boxes, deskew: bool = True) -> CompiledNote:
        notes: list[str] = []
        if given_boxes is None:
            pre = preprocess(image, deskew=deskew)
            grey = pre.grey
            boxes = segment(pre.ink)
            pre_ms = pre.total_ms
        else:
            grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes, pre_ms = [tuple(b) for b in given_boxes], 0.0

        probs, infer_ms, ok = self.classify(grey, boxes)
        boxes = [b for b, good in zip(boxes, ok) if good]
        n = max(1, len(boxes))
        share_pre, share_infer = pre_ms / n, infer_ms / n

        blocks: list[Block] = []
        for i, (box, p) in enumerate(zip(boxes, probs)):
            cls = CLASS_NAMES[int(p.argmax())]
            confidence = float(p.max())
            gate = decide(cls, confidence)
            blocks.append(Block(
                id=f"r{i + 1}", region_class=cls, confidence=round(confidence, 4),
                threshold=gate.threshold, held=gate.held, hold_reason=gate.reason,
                box=Box(x=int(box[0]), y=int(box[1]), w=int(box[2]), h=int(box[3])),
                timing_ms=Timing(preprocess_ms=share_pre, infer_ms=share_infer),
            ))
        if not blocks:
            notes.append("no regions were found on this page")

        for block in blocks:
            self.activity.append(
                source="compile", backbone=self.active, device=self.device,
                region_class=block.region_class, confidence=block.confidence,
                threshold=block.threshold, held=block.held,
                preprocess_ms=share_pre, infer_ms=share_infer,
                w=block.box.w, h=block.box.h, title=title[:60],
            )

        # ---- pass 1: read every admitted region
        crops_for = {b.id: self._reading_crop(grey, b.box) for b in blocks}
        for block in blocks:
            if block.held or block.region_class not in ("THEORY", "FORMULA", "TABLE"):
                continue
            began = time.perf_counter()
            try:
                reader = self.reader()
                crop = crops_for[block.id]
                if block.region_class == "THEORY":
                    block.content = reader.read_prose(crop)
                elif block.region_class == "FORMULA":
                    block.content = reader.read_formula(crop)
                else:
                    block.content = reader.read_table(crop)
            except Exception as exc:
                block.held = True
                block.content = None
                block.hold_reason = f"the reader could not read this region: {exc}"[:300]
            block.timing_ms.route_ms = (time.perf_counter() - began) * 1000

        # ---- pass 2: symbol table, then resolve formulas against it
        table = SymbolTable()
        for block in blocks:
            if isinstance(block.content, ProseContent):
                table.extend_from_text(block.content.markdown, block.box.y_centre, block.id)
            elif isinstance(block.content, FormulaContent):
                table.extend_from_text(block.content.expr or block.content.tex,
                                       block.box.y_centre, block.id)

        solved_by_block = {}
        for block in blocks:
            if not isinstance(block.content, FormulaContent) or not block.content.expr:
                continue
            began = time.perf_counter()
            solved = solve(block.content.expr, table.resolve(block.box.y_centre))
            block.content.answer = solved.answer
            block.content.bound = {display(k): v for k, v in solved.bound.items()}
            block.content.free = [display(k) for k in solved.free]
            if solved.error:
                notes.append(f"{block.id}: formula did not parse ({solved.error})")
            solved_by_block[block.id] = (solved, table.resolve(block.box.y_centre))
            block.timing_ms.route_ms += (time.perf_counter() - began) * 1000

        # ---- pass 3: a plot sketch is drawn from the nearest plottable formula
        for block in blocks:
            if block.held or block.region_class != "PLOT_SKETCH":
                continue
            began = time.perf_counter()
            block.content = self._figure_for(block, blocks, solved_by_block)
            block.timing_ms.route_ms += (time.perf_counter() - began) * 1000

        budget = CompileBudget(
            regions=len(blocks),
            service_ms=sum(b.timing_ms.total_ms for b in blocks),
            preprocess_ms=pre_ms,
            infer_ms=infer_ms,
            route_ms=sum(b.timing_ms.route_ms for b in blocks),
        )
        note = CompiledNote(
            note_id=time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4],
            title=title or "Untitled page",
            source_width=int(grey.shape[1]), source_height=int(grey.shape[0]),
            blocks=blocks, budget=budget,
            symbols={display(k): v for k, v in table.flat().items()},
            backbone=self.active or "", model_notes=notes,
        )
        self._write_vault(note, grey)
        return note

    @staticmethod
    def _reading_crop(grey: np.ndarray, box: Box) -> np.ndarray:
        """The reader gets the region at full working resolution with a little
        margin, not the 224 px classifier crop, which is too small to read."""
        pad = 8
        H, W = grey.shape
        return grey[max(0, box.y - pad):min(H, box.y + box.h + pad),
                    max(0, box.x - pad):min(W, box.x + box.w + pad)]

    @staticmethod
    def _figure_for(block: Block, blocks: list[Block], solved_by_block: dict) -> FigureContent:
        # Nearest formula by vertical distance that determines a curve.
        candidates = sorted(
            (abs(b.box.y_centre - block.box.y_centre), b.id) for b in blocks if b.id in solved_by_block
        )
        reasons = []
        for _, bid in candidates:
            solved, bindings = solved_by_block[bid]
            reason = refuse_reason(solved)
            if reason:
                reasons.append(f"{bid}: {reason}")
                continue
            plot_range = choose_plot_range(solved, bindings)
            if plot_range is None:
                reasons.append(f"{bid}: no usable range")
                continue
            xs, ys = plot_series(solved, plot_range)
            return FigureContent(
                caption=f"Plotted from {bid}", source_expr=solved.expr_str,
                free_symbol=display(solved.free[0]),
                x_range=(plot_range.x_min, plot_range.x_max),
                x_limits=(plot_range.limit_min, plot_range.limit_max),
                range_source=plot_range.source, series_x=xs, series_y=ys,
                y_clip=suggest_y_clip(ys),
            )
        return FigureContent(refused="no formula on this page determines a curve"
                             + (f" ({'; '.join(reasons)})" if reasons else ""))

    # ------------------------------------------------------------- vault

    def _write_vault(self, note: CompiledNote, grey: np.ndarray) -> None:
        folder = self.vault / note.note_id
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / "page.png"), grey)
        (folder / "note.json").write_text(note.model_dump_json(indent=1))
        (folder / "note.md").write_text(compose_markdown(note))


def compose_markdown(note: CompiledNote) -> str:
    """The note as plain markdown, the way an Obsidian vault keeps it."""
    out = [f"# {note.title}", "",
           f"<!-- compiled {note.note_id} · {note.backbone} · "
           f"{note.budget.service_ms:.0f} ms, load factor {note.budget.load_factor:.2f} -->", ""]
    for b in note.blocks:
        if b.held:
            out += [f"> [!warning] Held for review: {b.region_class} at {b.confidence:.2f}",
                    f"> {b.hold_reason or ''}", ""]
            continue
        c = b.content
        if isinstance(c, ProseContent):
            out += [c.markdown, ""]
        elif isinstance(c, FormulaContent):
            out += ["$$", c.tex, "$$", ""]
            if c.answer is not None:
                out += [f"*Resolved:* `{c.answer:.4g}`", ""]
        elif isinstance(c, TableContent):
            out += [c.markdown, ""]
        elif isinstance(c, FigureContent):
            out += [f"*Figure:* {c.refused or c.caption}", ""]
    return "\n".join(out)


def list_vault(vault: Path) -> list[dict]:
    out = []
    for path in sorted(vault.glob("*/note.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out.append({"note_id": data["note_id"], "title": data["title"],
                    "regions": len(data["blocks"]),
                    "held": sum(1 for b in data["blocks"] if b["held"])})
    return out
