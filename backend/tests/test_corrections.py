"""Telling the system it got it wrong.

The case these come from: someone wrote `x^2 = 49` and the note said
`x^2 = 4`. The class was right, the gate was confident and correct to be, and
the reading was wrong, so nothing already in the pipeline had anywhere to put
it.
"""

import json

import numpy as np
import pytest

from app import corrections as c


class TestRecord:
    def test_a_wrong_reading_is_a_reading_error(self, tmp_path):
        entry = c.record(tmp_path / "c.jsonl", note_id="n", block_id="r1", box=[0, 0, 9, 9],
                         region_class="FORMULA", read_as="x^2 = 4", should_be="x^2 = 49")
        assert entry["kind"] == "reading"

    def test_a_wrong_class_is_a_class_error(self, tmp_path):
        entry = c.record(tmp_path / "c.jsonl", note_id="n", block_id="r1", box=[0, 0, 9, 9],
                         region_class="FORMULA", correct_class="THEORY")
        assert entry["kind"] == "class"

    def test_both_wrong_is_recorded_as_both(self, tmp_path):
        entry = c.record(tmp_path / "c.jsonl", note_id="n", block_id="r1", box=[0, 0, 9, 9],
                         region_class="FORMULA", correct_class="THEORY",
                         read_as="x^2", should_be="Parth")
        assert entry["kind"] == "class+reading"

    def test_agreeing_with_the_reading_is_a_confirmation(self, tmp_path):
        """A file of nothing but complaints cannot be scored, so blocks that
        were right have to be recordable too."""
        entry = c.record(tmp_path / "c.jsonl", note_id="n", block_id="r1", box=[0, 0, 9, 9],
                         region_class="FORMULA", read_as="x^2 = 49", should_be="x^2 = 49")
        assert entry["kind"] == "confirmed"

    def test_the_file_is_append_only(self, tmp_path):
        path = tmp_path / "c.jsonl"
        for i in range(4):
            c.record(path, note_id="n", block_id=f"r{i}", box=[0, 0, 9, 9],
                     region_class="THEORY", should_be=str(i))
        assert len(c.load(path)) == 4

    def test_a_half_written_line_does_not_lose_the_rest(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c.record(path, note_id="n", block_id="r1", box=[0, 0, 9, 9], region_class="THEORY")
        with path.open("a") as handle:
            handle.write('{"truncated": \n')
        c.record(path, note_id="n", block_id="r2", box=[0, 0, 9, 9], region_class="THEORY")
        assert len(c.load(path)) == 2

    def test_the_box_is_kept_so_the_crop_can_be_cut_again(self, tmp_path):
        entry = c.record(tmp_path / "c.jsonl", note_id="n", block_id="r1",
                         box=[12, 34, 56, 78], region_class="THEORY")
        assert entry["box"] == [12, 34, 56, 78]


class TestReport:
    def test_confusions_are_counted_by_direction(self, tmp_path):
        path = tmp_path / "c.jsonl"
        for _ in range(3):
            c.record(path, note_id="n", block_id="r", box=[0, 0, 9, 9],
                     region_class="FORMULA", correct_class="THEORY")
        c.record(path, note_id="n", block_id="r", box=[0, 0, 9, 9],
                 region_class="THEORY", correct_class="TABLE")
        report = c.report(path)
        assert report["confusions"]["FORMULA -> THEORY"] == 3
        assert report["class_errors"] == 4

    def test_eval_set_is_only_the_regions_with_known_text(self, tmp_path):
        path = tmp_path / "c.jsonl"
        c.record(path, note_id="n", block_id="r1", box=[0, 0, 9, 9],
                 region_class="FORMULA", read_as="x^2 = 4", should_be="x^2 = 49")
        c.record(path, note_id="n", block_id="r2", box=[0, 0, 9, 9],
                 region_class="FORMULA", correct_class="THEORY")
        assert [e["should_be"] for e in c.eval_set(path)] == ["x^2 = 49"]

    def test_empty_file_reports_nothing_rather_than_dividing_by_zero(self, tmp_path):
        assert c.report(tmp_path / "missing.jsonl")["entries"] == 0

    def test_normalise_ignores_spacing(self):
        assert c.normalise("x^2 = 49") == c.normalise("X^2=49")


class TestEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from app import main

        monkeypatch.setattr(c, "CORRECTIONS_PATH", tmp_path / "c.jsonl")
        monkeypatch.setattr(main.compiler, "vault", tmp_path / "vault")
        main.store.root = tmp_path / "real"
        (main.store.root / "pages").mkdir(parents=True, exist_ok=True)
        folder = tmp_path / "vault" / "note1"
        folder.mkdir(parents=True)
        note = {
            "note_id": "note1", "title": "t", "source_width": 200, "source_height": 200,
            "blocks": [{"id": "r1", "region_class": "FORMULA", "confidence": 1.0,
                        "threshold": 0.88, "held": False, "hold_reason": None,
                        "box": {"x": 10, "y": 10, "w": 80, "h": 40},
                        "content": {"kind": "formula", "tex": "x^2 = 4", "expr": "x**2 = 4"}}],
            "budget": {"regions": 1, "service_ms": 1.0, "preprocess_ms": 0.0,
                       "infer_ms": 1.0, "route_ms": 0.0},
            "symbols": {}, "backbone": "x", "model_notes": [], "held_count": 0,
            "mean_confidence": 1.0,
        }
        (folder / "note.json").write_text(json.dumps(note))
        import cv2

        cv2.imwrite(str(folder / "page.png"), np.full((200, 200), 235, np.uint8))
        return TestClient(main.app)

    def test_a_reading_correction_is_recorded(self, client, tmp_path):
        reply = client.post("/api/notes/note1/blocks/r1/correction",
                            json={"should_be": "x^2 = 49"})
        assert reply.status_code == 200
        entry = reply.json()["correction"]
        assert entry["kind"] == "reading"
        assert entry["read_as"] == "x^2 = 4" and entry["should_be"] == "x^2 = 49"

    def test_a_reading_correction_creates_no_label(self, client):
        """It says nothing about what kind of region it is, so it must not
        invent a training label that claims it does."""
        reply = client.post("/api/notes/note1/blocks/r1/correction",
                            json={"should_be": "x^2 = 49"})
        assert reply.json()["labelled_page"] is None

    def test_a_class_correction_becomes_a_hand_label(self, client):
        from app import main

        reply = client.post("/api/notes/note1/blocks/r1/correction",
                            json={"correct_class": "THEORY"})
        assert reply.json()["labelled_page"]
        pages = [p for p in main.store.all() if p["source_name"] == "note:note1"]
        assert pages and pages[0]["regions"][0]["class"] == "THEORY"

    def test_the_correction_is_kept_on_the_note(self, client, tmp_path):
        client.post("/api/notes/note1/blocks/r1/correction", json={"should_be": "x^2 = 49"})
        stored = json.loads((tmp_path / "vault" / "note1" / "note.json").read_text())
        assert stored["blocks"][0]["correction"]["should_be"] == "x^2 = 49"

    def test_an_unknown_class_is_refused(self, client):
        reply = client.post("/api/notes/note1/blocks/r1/correction",
                            json={"correct_class": "SCRIBBLE"})
        assert reply.status_code == 422

    def test_the_endpoint_writes_where_the_constant_points(self, client, tmp_path):
        """A default argument is bound at import, so `record(path=CONST)` in a
        signature ignores a redirected constant and writes to the real file.
        Nine fixture rows reached `data/corrections.jsonl` that way."""
        client.post("/api/notes/note1/blocks/r1/correction", json={"should_be": "x^2 = 49"})
        assert (tmp_path / "c.jsonl").exists(), "the correction went somewhere else"
        assert len(c.load(tmp_path / "c.jsonl")) == 1

    def test_the_review_queue_puts_confident_regions_first(self, client):
        """The tempting queue is the held regions, and it is the wrong one. A
        held region is already handled: nothing was written into the note.
        The expensive errors are the confident ones, written in as fact."""
        rows = client.get("/api/review").json()["regions"]
        assert rows and rows[0]["confidence"] == max(r["confidence"] for r in rows)

    def test_a_corrected_region_leaves_the_queue(self, client):
        before = client.get("/api/review").json()["pending"]
        client.post("/api/notes/note1/blocks/r1/correction", json={"should_be": "x^2 = 49"})
        assert client.get("/api/review").json()["pending"] == before - 1

    def test_the_queue_can_be_filtered_to_confident_regions(self, client):
        rows = client.get("/api/review?only=confident").json()["regions"]
        assert all(r["confidence"] >= 0.9 for r in rows)

    def test_the_queue_carries_what_was_read_and_where(self, client):
        row = client.get("/api/review").json()["regions"][0]
        assert row["read_as"] == "x^2 = 4"
        assert set(row["box"]) == {"x", "y", "w", "h"}
        assert row["note_id"] and row["block_id"]

    def test_an_unknown_block_is_a_404(self, client):
        assert client.post("/api/notes/note1/blocks/r9/correction",
                           json={"should_be": "x"}).status_code == 404
