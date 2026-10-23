"""The live feed between the server and palimtop.

One file, appended by the compiler and tailed by the monitor. What these
check is the behaviour a tail depends on and a naive implementation gets
wrong: a partial line, a trimmed file, and a writer that must never be able
to fail the compile it is reporting on.
"""

import json

import pytest

from app.metrics.activity import ActivityLog
from app.tui import panels


class TestAppendAndRead:
    def test_a_reader_sees_what_was_appended(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(region_class="THEORY", confidence=0.9)
        records, offset = log.read(0)
        assert [r["region_class"] for r in records] == ["THEORY"]
        assert offset > 0

    def test_reading_again_from_the_offset_returns_only_new_records(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(region_class="THEORY")
        _, offset = log.read(0)
        log.append(region_class="FORMULA")
        records, _ = log.read(offset)
        assert [r["region_class"] for r in records] == ["FORMULA"]

    def test_an_idle_server_yields_nothing_not_an_error(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(region_class="THEORY")
        _, offset = log.read(0)
        for _ in range(3):
            records, offset = log.read(offset)
            assert records == []

    def test_a_half_written_line_is_left_for_next_time(self, tmp_path):
        """The compiler may be mid-write when the monitor reads. A line with
        no newline yet must not be parsed, and must not be skipped either."""
        path = tmp_path / "a.jsonl"
        log = ActivityLog(path)
        log.append(region_class="THEORY")
        with path.open("a") as handle:
            handle.write('{"region_class": "FORM')
        records, offset = log.read(0)
        assert [r["region_class"] for r in records] == ["THEORY"]
        with path.open("a") as handle:
            handle.write('ULA"}\n')
        more, _ = log.read(offset)
        assert [r["region_class"] for r in more] == ["FORMULA"]

    def test_a_trimmed_file_restarts_rather_than_seeking_past_the_end(self, tmp_path):
        path = tmp_path / "a.jsonl"
        log = ActivityLog(path)
        for i in range(5):
            log.append(region_class="THEORY", i=i)
        _, offset = log.read(0)
        path.write_text(json.dumps({"region_class": "TABLE"}) + "\n")
        records, _ = log.read(offset)
        assert [r["region_class"] for r in records] == ["TABLE"]

    def test_the_file_is_capped(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl", max_lines=50)
        for i in range(400):
            log.append(region_class="THEORY", i=i, pad="x" * 200)
        assert len(log.read(0)[0]) <= 50

    def test_a_timestamp_is_added(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(region_class="THEORY")
        assert log.read(0)[0][0]["t"] > 0

    def test_a_failed_write_never_raises(self, tmp_path):
        """Monitoring is not worth a 500 on someone's page."""
        log = ActivityLog(tmp_path / "nope" / "a.jsonl")
        log.path = tmp_path  # a directory: every write will fail
        log.append(region_class="THEORY")  # must not raise

    def test_reading_a_file_that_does_not_exist_is_empty(self, tmp_path):
        assert ActivityLog(tmp_path / "missing.jsonl").read(0) == ([], 0)

    def test_end_skips_what_is_already_there(self, tmp_path):
        log = ActivityLog(tmp_path / "a.jsonl")
        log.append(region_class="THEORY")
        records, _ = log.read(log.end())
        assert records == []


class TestLivePanels:
    def test_activity_panel_says_so_when_nothing_has_happened(self):
        assert panels.activity_panel([]) is not None

    def test_activity_panel_renders_records(self):
        rows = [{"t": 0, "region_class": "FORMULA", "confidence": 0.99, "held": False,
                 "w": 10, "h": 10, "title": "page"}]
        assert panels.activity_panel(rows) is not None

    def test_mix_panel_handles_an_empty_mix_without_dividing_by_zero(self):
        assert panels.mix_panel({}, {}) is not None

    def test_mix_panel_renders_counts(self):
        assert panels.mix_panel({"THEORY": 3, "FORMULA": 1},
                                {"THEORY": 0.8, "FORMULA": 0.99}) is not None


class TestCompilerWritesActivity:
    def test_compiling_a_page_appends_one_record_per_region(self, tmp_path, monkeypatch):
        import numpy as np

        from app.pipeline.compile import Compiler

        compiler = Compiler(vault=tmp_path / "vault")
        compiler.activity = ActivityLog(tmp_path / "a.jsonl")
        if compiler.active is None:
            pytest.skip("no trained checkpoint to compile with")

        class Silent:
            def read_prose(self, image): raise RuntimeError("offline")
            def read_formula(self, image): raise RuntimeError("offline")
            def read_table(self, image): raise RuntimeError("offline")

        compiler._reader = Silent()
        page = np.full((600, 800), 235, np.uint8)
        page[100:160, 80:500] = 40
        note = compiler.compile(page, title="t", deskew=False)
        records, _ = compiler.activity.read(0)
        assert len(records) == len(note.blocks)
        if records:
            assert records[0]["source"] == "compile"
            assert records[0]["region_class"] in {b.region_class for b in note.blocks}
