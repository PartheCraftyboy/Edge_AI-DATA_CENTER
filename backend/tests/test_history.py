"""The append-only learning curve.

The point of the file is that it survives runs. These check the two ways a
long-lived CSV goes wrong: a column quietly shifting, and a row overwriting
the history it was supposed to extend.
"""

import csv

import pytest

from app import CLASS_NAMES
from app.history import FIELDS, append_run, read_history


class TestAppend:
    def test_first_run_writes_a_header(self, tmp_path):
        path = append_run(tmp_path / "history.csv", run="distil", backbone="squeezenet1_1")
        with path.open() as handle:
            assert next(csv.reader(handle)) == list(FIELDS)

    def test_runs_accumulate_rather_than_replace(self, tmp_path):
        path = tmp_path / "history.csv"
        for i in range(3):
            append_run(path, run="distil", round=i, regions_hand=10 * i)
        rows = read_history(path)
        assert [r["round"] for r in rows] == ["0", "1", "2"]
        assert [r["regions_hand"] for r in rows] == ["0", "10", "20"]

    def test_every_row_has_every_column(self, tmp_path):
        """A row written with only some fields must still fill the rest, or
        every column to the right of the gap shifts and the plot is wrong."""
        path = tmp_path / "history.csv"
        append_run(path, run="finetune", hand_accuracy=0.8)
        append_run(path, run="distil", round=2, teacher_agreement=0.9)
        for row in read_history(path):
            assert set(row) == set(FIELDS)
            assert row["timestamp"]

    def test_unknown_field_is_dropped_not_appended(self, tmp_path):
        """Silently widening the schema would break anything already plotting
        the file, so an unknown key is ignored rather than added."""
        path = append_run(tmp_path / "history.csv", run="distil", invented_metric=1.23)
        rows = read_history(path)
        assert "invented_metric" not in rows[0]
        assert len(rows) == 1

    def test_none_is_written_as_empty_not_the_string_none(self, tmp_path):
        """`float('None')` raises; an unscored run has to read back as blank."""
        path = append_run(tmp_path / "history.csv", run="distil", hand_accuracy=None)
        assert read_history(path)[0]["hand_accuracy"] == ""

    def test_a_class_count_column_exists_for_every_class(self):
        for name in CLASS_NAMES:
            assert f"n_{name.lower()}" in FIELDS

    def test_reading_a_missing_file_is_empty_not_an_error(self, tmp_path):
        assert read_history(tmp_path / "nothing.csv") == []


class TestPlot:
    def test_plotting_an_empty_history_says_so(self, tmp_path):
        from app.history import plot

        with pytest.raises(SystemExit):
            plot(tmp_path / "missing.csv", tmp_path / "out.png")

    def test_plot_writes_a_png(self, tmp_path):
        pytest.importorskip("matplotlib")
        from app.history import plot

        path = tmp_path / "history.csv"
        for i in range(3):
            append_run(path, run="distil", round=i, regions_hand=10 * i,
                       hand_accuracy=0.5 + 0.1 * i, teacher_agreement=0.6 + 0.1 * i,
                       teacher_vs_hand=0.9, n_theory=4 * i, n_formula=2 * i)
        out = plot(path, tmp_path / "history.png")
        assert out.exists() and out.stat().st_size > 1000

    def test_plot_tolerates_rows_with_gaps(self, tmp_path):
        """A fine-tune row has no teacher columns and a distil row has no
        synthetic ratio; the curve has to skip the gaps, not crash on them."""
        pytest.importorskip("matplotlib")
        from app.history import plot

        path = tmp_path / "history.csv"
        append_run(path, run="finetune", regions_hand=10, hand_accuracy=0.4)
        append_run(path, run="distil", round=1, regions_hand=10, teacher_agreement=0.7)
        assert plot(path, tmp_path / "h.png").exists()
