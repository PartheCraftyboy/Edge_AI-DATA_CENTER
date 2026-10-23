"""Ingesting public page corpora.

These are acquisition shims, so what is worth testing is not the download but
the filtering: a corpus that quietly contributes thumbnails, non-images or
more pages than asked for would waste a night of teacher time before anyone
noticed. The remote filesystem is replaced with a local file throughout, so
nothing here touches the network.
"""

import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.data import gnhk, iam


def page_png(h=900, w=700, value=200) -> bytes:
    image = np.full((h, w), value, np.uint8)
    cv2.putText(image, "handwriting", (40, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, 20, 3)
    return cv2.imencode(".png", image)[1].tobytes()


class LocalFS:
    """Stands in for HfFileSystem: one local file per remote path."""

    def __init__(self, mapping):
        self.mapping = mapping

    def open(self, path, mode="rb"):
        for key, local in self.mapping.items():
            if key in path:
                return open(local, mode)
        raise FileNotFoundError(path)


# ------------------------------------------------------------------- IAM

def make_tar(path: Path, entries) -> Path:
    with tarfile.open(path, "w:gz") as tar:
        for name, payload in entries:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return path


@pytest.fixture
def iam_fs(tmp_path, monkeypatch):
    archive = make_tar(tmp_path / "formsA-D.tgz", [
        ("forms/a01-000.png", page_png()),
        ("forms/readme.txt", b"not an image"),
        ("forms/a01-001.png", page_png(h=60, w=50)),   # too small
        ("forms/a01-002.png", page_png()),
        ("forms/a01-003.png", page_png()),
    ])
    fs = LocalFS({"formsA-D.tgz": archive})
    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda *a, **k: fs)
    monkeypatch.setattr(iam, "ARCHIVES", ("formsA-D.tgz",))
    return fs


class TestIAM:
    def test_pages_are_written_as_png(self, iam_fs, tmp_path):
        manifest = iam.ingest(tmp_path / "out", limit=10)
        written = sorted(p.name for p in (tmp_path / "out" / "pages").glob("*.png"))
        assert written == ["a01-000.png", "a01-002.png", "a01-003.png"]
        assert manifest["pages"] == 3

    def test_a_non_image_member_is_ignored(self, iam_fs, tmp_path):
        iam.ingest(tmp_path / "out", limit=10)
        assert not (tmp_path / "out" / "pages" / "readme.png").exists()

    def test_a_thumbnail_is_skipped_not_upscaled(self, iam_fs, tmp_path):
        """The working size is 1600 px; a 60 px page would be blown up into a
        blur that teaches the classifier nothing true about handwriting."""
        manifest = iam.ingest(tmp_path / "out", limit=10)
        assert manifest["skipped"] == 1

    def test_limit_stops_the_stream_early(self, iam_fs, tmp_path):
        assert iam.ingest(tmp_path / "out", limit=2)["pages"] == 2

    def test_the_manifest_states_the_real_licence(self, iam_fs, tmp_path):
        manifest = iam.ingest(tmp_path / "out", limit=1)
        assert "non-commercial" in manifest["licence"]

    def test_the_manifest_indexes_what_was_written(self, iam_fs, tmp_path):
        iam.ingest(tmp_path / "out", limit=2)
        index = json.loads((tmp_path / "out" / "manifest.json").read_text())["index"]
        assert len(index) == 2 and all(e["w"] and e["h"] for e in index)


# ------------------------------------------------------------------ GNHK

@pytest.fixture
def gnhk_fs(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "image": [{"bytes": page_png(), "path": "eng_AA_001.jpg"},
                  {"bytes": page_png(h=50, w=40), "path": "eng_AA_002.jpg"},
                  {"bytes": None, "path": "eng_AA_003.jpg"},
                  {"bytes": page_png(), "path": "eng_AA_004.jpg"}],
        "text": ["one", "two", "three", "four"],
    }
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table(rows), shard)
    fs = LocalFS({"train-00000": shard})
    monkeypatch.setattr("huggingface_hub.HfFileSystem", lambda *a, **k: fs)
    monkeypatch.setattr(gnhk, "SHARDS", ("data/train-00000-of-00002.parquet",))
    return fs


class TestGNHK:
    def test_pages_are_written_named_after_the_source(self, gnhk_fs, tmp_path):
        manifest = gnhk.ingest(tmp_path / "out", limit=10)
        written = sorted(p.name for p in (tmp_path / "out" / "pages").glob("*.png"))
        assert written == ["eng_AA_001.png", "eng_AA_004.png"]
        assert manifest["pages"] == 2

    def test_a_row_with_no_image_is_skipped_not_fatal(self, gnhk_fs, tmp_path):
        assert gnhk.ingest(tmp_path / "out", limit=10)["skipped"] == 2

    def test_limit_is_respected(self, gnhk_fs, tmp_path):
        assert gnhk.ingest(tmp_path / "out", limit=1)["pages"] == 1

    def test_the_manifest_records_the_licence(self, gnhk_fs, tmp_path):
        assert gnhk.ingest(tmp_path / "out", limit=1)["licence"] == "CC-BY-4.0"

    def test_written_pages_are_readable_greyscale(self, gnhk_fs, tmp_path):
        gnhk.ingest(tmp_path / "out", limit=1)
        page = next((tmp_path / "out" / "pages").glob("*.png"))
        assert cv2.imread(str(page), cv2.IMREAD_GRAYSCALE) is not None


class TestIngestedPagesFeedTheTeacher:
    def test_a_folder_of_pages_becomes_regions_to_label(self, gnhk_fs, tmp_path):
        """`app.distil --images <dir>` has to find these and segment them."""
        from app.data.real import PageStore
        from app.distil import page_images

        gnhk.ingest(tmp_path / "out", limit=1)
        store = PageStore(root=tmp_path / "real")
        found = list(page_images(store, tmp_path / "out" / "pages"))
        assert len(found) == 1
        page_id, split, grey, boxes, hand = found[0]
        assert page_id.startswith("ext-") and grey is not None and hand == {}
        assert split in ("train", "val", "test")
