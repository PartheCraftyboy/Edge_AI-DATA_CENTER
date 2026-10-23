"""HTTP surface tests, through FastAPI's TestClient.

The page store and vault are redirected to a temporary folder and the reader
is a fake, so these never touch your labelled pages, Ollama or the network.
Routes that need a trained checkpoint are skipped on a clean checkout.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.contracts import ProseContent
from app.data.real import PageStore
from app.pipeline.compile import discover_checkpoints

HAVE_CKPT = bool(discover_checkpoints())
needs_ckpt = pytest.mark.skipif(not HAVE_CKPT, reason="no trained checkpoint on this machine")


class Reader:
    def available(self):
        return True

    def read_prose(self, crop):
        return ProseContent(markdown="some prose", raw_lines=["some prose"])

    read_formula = read_table = read_prose


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", PageStore(tmp_path / "real"))
    monkeypatch.setattr(main.compiler, "vault", tmp_path / "vault")
    (tmp_path / "vault").mkdir()
    monkeypatch.setattr(main.compiler, "_reader", Reader())
    # `compiler` is module level and shared, so the selected reader leaks
    # between tests unless it is restored alongside the instance. Without
    # this, a test that switched readers left the next one's `set_reader(...)`
    # returning early onto the fake above.
    monkeypatch.setattr(main.compiler, "_reader_name", None)
    return TestClient(main.app)


def page_png() -> bytes:
    img = np.full((500, 700), 235, np.uint8)
    for y in (80, 110, 140, 300, 330):
        for x in range(60, 560, 14):
            cv2.line(img, (x, y), (x + 8, y - 10), 40, 2)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["available"] is True and len(body["classes"]) == 5


def test_frontend_is_served(client):
    if not main.FRONTEND.exists():
        pytest.skip("frontend folder missing")
    assert "Palimpsest" in client.get("/").text


def test_bad_upload_is_415(client):
    r = client.post("/api/compile", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 415


@needs_ckpt
def test_compile_and_read_back(client):
    r = client.post("/api/compile", files={"file": ("p.png", page_png(), "image/png")}, data={"title": "T"})
    assert r.status_code == 200, r.text
    note = r.json()
    assert note["title"] == "T" and "load_factor" in note["budget"]
    assert client.get(f"/api/notes/{note['note_id']}").json()["note_id"] == note["note_id"]
    assert client.get(f"/api/notes/{note['note_id']}/page.png").status_code == 200
    assert client.get("/api/notes").json()[0]["note_id"] == note["note_id"]


def test_note_path_traversal_is_404(client):
    assert client.get("/api/notes/..%2F..%2Fetc").status_code == 404


@needs_ckpt
def test_label_loop(client):
    rec = client.post("/api/pages", files={"file": ("p.png", page_png(), "image/png")}).json()
    pid = rec["id"]
    assert not rec["labelled"] and isinstance(rec["regions"], list)
    assert client.get(f"/api/pages/{pid}/image.png").status_code == 200

    bad = client.put(f"/api/pages/{pid}/labels", json={"regions": [{"box": [0, 0, 50, 50], "region_class": "DOODLE"}]})
    assert bad.status_code == 422

    good = client.put(f"/api/pages/{pid}/labels", json={"regions": [
        {"box": [50, 60, 520, 100], "region_class": "THEORY"},
        {"box": [50, 280, 520, 60], "region_class": "FORMULA"}]})
    assert good.status_code == 200 and good.json()["labelled"]
    stats = client.get("/api/pages/stats").json()
    assert stats["labelled_pages"] == 1 and stats["regions"] == 2
    assert client.get(f"/api/pages/{pid}/crop/0.png").status_code == 200

    note = client.post(f"/api/pages/{pid}/compile").json()
    assert [b["box"]["y"] for b in note["blocks"]] == [60, 280]  # your boxes, not segmentation

    assert client.delete(f"/api/pages/{pid}").status_code == 200
    assert client.get(f"/api/pages/{pid}").status_code == 404


@needs_ckpt
def test_models_and_switching(client):
    models = client.get("/api/models").json()
    assert models and sum(m["active"] for m in models) == 1
    other = next(m for m in models if not m["active"])
    assert client.post("/api/models", json={"key": other["key"]}).json()["active"] == other["key"]
    assert client.post("/api/models", json={"key": "nope/nothing"}).status_code == 404
    client.post("/api/models", json={"key": next(m for m in models if m["active"])["key"]})


@needs_ckpt
def test_metrics_and_pareto(client):
    m = client.get("/api/metrics").json()
    assert len(m["confusion"]) == 5 and 0 <= m["macro_f1"] <= 1
    assert all("latency_ms" in p for p in client.get("/api/pareto").json())


def test_train_rejects_unknown_backbone(client):
    assert client.post("/api/train", json={"backbone": "alexnet"}).status_code == 404


@needs_ckpt
def test_drawn_page_keeps_canvas_coordinates(client):
    """The Write view sends deskew=false: a drawn page is straight, and its
    boxes must land on the strokes in the canvas's own pixels."""
    img = np.full((1200, 1600), 255, np.uint8)
    for x in range(200, 900, 20):
        cv2.line(img, (x, 400), (x + 12, 380), 30, 4)
    r = client.post("/api/compile", files={"file": ("d.png", cv2.imencode(".png", img)[1].tobytes(), "image/png")},
                    data={"deskew": "false"})
    note = r.json()
    assert note["source_width"] == 1600
    box = note["blocks"][0]["box"]
    assert abs(box["x"] - 200) <= 4 and abs(box["y"] - 378) <= 6


class TestReaderSelection:
    """Switching the transcription model from the interface.

    The sidebar redraws itself from whichever of these two routes answered
    last, so the shape invariant below is not pedantry: the first cut of this
    returned only the reader fields, and the missing `device` turned every
    switch into a TypeError in the browser with a green test suite behind it.
    """

    # `none` is the one choice that builds without touching anything: `qwen`
    # opens a socket to Ollama inside `available()` and `paddle` pulls
    # weights, and this suite promises neither.
    OFFLINE = ("none",)

    def test_both_routes_answer_with_the_same_shape(self, client):
        health = client.get("/api/health").json()
        swapped = client.post("/api/reader", json={"name": "none"}).json()
        assert set(health) == set(swapped)

    @pytest.mark.parametrize("name", OFFLINE)
    def test_switching_reports_the_new_reader(self, client, name):
        body = client.post("/api/reader", json={"name": name})
        assert body.status_code == 200
        assert body.json()["name"] == name
        assert client.get("/api/health").json()["name"] == name

    def test_an_unknown_reader_is_refused_and_changes_nothing(self, client):
        before = client.get("/api/health").json()["name"]
        assert client.post("/api/reader", json={"name": "nope"}).status_code == 404
        assert client.get("/api/health").json()["name"] == before

    def test_every_offered_choice_is_one_the_server_accepts(self, client):
        """The dropdown is built from `choices`, so an entry the POST route
        rejects is a menu item that always errors."""
        offered = [c["name"] for c in client.get("/api/health").json()["choices"]]
        assert set(self.OFFLINE) <= set(offered)
        for name in offered:
            if name not in self.OFFLINE:
                continue  # building these would reach the network or disk
            assert client.post("/api/reader", json={"name": name}).status_code == 200
