"""The Ollama client, against a fake server: no network, no GPU, no model."""

from __future__ import annotations

import base64
import json

import cv2
import httpx
import numpy as np
import pytest

from app.vlm import OllamaVision, VisionReadError, VisionUnavailable
from app.vlm.ollama import table_shape


def fake(replies: list, seen: list | None = None) -> httpx.MockTransport:
    """A server that answers `/api/chat` with each reply in turn."""
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [
                {"name": "qwen2.5vl:3b", "size": 4 * 2**30, "size_vram": 3 * 2**30}]})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5vl:3b"}]})
        if seen is not None:
            seen.append(json.loads(request.content))
        item = queue.pop(0)
        if isinstance(item, httpx.Response):
            return item
        content = item if isinstance(item, str) else json.dumps(item)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content}})

    return httpx.MockTransport(handler)


@pytest.fixture
def crop():
    return np.full((224, 224), 240, dtype=np.uint8)


class TestRequest:
    def test_payload_carries_image_schema_and_zero_temperature(self, crop):
        seen: list = []
        vlm = OllamaVision(transport=fake([{"tex": "x", "expr": "x"}], seen))
        vlm.read_formula(crop)
        body = seen[0]
        assert body["model"] == "qwen2.5vl:3b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert body["options"]["num_ctx"] == 4096
        assert set(body["format"]["properties"]) == {"tex", "expr"}
        assert len(body["messages"][0]["images"]) == 1

    def test_small_crops_are_upscaled_before_sending(self, crop):
        seen: list = []
        vlm = OllamaVision(transport=fake([{"tex": "x", "expr": "x"}], seen), min_side=448)
        vlm.read_formula(crop)
        png = base64.b64decode(seen[0]["messages"][0]["images"][0])
        sent = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert max(sent.shape) == 448

    def test_large_crops_are_capped(self):
        vlm = OllamaVision(transport=fake([]), max_side=1280)
        png = base64.b64decode(vlm.encode(np.zeros((3000, 1500), np.uint8)))
        sent = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert sent.shape == (1280, 640)


class TestReaders:
    def test_formula_maps_to_contract_and_leaves_answer_to_solver(self, crop):
        vlm = OllamaVision(transport=fake([{"tex": "$\\rho = \\frac{\\lambda}{\\mu}$", "expr": "rho = lambda/mu"}]))
        out = vlm.read_formula(crop)
        assert out.kind == "formula"
        assert out.tex == "\\rho = \\frac{\\lambda}{\\mu}"
        assert out.expr == "rho = lambda/mu"
        assert out.answer is None and out.free == []

    def test_prose_keeps_raw_lines_beside_tidied_markdown(self, crop):
        vlm = OllamaVision(transport=fake([{"lines": ["arival rate", "", "lambda = 25"],
                                             "markdown": "Arrival rate, lambda = 25"}]))
        out = vlm.read_prose(crop)
        assert out.raw_lines == ["arival rate", "lambda = 25"]
        assert "lambda = 25" in out.markdown

    def test_table_counts_rows_and_columns(self, crop):
        cells = [["t", "v"], ["1", "2"], ["3", "4"], ["", ""]]
        out = OllamaVision(transport=fake([{"cells": cells}])).read_table(crop)
        assert (out.rows, out.cols) == (3, 2)
        assert out.markdown.splitlines()[1] == "| --- | --- |"

    def test_unruled_table_still_comes_back_as_a_grid(self, crop):
        # Regression: qwen2.5vl:3b read an unruled table as plain lines
        # ("GLOR 5\nSONIA 9\n...") when asked for markdown, and the reader
        # passed it through as a table of 0 rows and 0 columns. Asking for
        # cells makes the grid a property of the schema, not of the prompt.
        cells = [["GLOR", "5"], ["SONIA", "9"], ["OCEANE", "u2xpcysgwoi"]]
        out = OllamaVision(transport=fake([{"cells": cells}])).read_table(crop)
        assert (out.rows, out.cols) == (2, 2)
        assert "| SONIA | 9 |" in out.markdown

    def test_ragged_rows_are_padded(self, crop):
        out = OllamaVision(transport=fake([{"cells": [["a", "b", "c"], ["1"]]}])).read_table(crop)
        assert (out.rows, out.cols) == (1, 3)
        assert "| 1 |  |  |" in out.markdown

    def test_pipes_in_cells_are_escaped(self, crop):
        out = OllamaVision(transport=fake([{"cells": [["x"], ["a|b"]]}])).read_table(crop)
        assert "a\\|b" in out.markdown and out.cols == 1


class TestFailures:
    def test_bad_reply_is_retried_once(self, crop):
        vlm = OllamaVision(transport=fake(["not json", {"tex": "x", "expr": "x"}]))
        assert vlm.read_formula(crop).tex == "x"

    def test_two_bad_replies_hold_the_region(self, crop):
        vlm = OllamaVision(transport=fake(["nope", {"tex": "missing expr"}]))
        with pytest.raises(VisionReadError):
            vlm.read_formula(crop)

    def test_missing_model_is_unavailable_not_a_read_error(self, crop):
        vlm = OllamaVision(transport=fake([httpx.Response(404, text="model not found")]))
        with pytest.raises(VisionUnavailable, match="ollama pull"):
            vlm.read_formula(crop)

    def test_server_down_is_unavailable(self, crop):
        def refuse(request):
            raise httpx.ConnectError("refused", request=request)
        vlm = OllamaVision(transport=httpx.MockTransport(refuse))
        with pytest.raises(VisionUnavailable):
            vlm.read_formula(crop)
        assert vlm.available() is False

    def test_available_sees_the_pulled_model(self):
        assert OllamaVision(transport=fake([])).available() is True


def test_table_shape_ignores_separator_and_prose():
    assert table_shape("notes\n| a | b | c |\n| :-- | --: | --- |\n| 1 | 2 | 3 |") == (1, 3)
    assert table_shape("no table here") == (0, 0)


def test_placement_reports_spill_to_system_ram():
    where = OllamaVision(transport=fake([])).placement()
    assert where == {"size_mib": 4096, "vram_mib": 3072, "ram_mib": 1024, "gpu_fraction": 0.75}


class TestNoReader:
    """Running the app with no model at all, for frontend work while a
    training pass holds the GPU. One press of Compile would otherwise load
    PaddleOCR-VL and evict the model the run is using."""

    def test_make_reader_none_gives_a_reader_that_reads_nothing(self):
        from app.vlm import NoReader, make_reader

        assert isinstance(make_reader("none"), NoReader)

    def test_it_reports_itself_unavailable(self):
        from app.vlm import NoReader

        assert NoReader().available() is False

    @pytest.mark.parametrize("method", ["read_prose", "read_formula", "read_table"])
    def test_every_reader_refuses_the_way_an_offline_model_does(self, method):
        """`VisionUnavailable`, so the compile path holds the region with a
        reason rather than needing a special case for this mode."""
        from app.vlm import NoReader, VisionUnavailable

        with pytest.raises(VisionUnavailable):
            getattr(NoReader(), method)(None)

    def test_it_cannot_be_used_as_a_judge_either(self):
        from app.vlm import NoReader, VisionUnavailable

        with pytest.raises(VisionUnavailable):
            NoReader().classify_region(None)

    def test_an_unknown_reader_name_still_says_what_is_available(self):
        from app.vlm import make_reader

        with pytest.raises(ValueError, match="none"):
            make_reader("nonesuch")
