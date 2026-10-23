"""Architecture profiler tests (Module 3).

The module is about being able to account for what an architecture costs and
why. A profiler that reports numbers nobody checked is no better than quoting
them from a paper, so each figure here is verified against something known:
MACs against a hand-computed convolution, the downsampling cascade against the
224-to-7 progression, and the separable-convolution saving against the dense
convolution it replaces.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from app.models.profiler import count_macs, layer_cascade, profile


class TestMacCounting:
    def test_a_single_convolution_matches_the_hand_calculation(self):
        """A 3x3 convolution, 8 in, 16 out, over a 32x32 map with padding
        keeps the map size, so it is 3*3*8*16 multiply-accumulates per output
        position and there are 32*32 of them."""
        conv = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        macs = count_macs(conv, (1, 8, 32, 32))
        assert macs == 3 * 3 * 8 * 16 * 32 * 32

    def test_a_linear_layer_is_in_times_out(self):
        macs = count_macs(nn.Linear(64, 10), (1, 64))
        assert macs == 64 * 10

    def test_stride_two_quarters_the_cost(self):
        dense = nn.Conv2d(8, 16, 3, padding=1, bias=False)
        strided = nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False)
        assert count_macs(strided, (1, 8, 32, 32)) == count_macs(dense, (1, 8, 32, 32)) // 4

    def test_depthwise_separable_is_far_cheaper_than_dense(self):
        """The arithmetic the whole of PalimpsestNet rests on."""
        dense = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        separable = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, 128, 1, bias=False),
        )
        shape = (1, 64, 28, 28)
        assert count_macs(separable, shape) < count_macs(dense, shape) * 0.25


class TestCascade:
    def test_reports_the_224_to_7_progression(self):
        from app.models.zoo import PalimpsestNet

        stages = layer_cascade(PalimpsestNet(), (1, 3, 224, 224))
        sizes = [s["output"][-1] for s in stages if len(s["output"]) == 4]
        assert sizes[0] <= 112, sizes[:3]
        assert min(sizes) <= 7, sizes[-3:]

    def test_spatial_extent_never_grows(self):
        from app.models.zoo import PalimpsestNet

        stages = layer_cascade(PalimpsestNet(), (1, 3, 224, 224))
        sizes = [s["output"][-1] for s in stages if len(s["output"]) == 4]
        assert sizes == sorted(sizes, reverse=True), sizes

    def test_each_stage_names_its_module_type(self):
        from app.models.zoo import PalimpsestNet

        stages = layer_cascade(PalimpsestNet(), (1, 3, 224, 224))
        assert all(stage["type"] for stage in stages)


class TestProfile:
    @pytest.mark.parametrize("name", ["palimpsest_net", "squeezenet1_1"])
    def test_reports_the_expected_fields(self, name):
        report = profile(name, runs=3, warmup=1)
        for key in ("name", "parameters", "macs", "latency_ms", "device"):
            assert key in report, (name, report.keys())
        assert report["parameters"] > 0
        assert report["macs"] > 0
        assert report["latency_ms"]["p50"] > 0

    def test_the_designed_network_is_the_cheapest(self):
        """It has to sit at the cheap end of the frontier or it makes no
        point: a from-scratch model that is also expensive is just worse."""
        small = profile("palimpsest_net", runs=3, warmup=1)
        big = profile("resnet18", runs=3, warmup=1)
        assert small["parameters"] < big["parameters"]
        assert small["macs"] < big["macs"]

    def test_mobilenet_is_cheaper_than_resnet_in_macs(self):
        """Inverted residuals exist precisely to buy this."""
        assert profile("mobilenet_v2", runs=3, warmup=1)["macs"] < \
               profile("resnet18", runs=3, warmup=1)["macs"]
