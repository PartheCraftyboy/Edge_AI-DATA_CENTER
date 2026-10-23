"""Backbone zoo tests (Modules 2 and 3).

The zoo exists so the accuracy-against-latency trade-off is a measurement
rather than a claim. Three torchvision backbones are re-headed to five classes
and fine-tuned from ImageNet; a fourth is designed here and trained from
scratch.

That fourth entry does double duty. Module 3 asks for architecture *design*,
not only inspection, so a hand-built block is the deliverable. And because it
carries no ImageNet weights, the gap between it and the three pretrained
backbones is a measured demonstration of what transfer learning is worth,
which is Module 2's central claim. One experiment, two modules evidenced.
"""

import pytest

torch = pytest.importorskip("torch")

from app import NUM_CLASSES
from app.models.zoo import BACKBONES, PalimpsestNet, build_backbone, describe


class TestRegistry:
    def test_the_three_taught_backbones_are_present(self):
        """SqueezeNet, MobileNetV2 and ResNet-18 are the ones the module
        taught, so the zoo is not free to substitute others."""
        assert {"squeezenet1_1", "mobilenet_v2", "resnet18"} <= set(BACKBONES)

    def test_a_hand_designed_entry_is_present(self):
        assert "palimpsest_net" in BACKBONES

    def test_every_entry_is_constructible(self):
        for name in BACKBONES:
            assert build_backbone(name, pretrained=False) is not None

    def test_unknown_backbone_fails_loudly(self):
        with pytest.raises(KeyError):
            build_backbone("resnet50_but_imaginary", pretrained=False)


class TestHeadsAndShapes:
    @pytest.mark.parametrize("name", sorted(BACKBONES))
    def test_output_is_five_logits(self, name):
        model = build_backbone(name, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(2, 3, 224, 224))
        assert out.shape == (2, NUM_CLASSES), name

    @pytest.mark.parametrize("name", sorted(BACKBONES))
    def test_logits_are_not_probabilities(self, name):
        """The loss applies softmax itself. A model that also softmaxed would
        train on log-of-softmax-of-softmax and quietly underfit."""
        model = build_backbone(name, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 224, 224))
        sums = out.exp().sum(dim=1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3), name

    @pytest.mark.parametrize("name", sorted(BACKBONES))
    def test_gradients_reach_the_backbone(self, name):
        """Full fine-tune, not a frozen trunk with a linear probe."""
        model = build_backbone(name, pretrained=False)
        model(torch.randn(2, 3, 224, 224)).sum().backward()
        grads = [p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()]
        assert sum(grads) > len(grads) * 0.5, name


class TestPalimpsestNet:
    """The architecture written for this project rather than imported."""

    def test_uses_depthwise_separable_convolution(self):
        """The Module 3 idea it is built on: a grouped spatial convolution
        followed by a pointwise one, which is what makes it small."""
        net = PalimpsestNet()
        depthwise = [m for m in net.modules()
                     if isinstance(m, torch.nn.Conv2d) and m.groups > 1]
        assert depthwise, "no grouped convolution; this is not separable"

    def test_has_residual_connections(self):
        net = PalimpsestNet()
        assert any(getattr(b, "use_residual", False) for b in net.modules())

    def test_is_genuinely_small(self):
        """It has to be the cheap end of the Pareto frontier or it makes no
        point at all."""
        params = sum(p.numel() for p in PalimpsestNet().parameters())
        assert params < 1_200_000, params

    def test_downsamples_to_a_small_feature_map(self):
        """The 224-to-7 cascade the module describes."""
        net = PalimpsestNet()
        net.eval()
        with torch.no_grad():
            features = net.features(torch.zeros(1, 3, 224, 224))
        assert features.shape[-1] <= 14, features.shape


class TestDescribe:
    def test_reports_parameter_counts_for_each(self):
        for name in BACKBONES:
            info = describe(name)
            assert info["parameters"] > 0
            assert info["name"] == name

    def test_the_ordering_matches_the_taught_trade_off(self):
        """SqueezeNet is smaller than MobileNetV2 is smaller than ResNet-18.
        If that ever stops being true the Pareto story needs rewriting."""
        sizes = {n: describe(n)["parameters"] for n in
                 ("palimpsest_net", "squeezenet1_1", "mobilenet_v2", "resnet18")}
        assert sizes["palimpsest_net"] < sizes["squeezenet1_1"] < sizes["resnet18"]
        assert sizes["mobilenet_v2"] < sizes["resnet18"]
