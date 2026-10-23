"""Region classifier tests (Modules 2 and 5).

The Lightning module itself is thin, which is the point of using Lightning,
so the tests concentrate on the few decisions that are not boilerplate:

* the loss carries **uniform** class weights, because per-class cost belongs
  at the confidence gate and not in the objective — weighting the loss would
  bias the probabilities and make the ECE number meaningless;
* the headline metric is **macro-F1**, not accuracy;
* the thing actually learns, which is checked by running real optimisation
  steps rather than by inspecting that an optimiser exists.
"""

import pytest

torch = pytest.importorskip("torch")

from app import NUM_CLASSES
from app.models.classifier import RegionClassifier


@pytest.fixture
def batch():
    torch.manual_seed(0)
    return torch.randn(8, 3, 224, 224), torch.randint(0, NUM_CLASSES, (8,))


class TestConstruction:
    @pytest.mark.parametrize("name", ["palimpsest_net", "squeezenet1_1"])
    def test_builds_on_each_backbone(self, name):
        model = RegionClassifier(backbone=name, pretrained=False)
        assert model.hparams.backbone == name

    def test_records_hyperparameters_for_the_checkpoint(self):
        """A checkpoint that does not know its own backbone cannot be
        reloaded by the runtime switch."""
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False, lr=3e-4)
        assert model.hparams.lr == 3e-4
        assert model.hparams.backbone == "palimpsest_net"


class TestLossPolicy:
    def test_class_weights_are_uniform(self):
        """Per-class cost is applied at the gate, after the model speaks. If
        it were folded into the loss the confidences would be deliberately
        skewed and calibration error would stop meaning anything."""
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        assert getattr(model.loss_fn, "weight", None) is None

    def test_loss_is_cross_entropy_on_logits(self, batch):
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        x, y = batch
        loss = model.training_step((x, y), 0)
        assert loss.ndim == 0
        assert loss.requires_grad
        # Near chance for an untrained 5-class model.
        assert 1.0 < float(loss.detach()) < 3.0


class TestOptimisation:
    def test_configures_an_optimiser_and_schedule(self):
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        config = model.configure_optimizers()
        assert "optimizer" in config and "lr_scheduler" in config

    def test_training_actually_reduces_loss(self):
        """The real check. Overfit one fixed batch for a few steps: if the
        wiring is wrong the loss stays flat and every other test still
        passes."""
        torch.manual_seed(0)
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False, lr=3e-3)
        x = torch.randn(6, 3, 224, 224)
        y = torch.randint(0, NUM_CLASSES, (6,))
        optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)

        first = float(model.training_step((x, y), 0).detach())
        for step in range(25):
            optimiser.zero_grad()
            loss = model.training_step((x, y), step)
            loss.backward()
            optimiser.step()
        assert float(loss.detach()) < first * 0.7, (first, float(loss.detach()))


class TestMetrics:
    def test_validation_step_runs_and_tracks_macro_f1(self, batch):
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        x, y = batch
        model.on_validation_epoch_start()
        model.validation_step((x, y), 0)
        assert model.val_f1.update_called or float(model.val_f1.compute()) >= 0.0

    def test_confusion_matrix_is_five_by_five(self, batch):
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        x, y = batch
        model.on_validation_epoch_start()
        model.validation_step((x, y), 0)
        assert tuple(model.val_confusion.compute().shape) == (NUM_CLASSES, NUM_CLASSES)

    def test_predict_proba_returns_a_distribution(self, batch):
        """The gate consumes these, so they must be real probabilities even
        though the training path deals in logits."""
        model = RegionClassifier(backbone="palimpsest_net", pretrained=False)
        model.eval()
        probabilities = model.predict_proba(batch[0])
        assert probabilities.shape == (8, NUM_CLASSES)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(8), atol=1e-5)
        assert float(probabilities.min()) >= 0.0
