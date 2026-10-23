"""
The region classifier as a Lightning module — Modules 2 and 5.

Lightning rather than a hand-rolled loop, deliberately: it supplies
checkpointing, the validation hooks the metrics need, and device-agnostic
execution, so the same code runs on CPU while the dGPU is switched off and on
CUDA when it is back, with nothing to change.

Two choices here are policy rather than convention.

**The loss uses uniform class weights.** A misread formula costs nine times a
misread paragraph in this system, and the obvious move is to tell the loss so.
That is wrong here. Weighting the loss biases the predicted probabilities, and
once they are biased a reported confidence of 0.91 no longer means 0.91, which
makes the expected-calibration-error figure meaningless and undermines the
per-class threshold policy it is supposed to justify. The cost asymmetry is
applied afterwards, at the gate, where it can be reasoned about separately.
The model's job is to be honest; the gate's job is to be cautious.

**Disagreement with a teacher can be penalised, and defaults off.** When the
labels come from the vision-language model rather than from a person,
`disagreement_penalty` scales up the loss on exactly the crops the student
currently gets wrong. It is self-annealing: every crop the student comes to
agree on drops back to weight one, so the objective slides into ordinary
cross-entropy as the two models converge, and "train until they agree" is a
property of the loss rather than a stopping rule bolted on outside it. Left
at 1.0 for hand-labelled data, where every label is equally trustworthy and
there is nothing to anneal towards.

**The headline metric is macro-F1.** The synthetic set is balanced by
construction, but real pages are dominated by prose, and accuracy would read
high for a model that never finds a formula. Macro-F1 gives the rare classes
equal say.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn as nn
import torchmetrics as tm

from .. import NUM_CLASSES
from .zoo import build_backbone


class RegionClassifier(L.LightningModule):
    """Five-class region classifier over 224x224 crops."""

    def __init__(
        self,
        backbone: str = "mobilenet_v2",
        pretrained: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.0,
        disagreement_penalty: float = 1.0,
        max_epochs: int = 20,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = build_backbone(backbone, pretrained=pretrained, num_classes=num_classes)

        # No `weight=`. See the module docstring: cost lives at the gate.
        # Label smoothing defaults off for the same reason — it deliberately
        # softens the peak probability, which moves calibration.
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Per-sample, so the training step can weight the crops the student
        # and the teacher disagree on. Validation keeps the plain mean.
        self.per_sample_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing,
                                                   reduction="none")

        metric_args = {"task": "multiclass", "num_classes": num_classes}
        self.train_acc = tm.Accuracy(**metric_args)
        self.val_acc = tm.Accuracy(**metric_args)
        self.val_f1 = tm.F1Score(average="macro", **metric_args)
        self.val_confusion = tm.ConfusionMatrix(**metric_args)
        self.test_acc = tm.Accuracy(**metric_args)
        self.test_f1 = tm.F1Score(average="macro", **metric_args)
        self.test_confusion = tm.ConfusionMatrix(**metric_args)
        # Calibration: the number that justifies the whole threshold policy.
        self.val_ece = tm.CalibrationError(n_bins=15, norm="l1", **metric_args)
        self.test_ece = tm.CalibrationError(n_bins=15, norm="l1", **metric_args)

    # ------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Class probabilities, which is what the confidence gate consumes."""
        return torch.softmax(self(x), dim=1)

    # ------------------------------------------------------------- training

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        penalty = float(self.hparams.disagreement_penalty)
        if penalty == 1.0:
            loss = self.loss_fn(logits, y)
        else:
            # Weight by whether the student currently contradicts the label.
            # Detached: this selects which crops to learn from, it is not a
            # second thing to optimise, and letting a gradient through the
            # comparison would reward the model for agreeing with itself.
            per_sample = self.per_sample_loss(logits, y)
            disagrees = (logits.detach().argmax(dim=1) != y)
            weights = torch.where(disagrees, penalty, 1.0).to(per_sample.dtype)
            loss = (weights * per_sample).sum() / weights.sum().clamp(min=1e-8)
            self.log("train/agreement", 1.0 - disagrees.float().mean(),
                     prog_bar=True, on_step=False, on_epoch=True)
        self.train_acc(logits, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        for metric in (self.val_acc, self.val_f1, self.val_confusion, self.val_ece):
            metric.reset()

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.val_confusion(logits, y)
        self.val_ece(probabilities, y)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/macro_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/ece", self.val_ece, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        probabilities = torch.softmax(logits, dim=1)

        self.test_acc(logits, y)
        self.test_f1(logits, y)
        self.test_confusion(logits, y)
        self.test_ece(probabilities, y)

        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc)
        self.log("test/macro_f1", self.test_f1)
        self.log("test/ece", self.test_ece)
        return loss

    # ---------------------------------------------------------- optimisation

    def configure_optimizers(self) -> dict:
        """AdamW at a low learning rate, with cosine decay.

        Low because three of the four backbones start from ImageNet weights
        and the job is to adapt them, not to overwrite them. The same rate is
        used for the from-scratch model so the comparison between them is not
        confounded by a second difference.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.hparams.max_epochs)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
