"""
Module 5 — confusion matrix, macro-F1, calibration, held rate, Pareto.

Deliberately plain numpy rather than torchmetrics. Two reasons: this runs in
the live monitor over inference on real pages, where there is no Lightning
trainer and often no torch tensor in sight, and a rolling window over the last
N samples is not something torchmetrics offers. The training loop keeps its
torchmetrics instances, and these numbers are checked against them.

**Macro-F1 is the headline, not accuracy.** The synthetic set is balanced by
construction, but real pages are mostly prose: a model that never predicts
FORMULA can post 80% accuracy while being useless at the job. Macro-F1 gives
every class an equal vote and exposes that immediately.

**Calibration is what justifies the thresholds.** The per-class confidence
policy only means something if a reported 0.9 corresponds to being right nine
times in ten. Expected calibration error measures exactly that, which is why
the loss is left unweighted: biasing the objective to reflect per-class cost
would also bias the probabilities and make this number a fiction.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .. import CLASS_NAMES, NUM_CLASSES, THRESHOLDS


class MetricCore:
    """Accumulates predictions and computes every reported figure.

    `window` bounds how many recent samples count. The live monitor uses a
    window so that switching backbone visibly changes the matrix within a few
    seconds instead of being diluted by everything already seen; evaluation
    leaves it unbounded.
    """

    def __init__(self, window: int | None = None, n_bins: int = 15):
        self.window = window
        self.n_bins = n_bins
        self._probabilities: deque[np.ndarray] = deque(maxlen=window)
        self._targets: deque[int] = deque(maxlen=window)
        self._latencies: deque[float] = deque(maxlen=window)

    # --------------------------------------------------------------- input

    def update(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        latency_ms: float | None = None,
    ) -> None:
        """Add a batch. `probabilities` is (N, C), `targets` is (N,)."""
        probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1, NUM_CLASSES)
        targets = np.asarray(targets).reshape(-1)
        for row, target in zip(probabilities, targets):
            self._probabilities.append(row)
            self._targets.append(int(target))
            if latency_ms is not None:
                self._latencies.append(float(latency_ms))

    def reset(self) -> None:
        self._probabilities.clear()
        self._targets.clear()
        self._latencies.clear()

    def __len__(self) -> int:
        return len(self._targets)

    # ------------------------------------------------------------ internals

    def _arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._targets:
            empty = np.zeros((0, NUM_CLASSES))
            return empty, np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        probabilities = np.stack(self._probabilities)
        targets = np.asarray(self._targets, dtype=int)
        predictions = probabilities.argmax(axis=1)
        return probabilities, targets, predictions

    # ------------------------------------------------------------ confusion

    def confusion_counts(self) -> np.ndarray:
        """Rows are truth, columns are prediction.

        That orientation is not a convention to pick freely: row-normalised,
        the diagonal reads as recall, which is the question a reader of the
        matrix is actually asking. Transposed it reads as precision while
        looking identical.
        """
        _, targets, predictions = self._arrays()
        matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for t, p in zip(targets, predictions):
            matrix[t, p] += 1
        return matrix

    def confusion_row_normalised(self) -> np.ndarray:
        counts = self.confusion_counts().astype(np.float64)
        totals = counts.sum(axis=1, keepdims=True)
        return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    # ------------------------------------------------------------------ f1

    def accuracy(self) -> float:
        _, targets, predictions = self._arrays()
        if not len(targets):
            return 0.0
        return float((targets == predictions).mean())

    def per_class_f1(self) -> list[float]:
        counts = self.confusion_counts()
        scores: list[float] = []
        for c in range(NUM_CLASSES):
            true_positive = float(counts[c, c])
            false_negative = float(counts[c, :].sum() - true_positive)
            false_positive = float(counts[:, c].sum() - true_positive)
            denominator = 2 * true_positive + false_positive + false_negative
            scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
        return scores

    def macro_f1(self) -> float:
        return float(np.mean(self.per_class_f1()))

    # --------------------------------------------------------- calibration

    def _ece(self, confidences: np.ndarray, correct: np.ndarray) -> float:
        """Binned absolute gap between confidence and accuracy."""
        if not len(confidences):
            return 0.0
        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = (confidences > lo) & (confidences <= hi)
            if not in_bin.any():
                continue
            weight = in_bin.mean()
            total += weight * abs(correct[in_bin].mean() - confidences[in_bin].mean())
        return float(total)

    def ece(self) -> float:
        probabilities, targets, predictions = self._arrays()
        if not len(targets):
            return 0.0
        return self._ece(probabilities.max(axis=1), (predictions == targets).astype(float))

    def ece_per_class(self) -> dict[str, float]:
        """Per class, because one global figure can hide a single badly
        calibrated class, and FORMULA is the one that matters."""
        probabilities, targets, predictions = self._arrays()
        out: dict[str, float] = {}
        for index, name in enumerate(CLASS_NAMES):
            mask = predictions == index
            if not mask.any():
                out[name] = 0.0
                continue
            out[name] = self._ece(
                probabilities[mask].max(axis=1),
                (predictions[mask] == targets[mask]).astype(float),
            )
        return out

    # ----------------------------------------------------------- held rate

    def held_rate(self) -> dict[str, float]:
        """Share of predictions per class that fall below their threshold.

        This is the policy's user-facing consequence: a held region is never
        written into the note, it goes to the review panel. A high FORMULA
        held rate means a cautious notebook and a lot of review friction, and
        that trade is the whole point of setting thresholds per class.
        """
        probabilities, _, predictions = self._arrays()
        out: dict[str, float] = {}
        for index, name in enumerate(CLASS_NAMES):
            mask = predictions == index
            if not mask.any():
                out[name] = 0.0
                continue
            confidences = probabilities[mask].max(axis=1)
            out[name] = float((confidences < THRESHOLDS[name]).mean())
        return out

    # ------------------------------------------------------------- latency

    def latency(self) -> dict[str, float]:
        if not self._latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        values = np.asarray(self._latencies, dtype=np.float64)
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()),
        }

    # -------------------------------------------------------------- report

    def report(self) -> dict:
        """Everything the dashboard and the deck need, in one call."""
        return {
            "samples": len(self),
            "accuracy": self.accuracy(),
            "macro_f1": self.macro_f1(),
            "per_class_f1": dict(zip(CLASS_NAMES, self.per_class_f1())),
            "ece": self.ece(),
            "ece_per_class": self.ece_per_class(),
            "held_rate": self.held_rate(),
            "latency_ms": self.latency(),
            "confusion": self.confusion_counts().tolist(),
            "confusion_row_normalised": self.confusion_row_normalised().tolist(),
        }


class ParetoAccumulator:
    """Accuracy against latency across backbones, keeping the frontier.

    A backbone that is both slower and less accurate than another is never the
    right choice, so it is dropped. What remains is the set of genuine
    trade-offs, which is the Module 2 deliverable: not "which model is best"
    but "which models are worth considering, and what does each cost".
    """

    def __init__(self):
        self.points: list[dict] = []

    def add(
        self,
        name: str,
        accuracy: float,
        latency_ms: float,
        energy_j: float | None = None,
        **extra,
    ) -> None:
        self.points.append({
            "name": name,
            "accuracy": float(accuracy),
            "latency_ms": float(latency_ms),
            "energy_j": energy_j,
            **extra,
        })

    def frontier(self) -> list[dict]:
        """Non-dominated points, cheapest first."""
        keep = [
            point for point in self.points
            if not any(
                other is not point
                and other["latency_ms"] <= point["latency_ms"]
                and other["accuracy"] >= point["accuracy"]
                and (other["latency_ms"] < point["latency_ms"]
                     or other["accuracy"] > point["accuracy"])
                for other in self.points
            )
        ]
        return sorted(keep, key=lambda p: p["latency_ms"])
