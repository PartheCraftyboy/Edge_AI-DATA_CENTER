"""
The per-class confidence gate.

The one rule that matters: **nothing below threshold is written into the note.**
A held region renders as a review panel, not as content the reader might trust.

Why per class rather than one global number: holding is a decision under
asymmetric cost. A garbled sentence is self-evident and trivially corrected. A
wrong subscript is invisible and silently corrupts the note. So the bar is set
per class by harm, and by how likely the reader is to notice the error.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import THRESHOLDS


@dataclass(frozen=True)
class GateDecision:
    held: bool
    threshold: float
    reason: str | None = None


# Classes with no handler in this build. They are held regardless of how
# confident the classifier is, because confidence in the *label* says nothing
# about our ability to do anything useful with the region.
UNHANDLED: frozenset[str] = frozenset({"DIAGRAM"})


def decide(region_class: str, confidence: float) -> GateDecision:
    """Hold or admit one region.

    `confidence` is the softmax probability of the predicted class. It is only
    meaningful if the model is calibrated, which is why per-class ECE is a
    reported metric rather than a decorative one.
    """
    if region_class not in THRESHOLDS:
        raise KeyError(f"unknown region class: {region_class!r}")

    threshold = THRESHOLDS[region_class]

    if region_class in UNHANDLED:
        return GateDecision(
            held=True,
            threshold=threshold,
            reason=f"{region_class} has no handler in this build, so it is always reviewed",
        )

    if confidence < threshold:
        return GateDecision(
            held=True,
            threshold=threshold,
            reason=(
                f"confidence {confidence:.2f} is below the {threshold:.2f} "
                f"threshold for {region_class}"
            ),
        )

    return GateDecision(held=False, threshold=threshold)


def held_rate(decisions: list[GateDecision]) -> float:
    """Share of regions held. The user-facing consequence of the thresholds, and
    the number that tells you whether the policy is too cautious in practice."""
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d.held) / len(decisions)
