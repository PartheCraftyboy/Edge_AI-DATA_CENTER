"""
The compile budget and its queueing model — Module 1.

A page should compile fast enough to feel immediate. That is an interactive
latency budget, and with N regions sharing it the system is a queue: regions
arrive, each takes a service time, and if they arrive faster than they can be
served the backlog grows. On this project the backlog is your own ink rather
than a camera feed.

**Why M/D/1 and not M/M/1.** The `D` is deterministic service. Every region is
resized to the same 224x224 and pushed through the same fixed graph, so the
time to classify one barely varies: it is nothing like the exponential spread
M/M/1 assumes. Choosing the right model matters because the two give different
answers, and the difference is the interesting part:

    M/M/1   Wq = rho / (mu * (1 - rho))
    M/D/1   Wq = rho / (2 * mu * (1 - rho))

Deterministic service halves the wait at the same utilisation. The variance of
the service time, not just its mean, is what fills a queue, and removing the
variance buys back half the waiting without making anything faster.

**The load factor and the utilisation are the same number.** With N regions
presented inside one budget B, the arrival rate is N/B and the service rate is
1/s, so rho = lambda/mu = N*s/B, which is exactly total service time over
budget. Worth noticing rather than computing twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import COMPILE_BUDGET_MS


@dataclass(frozen=True)
class QueueModel:
    """A single-server queue, in regions per second."""

    arrival_rate: float     # lambda
    service_rate: float     # mu

    @property
    def rho(self) -> float:
        """Utilisation. At or above 1 the queue is unstable."""
        if self.service_rate <= 0:
            return math.inf
        return self.arrival_rate / self.service_rate

    @property
    def is_stable(self) -> bool:
        return self.rho < 1.0

    @property
    def wait_md1_ms(self) -> float:
        """Mean wait in queue with deterministic service."""
        if not self.is_stable:
            return math.inf
        return self.rho / (2 * self.service_rate * (1 - self.rho)) * 1000.0

    @property
    def wait_mm1_ms(self) -> float:
        """Mean wait with exponential service, for comparison only."""
        if not self.is_stable:
            return math.inf
        return self.rho / (self.service_rate * (1 - self.rho)) * 1000.0

    @property
    def queue_length_md1(self) -> float:
        """Little's law: Lq = lambda * Wq."""
        if not self.is_stable:
            return math.inf
        return self.arrival_rate * self.wait_md1_ms / 1000.0

    def report(self) -> dict:
        return {
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "rho": self.rho,
            "stable": self.is_stable,
            "wait_md1_ms": self.wait_md1_ms,
            "wait_mm1_ms": self.wait_mm1_ms,
            "determinism_saving": (
                self.wait_mm1_ms / self.wait_md1_ms if self.wait_md1_ms > 0 else 0.0
            ),
            "queue_length_md1": self.queue_length_md1,
        }


@dataclass
class CompileBudget:
    """One page's compile, measured against the interactive budget.

    `stage_ms` holds the measured per-region cost of each pipeline stage,
    typically preprocess, infer and route. Reporting the split is the point:
    the expected finding is that classification costs tens of milliseconds
    while the router costs hundreds, so the backbone choice moves the total
    but the router is the real lever.
    """

    n_regions: int
    stage_ms: dict[str, float] = field(default_factory=dict)
    budget_ms: float = COMPILE_BUDGET_MS

    @property
    def service_ms_per_region(self) -> float:
        return float(sum(self.stage_ms.values()))

    @property
    def share_ms_per_region(self) -> float:
        """What each region may spend if the budget is divided evenly."""
        if self.n_regions <= 0:
            return self.budget_ms
        return self.budget_ms / self.n_regions

    @property
    def total_service_ms(self) -> float:
        return self.service_ms_per_region * max(0, self.n_regions)

    @property
    def load_factor(self) -> float:
        if self.budget_ms <= 0 or self.n_regions <= 0:
            return 0.0
        return self.total_service_ms / self.budget_ms

    @property
    def within_budget(self) -> bool:
        return self.load_factor < 1.0

    @property
    def queue(self) -> QueueModel:
        """The same situation expressed as a queue.

        Arrival rate is the whole page presented across the budget; service
        rate is the reciprocal of one region's service time. Their ratio is
        the load factor, which is the identity noted in the module docstring.
        """
        service = self.service_ms_per_region
        if service <= 0 or self.n_regions <= 0:
            return QueueModel(arrival_rate=0.0, service_rate=1.0)
        return QueueModel(
            arrival_rate=self.n_regions / (self.budget_ms / 1000.0),
            service_rate=1000.0 / service,
        )

    def stage_split(self) -> dict[str, float]:
        """Each stage's share of service time."""
        total = self.service_ms_per_region
        if total <= 0:
            return {name: 0.0 for name in self.stage_ms}
        return {name: ms / total for name, ms in self.stage_ms.items()}

    def bottleneck(self) -> str | None:
        if not self.stage_ms:
            return None
        return max(self.stage_ms, key=lambda name: self.stage_ms[name])

    def report(self) -> dict:
        return {
            "n_regions": self.n_regions,
            "budget_ms": self.budget_ms,
            "share_ms_per_region": self.share_ms_per_region,
            "service_ms_per_region": self.service_ms_per_region,
            "total_service_ms": self.total_service_ms,
            "load_factor": self.load_factor,
            "within_budget": self.within_budget,
            "stage_ms": dict(self.stage_ms),
            "stage_split": self.stage_split(),
            "bottleneck": self.bottleneck(),
            "queue": self.queue.report(),
        }
