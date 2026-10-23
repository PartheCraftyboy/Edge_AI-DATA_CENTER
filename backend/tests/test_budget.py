"""Compile budget and queueing tests (Module 1).

The module taught M/D/1 specifically, and M/D/1 rather than M/M/1 is the right
model for this pipeline: every region is resized to the same 224x224 and goes
through the same fixed graph, so service time is near-deterministic rather
than exponentially distributed.

The pedagogical result worth landing is that deterministic service halves the
queueing wait at the same utilisation. That factor of two is the whole reason
the distinction is taught, so it is asserted exactly rather than approximately.
"""

import math

import pytest

from app import COMPILE_BUDGET_MS
from app.pipeline.budget import CompileBudget, QueueModel


class TestUtilisation:
    def test_rho_is_arrival_over_service_rate(self):
        q = QueueModel(arrival_rate=10.0, service_rate=40.0)
        assert abs(q.rho - 0.25) < 1e-12

    def test_a_system_at_capacity_is_unstable(self):
        assert not QueueModel(arrival_rate=40.0, service_rate=40.0).is_stable

    def test_an_overloaded_system_reports_infinite_wait(self):
        q = QueueModel(arrival_rate=50.0, service_rate=40.0)
        assert not q.is_stable
        assert math.isinf(q.wait_md1_ms)


class TestQueueingFormulas:
    def test_md1_wait_matches_the_closed_form(self):
        lam, mu = 10.0, 20.0
        q = QueueModel(arrival_rate=lam, service_rate=mu)
        expected = (lam / mu) / (2 * mu * (1 - lam / mu)) * 1000.0
        assert abs(q.wait_md1_ms - expected) < 1e-9

    def test_mm1_wait_matches_the_closed_form(self):
        lam, mu = 10.0, 20.0
        q = QueueModel(arrival_rate=lam, service_rate=mu)
        expected = (lam / mu) / (mu * (1 - lam / mu)) * 1000.0
        assert abs(q.wait_mm1_ms - expected) < 1e-9

    @pytest.mark.parametrize("rho", [0.1, 0.35, 0.5, 0.8, 0.95])
    def test_deterministic_service_halves_the_wait(self, rho):
        """The entire point of teaching M/D/1 alongside M/M/1."""
        mu = 25.0
        q = QueueModel(arrival_rate=rho * mu, service_rate=mu)
        assert abs(q.wait_mm1_ms / q.wait_md1_ms - 2.0) < 1e-9

    def test_wait_grows_without_bound_approaching_capacity(self):
        mu = 20.0
        waits = [QueueModel(arrival_rate=r * mu, service_rate=mu).wait_md1_ms
                 for r in (0.5, 0.8, 0.95, 0.99)]
        assert waits == sorted(waits)
        assert waits[-1] > 10 * waits[0]

    def test_queue_length_follows_little_s_law(self):
        """Lq = lambda * Wq, which is the cross-check that the wait is right."""
        q = QueueModel(arrival_rate=12.0, service_rate=20.0)
        assert abs(q.queue_length_md1 - 12.0 * q.wait_md1_ms / 1000.0) < 1e-9


class TestCompileBudget:
    def test_load_factor_is_service_over_budget(self):
        budget = CompileBudget(n_regions=8, stage_ms={"preprocess": 10.0, "infer": 15.0})
        assert abs(budget.service_ms_per_region - 25.0) < 1e-9
        assert abs(budget.load_factor - (8 * 25.0 / COMPILE_BUDGET_MS)) < 1e-9

    def test_the_load_factor_is_the_utilisation(self):
        """Not a coincidence worth hiding: with N regions arriving inside one
        budget, lambda/mu reduces exactly to total service time over budget.
        The two Module 1 ideas are the same number."""
        budget = CompileBudget(n_regions=8, stage_ms={"preprocess": 10.0, "infer": 15.0})
        assert abs(budget.load_factor - budget.queue.rho) < 1e-9

    def test_per_region_share_divides_the_budget(self):
        budget = CompileBudget(n_regions=10, stage_ms={"infer": 5.0})
        assert abs(budget.share_ms_per_region - COMPILE_BUDGET_MS / 10) < 1e-9

    def test_within_budget_when_service_fits(self):
        assert CompileBudget(n_regions=4, stage_ms={"infer": 20.0}).within_budget

    def test_over_budget_when_it_does_not(self):
        assert not CompileBudget(n_regions=40, stage_ms={"infer": 90.0}).within_budget

    def test_stage_split_sums_to_one(self):
        budget = CompileBudget(
            n_regions=8,
            stage_ms={"preprocess": 42.0, "infer": 18.0, "route": 87.0},
        )
        split = budget.stage_split()
        assert abs(sum(split.values()) - 1.0) < 1e-9
        assert split["route"] > split["infer"], "routing should dominate"

    def test_identifies_the_dominant_stage(self):
        """The honest finding the measurement is expected to produce: the CNN
        is not the bottleneck, the router is."""
        budget = CompileBudget(
            n_regions=8,
            stage_ms={"preprocess": 42.0, "infer": 18.0, "route": 87.0},
        )
        assert budget.bottleneck() == "route"

    def test_zero_regions_does_not_divide_by_zero(self):
        budget = CompileBudget(n_regions=0, stage_ms={"infer": 10.0})
        assert budget.load_factor == 0.0
        assert budget.within_budget

    def test_report_is_serialisable(self):
        import json

        budget = CompileBudget(n_regions=8, stage_ms={"infer": 18.0, "route": 87.0})
        json.dumps(budget.report())
