"""Robustness sweep tests (Modules 4 and 5).

Written after the module rather than before it, which is worth recording
honestly: these can only confirm what the code already does, not prove the
tests could have caught it being wrong. The ones below were chosen to be the
kind that would have failed on a plausible mistake — a severity that is
silently ignored, an ablation that degrades everything at once, a sweep that
reports a drop it did not measure.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.data.augment import PHOTOMETRIC_ORDER
from app.robustness import _degrade

RUNS = Path(__file__).resolve().parent.parent / "runs"


def crops(n: int = 6) -> np.ndarray:
    base = np.full((224, 224), 243, np.uint8)
    for i in range(6):
        base[20 + i * 30: 32 + i * 30, 18:200] = 40
    return np.stack([base] * n)


class TestDegrade:
    def test_zero_severity_is_a_no_op(self):
        source = crops()
        assert np.array_equal(_degrade(source, 0.0, seed=0), source)

    def test_degradation_changes_the_pixels(self):
        source = crops()
        assert not np.array_equal(_degrade(source, 1.0, seed=0), source)

    def test_shape_and_dtype_survive(self):
        out = _degrade(crops(), 0.6, seed=0)
        assert out.shape == (6, 224, 224)
        assert out.dtype == np.uint8

    def test_degradation_increases_at_every_step_of_the_sweep(self):
        """Severity must be the *only* variable along the curve.

        The bank normally samples a random subset of one to four kernels,
        which is right for training but wrong for a sweep: it means each
        point on the curve is degraded by a different set of effects, so the
        curve measures kernel choice as much as severity. The first run of
        this showed macro-F1 *rising* from 0.9897 to 0.9960 between severity
        0 and 0.25, which is not a thing stronger degradation can do.
        """
        source = crops(12)

        def distance(severity):
            out = _degrade(source, severity, seed=3)
            return float(np.abs(out.astype(float) - source.astype(float)).mean())

        steps = [distance(s) for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
        assert steps == sorted(steps), steps

    def test_is_deterministic_for_a_seed(self):
        assert np.array_equal(_degrade(crops(), 0.7, seed=5),
                              _degrade(crops(), 0.7, seed=5))

    def test_ablation_applies_exactly_one_kernel(self):
        """`only=` must isolate a single augmentation. If it silently fell
        through to the whole bank, every row of the ablation table would be
        the same number and the ranking would be meaningless."""
        source = crops(8)
        results = {
            kernel.name: _degrade(source, 1.0, seed=1, only=kernel.name)
            for kernel in PHOTOMETRIC_ORDER
        }
        distances = {
            name: round(float(np.abs(out.astype(float) - source.astype(float)).mean()), 3)
            for name, out in results.items()
        }
        assert len(set(distances.values())) > 4, distances

    def test_an_unknown_kernel_fails_loudly(self):
        with pytest.raises(StopIteration):
            _degrade(crops(), 1.0, seed=0, only="not_a_real_augmentation")


@pytest.mark.skipif(not (RUNS / "palimpsest_net" / "best.ckpt").exists(),
                    reason="no trained checkpoint")
class TestSweepAgainstARealCheckpoint:
    def test_sweep_reports_a_point_per_severity(self):
        from app.robustness import severity_sweep

        result = severity_sweep(RUNS / "palimpsest_net" / "best.ckpt",
                                severities=(0.0, 1.0), limit=64)
        assert [p["severity"] for p in result["points"]] == [0.0, 1.0]

    def test_reported_drop_matches_the_measured_points(self):
        from app.robustness import severity_sweep

        result = severity_sweep(RUNS / "palimpsest_net" / "best.ckpt",
                                severities=(0.0, 1.0), limit=64)
        expected = result["points"][0]["macro_f1"] - result["points"][-1]["macro_f1"]
        assert abs(result["robustness_drop"] - expected) < 1e-9

    def test_ablation_ranks_every_augmentation(self):
        from app.robustness import augmentation_ablation

        result = augmentation_ablation(RUNS / "palimpsest_net" / "best.ckpt", limit=64)
        assert {r["augmentation"] for r in result["rows"]} == {a.name for a in PHOTOMETRIC_ORDER}
        drops = [r["drop"] for r in result["rows"]]
        assert drops == sorted(drops, reverse=True), "rows are not worst-first"
