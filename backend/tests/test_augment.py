"""Augmentation bank tests (Module 4).

Augmentation is where a phone photograph gets simulated, and the split between
the two halves is deliberate:

* **Geometric** augmentation happens once, at generation time, to the whole
  page. A camera applies one perspective to a sheet of paper, not a different
  one to each region, and the region boxes go through the same homography so
  the labels stay exact by construction.
* **Photometric** augmentation happens every epoch, per 224px crop. Glare,
  shadow, bleed, blur, speckle and compression are cheap and genuinely vary
  per sample.

The order the photometric kernels run in is not arbitrary and is tested: ink
soaks into paper before the photo is taken, lighting happens before the lens,
the lens before the sensor, and the sensor before the JPEG encoder. Running
them in a different order simulates a camera that cannot exist.
"""

import numpy as np
import pytest

from app.data.augment import (
    PHOTOMETRIC_ORDER,
    AugmentationBank,
    Glare,
    IlluminationGradient,
    InkBleed,
    JpegArtifacts,
    MotionBlur,
    Shadow,
    Speckle,
    camera_homography,
    warp_page,
)

ALL = [InkBleed(), IlluminationGradient(), Shadow(), Glare(), MotionBlur(), Speckle(), JpegArtifacts()]


def crop(size=224):
    """A plausible region crop: light paper with dark marks on it."""
    img = np.full((size, size), 243, np.uint8)
    for i in range(6):
        y = 20 + i * 30
        img[y:y + 12, 18:200] = 40
    return img


class TestKernelContract:
    def test_shape_and_dtype_are_preserved(self):
        for aug in ALL:
            out = aug.apply(crop(), np.random.default_rng(0), 0.7)
            assert out.shape == (224, 224), aug.name
            assert out.dtype == np.uint8, aug.name

    def test_output_stays_in_range(self):
        for aug in ALL:
            out = aug.apply(crop(), np.random.default_rng(1), 1.0)
            assert int(out.min()) >= 0 and int(out.max()) <= 255, aug.name

    def test_same_seed_same_result(self):
        for aug in ALL:
            a = aug.apply(crop(), np.random.default_rng(2), 0.8)
            b = aug.apply(crop(), np.random.default_rng(2), 0.8)
            assert np.array_equal(a, b), aug.name

    def test_zero_severity_is_near_identity(self):
        for aug in ALL:
            out = aug.apply(crop(), np.random.default_rng(3), 0.0)
            drift = float(np.abs(out.astype(float) - crop().astype(float)).mean())
            assert drift < 2.0, f"{aug.name} changed the image at severity 0 ({drift:.2f})"

    def test_full_severity_changes_the_image(self):
        for aug in ALL:
            out = aug.apply(crop(), np.random.default_rng(4), 1.0)
            drift = float(np.abs(out.astype(float) - crop().astype(float)).mean())
            assert drift > 0.5, f"{aug.name} did nothing at severity 1"

    def test_does_not_mutate_its_input(self):
        for aug in ALL:
            original = crop()
            aug.apply(original, np.random.default_rng(5), 1.0)
            assert np.array_equal(original, crop()), f"{aug.name} wrote to its input"


class TestKernelsDoWhatTheyClaim:
    """A kernel that runs without error but does the wrong thing is the real
    risk: every shape assertion above would still pass."""

    def test_ink_bleed_adds_ink_rather_than_removing_it(self):
        out = InkBleed().apply(crop(), np.random.default_rng(0), 1.0)
        assert int((out < 128).sum()) > int((crop() < 128).sum())

    def test_glare_brightens_somewhere(self):
        out = Glare().apply(crop(), np.random.default_rng(0), 1.0)
        assert int(out.max()) >= int(crop().max())
        assert float(out.mean()) > float(crop().mean())

    def test_shadow_darkens_overall(self):
        out = Shadow().apply(crop(), np.random.default_rng(0), 1.0)
        assert float(out.mean()) < float(crop().mean())

    def test_motion_blur_reduces_sharpness(self):
        """Laplacian variance, not a single-axis gradient. A directional blur
        only softens edges perpendicular to its direction, so measuring one
        axis passes or fails on the randomly drawn angle rather than on
        whether the kernel blurred anything."""
        import cv2

        def sharpness(x):
            return float(cv2.Laplacian(x.astype(np.float32), cv2.CV_32F).var())

        for seed in range(6):
            blurred = MotionBlur().apply(crop(), np.random.default_rng(seed), 1.0)
            assert sharpness(blurred) < sharpness(crop()), seed

    def test_speckle_raises_local_variance_in_flat_paper(self):
        patch = (slice(150, 220), slice(150, 220))   # a blank corner
        before = float(crop()[patch].std())
        after = float(Speckle().apply(crop(), np.random.default_rng(0), 1.0)[patch].std())
        assert after > before


class TestBank:
    def test_runs_in_physical_order_regardless_of_sampling(self):
        """Ink soaks into paper before the shutter opens; JPEG happens last,
        inside the camera. A bank that applied these in sampled order would be
        simulating a camera that compresses before it focuses."""
        bank = AugmentationBank()
        _, log = bank.apply(crop(), np.random.default_rng(0), always_all=True)
        names = [entry["name"] for entry in log]
        assert names == [a.name for a in PHOTOMETRIC_ORDER]

    def test_log_records_name_and_severity_for_each(self):
        bank = AugmentationBank()
        _, log = bank.apply(crop(), np.random.default_rng(1))
        assert log, "nothing was applied and nothing was logged"
        for entry in log:
            assert set(entry) == {"name", "severity"}
            assert 0.0 <= entry["severity"] <= 1.0

    def test_applies_a_subset_not_everything(self):
        """All eight every time is not a photograph, it is a worst case."""
        bank = AugmentationBank(max_applied=3)
        counts = {len(bank.apply(crop(), np.random.default_rng(s))[1]) for s in range(25)}
        assert max(counts) <= 3

    def test_is_deterministic(self):
        bank = AugmentationBank()
        a, la = bank.apply(crop(), np.random.default_rng(7))
        b, lb = bank.apply(crop(), np.random.default_rng(7))
        assert np.array_equal(a, b) and la == lb

    def test_output_is_still_a_valid_crop(self):
        bank = AugmentationBank()
        out, _ = bank.apply(crop(), np.random.default_rng(8))
        assert out.shape == (224, 224) and out.dtype == np.uint8


class TestGeometry:
    """Labels survive the warp only if the boxes go through the same matrix as
    the pixels. This is the property the whole 'labels are free' claim rests
    on, so it is tested directly rather than assumed."""

    def test_zero_severity_is_the_identity(self):
        H = camera_homography((768, 1024), np.random.default_rng(0), 0.0)
        assert np.allclose(H, np.eye(3), atol=1e-6)

    def test_homography_is_invertible(self):
        for seed in range(20):
            H = camera_homography((768, 1024), np.random.default_rng(seed), 1.0)
            assert H.shape == (3, 3)
            assert abs(float(np.linalg.det(H))) > 1e-6

    def test_identity_warp_leaves_image_and_boxes_alone(self):
        page = crop(256)
        boxes = [(20, 30, 80, 40)]
        out, moved = warp_page(page, np.eye(3), boxes)
        assert np.array_equal(out, page)
        assert moved == boxes

    def test_boxes_follow_the_pixels(self):
        """Mark one region, warp, and confirm the ink lands inside the box the
        warp reports. If boxes and pixels ever diverge, every label is wrong
        and nothing downstream can detect it."""
        page = np.full((256, 256), 240, np.uint8)
        page[60:110, 40:150] = 20
        box = (40, 60, 110, 50)
        H = camera_homography(page.shape, np.random.default_rng(3), 0.8)
        warped, (moved,) = warp_page(page, H, [box])

        x, y, w, h = moved
        inside = warped[y:y + h, x:x + w]
        dark_total = int((warped < 128).sum())
        dark_inside = int((inside < 128).sum())
        assert dark_total > 0
        assert dark_inside / dark_total > 0.9, "the ink left the box the warp reported"

    def test_boxes_stay_within_the_page(self):
        page = np.full((256, 256), 240, np.uint8)
        boxes = [(10, 10, 60, 40), (180, 190, 70, 50)]
        H = camera_homography(page.shape, np.random.default_rng(5), 1.0)
        _, moved = warp_page(page, H, boxes)
        for x, y, w, h in moved:
            assert 0 <= x and 0 <= y
            assert x + w <= 256 and y + h <= 256
            assert w > 0 and h > 0

    @pytest.mark.parametrize("severity", [0.25, 0.5, 1.0])
    def test_stronger_severity_moves_corners_further(self, severity):
        corner = np.array([1023.0, 767.0, 1.0])
        shifts = []
        for seed in range(12):
            H = camera_homography((768, 1024), np.random.default_rng(seed), severity)
            p = H @ corner
            shifts.append(float(np.hypot(p[0] / p[2] - 1023.0, p[1] / p[2] - 767.0)))
        assert float(np.mean(shifts)) > 1.0 * severity
