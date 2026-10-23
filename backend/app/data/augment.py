"""
The augmentation bank — Module 4.

A phone photograph of a page *is* augmentation. The camera warps the sheet,
the flash blows out one corner, a hand drops a shadow, the lens smears a little
from the shake, the sensor adds speckle, and the encoder throws away detail.
Training on clean renders and testing on photographs is the domain gap, and
this module is the thing that closes it.

Two properties of this file are deliberate and are worth stating, because the
obvious implementation of each is wrong.

**Geometry is applied to the page, photometry to the crop.** A camera applies
one perspective transform to a whole sheet of paper. Warping each 224px region
independently simulates five cameras photographing five parts of one page from
five angles, which is a geometry no photograph has ever had. So the perspective
and rotation live at generation time, applied once to the full page, with the
region boxes pushed through the same homography so the labels stay exact. The
lighting, blur, noise and compression are cheap and genuinely vary per sample,
so they run per crop, every epoch.

**The kernels run in physical order, not in sampled order.** Ink soaks into
paper before the shutter opens. Light lands on the page before it reaches the
lens. The lens blurs before the sensor adds noise. The encoder runs last,
inside the camera. Shuffling that order models a camera that compresses an
image before it focuses it, and the resulting artefacts are subtly wrong in a
way that is very hard to see and trivially avoided.

Each kernel takes a severity in [0, 1], and every application is logged with
its severity so degradation can be reported per augmentation rather than
asserted in general.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import cv2
import numpy as np

Box = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# photometric kernels, applied per crop at training time
# ---------------------------------------------------------------------------

class Augmentation(ABC):
    """One photometric effect, parameterised by severity."""

    name: str = "abstract"

    @abstractmethod
    def apply(self, img: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
        """Return a new uint8 image. Never writes to `img`."""

    @staticmethod
    def _finish(x: np.ndarray) -> np.ndarray:
        return np.clip(x, 0, 255).astype(np.uint8)


class InkBleed(Augmentation):
    """Pen on absorbent paper: strokes thicken and their edges soften.

    Runs first, because this happens to the page itself long before anyone
    photographs it.
    """

    name = "ink_bleed"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        # Ink is dark, so spreading ink is a grey *erosion*, not a dilation.
        k = 3 if severity < 0.6 else 5
        spread = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        blended = img * (1.0 - severity) + spread * severity
        wet = cv2.GaussianBlur(blended, (3, 3), 0.4 + 0.5 * severity)
        return self._finish(wet)


class IlluminationGradient(Augmentation):
    """Window light or an off-axis flash: a smooth brightness ramp."""

    name = "illumination"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        h, w = img.shape
        angle = rng.uniform(0, 2 * math.pi)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        ramp = (math.cos(angle) * xs / w) + (math.sin(angle) * ys / h)
        ramp = (ramp - ramp.min()) / max(1e-6, float(np.ptp(ramp)))
        gain = 1.0 + severity * 0.45 * (ramp - 0.5) * 2.0
        return self._finish(img.astype(np.float32) * gain)


class Glare(Augmentation):
    """A specular highlight: the flash reflecting off the paper."""

    name = "glare"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        h, w = img.shape
        cx, cy = rng.uniform(0.15, 0.85) * w, rng.uniform(0.15, 0.85) * h
        radius = max(h, w) * rng.uniform(0.18, 0.42)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        falloff = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * radius ** 2)))
        # Additive, because a specular highlight adds light rather than
        # scaling what is already there: it washes ink out towards white.
        return self._finish(img.astype(np.float32) + falloff * severity * 120.0)


class Shadow(Augmentation):
    """A hand or the phone itself between the lamp and the page."""

    name = "shadow"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        h, w = img.shape
        mask = np.zeros((h, w), np.float32)
        # A straight-edged occluder, which is what a hand or a phone body is.
        pts = np.array([
            [rng.uniform(-0.2, 0.6) * w, -0.1 * h],
            [rng.uniform(0.4, 1.2) * w, -0.1 * h],
            [rng.uniform(0.4, 1.2) * w, rng.uniform(0.3, 1.1) * h],
            [rng.uniform(-0.2, 0.6) * w, rng.uniform(0.3, 1.1) * h],
        ], np.int32)
        cv2.fillConvexPoly(mask, pts, 1.0)
        # A shadow edge is soft because the light source is not a point.
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, min(h, w) * 0.06))
        return self._finish(img.astype(np.float32) * (1.0 - mask * severity * 0.55))


class MotionBlur(Augmentation):
    """Hand shake: a directional smear along one axis."""

    name = "motion_blur"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        length = int(round(1 + severity * 10))
        if length < 2:
            return img.copy()
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0
        angle = rng.uniform(0, 180)
        matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (length, length))
        total = kernel.sum()
        if total <= 0:
            return img.copy()
        return self._finish(cv2.filter2D(img.astype(np.float32), -1, kernel / total))


class Speckle(Augmentation):
    """Paper grain and sensor noise."""

    name = "speckle"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        noisy = img.astype(np.float32) + rng.normal(0.0, 2.0 + severity * 12.0, img.shape)
        # A few dark specks: dust and paper fibre, not Gaussian.
        count = int(severity * img.size * 0.0012)
        if count:
            ys = rng.integers(0, img.shape[0], count)
            xs = rng.integers(0, img.shape[1], count)
            noisy[ys, xs] -= rng.uniform(30, 90, count)
        return self._finish(noisy)


class JpegArtifacts(Augmentation):
    """Phone camera output: blocking and ringing around strokes.

    Runs last, because the encoder is the last thing in the camera.
    """

    name = "jpeg"

    def apply(self, img, rng, severity):
        if severity <= 0.01:
            return img.copy()
        quality = int(round(92 - severity * 62))          # 92 down to 30
        ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return img.copy()
        return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)


# The physical order. This is the pipeline light actually travels.
PHOTOMETRIC_ORDER: tuple[Augmentation, ...] = (
    InkBleed(),               # on the paper, before the photo exists
    IlluminationGradient(),   # the light in the room
    Shadow(),                 # an occluder in that light
    Glare(),                  # a specular reflection off the sheet
    MotionBlur(),             # the lens and the shutter
    Speckle(),                # the sensor
    JpegArtifacts(),          # the encoder
)


class AugmentationBank:
    """Samples a subset of the photometric kernels and applies them in order.

    A subset rather than all of them: a photograph is not a worst case, and
    training only on maximally degraded crops shifts the input distribution
    away from the ordinary photo the system will usually see.
    """

    def __init__(
        self,
        augmentations: tuple[Augmentation, ...] = PHOTOMETRIC_ORDER,
        max_applied: int = 4,
        severity_range: tuple[float, float] = (0.15, 0.95),
    ):
        self.augmentations = augmentations
        self.max_applied = max_applied
        self.severity_range = severity_range

    def apply(
        self,
        img: np.ndarray,
        rng: np.random.Generator,
        always_all: bool = False,
    ) -> tuple[np.ndarray, list[dict]]:
        """Return `(augmented, log)`, the log naming each kernel and severity."""
        if always_all:
            chosen = set(range(len(self.augmentations)))
        else:
            count = int(rng.integers(1, self.max_applied + 1))
            chosen = set(rng.choice(len(self.augmentations), size=count, replace=False).tolist())

        lo, hi = self.severity_range
        out = img
        log: list[dict] = []
        # Iterating the tuple rather than the sample is what enforces physical
        # order: the sample decides *whether*, never *when*.
        for i, augmentation in enumerate(self.augmentations):
            if i not in chosen:
                continue
            severity = float(rng.uniform(lo, hi))
            out = augmentation.apply(out, rng, severity)
            log.append({"name": augmentation.name, "severity": round(severity, 4)})
        return out, log


# ---------------------------------------------------------------------------
# geometry, applied once per page at generation time
# ---------------------------------------------------------------------------

def camera_homography(
    shape: tuple[int, int],
    rng: np.random.Generator,
    severity: float,
) -> np.ndarray:
    """A plausible camera pose over a flat page, as a 3x3 homography.

    Built by displacing the four page corners rather than by composing named
    rotations, because that is the parameterisation where "how far off square
    was the phone" maps onto a single number. Severity scales the maximum
    corner displacement, plus a small in-plane rotation for a page not square
    to the lens.
    """
    h, w = shape[:2]
    if severity <= 1e-6:
        return np.eye(3, dtype=np.float64)

    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    reach = severity * 0.055 * min(h, w)
    jitter = rng.uniform(-reach, reach, size=(4, 2)).astype(np.float32)

    angle = rng.normal(0.0, 1.8 * severity)
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.transform(source.reshape(-1, 1, 2), rotation).reshape(-1, 2)

    return cv2.getPerspectiveTransform(source, (rotated + jitter).astype(np.float32)).astype(np.float64)


def _warp_box(H: np.ndarray, box: Box, shape: tuple[int, int]) -> Box:
    """Push a box through `H` and take the bounding box of the result.

    A warped rectangle is a quadrilateral, and the region crop is axis-aligned,
    so the enclosing axis-aligned box is what the label becomes. Slightly
    generous at a steep angle, which is the right direction to err: a crop with
    a little extra margin still shows the region, while a tight crop that
    clipped the warped corner would cut ink off the label.
    """
    h, w = shape[:2]
    x, y, bw, bh = box
    corners = np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], dtype=np.float64)
    transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)

    x0 = int(np.floor(transformed[:, 0].min()))
    y0 = int(np.floor(transformed[:, 1].min()))
    x1 = int(np.ceil(transformed[:, 0].max()))
    y1 = int(np.ceil(transformed[:, 1].max()))

    x0, y0 = max(0, min(x0, w - 1)), max(0, min(y0, h - 1))
    x1, y1 = max(x0 + 1, min(x1, w)), max(y0 + 1, min(y1, h))
    return x0, y0, x1 - x0, y1 - y0


def warp_page(
    img: np.ndarray,
    H: np.ndarray,
    boxes: list[Box],
) -> tuple[np.ndarray, list[Box]]:
    """Apply one camera homography to a page and to its region boxes.

    Pixels and labels go through the same matrix, which is the whole reason
    the labels on a generated page remain exact after augmentation.
    """
    if np.allclose(H, np.eye(3)):
        return img, list(boxes)

    h, w = img.shape[:2]
    border = int(np.median(img[0, :])) if img.size else 245
    warped = cv2.warpPerspective(
        img, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,          # the page sits on more paper, not on black
    )
    return warped, [_warp_box(H, b, img.shape) for b in boxes]
