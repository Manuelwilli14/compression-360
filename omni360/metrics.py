"""Quality metrics for omnidirectional images (Sec. 4, 5 and 6.2 of the paper)."""
import numpy as np

from .projections import equirectangular
from .sampling import render, sample
from .sphere import fibonacci_sphere
from .viewport import viewport_directions

PEAK = 255.0


def to_luma(rgb):
    """BT.709 luma of an RGB image; grayscale input is returned unchanged."""
    rgb = np.asarray(rgb, np.float32)
    if rgb.ndim == 2:
        return rgb
    return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def mse_to_psnr(mse, peak=PEAK):
    return float("inf") if mse == 0 else float(10.0 * np.log10(peak * peak / mse))


def psnr(ref, test, weights=None, peak=PEAK):
    """PSNR, optionally with per-sample weights (averaged over channels if any)."""
    err = (np.asarray(ref, np.float64) - np.asarray(test, np.float64)) ** 2
    if weights is None:
        return mse_to_psnr(err.mean(), peak)
    weights = np.asarray(weights, np.float64)
    if err.ndim == weights.ndim + 1:
        err = err.mean(axis=-1)
    return mse_to_psnr((err * weights).sum() / weights.sum(), peak)


def s_psnr(ref, ref_proj, test, test_proj, n_points=655362, weight=None, peak=PEAK):
    """S-PSNR (Sec. 5): compare two panoramas on uniformly spread sphere points.

    The panoramas may use different projections and resolutions. ``weight`` is an
    optional callable (theta, phi) -> weights, e.g. ``HeadMotionStats.point_weights``
    (WeightSph) or ``HeadMotionStats.latitude_weights`` (LatSph).
    """
    theta, phi = fibonacci_sphere(n_points)
    w = None if weight is None else weight(theta, phi)
    return psnr(sample(ref, ref_proj, theta, phi), sample(test, test_proj, theta, phi), w, peak)


def viewport_psnr(ref, ref_proj, test, test_proj, orientations, peak=PEAK, **viewport):
    """Average PSNR of the viewports displayed along a head-motion trajectory (Sec. 4)."""
    scores = []
    for yaw, pitch, roll in orientations:
        theta, phi = viewport_directions(yaw, pitch, roll, **viewport)
        scores.append(psnr(sample(ref, ref_proj, theta, phi), sample(test, test_proj, theta, phi), peak=peak))
    return float(np.mean(scores))


class Evaluator:
    """All the metrics of Sec. 6.2 against one ground-truth panorama.

    Ground-truth samples are computed once, so many coded versions (projections,
    QPs, codecs) can be scored quickly. Returns a dict with:
      * ``Viewport``: average viewport PSNR over ``orientations``;
      * ``WeightSph`` / ``LatSph``: S-PSNR weighted by ``stats`` (HeadMotionStats);
      * ``Sph``: unweighted S-PSNR;
      * ``Quad``: PSNR after mapping the coded panorama back to the ground-truth ERP.
    """

    def __init__(self, ref, ref_proj, orientations=(), stats=None, n_points=655362,
                 quad=True, peak=PEAK, **viewport):
        self.ref = np.asarray(ref, np.float32)
        self.ref_proj = ref_proj
        self.peak = peak
        self.quad = quad
        self.theta, self.phi = fibonacci_sphere(n_points)
        self.ref_points = sample(self.ref, ref_proj, self.theta, self.phi)
        self.weights = {}
        if stats is not None:
            self.weights["WeightSph"] = stats.point_weights(self.theta, self.phi)
            self.weights["LatSph"] = stats.latitude_weights(self.theta, self.phi)
        self.viewport_dirs = [viewport_directions(*o, **viewport) for o in orientations]
        self.ref_viewports = [sample(self.ref, ref_proj, *d) for d in self.viewport_dirs]

    def __call__(self, test, test_proj):
        scores = {}
        if self.viewport_dirs:
            scores["Viewport"] = float(np.mean([
                psnr(ref_vp, sample(test, test_proj, *d), peak=self.peak)
                for ref_vp, d in zip(self.ref_viewports, self.viewport_dirs)
            ]))
        points = sample(test, test_proj, self.theta, self.phi)
        for name, w in self.weights.items():
            scores[name] = psnr(self.ref_points, points, w, self.peak)
        scores["Sph"] = psnr(self.ref_points, points, peak=self.peak)
        if self.quad and self.ref_proj.name == "erp":
            h, w = self.ref_proj.frame_size
            scores["Quad"] = psnr(self.ref, render(test, test_proj, equirectangular(w, h)), peak=self.peak)
        return scores
