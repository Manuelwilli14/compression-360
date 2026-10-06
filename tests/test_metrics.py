import numpy as np
import pytest

from omni360.bdrate import bd_psnr, bd_rate
from omni360.metrics import Evaluator, psnr, s_psnr, to_luma, viewport_psnr
from omni360.projections import make_projection
from omni360.sampling import render
from omni360.viewport import HeadMotionStats, synthetic_orientations

from .test_projections import smooth_signal


def test_psnr_of_unit_error():
    ref = np.zeros((8, 8))
    assert psnr(ref, ref + 1) == pytest.approx(10 * np.log10(255 ** 2))
    assert psnr(ref, ref) == float("inf")


def test_weighted_psnr_with_channels():
    ref = np.zeros((4, 4, 3))
    test = ref.copy()
    test[0] = 10.0
    weights = np.ones((4, 4))
    weights[0] = 0.0
    assert psnr(ref, test, weights) == float("inf")


def test_to_luma_of_gray_is_identity():
    rgb = np.full((2, 2, 3), 100.0)
    np.testing.assert_allclose(to_luma(rgb), 100.0, rtol=1e-6)


def test_s_psnr_compares_different_projections():
    erp = make_projection("erp", 1024)
    gt = smooth_signal(*erp.pixel_to_sphere()).astype(np.float32)
    cmp = make_projection("cmp", 512)
    coded = render(gt, erp, cmp)
    clean = s_psnr(gt, erp, coded, cmp, n_points=50000)
    assert clean > 50
    noisy = coded + np.random.default_rng(0).normal(0, 5, coded.shape)
    assert s_psnr(gt, erp, noisy, cmp, n_points=50000) < clean - 10


def test_evaluator_matches_standalone_metrics():
    erp = make_projection("erp", 512)
    gt = smooth_signal(*erp.pixel_to_sphere()).astype(np.float32)
    eap = make_projection("eap", 256)
    coded = render(gt, erp, eap) + np.random.default_rng(1).normal(0, 3, eap.frame_size)
    orientations = synthetic_orientations(3, seed=0)
    stats = HeadMotionStats(synthetic_orientations(20, seed=1), bins=(45, 90), width=64, height=64)
    vp = {"width": 64, "height": 64}
    scores = Evaluator(gt, erp, orientations, stats, n_points=20000, **vp)(coded, eap)
    assert scores["Sph"] == pytest.approx(s_psnr(gt, erp, coded, eap, n_points=20000))
    assert scores["Viewport"] == pytest.approx(viewport_psnr(gt, erp, coded, eap, orientations, **vp))
    assert scores["WeightSph"] == pytest.approx(
        s_psnr(gt, erp, coded, eap, n_points=20000, weight=stats.point_weights))
    assert set(scores) == {"Viewport", "WeightSph", "LatSph", "Sph", "Quad"}


def test_bd_rate_of_identical_curves_is_zero():
    rate, quality = [1000, 2000, 4000, 8000], [30, 33, 36, 38.5]
    assert bd_rate(rate, quality, rate, quality) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("method", ["cubic", "pchip"])
def test_bd_rate_of_a_constant_saving(method):
    rate, quality = np.array([1000, 2000, 4000, 8000]), [30, 33, 36, 38.5]
    assert bd_rate(rate, quality, 0.9 * rate, quality, method) == pytest.approx(-10, abs=1e-6)


def test_bd_psnr_of_a_constant_gain():
    rate, quality = [1000, 2000, 4000, 8000], np.array([30, 33, 36, 38.5])
    assert bd_psnr(rate, quality, rate, quality + 0.5) == pytest.approx(0.5)


def test_bd_rate_needs_overlapping_curves():
    with pytest.raises(ValueError):
        bd_rate([1, 2, 3, 4], [30, 31, 32, 33], [1, 2, 3, 4], [40, 41, 42, 43])
