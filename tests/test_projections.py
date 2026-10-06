import numpy as np
import pytest

from omni360.projections import PROJECTIONS, make_projection
from omni360.sampling import render, sample
from omni360.sphere import fibonacci_sphere, sph_to_dir


def smooth_signal(theta, phi):
    """A smooth function on the sphere (continuous across the poles and the date line)."""
    d = sph_to_dir(theta, phi)
    return 128 + 50 * d[..., 0] + 30 * d[..., 1] * d[..., 2] + 20 * np.sin(2 * d[..., 1])


@pytest.mark.parametrize("name", PROJECTIONS)
def test_pixel_to_sphere_round_trip(name):
    proj = make_projection(name, 256)
    for k, r in enumerate(proj.regions):
        y, x = np.mgrid[0:r.h, 0:r.w].astype(float)
        kk, xx, yy = proj.sphere_to_region(*proj.region_to_sphere(k, x, y))
        assert np.all(kk == k)
        np.testing.assert_allclose(xx, x, atol=1e-6)
        np.testing.assert_allclose(yy, y, atol=1e-6)


@pytest.mark.parametrize("name", PROJECTIONS)
def test_regions_tile_the_frame(name):
    proj = make_projection(name, 256)
    covered = np.zeros(proj.frame_size, int)
    for r in proj.regions:
        covered[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w] += 1
    assert np.all(covered == 1)


def test_dyadic_and_cubemap_sizes():
    assert make_projection("dyadic", 2048).frame_size == (840, 2048)
    assert make_projection("cmp", 2048).frame_size == (1184, 1776)


@pytest.mark.parametrize("name", PROJECTIONS)
def test_sampling_recovers_a_smooth_signal(name):
    proj = make_projection(name, 512)
    frame = smooth_signal(*proj.pixel_to_sphere()).astype(np.float32)
    theta, phi = fibonacci_sphere(20000)
    err = sample(frame, proj, theta, phi) - smooth_signal(theta, phi)
    # Equal-area rows get very sparse near the poles (Fig. 1b of the paper), so
    # accuracy is only required away from them; elsewhere the bound is global.
    polar = np.abs(phi) > np.radians(75)
    assert np.sqrt(np.mean(err[~polar] ** 2)) < 0.05
    assert np.sqrt(np.mean(err ** 2)) < (0.5 if name == "eap" else 0.05)


@pytest.mark.parametrize("name", PROJECTIONS)
def test_render_from_erp(name):
    src_proj = make_projection("erp", 1024)
    src = smooth_signal(*src_proj.pixel_to_sphere()).astype(np.float32)
    dst_proj = make_projection(name, 512)
    out = render(np.stack([src] * 3, -1), src_proj, dst_proj)
    assert out.shape == dst_proj.frame_size + (3,)
    expected = smooth_signal(*dst_proj.pixel_to_sphere())
    assert np.sqrt(np.mean((out[..., 1] - expected) ** 2)) < 0.05
