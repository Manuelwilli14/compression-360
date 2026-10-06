import numpy as np
import pytest

from omni360.sphere import dir_to_sph, rotation, sph_to_dir
from omni360.viewport import HeadMotionStats, synthetic_orientations, viewport_directions


def test_rotation_points_the_head():
    yaw, pitch = np.radians(40), np.radians(-25)
    d = rotation(yaw, pitch, np.radians(10)) @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(d, sph_to_dir(yaw, pitch), atol=1e-12)


def test_viewport_centre_and_orientation():
    yaw, pitch = np.radians(40), np.radians(25)
    theta, phi = viewport_directions(yaw, pitch, width=101, height=101)
    assert theta[50, 50] == pytest.approx(yaw)
    assert phi[50, 50] == pytest.approx(pitch)
    assert phi[0, 50] > pitch > phi[100, 50]          # top of the image looks up
    assert theta[50, 0] < yaw < theta[50, 100]        # left of the image looks left


def test_viewport_field_of_view():
    theta, _ = viewport_directions(0.0, 0.0, width=101, height=101, fov_x=np.radians(90))
    # Pixel centre 50 px left of the principal point, with fx = 50.5 px.
    assert np.degrees(theta[50, 0]) == pytest.approx(-np.degrees(np.arctan(50 / 50.5)))


def test_fibonacci_points_are_balanced():
    from omni360.sphere import fibonacci_sphere
    d = sph_to_dir(*fibonacci_sphere(10000))
    np.testing.assert_allclose(d.mean(axis=0), 0, atol=1e-3)
    theta, phi = dir_to_sph(d)
    assert theta.min() >= -np.pi and theta.max() <= np.pi


def test_head_motion_density_is_normalised_and_peaks_off_the_poles():
    stats = HeadMotionStats(synthetic_orientations(200), bins=(90, 180), width=128, height=128)
    rows, cols = stats.bins
    edges = np.sin(np.pi / 2 - np.arange(rows + 1) * np.pi / rows)
    area = (edges[:-1] - edges[1:]) * 2 * np.pi / cols
    assert (stats.density * area[:, None]).sum() == pytest.approx(4 * np.pi)
    assert stats.latitude_density[rows // 2] > 5 * stats.latitude_density[2]
    assert stats.latitude_frequency.sum() == pytest.approx(1.0)
