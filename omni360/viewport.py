"""Viewport rendering and head-motion statistics (Sec. 4 and 5 of the paper)."""
import numpy as np

from .sampling import sample
from .sphere import dir_to_sph, rotation


def intrinsics(width, height, fov_x, fov_y=None):
    """Pinhole intrinsic matrix K (Eq. 1), with W / (2 fx) = tan(fov_x / 2)."""
    fx = width / (2.0 * np.tan(fov_x / 2.0))
    fy = height / (2.0 * np.tan(fov_y / 2.0)) if fov_y else fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def viewport_directions(yaw, pitch, roll=0.0, width=1024, height=1024, fov_x=np.pi / 2, fov_y=None):
    """Sphere point (theta, phi) seen by each viewport pixel (reverse projection, Eq. 3).

    E = R K^-1 e' / ||K^-1 e'||, where e' runs over the viewport pixel centres.
    """
    k_inv = np.linalg.inv(intrinsics(width, height, fov_x, fov_y))
    v, u = np.mgrid[0:height, 0:width] + 0.5
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ k_inv.T
    # Image axes (right, down, forward) to sphere axes (x right, y up, looking down -z).
    rays *= np.array([1.0, -1.0, -1.0])
    return dir_to_sph(rays @ rotation(yaw, pitch, roll).T)


def render_viewport(frame, proj, yaw, pitch, roll=0.0, **viewport):
    """What a head-mounted display shows for a given head orientation."""
    return sample(frame, proj, *viewport_directions(yaw, pitch, roll, **viewport))


def synthetic_orientations(n, pitch_std_deg=15.0, seed=0):
    """Random head orientations (yaw, pitch, roll) in radians.

    Yaw is uniform and pitch is Gaussian around the equator, without roll. This is
    only a stand-in for recorded head motion (the paper used 10 users wearing an
    Oculus Rift DK2); load real trajectories, e.g. from Salient360!, when available.
    With viewers staying near the horizon, the access density per unit area peaks
    away from the equator (about +/-20 to 25 degrees here), the perspective effect the
    paper reports in Fig. 5b.
    """
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi, np.pi, n)
    pitch = np.clip(rng.normal(0.0, np.radians(pitch_std_deg), n), -np.pi / 2, np.pi / 2)
    return np.stack([yaw, pitch, np.zeros(n)], axis=1)


class HeadMotionStats:
    """How often each sphere point is displayed, estimated from head orientations.

    Every viewport pixel of every orientation is accumulated into an
    equirectangular histogram (Fig. 5a). Dividing by the solid angle of each bin
    gives a density, normalised to a mean of 1 over the sphere, so that weighting
    uniformly spread sphere points by it reproduces the average viewport error.
    """

    def __init__(self, orientations, bins=(180, 360), **viewport):
        viewport = {"width": 256, "height": 256, **viewport}
        rows, cols = bins
        counts = np.zeros(rows * cols)
        for yaw, pitch, roll in orientations:
            theta, phi = viewport_directions(yaw, pitch, roll, **viewport)
            counts += np.bincount(self._bin(theta, phi, bins).ravel(), minlength=rows * cols)
        self.bins = bins
        self.counts = counts.reshape(bins)
        edges = np.sin(np.pi / 2 - np.arange(rows + 1) * np.pi / rows)
        area = np.repeat(((edges[:-1] - edges[1:]) * 2.0 * np.pi / cols)[:, None], cols, axis=1)
        density = self.counts / area
        self.density = density * 4.0 * np.pi / (density * area).sum()
        self.latitude_density = (self.density * area).sum(axis=1) / area.sum(axis=1)

    @staticmethod
    def _bin(theta, phi, bins):
        rows, cols = bins
        r = np.clip(((np.pi / 2 - phi) / np.pi * rows).astype(np.int64), 0, rows - 1)
        c = np.clip(((theta + np.pi) / (2.0 * np.pi) * cols).astype(np.int64), 0, cols - 1)
        return r * cols + c

    @property
    def latitude_frequency(self):
        """Share of pixel accesses per latitude row (Fig. 5b)."""
        return self.counts.sum(axis=1) / self.counts.sum()

    def point_weights(self, theta, phi):
        """Weights of the WeightSph metric (per-point access density)."""
        return self.density.ravel()[self._bin(theta, phi, self.bins)]

    def latitude_weights(self, theta, phi):
        """Weights of the LatSph metric (access density averaged over longitude)."""
        r = self._bin(theta, phi, self.bins) // self.bins[1]
        return self.latitude_density[r]
