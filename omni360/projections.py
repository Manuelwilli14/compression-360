"""Sphere-to-plane mappings compared in the paper (Sec. 3).

A projection describes a packed 2D frame made of one or more rectangular
regions. For every pixel centre of the frame it gives the sphere point it
represents (used to *generate* a panorama, Fig. 2 of the paper) and, for any
sphere point, where it lands in the frame (used to *evaluate* a coded
panorama on the sphere, Fig. 4).

Pixel coordinates follow the "pixel centre" convention: the centre of pixel
(column u, row v) is at x = u, y = v, and the region spans [-0.5, w - 0.5].
"""
from dataclasses import dataclass

import numpy as np

from .sphere import dir_to_sph, sph_to_dir

PROJECTIONS = ("erp", "eap", "dyadic", "cmp")


@dataclass(frozen=True)
class Region:
    x0: int
    y0: int
    w: int
    h: int
    wrap_x: bool = False  # left and right edges are neighbours (longitude bands)


class Projection:
    name = "base"
    frame_size = (0, 0)  # (height, width)
    regions = ()

    def region_to_sphere(self, k, x, y):
        """Sphere point (theta, phi) at local pixel coordinates of region ``k``."""
        raise NotImplementedError

    def sphere_to_region(self, theta, phi):
        """Region index and local pixel coordinates (k, x, y) of sphere points."""
        raise NotImplementedError

    def pixel_to_sphere(self):
        """(theta, phi) of every pixel centre of the frame, NaN outside regions."""
        theta = np.full(self.frame_size, np.nan)
        phi = np.full(self.frame_size, np.nan)
        for k, r in enumerate(self.regions):
            y, x = np.mgrid[0:r.h, 0:r.w].astype(np.float64)
            t, p = self.region_to_sphere(k, x, y)
            theta[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w] = t
            phi[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w] = p
        return theta, phi

    def __repr__(self):
        h, w = self.frame_size
        return f"{type(self).__name__}({self.name}, {w}x{h})"


class CylindricalProjection(Projection):
    """Latitude bands, each sampled uniformly in longitude.

    Rows are uniform in latitude (equirectangular, dyadic) or in sin(latitude)
    (Lambert cylindrical equal-area).
    """

    def __init__(self, name, frame_size, regions, phi_edges, equal_area):
        self.name = name
        self.frame_size = frame_size
        self.regions = tuple(regions)
        self.phi_edges = tuple(phi_edges)  # (phi_top, phi_bottom) of each region
        self.equal_area = equal_area

    def _t(self, phi):
        return np.sin(phi) if self.equal_area else phi

    def _t_inv(self, t):
        return np.arcsin(np.clip(t, -1.0, 1.0)) if self.equal_area else t

    def _t_range(self, k):
        top, bottom = self.phi_edges[k]
        return self._t(top), self._t(bottom)

    def region_to_sphere(self, k, x, y):
        r = self.regions[k]
        t_top, t_bottom = self._t_range(k)
        theta = (np.asarray(x) + 0.5) / r.w * 2.0 * np.pi - np.pi
        phi = self._t_inv(t_top - (np.asarray(y) + 0.5) / r.h * (t_top - t_bottom))
        return theta, phi

    def sphere_to_region(self, theta, phi):
        theta, phi = np.broadcast_arrays(np.asarray(theta, np.float64), np.asarray(phi, np.float64))
        k = np.zeros(theta.shape, np.int64)
        for i, (top, bottom) in enumerate(self.phi_edges):
            k[(phi <= top) & (phi >= bottom)] = i
        x = np.empty(theta.shape)
        y = np.empty(theta.shape)
        for i, r in enumerate(self.regions):
            m = k == i
            t_top, t_bottom = self._t_range(i)
            x[m] = (theta[m] + np.pi) / (2.0 * np.pi) * r.w - 0.5
            y[m] = (t_top - self._t(phi[m])) / (t_top - t_bottom) * r.h - 0.5
        return k, x, y


# Cube faces as (centre, right, up) unit vectors; a face pixel at normalised
# coordinates (a, b) in [-1, 1]^2 looks in direction centre + a*right + b*up.
_FACES = {
    "front": ((0, 0, -1), (1, 0, 0), (0, 1, 0)),
    "right": ((1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "back": ((0, 0, 1), (-1, 0, 0), (0, 1, 0)),
    "left": ((-1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "top": ((0, 1, 0), (1, 0, 0), (0, 0, 1)),
    "bottom": ((0, -1, 0), (1, 0, 0), (0, 0, -1)),
}
# 3x2 packing. The first row (left, front, right) is continuous across face edges.
_CUBE_LAYOUT = (("left", "front", "right"), ("back", "top", "bottom"))


class Cubemap(Projection):
    """Six rectilinear 90 degree views packed in a 3x2 frame (Sec. 3, "Cubic")."""

    name = "cmp"

    def __init__(self, face):
        self.face = face
        self.frame_size = (2 * face, 3 * face)
        self.face_names = tuple(n for row in _CUBE_LAYOUT for n in row)
        self.regions = tuple(
            Region(c * face, r * face, face, face)
            for r, row in enumerate(_CUBE_LAYOUT) for c in range(len(row))
        )
        self._basis = np.array([_FACES[n] for n in self.face_names], np.float64)  # (6, 3, 3)

    def region_to_sphere(self, k, x, y):
        centre, right, up = self._basis[k]
        a = 2.0 * (np.asarray(x) + 0.5) / self.face - 1.0
        b = 1.0 - 2.0 * (np.asarray(y) + 0.5) / self.face
        return dir_to_sph(centre + a[..., None] * right + b[..., None] * up)

    def sphere_to_region(self, theta, phi):
        d = sph_to_dir(theta, phi)
        k = np.argmax(d @ self._basis[:, 0].T, axis=-1)
        centre, right, up = self._basis[k, 0], self._basis[k, 1], self._basis[k, 2]
        depth = np.sum(d * centre, axis=-1)
        a = np.sum(d * right, axis=-1) / depth
        b = np.sum(d * up, axis=-1) / depth
        return k, (a + 1.0) / 2.0 * self.face - 0.5, (1.0 - b) / 2.0 * self.face - 0.5


def equirectangular(width, height=None):
    height = height or width // 2
    return CylindricalProjection(
        "erp", (height, width), [Region(0, 0, width, height, True)], [(np.pi / 2, -np.pi / 2)], False)


def equal_area(width, height=None):
    height = height or width // 2
    return CylindricalProjection(
        "eap", (height, width), [Region(0, 0, width, height, True)], [(np.pi / 2, -np.pi / 2)], True)


def dyadic(width, height=None):
    """Equirectangular with half the horizontal resolution for |phi| >= pi/3.

    The two polar bands (each 30 degrees of latitude, ``width/2`` columns) are
    packed side by side above the equatorial band, so the frame has 5/6 of the
    rows of the matching equirectangular frame. Band heights are rounded to a
    multiple of 8 rows to keep the frame codec friendly.
    """
    height = height or width // 2
    if width % 4:
        raise ValueError("dyadic projection needs a width divisible by 4")
    p = 8 * max(1, round(height / 48))
    regions = [Region(0, 0, width // 2, p, True), Region(width // 2, 0, width // 2, p, True),
               Region(0, p, width, 4 * p, True)]
    edges = [(np.pi / 2, np.pi / 3), (-np.pi / 3, -np.pi / 2), (np.pi / 3, -np.pi / 3)]
    return CylindricalProjection("dyadic", (5 * p, width), regions, edges, False)


def cubemap(width, height=None):
    """Cubemap with (about) the same pixel budget as a ``width x height`` ERP."""
    height = height or width // 2
    return Cubemap(8 * max(1, round(np.sqrt(width * height / 6) / 8)))


def make_projection(name, width, height=None):
    """Build a projection by its JVET-style short name: erp, eap, dyadic, cmp."""
    factories = {"erp": equirectangular, "eap": equal_area, "dyadic": dyadic, "cmp": cubemap}
    if name not in factories:
        raise ValueError(f"unknown projection {name!r}, expected one of {PROJECTIONS}")
    return factories[name](width, height)
