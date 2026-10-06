"""Spherical geometry helpers.

Conventions (same as Yu et al., ISMAR 2015):
  * the viewer sits at the centre of a unit sphere and, in the canonical head
    position, looks down the negative z axis, with x pointing right and y up;
  * longitude ``theta`` in [-pi, pi], positive to the right of the front direction;
  * latitude ``phi`` in [-pi/2, pi/2], positive above the equator.
"""
import numpy as np


def sph_to_dir(theta, phi):
    """Unit vectors for spherical coordinates, shape (..., 3)."""
    theta, phi = np.asarray(theta, np.float64), np.asarray(phi, np.float64)
    cos_phi = np.cos(phi)
    return np.stack([cos_phi * np.sin(theta), np.sin(phi), -cos_phi * np.cos(theta)], axis=-1)


def dir_to_sph(d):
    """Spherical coordinates (theta, phi) of (not necessarily unit) vectors."""
    d = np.asarray(d, np.float64)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    theta = np.arctan2(d[..., 0], -d[..., 2])
    phi = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return theta, phi


def rotation(yaw, pitch, roll=0.0):
    """Head rotation R (Sec. 4 of the paper).

    R maps the canonical viewing direction (0, 0, -1) to ``sph_to_dir(yaw, pitch)``;
    ``roll`` turns the head around the viewing axis.
    """
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    r_yaw = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    r_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
    return r_yaw @ r_pitch @ r_roll


def fibonacci_sphere(n):
    """``n`` points spread almost uniformly over the sphere, as (theta, phi).

    Used as the uniformly sampled sphere points of S-PSNR (Sec. 5, Fig. 4).
    """
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arcsin(1.0 - 2.0 * i / n)
    theta = np.mod(np.pi * (1.0 + 5.0 ** 0.5) * i, 2.0 * np.pi) - np.pi
    return theta, phi
