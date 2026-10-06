"""Image loading and saving (Unicode-safe on Windows) and a synthetic test panorama."""
from pathlib import Path

import cv2
import numpy as np

from .projections import equirectangular
from .sphere import sph_to_dir


def load_image(path, width=None):
    """Load an RGB image as float32 in [0, 255], optionally resized to ``width`` (aspect kept)."""
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read image {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if width and img.shape[1] != width:
        height = round(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def to_uint8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save_image(path, img):
    """Save a float or uint8 RGB (or grayscale) image; the format follows the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = to_uint8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"cannot encode image {path}")
    buf.tofile(str(path))


def synthetic_panorama(width=2048):
    """Equirectangular test pattern: a 15 degree checkerboard over smooth colour gradients.

    Handy to see how each projection distorts the sphere, and to run the
    pipeline before downloading real panoramas.
    """
    proj = equirectangular(width)
    theta, phi = proj.pixel_to_sphere()
    d = sph_to_dir(theta, phi)
    cell = np.radians(15.0)
    checker = ((np.floor(theta / cell) + np.floor(phi / cell)) % 2) * 2.0 - 1.0
    rgb = np.stack([
        128 + 70 * d[..., 0] + 25 * checker,
        128 + 70 * d[..., 1] + 25 * checker,
        128 - 70 * d[..., 2] + 25 * checker,
    ], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.float32)
