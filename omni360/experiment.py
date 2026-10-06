"""Shared plumbing for the experiment scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_image, synthetic_panorama
from .metrics import Evaluator, to_luma
from .projections import equirectangular
from .viewport import HeadMotionStats, synthetic_orientations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLUMNS = ["image", "codec", "projection", "param", "bits", "bpp",
           "Viewport", "WeightSph", "LatSph", "Sph", "Quad", "seconds"]


def list_panoramas(inputs):
    """Expand files and folders into image paths; the word ``synthetic`` is kept as is."""
    items = []
    for item in inputs:
        if item == "synthetic":
            items.append(item)
            continue
        path = Path(item)
        if path.is_dir():
            items += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            items.append(path)
    if not items:
        raise SystemExit("no panorama found")
    return items


def panorama_name(item):
    return item if item == "synthetic" else Path(item).stem


def load_panorama(item, width):
    """Ground-truth equirectangular panorama resized to ``width`` x ``width / 2``."""
    if item == "synthetic":
        return synthetic_panorama(width)
    img = load_image(item, width)
    if img.shape[0] * 2 != img.shape[1]:
        raise ValueError(f"{item}: expected a 2:1 equirectangular panorama, got {img.shape[1]}x{img.shape[0]}")
    return img


def make_evaluator(gt_rgb, n_viewports=30, n_points=655362, viewport_size=1024, fov_deg=90.0, seed=0):
    """Evaluator on the luma of a ground-truth ERP panorama.

    Viewport PSNR uses ``n_viewports`` synthetic head orientations. WeightSph and
    LatSph use statistics estimated from a *different* synthetic set, just as the
    paper estimates them over other users than the trajectory being evaluated.
    """
    gt = to_luma(gt_rgb)
    h, w = gt.shape
    return Evaluator(
        gt, equirectangular(w, h),
        orientations=synthetic_orientations(n_viewports, seed=seed),
        stats=HeadMotionStats(synthetic_orientations(500, seed=seed + 1)),
        n_points=n_points, width=viewport_size, height=viewport_size, fov_x=np.radians(fov_deg))


def append_rows(path, rows):
    """Append result rows to a CSV file (created with a header if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=COLUMNS)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
