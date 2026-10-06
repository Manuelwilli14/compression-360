"""Reading panoramas at arbitrary sphere points (bicubic interpolation).

The paper uses bicubic interpolation for every sub-pixel access (Sec. 3). It
is implemented here in NumPy rather than with ``cv2.remap``, whose fixed-point
maps round positions to 1/32 pixel and would bias high-PSNR comparisons.
"""
import numpy as np

_CHUNK = 1 << 19


def _keys(s, a=-0.5):
    """Keys cubic convolution kernel (a = -0.5, as in MATLAB's bicubic)."""
    s = np.abs(s)
    s2, s3 = s * s, s * s * s
    return np.where(s <= 1.0, (a + 2.0) * s3 - (a + 3.0) * s2 + 1.0,
                    np.where(s < 2.0, a * s3 - 5.0 * a * s2 + 8.0 * a * s - 4.0 * a, 0.0))


def _bicubic(img, x, y, wrap_x):
    """Sample ``img`` (h, w, c) at 1-D pixel coordinates ``x``, ``y``."""
    h, w = img.shape[:2]
    x_floor, y_floor = np.floor(x), np.floor(y)
    tx, ty = (x - x_floor)[:, None], (y - y_floor)[:, None]
    xi, yi = x_floor.astype(np.int64), y_floor.astype(np.int64)
    wx = [_keys(tx - i) for i in range(-1, 3)]
    cols = [(xi + i) % w if wrap_x else np.clip(xi + i, 0, w - 1) for i in range(-1, 3)]
    out = np.zeros((x.size, img.shape[2]), np.float64)
    for j in range(-1, 3):
        rows = np.clip(yi + j, 0, h - 1)
        line = sum(wx[i] * img[rows, cols[i]] for i in range(4))
        out += _keys(ty - j) * line
    return out


def sample(frame, proj, theta, phi):
    """Values of a panorama ``frame`` stored in ``proj`` at sphere points (theta, phi)."""
    frame = np.asarray(frame, np.float32)
    gray = frame.ndim == 2
    if gray:
        frame = frame[..., None]
    theta, phi = np.broadcast_arrays(np.asarray(theta, np.float64), np.asarray(phi, np.float64))
    shape = theta.shape
    k, x, y = proj.sphere_to_region(theta.ravel(), phi.ravel())
    out = np.empty((theta.size, frame.shape[2]), np.float32)
    for idx, r in enumerate(proj.regions):
        points = np.flatnonzero(k == idx)
        sub = frame[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w]
        for start in range(0, points.size, _CHUNK):
            m = points[start:start + _CHUNK]
            out[m] = _bicubic(sub, x[m], y[m], r.wrap_x)
    out = out.reshape(shape + (frame.shape[2],))
    return out[..., 0] if gray else out


def render(src, src_proj, dst_proj):
    """Generate a panorama in ``dst_proj`` from one in ``src_proj`` (Fig. 2 of the paper).

    Each target pixel is mapped to the sphere, then read from the source.
    """
    theta, phi = dst_proj.pixel_to_sphere()
    src = np.asarray(src, np.float32)
    out = np.zeros(dst_proj.frame_size + src.shape[2:], np.float32)
    valid = ~np.isnan(theta)
    out[valid] = sample(src, src_proj, theta[valid], phi[valid])
    return out
