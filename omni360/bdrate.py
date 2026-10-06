"""Bjontegaard delta metrics (G. Bjontegaard, ITU-T VCEG-M33, 2001), used in Tables 1 and 2."""
import numpy as np


def _integral(x, y, lo, hi, method):
    order = np.argsort(x)
    x, y = np.asarray(x, np.float64)[order], np.asarray(y, np.float64)[order]
    if method == "cubic":
        poly = np.polyint(np.polyfit(x, y, 3))
        return np.polyval(poly, hi) - np.polyval(poly, lo)
    if method == "pchip":
        from scipy.interpolate import PchipInterpolator
        return PchipInterpolator(x, y).integrate(lo, hi)
    raise ValueError(f"unknown method {method!r}")


def _overlap(a, b):
    lo, hi = max(np.min(a), np.min(b)), min(np.max(a), np.max(b))
    if lo >= hi:
        raise ValueError("the two RD curves do not overlap")
    return lo, hi


def bd_rate(rate_ref, quality_ref, rate_test, quality_test, method="cubic"):
    """Average bitrate difference (%) of ``test`` vs ``ref`` at equal quality.

    Negative values mean the test saves bitrate. ``method`` is "cubic" (the
    original polynomial fit) or "pchip" (piecewise cubic, as in recent JVET tools).
    """
    lo, hi = _overlap(quality_ref, quality_test)
    log_ref, log_test = np.log(rate_ref), np.log(rate_test)
    diff = (_integral(quality_test, log_test, lo, hi, method)
            - _integral(quality_ref, log_ref, lo, hi, method)) / (hi - lo)
    return float((np.exp(diff) - 1.0) * 100.0)


def bd_psnr(rate_ref, quality_ref, rate_test, quality_test, method="cubic"):
    """Average quality difference (dB) of ``test`` vs ``ref`` at equal bitrate."""
    log_ref, log_test = np.log(rate_ref), np.log(rate_test)
    lo, hi = _overlap(log_ref, log_test)
    return float((_integral(log_test, quality_test, lo, hi, method)
                  - _integral(log_ref, quality_ref, lo, hi, method)) / (hi - lo))
