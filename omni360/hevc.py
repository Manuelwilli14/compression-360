"""HEVC intra coding of a panorama with x265 (through FFmpeg).

Colour conversion is done here, not by FFmpeg, so that the decoded luma plane
can be compared exactly with ``metrics.to_luma`` of the ground truth: BT.709
full-range Y'CbCr, chroma subsampled 2x2 (4:2:0).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

_KR, _KB = 0.2126, 0.0722


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("FFmpeg not found: install it or `pip install imageio-ffmpeg`") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def rgb_to_yuv420(rgb):
    """RGB (h, w, 3) in [0, 255] to 8-bit planes Y (h, w), Cb and Cr (h/2, w/2)."""
    rgb = np.asarray(rgb, np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = _KR * r + (1 - _KR - _KB) * g + _KB * b
    cb = (b - y) / (2 * (1 - _KB)) + 128
    cr = (r - y) / (2 * (1 - _KR)) + 128
    h, w = y.shape

    def down(c):
        return c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    return [np.clip(np.rint(p), 0, 255).astype(np.uint8) for p in (y, down(cb), down(cr))]


def x265_intra(rgb, qp, preset="medium", ffmpeg=None):
    """Code one frame with x265 at a fixed QP.

    Returns ``(decoded_luma, bits)``: the decoded Y plane as float32 and the size
    of the HEVC bitstream in bits.
    """
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise ValueError("4:2:0 coding needs even frame dimensions")
    ffmpeg = ffmpeg or find_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        src, bitstream, dec = Path(tmp, "in.yuv"), Path(tmp, "out.hevc"), Path(tmp, "dec.yuv")
        src.write_bytes(b"".join(p.tobytes() for p in rgb_to_yuv420(rgb)))
        raw = ["-f", "rawvideo", "-pix_fmt", "yuv420p", "-s:v", f"{w}x{h}"]
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", *raw, "-i", str(src), "-frames:v", "1",
             "-c:v", "libx265", "-preset", preset, "-x265-params", f"qp={qp}:info=0:log-level=error",
             str(bitstream)],
            check=True)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(bitstream), "-f", "rawvideo",
             "-pix_fmt", "yuv420p", str(dec)],
            check=True)
        luma = np.frombuffer(dec.read_bytes(), np.uint8, count=h * w).reshape(h, w)
        return luma.astype(np.float32), bitstream.stat().st_size * 8
