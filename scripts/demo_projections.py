"""Step 1: the projections of the paper on one panorama, plus Fig. 1 and Fig. 5.

Writes to the output folder: the panorama in each projection, viewports looking
at the horizon and near the north pole rendered from each projection, the
sampling densities of Fig. 1 and head-motion statistics in the style of Fig. 5.
It also prints how much each projection loses before any coding (S-PSNR of the
resampled panorama against the ground truth).
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from omni360.experiment import load_panorama, panorama_name  # noqa: E402
from omni360.io import save_image, to_uint8  # noqa: E402
from omni360.metrics import s_psnr, to_luma  # noqa: E402
from omni360.projections import PROJECTIONS, equirectangular, make_projection  # noqa: E402
from omni360.sampling import render  # noqa: E402
from omni360.viewport import HeadMotionStats, render_viewport, synthetic_orientations  # noqa: E402


def plot_sampling_density(path):
    """Fig. 1: sampling density relative to the equator of the equirectangular projection."""
    phi = np.linspace(-1.5, 1.5, 601)
    inv_cos = 1 / np.cos(phi)
    dyadic_h = np.where(np.abs(phi) >= np.pi / 3, 0.5, 1.0) * inv_cos
    panels = [
        ("(a) horizontal", {"Equirectangular, Equal-area": inv_cos, "Dyadic": dyadic_h}),
        ("(b) vertical", {"Equirectangular, Dyadic": np.ones_like(phi), "Equal-area": np.cos(phi)}),
        ("(c) combined", {"Equirectangular": inv_cos, "Equal-area": np.ones_like(phi), "Dyadic": dyadic_h}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (title, curves) in zip(axes, panels):
        for label, values in curves.items():
            ax.plot(phi, values, label=label)
        ax.set(title=title, xlabel="latitude phi (rad)", ylim=(0, 5))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("relative sampling density")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_head_motion(path, stats):
    """Fig. 5 style: where viewers look, from synthetic head orientations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    im = ax1.imshow(stats.density, extent=(-180, 180, -90, 90), aspect="auto", cmap="viridis")
    ax1.set(title="(a) access density (synthetic head motion)", xlabel="longitude (deg)", ylabel="latitude (deg)")
    fig.colorbar(im, ax=ax1)
    rows = stats.bins[0]
    lat = 90 - (np.arange(rows) + 0.5) * 180 / rows
    ax2.plot(lat, stats.latitude_density)
    ax2.set(title="(b) averaged over longitude", xlabel="latitude (deg)",
            ylabel="access density (sphere mean = 1)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="synthetic", help="ERP panorama, or 'synthetic' (default)")
    ap.add_argument("--width", type=int, default=2048, help="width of the projected ERP (paper: 4096)")
    ap.add_argument("--gt-width", type=int, default=3072, help="width of the ground-truth ERP (paper: 6144)")
    ap.add_argument("--out", default="results/demo")
    args = ap.parse_args()

    out = Path(args.out)
    gt = load_panorama(args.input, args.gt_width)
    gt_proj = equirectangular(args.gt_width)
    save_image(out / f"{panorama_name(args.input)}_ground_truth.jpg", gt)
    erp_pixels = args.width * (args.width // 2)

    print(f"{'projection':<10} {'frame':>11} {'pixels vs ERP':>14} {'S-PSNR before coding':>21}")
    for name in PROJECTIONS:
        proj = make_projection(name, args.width)
        frame = to_uint8(render(gt, gt_proj, proj))
        save_image(out / f"projection_{name}.png", frame)
        h, w = proj.frame_size
        quality = s_psnr(to_luma(gt), gt_proj, to_luma(frame), proj)
        print(f"{name:<10} {w:>5}x{h:<5} {h * w / erp_pixels:>13.0%} {quality:>18.2f} dB")
        for label, pitch in (("horizon", 0.0), ("pole", 80.0)):
            view = render_viewport(frame, proj, np.radians(30), np.radians(pitch), width=640, height=640)
            save_image(out / f"viewport_{label}_{name}.png", view)

    plot_sampling_density(out / "fig1_sampling_density.png")
    plot_head_motion(out / "fig5_head_motion.png", HeadMotionStats(synthetic_orientations(500, seed=1)))
    print(f"images and figures written to {out.resolve()}")


if __name__ == "__main__":
    main()
