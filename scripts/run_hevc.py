"""Step 3: code panoramas in each projection with HEVC (x265 intra) and score them.

Reproduces the protocol of Sec. 6 of the paper on still images: a ground truth
ERP at 1.5x the coded resolution is resampled to every projection, coded at
four QPs, then compared on the sphere and on viewports. Scores are computed on
luma. Results are appended to a CSV file read by scripts/report.py.
"""
import argparse
import time

from omni360.experiment import append_rows, list_panoramas, load_panorama, make_evaluator, panorama_name
from omni360.hevc import find_ffmpeg, x265_intra
from omni360.io import to_uint8
from omni360.projections import PROJECTIONS, equirectangular, make_projection
from omni360.sampling import render


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="ERP panoramas or folders ('synthetic' for the test pattern)")
    ap.add_argument("--width", type=int, default=2048, help="width of the coded ERP (paper: 4096)")
    ap.add_argument("--gt-width", type=int, default=3072, help="width of the ground-truth ERP (paper: 6144)")
    ap.add_argument("--projections", nargs="+", default=list(PROJECTIONS), choices=PROJECTIONS)
    ap.add_argument("--qps", type=int, nargs="+", default=[22, 27, 32, 37])
    ap.add_argument("--preset", default="medium", help="x265 preset")
    ap.add_argument("--viewports", type=int, default=30, help="number of synthetic head orientations")
    ap.add_argument("--viewport-size", type=int, default=1024)
    ap.add_argument("--out", default="results/hevc.csv")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
    gt_proj = equirectangular(args.gt_width)
    coded_pixels = args.width * (args.width // 2)  # bpp is relative to the coded ERP grid
    for item in list_panoramas(args.inputs):
        gt = load_panorama(item, args.gt_width)
        evaluate = make_evaluator(gt, n_viewports=args.viewports, viewport_size=args.viewport_size)
        rows = []
        for name in args.projections:
            proj = make_projection(name, args.width)
            frame = to_uint8(render(gt, gt_proj, proj))
            for qp in args.qps:
                start = time.time()
                luma, bits = x265_intra(frame, qp, args.preset, ffmpeg)
                scores = evaluate(luma, proj)
                rows.append({"image": panorama_name(item), "codec": "hevc", "projection": name, "param": qp,
                             "bits": bits, "bpp": bits / coded_pixels, **scores,
                             "seconds": time.time() - start})
                print(f"{panorama_name(item)} {name:<6} QP{qp}  {bits / coded_pixels:.3f} bpp  "
                      f"viewport {scores['Viewport']:.2f} dB  S-PSNR {scores['Sph']:.2f} dB")
        append_rows(args.out, rows)
    print(f"results appended to {args.out}")


if __name__ == "__main__":
    main()
