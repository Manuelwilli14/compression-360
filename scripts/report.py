"""BD-rate tables and RD curves from the result CSVs (Tables 1 and 2, Fig. 6 and 7 of the paper).

Table 1 style: BD-rate of every codec and projection against a reference
(HEVC on ERP by default), measured with one metric (viewport PSNR by default).
Negative values are bitrate savings.

Table 2 style: how far each cheaper metric is from viewport PSNR, as the
absolute BD-rate between the two RD curves of the same coded data.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from omni360.bdrate import bd_rate  # noqa: E402

APPROXIMATIONS = ["WeightSph", "LatSph", "Sph", "Quad"]


def safe_bd_rate(rate_ref, q_ref, rate_test, q_test):
    try:
        return bd_rate(rate_ref, q_ref, rate_test, q_test)
    except ValueError:  # curves without a common quality range
        return np.nan


def table_vs_reference(df, reference, metric):
    ref_codec, ref_proj = reference.split(":")
    table = {}
    for (image, codec, proj), curve in df.groupby(["image", "codec", "projection"]):
        ref = df[(df.image == image) & (df.codec == ref_codec) & (df.projection == ref_proj)]
        if (codec, proj) == (ref_codec, ref_proj) or ref.empty:
            continue
        table.setdefault(image, {})[f"{codec}:{proj}"] = safe_bd_rate(ref.bits, ref[metric], curve.bits, curve[metric])
    table = pd.DataFrame(table).T.sort_index()
    table.loc["Average"] = table.mean()
    return table


def table_metric_error(df):
    errors = {}
    for (_, codec, proj), curve in df.groupby(["image", "codec", "projection"]):
        for metric in APPROXIMATIONS:
            if curve[metric].notna().all():
                value = abs(safe_bd_rate(curve.bits, curve.Viewport, curve.bits, curve[metric]))
                errors.setdefault(f"{codec}:{proj}", {}).setdefault(metric, []).append(value)
    table = pd.DataFrame({k: {m: np.nanmean(v) for m, v in d.items()} for k, d in errors.items()}).T
    table.loc["Average"] = table.mean()
    return table


def to_markdown(table, fmt="{:+.2f} %"):
    header = "| | " + " | ".join(table.columns) + " |"
    lines = [header, "|" + "---|" * (len(table.columns) + 1)]
    for name, row in table.iterrows():
        cells = ["n/a" if pd.isna(v) else fmt.format(v) for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_rd(df, image, reference, path):
    data = df[df.image == image]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    for (codec, proj), curve in data.groupby(["codec", "projection"]):
        curve = curve.sort_values("bpp")
        ax1.plot(curve.bpp, curve.Viewport, "o-", ms=4, label=f"{codec}:{proj}")
    ax1.set(title=f"{image}: viewport PSNR", xlabel="bits per ERP pixel", ylabel="PSNR (dB)")
    ref_codec, ref_proj = reference.split(":")
    ref = data[(data.codec == ref_codec) & (data.projection == ref_proj)].sort_values("bpp")
    for metric in ["Viewport", *APPROXIMATIONS]:
        if ref[metric].notna().all():
            ax2.plot(ref.bpp, ref[metric], "o-", ms=4, label=metric)
    ax2.set(title=f"{image}: metrics on {reference}", xlabel="bits per ERP pixel", ylabel="PSNR (dB)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="result files from run_hevc.py")
    ap.add_argument("--reference", default="hevc:erp", help="codec:projection used as BD-rate anchor")
    ap.add_argument("--metric", default="Viewport", help="quality metric for the BD-rate table")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = [
        f"## BD-rate vs {args.reference} ({args.metric} PSNR, negative = saving)",
        to_markdown(table_vs_reference(df, args.reference, args.metric)),
        "",
        "## Distance of each metric to viewport PSNR (|BD-rate|)",
        to_markdown(table_metric_error(df), "{:.2f} %"),
    ]
    text = "\n\n".join(report)
    print(text)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    for image in df.image.unique():
        plot_rd(df, image, args.reference, out / f"rd_{image}.png")
    print(f"\nreport and RD curves written to {out.resolve()}")


if __name__ == "__main__":
    main()
