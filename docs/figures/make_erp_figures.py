"""Explanatory figures: why the equirectangular projection oversamples the poles."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from omni360.io import synthetic_panorama
from omni360.projections import equirectangular
from omni360.viewport import render_viewport

OUT = Path(r"C:\Users\HP\Documents\compression-360\docs\figures")
OUT.mkdir(parents=True, exist_ok=True)

SURF, TXT, TXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 12, "text.color": TXT, "axes.labelcolor": TXT2, "axes.titlesize": 13,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.edgecolor": MUTED,
})

# Orthographic camera looking at longitude 0 from 22 degrees above the equator.
EL = np.radians(22)
VIEW = np.array([np.cos(EL), 0.0, np.sin(EL)])
RIGHT = np.array([0.0, 1.0, 0.0])
UP = np.array([-np.sin(EL), 0.0, np.cos(EL)])


def point(lat, lon):
    lat, lon = np.broadcast_arrays(np.radians(lat), np.radians(lon))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)


def screen(p):
    return p @ RIGHT, p @ UP, p @ VIEW


def draw_sphere(ax):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=1.2)
    for lat in range(-60, 90, 30):
        x, y, d = screen(point(lat, np.linspace(-180, 180, 400)))
        ax.plot(np.where(d > 0, x, np.nan), np.where(d > 0, y, np.nan), color=GRID, lw=0.8)
    for lon in range(-180, 180, 30):
        x, y, d = screen(point(np.linspace(-90, 90, 200), lon))
        ax.plot(np.where(d > 0, x, np.nan), np.where(d > 0, y, np.nan), color=GRID, lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")


def erp_frame(ax):
    ax.add_patch(Rectangle((-180, -90), 360, 180, fill=False, ec=MUTED, lw=1.2))
    for lat in range(-60, 90, 30):
        ax.axhline(lat, color=GRID, lw=0.8, zorder=0)
    for lon in range(-150, 180, 30):
        ax.axvline(lon, color=GRID, lw=0.8, zorder=0)
    ax.set(xlim=(-185, 185), ylim=(-95, 95), xticks=[-180, -90, 0, 90, 180], yticks=[-90, -60, -30, 0, 30, 60, 90])
    ax.set_xlabel("longitude (°)  ->  colonnes de l'image")
    ax.set_ylabel("latitude (°)  ->  lignes de l'image")
    ax.set_aspect("equal")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def fig_circles():
    rings = [(0, BLUE, "équateur"), (60, ORANGE, "latitude 60°"), (80, AQUA, "latitude 80°")]
    n = 24
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.55]})
    draw_sphere(ax1)
    for lat, color, label in rings:
        x, y, d = screen(point(lat, np.linspace(-180, 180, 400)))
        ax1.plot(np.where(d > 0, x, np.nan), np.where(d > 0, y, np.nan), color=color, lw=2.2)
        ax1.plot(np.where(d <= 0, x, np.nan), np.where(d <= 0, y, np.nan), color=color, lw=1, ls="--", alpha=0.45)
        px, py, pd_ = screen(point(lat, np.arange(n) * 360 / n - 180))
        ax1.scatter(px[pd_ > 0], py[pd_ > 0], s=34, color=color, ec=SURF, lw=1, zorder=3)
        ax1.scatter(px[pd_ <= 0], py[pd_ <= 0], s=14, color=color, alpha=0.35, zorder=2)
        lx, ly, _ = screen(point(lat, 75))
        ax1.annotate(label, (lx, ly), xytext=(1.12, ly), color=TXT, fontsize=11, va="center",
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    ax1.set_xlim(-1.1, 1.75)
    ax1.set_title("Sur la sphère : trois cercles de latitude\n(24 points par cercle)", color=TXT)

    erp_frame(ax2)
    notes = {0: "tour complet = 1      ->  24 pixels  ->  densité ×1",
             60: "tour = 0,50 (cos 60°)  ->  24 pixels  ->  densité ×2",
             80: "tour = 0,17 (cos 80°)  ->  24 pixels  ->  densité ×5,8"}
    for lat, color, _ in rings:
        ax2.plot([-180, 180], [lat, lat], color=color, lw=2.2)
        ax2.scatter((np.arange(n) + 0.5) * 360 / n - 180, np.full(n, lat), s=34, color=color, ec=SURF, lw=1, zorder=3)
        ax2.text(-176, lat + 2.5, notes[lat], fontsize=10.5, color=TXT, va="bottom",
                 bbox={"fc": SURF, "ec": "none", "pad": 1.0})
    ax2.set_title("Dans l'image équirectangulaire : chaque cercle devient une ligne\nde même largeur, avec autant de pixels",
                  color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "erp_1_cercles.png", dpi=140)
    plt.close(fig)


def cap(lat, lon, radius):
    c = point(lat, lon)
    e1 = np.cross(c, [0.0, 0.0, 1.0]) if abs(lat) < 89 else np.array([1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    t = np.linspace(0, 2 * np.pi, 120)[:, None]
    r = np.radians(radius)
    return np.cos(r) * c + np.sin(r) * (np.cos(t) * e1 + np.sin(t) * e2)


def fig_tissot():
    lats, lons, radius = [0, 30, 60, 80], [-60, 0, 60], 4.5
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.55]})
    draw_sphere(ax1)
    for lat in lats:
        for lon in lons:
            pts = cap(lat, lon, radius)
            x, y, d = screen(pts)
            if d.mean() > 0:
                ax1.add_patch(Polygon(np.c_[x, y], fc=BLUE, ec=SURF, lw=1, alpha=0.9))
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_title("Sur la sphère : 12 petits disques identiques\n(même taille, 9° de diamètre)", color=TXT)

    erp_frame(ax2)
    for lat in lats:
        for lon in lons:
            pts = cap(lat, lon, radius)
            plat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
            plon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            ax2.add_patch(Polygon(np.c_[plon, plat], fc=BLUE, ec=SURF, lw=1, alpha=0.9))
        stretch = 1 / np.cos(np.radians(lat))
        ax2.text(100, lat, f"{lat}° : largeur ×{stretch:.2f}".replace(".", ","), fontsize=11, va="center", color=TXT,
                 bbox={"fc": SURF, "ec": "none", "pad": 1.5})
    ax2.set_title("Dans l'image équirectangulaire : plus le disque est haut,\nplus il est étiré et plus il occupe de pixels",
                  color=TXT)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "erp_2_disques_etires.png", dpi=140)
    plt.close(fig)


def fig_budget():
    bands = ["0° à 30°", "30° à 60°", "60° à 90°"]
    pixels = np.array([1, 1, 1]) / 3 * 100
    area = np.array([np.sin(np.radians(30)), np.sin(np.radians(60)) - np.sin(np.radians(30)),
                     1 - np.sin(np.radians(60))]) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    y = np.arange(3)[::-1]
    h = 0.36
    ax1.barh(y + h / 2, pixels, h, color=ORANGE, ec=SURF, lw=2, label="part des pixels de l'image ERP")
    ax1.barh(y - h / 2, area, h, color=BLUE, ec=SURF, lw=2, label="part réelle de la sphère")
    for yy, p, a in zip(y, pixels, area):
        ax1.text(p + 1, yy + h / 2, f"{p:.0f} %", va="center", color=TXT, fontsize=11)
        ax1.text(a + 1, yy - h / 2, f"{a:.1f} %".replace(".", ","), va="center", color=TXT, fontsize=11)
    ax1.set_yticks(y, [f"{b}\n(nord + sud)" for b in bands])
    ax1.set_xlim(0, 62)
    ax1.set_xlabel("pourcentage")
    ax1.legend(loc="lower right", frameon=False, fontsize=10.5)
    ax1.set_title("Même nombre de pixels par bande de 30°,\nmais des surfaces très différentes", color=TXT)
    ax1.grid(axis="x", color=GRID, lw=0.8)
    ax1.set_axisbelow(True)

    lat = np.linspace(0, 86, 400)
    ax2.plot(lat, 1 / np.cos(np.radians(lat)), color=BLUE, lw=2)
    for l, label in [(30, "×1,15"), (60, "×2"), (80, "×5,8")]:
        v = 1 / np.cos(np.radians(l))
        ax2.scatter([l], [v], s=50, color=BLUE, ec=SURF, lw=1.5, zorder=3)
        ax2.annotate(f"{l}° : {label}", (l, v), xytext=(-78, 12), textcoords="offset points", color=TXT, fontsize=11)
    ax2.text(84, 13.2, "-> infini\nau pôle", color=TXT2, fontsize=10.5, ha="right", va="top")
    ax2.set(xlim=(0, 90), ylim=(0, 14), xlabel="latitude (°)", ylabel="pixels par unité de longueur\n(relatif à l'équateur)")
    ax2.set_title("Suréchantillonnage horizontal = 1 / cos(latitude)", color=TXT)
    ax2.grid(color=GRID, lw=0.8)
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "erp_3_pixels_vs_surface.png", dpi=140)
    plt.close(fig)


def fig_example():
    pano = synthetic_panorama(2048)
    rows = round(1024 * 15 / 180)
    pano[:rows] = 0.35 * pano[:rows] + 0.65 * np.array([235, 104, 52])
    view = render_viewport(pano, equirectangular(2048), 0.0, np.radians(72), width=700, height=700,
                           fov_x=np.radians(100))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax1.imshow(np.clip(pano, 0, 255).astype(np.uint8), extent=(-180, 180, -90, 90))
    ax1.set(xticks=[-180, -90, 0, 90, 180], yticks=[-90, -45, 0, 45, 90], xlabel="longitude (°)", ylabel="latitude (°)")
    pix = rows / 1024 * 100
    ax1.set_title(f"Image ERP : la bande orange (75° à 90°) occupe {pix:.1f} % des pixels".replace(".", ","), color=TXT)
    ax2.imshow(np.clip(view, 0, 255).astype(np.uint8))
    ax2.axis("off")
    area = (1 - np.sin(np.radians(75))) / 2 * 100
    ax2.set_title(f"Vue dans le casque vers le pôle : la même bande\nn'est qu'un petit disque, {area:.1f} % de la sphère".replace(".", ","),
                  color=TXT)
    fig.tight_layout()
    fig.savefig(OUT / "erp_4_exemple_pole.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_circles()
    fig_tissot()
    fig_budget()
    fig_example()
    print("written to", OUT)
