"""Explanatory figures: prediction in video coding (intra, inter, bit savings, the 360 degree case).

Run from the project root with the project venv:
    python docs/figures/make_prediction_figures.py
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from omni360.io import load_image
from omni360.metrics import to_luma
from omni360.projections import equirectangular
from omni360.sampling import sample
from omni360.sphere import dir_to_sph, rotation, sph_to_dir

OUT = Path(__file__).resolve().parent
BOATS = r"C:\Users\HP\Documents\ESIR Formation\Semestre 9\COV\TP\boats.bmp"
PANO = Path(__file__).resolve().parents[2] / "data" / "test" / "wide_street_01.jpg"

SURF, TXT, TXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#9a9893", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RESID = LinearSegmentedColormap.from_list("residual", [BLUE, "#f0efec", "#e34948"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 12, "text.color": TXT, "axes.labelcolor": TXT2, "axes.titlesize": 12.5,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.edgecolor": MUTED,
})


def load_gray(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float64)


def no_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def fr(x):
    """One-decimal number with a French decimal comma."""
    return f"{x:.1f}".replace(".", ",")


def entropy(values):
    _, counts = np.unique(np.rint(values).astype(int), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- block tools

def intra_modes(img, y, x, b):
    """Four simple intra predictions of the b x b block at (y, x) from its coded neighbours."""
    top, left, corner = img[y - 1, x:x + b], img[y:y + b, x - 1], img[y - 1, x - 1]
    i, j = np.mgrid[0:b, 0:b]
    ref = np.concatenate([left[::-1], [corner], top])  # pixel (i, j) on a 45 degree line hits ref[b + j - i]
    return {
        "DC (moyenne)": np.full((b, b), (top.sum() + left.sum()) / (2 * b)),
        "Vertical": np.tile(top, (b, 1)),
        "Horizontal": np.tile(left[:, None], (1, b)),
        "Diagonal 45°": ref[b + j - i],
    }


def intra_residual(img, b=16):
    """Residual of the best of the four intra modes, block by block (open loop, for illustration)."""
    res = np.zeros_like(img)
    for y in range(b, img.shape[0] - b + 1, b):
        for x in range(b, img.shape[1] - b + 1, b):
            block = img[y:y + b, x:x + b]
            res[y:y + b, x:x + b] = min((block - p for p in intra_modes(img, y, x, b).values()),
                                        key=lambda r: np.abs(r).sum())
    return res[b:, b:]


def block_matching(prev, cur, b=16, r=12, wrap_x=False, lam=4.0):
    """Full-search block matching (SAD + small vector cost). Returns vectors and the prediction."""
    h, w = cur.shape
    pad = np.pad(prev, ((r, r), (0, 0)), mode="edge")
    pad = np.pad(pad, ((0, 0), (r, r)), mode="wrap" if wrap_x else "edge")
    best = np.full((h // b, w // b), np.inf)
    vec = np.zeros((h // b, w // b, 2), int)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            shifted = pad[r + dy:r + dy + h, r + dx:r + dx + w]
            cost = np.abs(cur - shifted).reshape(h // b, b, w // b, b).sum(axis=(1, 3)) + lam * (abs(dx) + abs(dy))
            better = cost < best
            best[better] = cost[better]
            vec[better] = (dy, dx)
    pred = np.empty_like(cur)
    for by in range(h // b):
        for bx in range(w // b):
            dy, dx = vec[by, bx]
            y, x = by * b + r + dy, bx * b + r + dx
            pred[by * b:(by + 1) * b, bx * b:(bx + 1) * b] = pad[y:y + b, x:x + b]
    return vec, pred


# ---------------------------------------------------------------- figure 1

def fig_principle():
    top = np.array([120, 122, 124, 126])
    block = np.array([[121, 123, 124, 127], [122, 123, 125, 127], [121, 124, 126, 128], [122, 124, 125, 128]])
    pred = np.tile(top, (4, 1))
    resid = block - pred
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))

    def grid(ax, values, cmap, vmin, vmax, neighbours=None):
        shown = values if neighbours is None else np.vstack([neighbours, values])
        ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
        for (i, j), v in np.ndenumerate(shown):
            dark = cmap == "gray" and v < (vmin + vmax) / 2
            ax.text(j, i, f"{v:d}", ha="center", va="center", fontsize=15, color="white" if dark else TXT)
        for k in range(shown.shape[0] + 1):
            ax.axhline(k - 0.5, color=SURF, lw=2)
        for k in range(shown.shape[1] + 1):
            ax.axvline(k - 0.5, color=SURF, lw=2)
        if neighbours is not None:
            ax.add_patch(Rectangle((-0.5, -0.5), 4, 1, fill=False, ec=ORANGE, lw=3))
            ax.text(4.65, 0, "déjà\ncodée", color=ORANGE, va="center", fontsize=11)
        ax.set_xlim(-0.6, 5.4)
        no_ticks(ax)
        for s in ax.spines.values():
            s.set_visible(False)

    grid(axes[0], block, "gray", 100, 135, neighbours=top[None])
    axes[0].set_title("1. Le bloc à coder\n(sous la ligne déjà codée)")
    grid(axes[1], pred, "gray", 100, 135, neighbours=top[None])
    axes[1].set_title("2. La prédiction : on recopie\nla ligne du dessus vers le bas")
    grid(axes[2], resid, RESID, -4, 4)
    axes[2].set_title("3. Le résidu = bloc − prédiction\nC'est la seule chose envoyée")
    fig.text(0.333, 0.5, "−", fontsize=34, ha="center", va="center", color=TXT2)
    fig.text(0.655, 0.5, "=", fontsize=34, ha="center", va="center", color=TXT2)
    fig.text(0.5, 0.04, "Sans prédiction : il faut écrire 121, 123, 124… (valeurs de 0 à 255).   "
             "Avec prédiction : il suffit d'écrire 1, 1, 0… (petites valeurs, très peu de bits).\n"
             "Le décodeur fait le même calcul : il connaît la ligne du dessus, refait la même prédiction "
             "et lui ajoute le résidu reçu.", ha="center", fontsize=11.5, color=TXT)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "pred_1_principe.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- figure 2

def fig_intra(img):
    b = 16
    candidates = []
    for y in range(b, img.shape[0] - b, b):
        for x in range(b, img.shape[1] - b, b):
            block = img[y:y + b, x:x + b]
            if block.std() < 18:
                continue
            err = {k: np.abs(block - p).mean() for k, p in intra_modes(img, y, x, b).items()}
            candidates.append((err["DC (moyenne)"] / err["Vertical"], y, x))
    _, y, x = max(candidates)
    block = img[y:y + b, x:x + b]
    modes = intra_modes(img, y, x, b)
    errors = {k: np.abs(block - p).mean() for k, p in modes.items()}
    best = min(errors, key=errors.get)

    fig = plt.figure(figsize=(16, 7.4))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[2.1, 1.1, 1, 1, 1, 1], hspace=0.35, wspace=0.12)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), b, b, fill=False, ec=ORANGE, lw=2.5))
    ax.set_title("Image (boats.bmp, votre TP)\nle bloc étudié est encadré")
    no_ticks(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img[y - 1:y + b, x - 1:x + b], cmap="gray", vmin=0, vmax=255)
    ax.add_patch(Rectangle((-0.5, -0.5), b + 1, 1, fill=False, ec=ORANGE, lw=2))
    ax.add_patch(Rectangle((-0.5, -0.5), 1, b + 1, fill=False, ec=ORANGE, lw=2))
    ax.set_title("Bloc 16×16 agrandi\n+ voisins déjà codés", fontsize=11.5)
    no_ticks(ax)
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.text(0, 0.5, "En haut : la prédiction\nde chaque mode.\n\nEn bas : le résidu.\nbleu : prédiction trop claire\nrouge : prédiction trop sombre\ngris : prédiction parfaite\n\n"
            "L'encodeur essaie\ntous les modes et\ngarde le moins cher.", fontsize=11, va="center", color=TXT)

    for k, (name, p) in enumerate(modes.items()):
        ax = fig.add_subplot(gs[0, k + 2])
        ax.imshow(p, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name + ("\n(meilleur)" if name == best else "\n"), fontsize=11.5,
                     color=ORANGE if name == best else TXT, fontweight="bold" if name == best else "normal")
        no_ticks(ax)
        ax = fig.add_subplot(gs[1, k + 2])
        ax.imshow(block - p, cmap=RESID, vmin=-80, vmax=80)
        ax.set_title(f"erreur moyenne : {errors[name]:.0f}", fontsize=11.5,
                     color=ORANGE if name == best else TXT, fontweight="bold" if name == best else "normal")
        no_ticks(ax)
        if name == best:
            for a in fig.axes[-2:]:
                for s in a.spines.values():
                    s.set_edgecolor(ORANGE)
                    s.set_linewidth(3)
    fig.suptitle("Prédiction intra : deviner un bloc à partir de ses voisins dans la même image", fontsize=14, y=0.99)
    fig.savefig(OUT / "pred_2_intra.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3

def make_frames(img, noise=2.0, seed=0):
    """Two frames: the camera pans (content moves 5 px left, 2 px up) and a patch moves on its own."""
    rng = np.random.default_rng(seed)
    size, y0, x0 = 384, 40, 40
    prev = img[y0:y0 + size, x0:x0 + size].copy()
    cur = img[y0 + 2:y0 + 2 + size, x0 + 5:x0 + 5 + size].copy()
    obj = img[430:500, 20:110]
    prev[150:220, 100:190] = obj
    cur[156:226, 90:180] = obj  # the object moves 10 px left and 6 px down
    prev += rng.normal(0, noise, prev.shape)
    cur += rng.normal(0, noise, cur.shape)
    return np.clip(prev, 0, 255), np.clip(cur, 0, 255)


def fig_inter(prev, cur):
    b = 16
    vec, pred = block_matching(prev, cur, b=b, r=12)
    naive, comp = cur - prev, cur - pred
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.4))
    axes[0, 0].imshow(prev, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("a. Image précédente (déjà codée)")
    axes[0, 1].imshow(cur, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("b. Image actuelle (à coder)\nla caméra bouge, l'objet encadré aussi")
    axes[0, 1].add_patch(Rectangle((89.5, 155.5), 90, 70, fill=False, ec=ORANGE, lw=2))
    axes[0, 2].imshow(naive, cmap=RESID, vmin=-80, vmax=80)
    axes[0, 2].set_title(f"c. Sans compensation : actuelle − précédente\nerreur moyenne {fr(np.abs(naive).mean())}")

    axes[1, 0].imshow(cur, cmap="gray", vmin=0, vmax=255)
    cy, cx = np.mgrid[0:vec.shape[0], 0:vec.shape[1]] * b + b / 2
    axes[1, 0].quiver(cx, cy, -vec[..., 1] * 2, -vec[..., 0] * 2, color=ORANGE, angles="xy", scale_units="xy",
                      scale=1, width=0.004, headwidth=4)
    axes[1, 0].set_title("d. Vecteurs de mouvement trouvés\n(1 par bloc 16×16, flèches agrandies ×2)")
    axes[1, 1].imshow(pred, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("e. Prédiction : chaque bloc copié depuis\nl'image précédente, à l'endroit indiqué")
    axes[1, 2].imshow(comp, cmap=RESID, vmin=-80, vmax=80)
    axes[1, 2].set_title(f"f. Résidu après compensation : actuelle − prédiction\nerreur moyenne {fr(np.abs(comp).mean())}")
    for ax in axes.ravel():
        no_ticks(ax)
    fig.suptitle("Prédiction inter : deviner un bloc en le cherchant dans l'image précédente", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "pred_3_inter.png", dpi=130)
    plt.close(fig)
    return comp


# ---------------------------------------------------------------- figure 4

def fig_gain(cur, inter_res):
    intra_res = intra_residual(cur)
    sets = [("Pixels bruts (sans prédiction)", cur, (0, 255)),
            ("Résidu après prédiction intra", intra_res, (-80, 80)),
            ("Résidu après prédiction inter", inter_res, (-80, 80))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (title, values, (lo, hi)) in zip(axes, sets):
        v = np.rint(values)
        v = v[(v >= lo) & (v <= hi)]  # display range only; the entropy uses every value
        ax.hist(v.ravel(), bins=np.arange(lo, hi + 2) - 0.5, color=BLUE, lw=0)
        h = entropy(values)
        ax.set_title(f"{title}\nentropie ≈ {h:.1f} bits par pixel".replace(".", ","))
        ax.set_xlabel("valeur")
        ax.set_yticks([])
        ax.grid(axis="x", color=GRID, lw=0.8)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Pourquoi ça fait gagner des bits : les résidus se concentrent autour de 0, "
                 "le codage entropique les écrit en peu de bits", fontsize=13.5)
    fig.tight_layout()
    fig.savefig(OUT / "pred_4_gain_en_bits.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- figure 5

def rotated_erp(frame, proj, yaw=0.0, pitch=0.0):
    theta, phi = proj.pixel_to_sphere()
    src = sph_to_dir(theta, phi) @ rotation(yaw, pitch)
    return sample(frame, proj, *dir_to_sph(src))


def fig_360():
    proj = equirectangular(1024)
    f1 = to_luma(load_image(PANO, 1024)).astype(np.float64)
    b = 32
    angle = 8 * 360 / 1024  # 2.8125 degrees = exactly 8 pixels of the ERP grid
    cases = [("La caméra tourne sur le côté (lacet de 2,8°)", rotated_erp(f1, proj, yaw=np.radians(angle)), BLUE),
             ("La caméra se penche vers le haut (tangage de 2,8°)", rotated_erp(f1, proj, pitch=np.radians(angle)), ORANGE)]
    fig = plt.figure(figsize=(17, 8.8))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.45, 1.45, 1], hspace=0.28, wspace=0.12)
    ax_lat = fig.add_subplot(gs[:, 2])
    lat = 90 - (np.arange(f1.shape[0] // b) + 0.5) * 180 / (f1.shape[0] // b)
    for row, (title, f2, color) in enumerate(cases):
        vec, pred = block_matching(f1, f2, b=b, r=20, wrap_x=True, lam=8.0)
        res = f2 - pred
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(f2, cmap="gray", vmin=0, vmax=255, extent=(0, 1024, 512, 0))
        cy, cx = np.mgrid[0:vec.shape[0], 0:vec.shape[1]] * b + b / 2
        ax.quiver(cx, cy, -vec[..., 1] * 1.5, -vec[..., 0] * 1.5, color=color, angles="xy", scale_units="xy",
                  scale=1, width=0.0025, headwidth=4)
        ax.set_title(f"{title}\nvecteurs de mouvement (blocs 32×32)", fontsize=12)
        no_ticks(ax)
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(res, cmap=RESID, vmin=-60, vmax=60)
        ax.set_title(f"résidu après compensation\nerreur moyenne {fr(np.abs(res).mean())}", fontsize=12)
        no_ticks(ax)
        per_lat = np.abs(res).reshape(len(lat), b, -1).mean(axis=(1, 2))
        ax_lat.plot(per_lat, lat, color=color, lw=2)
        k = len(lat) - 3 if row else 2
        ax_lat.text(per_lat[k] + (0.15 if row else 0.9), lat[k], "tangage" if row else "lacet (erreur nulle)", color=TXT,
                    fontsize=11, va="center", ha="left")
    ax_lat.set(xlabel="erreur moyenne du résidu", ylabel="latitude (°)", ylim=(-90, 90))
    ax_lat.set_title("Erreur restante selon la latitude", fontsize=12)
    ax_lat.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax_lat.spines[s].set_visible(False)
    fig.suptitle("En 360° (image équirectangulaire wide_street_01) : un simple mouvement de caméra "
                 "n'est pas toujours une translation dans l'image", fontsize=14, y=0.99)
    fig.savefig(OUT / "pred_5_cas_360.png", dpi=125, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    boats = load_gray(BOATS)
    fig_principle()
    fig_intra(boats)
    prev, cur = make_frames(boats)
    inter_res = fig_inter(prev, cur)
    fig_gain(cur, inter_res)
    fig_360()
    print("figures written to", OUT)
