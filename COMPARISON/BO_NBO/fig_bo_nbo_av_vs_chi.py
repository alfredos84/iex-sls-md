"""
fig_bo_nbo_av_vs_chi.py
<CN_BO> y <CN_NBO> vs chi promediados sobre las 3 replicas, con barra de error entre replicas.

Para cada chi, serie y replica (r=1,2,3) se calcula la media de CN_BO (o CN_NBO) sobre todos los
cationes de esa replica. Punto = media de las 3 medias; barra de error = desviacion estandar
muestral (n-1) entre replicas. Linea = ajuste lineal sobre las medias.

Una sola figura 2x2: filas = CN_BO | CN_NBO, columnas = 10CaO | Ca-free. Paneles 2:1.

Clasificacion de oxigenos (r_cut Si-O = 2.0 A): BO = 2 vecinos Si, NBO = 1 vecino Si.
Radios de corte: Na-O 3.21, K-O 3.77 (AM) / 3.49 (IEX1), Ca-O 3.20 A.
Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
BASE = HERE.parent.parent

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

RCUT_SIO = 2.00
RCUT_NA  = 3.21
RCUT_K   = 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0
RCUT_CA  = 3.20
TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5
REPLICAS = (1, 2, 3)

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}
CHI = np.array(sorted(XX_CONFIG), dtype=float)


def read_data(path):
    box, lo = np.zeros(3), np.zeros(3)
    atoms, section = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if "xlo xhi" in s:
                lo[0], hi = float(s.split()[0]), float(s.split()[1])
                box[0] = hi - lo[0]
                continue
            elif "ylo yhi" in s:
                lo[1], hi = float(s.split()[0]), float(s.split()[1])
                box[1] = hi - lo[1]
                continue
            elif "zlo zhi" in s:
                lo[2], hi = float(s.split()[0]), float(s.split()[1])
                box[2] = hi - lo[2]
                continue
            if s[0].isalpha():
                section = "atoms" if s.startswith("Atoms") else None
                continue
            if section == "atoms":
                p = s.split()
                atoms.append((int(p[1]), float(p[3]), float(p[4]), float(p[5])))
    atoms = np.array(atoms)
    return atoms[:, 1:4] - lo, atoms[:, 0].astype(int), box


def replica_means(path, center_type, rcut):
    """(<CN_BO>, <CN_NBO>) de los cationes center_type en un archivo de datos."""
    pos, types, box = read_data(path)
    pos = np.mod(pos, box)
    o_pos = pos[types == TYPE_O]
    si_tree = cKDTree(pos[types == TYPE_SI], boxsize=box)
    n_si = si_tree.query_ball_point(o_pos, RCUT_SIO, return_length=True)
    bo_tree = cKDTree(o_pos[n_si == 2], boxsize=box)
    nbo_tree = cKDTree(o_pos[n_si == 1], boxsize=box)
    cat = pos[types == center_type]
    cn_bo = bo_tree.query_ball_point(cat, rcut, return_length=True)
    cn_nbo = nbo_tree.query_ball_point(cat, rcut, return_length=True)
    return float(np.mean(cn_bo)), float(np.mean(cn_nbo))


def series_stats(dirpath, name_pat, x, center_type, rcut):
    vals = np.array([replica_means(dirpath / name_pat.format(x=x, r=r), center_type, rcut) for r in REPLICAS])
    return vals.mean(axis=0), vals.std(axis=0, ddof=1)   # arrays (BO, NBO)


AM_PAT = "AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"
IEX_PAT = "IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"

# stats[sys][key] = (mean_bo[chi], sd_bo[chi], mean_nbo[chi], sd_nbo[chi])
KEYS = {"cao": ("na", "ka", "ki", "ca"), "noca": ("na", "ka", "ki")}
stats = {s: {k: np.full((4, len(CHI)), np.nan) for k in KEYS[s]} for s in KEYS}

for ix, (XX, (xna_c, xka_c, xt_c, xna_n, xka_n, xt_n)) in enumerate(sorted(XX_CONFIG.items())):
    print(f"XX={XX}%...")
    jobs = {
        ("cao", "na"):   (CAO_AM_DIR,   AM_PAT,  xna_c, TYPE_NA, RCUT_NA),
        ("cao", "ka"):   (CAO_AM_DIR,   AM_PAT,  xka_c, TYPE_K,  RCUT_K),
        ("cao", "ki"):   (CAO_IEX1_DIR, IEX_PAT, xt_c,  TYPE_K,  RCUT_IEX),
        ("cao", "ca"):   (CAO_AM_DIR,   AM_PAT,  xka_c, TYPE_CA, RCUT_CA),
        ("noca", "na"):  (NOCA_AM_DIR,   AM_PAT,  xna_n, TYPE_NA, RCUT_NA),
        ("noca", "ka"):  (NOCA_AM_DIR,   AM_PAT,  xka_n, TYPE_K,  RCUT_K),
        ("noca", "ki"):  (NOCA_IEX1_DIR, IEX_PAT, xt_n,  TYPE_K,  RCUT_IEX),
    }
    for (sys_, key), (d, pat, x, ctype, rcut) in jobs.items():
        m, s = series_stats(d, pat, x, ctype, rcut)
        stats[sys_][key][:, ix] = (m[0], s[0], m[1], s[1])

plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       17,
    "axes.linewidth":  0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

SERIES = {
    "na": ("#1a3a5c", "o", "Na (As-Melted)"),
    "ka": ("#2ca0c4", "s", "K (As-Melted)"),
    "ki": ("#c0392b", "^", "K (Ion-Exchanged)"),
    "ca": ("#7b3f00", "D", "Ca (As-Melted)"),
}
TICK_FS, LABEL_AX_FS, TITLE_FS, LETTER_FS = 16, 19, 20, 20

# Geometria (pulgadas): paneles 2:1
PW = 7.6
PH = PW / 2.0
LM, GAP_X, RM = 1.45, 0.75, 0.3
TM, GAP_Y, BM = 1.05, 0.62, 0.95
W = LM + 2 * PW + GAP_X + RM
H = TM + 2 * PH + GAP_Y + BM

YLIMS = {(0, "cao"): None, (0, "noca"): None, (1, "cao"): None, (1, "noca"): None}   # auto

fig = plt.figure(figsize=(W, H))
letters = {(0, 0): "(a)", (0, 1): "(b)", (1, 0): "(c)", (1, 1): "(d)"}
ylabels = {0: r"$\langle$CN$_\mathrm{BO}\rangle$", 1: r"$\langle$CN$_\mathrm{NBO}\rangle$"}
handles = None

for row in (0, 1):
    for col, (sys_, title) in enumerate((("cao", "10CaO"), ("noca", "Ca-free"))):
        x0 = LM + col * (PW + GAP_X)
        y0 = BM + (1 - row) * (PH + GAP_Y)
        ax = fig.add_axes([x0 / W, y0 / H, PW / W, PH / H])
        ymin, ymax = np.inf, -np.inf
        for key in KEYS[sys_]:
            color, marker, label = SERIES[key]
            m = stats[sys_][key][2 * row]
            e = stats[sys_][key][2 * row + 1]
            ok = ~np.isnan(m)
            ax.errorbar(CHI[ok], m[ok], yerr=e[ok], color=color, marker=marker, ls="none", ms=7, mew=0.8,
                        capsize=4, elinewidth=1.3, label=label, zorder=3)
            slope, icpt = np.polyfit(CHI[ok], m[ok], 1)
            xf = np.linspace(CHI[ok].min(), CHI[ok].max(), 200)
            ax.plot(xf, slope * xf + icpt, color=color, lw=1.6, label="_nolegend_", zorder=2)
            ymin = min(ymin, np.nanmin(m[ok] - e[ok]))
            ymax = max(ymax, np.nanmax(m[ok] + e[ok]))
        pad = 0.07 * (ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(8, 112)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(labelsize=TICK_FS, which="both")
        ax.grid(lw=0.35, color="#dddddd", zorder=0)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
        if row == 1:
            ax.set_xlabel(r"$\chi$ (%)", fontsize=LABEL_AX_FS)
        if col == 0:
            fig.text((x0 - 1.05) / W, (y0 + PH / 2) / H, ylabels[row], rotation=90, ha="center", va="center",
                     fontsize=LABEL_AX_FS)
        # titulo centrado en x, en el hueco vertical libre mas grande de la zona central (chi = 40..80)
        y_lo, y_hi = ax.get_ylim()
        rng = y_hi - y_lo
        busy = []
        for key in KEYS[sys_]:
            m = stats[sys_][key][2 * row]
            e = stats[sys_][key][2 * row + 1]
            for i in (1, 2, 3):
                if not np.isnan(m[i]):
                    busy.append((m[i] - e[i] - 0.03 * rng, m[i] + e[i] + 0.03 * rng))
        busy.sort()
        merged = []
        for lo_, hi_ in busy:
            if merged and lo_ <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi_)
            else:
                merged.append([lo_, hi_])
        edges = [y_lo] + [v for seg in merged for v in seg] + [y_hi]
        gaps = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
        g_lo, g_hi = max(gaps, key=lambda g: g[1] - g[0])
        ax.text(0.5, 0.5 * (g_lo + g_hi), title, transform=ax.get_yaxis_transform(), ha="center",
                va="center", fontsize=TITLE_FS, zorder=5)
        ax.text(0.012, 0.975, letters[(row, col)], transform=ax.transAxes, ha="left", va="top",
                fontsize=LETTER_FS, fontweight="bold", zorder=5)
        if row == 0 and col == 0:
            handles = ax.get_legend_handles_labels()

fig.legend(*handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, fontsize=17,
           frameon=True, framealpha=0.9, edgecolor="#cccccc")

out = HERE / "fig_bo_nbo_av_vs_chi.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}  (paneles {PW:.2f} x {PH:.2f} in = 2:1)")

for sys_ in KEYS:
    for key in KEYS[sys_]:
        a = stats[sys_][key]
        print(f"{sys_:4s} {key} BO :", " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(a[0], a[1])))
        print(f"{sys_:4s} {key} NBO:", " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(a[2], a[3])))
