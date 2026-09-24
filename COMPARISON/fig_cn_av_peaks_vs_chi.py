"""
fig_cn_av_peaks_vs_chi.py
Pico (moda) del CN_O vs chi — Na(AM), K(AM), K(IOX) — promediado sobre las 3 replicas.

Para cada chi, serie y replica (r=1,2,3) se calcula la moda de la distribucion de CN_O de los
cationes (O dentro del radio de corte). Punto = media de las 3 modas; barra de error =
desviacion estandar muestral (n-1) entre replicas. Ajuste lineal sobre las medias.

Radios de corte: Na-O 3.21 A, K-O 3.77 A (AM), K-O 3.49 A (IOX; promedio Na/K).
El Na solo se muestra en chi = 20-80 (en chi = 100 no hay Na; XX_CONFIG usa x=0 de relleno).
2 paneles: Ca | Ca-free.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
from paper_style import FS_LABEL, FS_TEXT, apply_rcparams, make_fig, panel_letter, relabel
apply_rcparams()

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

RCUT_NA, RCUT_K = 3.21, 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0
TYPE_O, TYPE_NA, TYPE_K = 2, 4, 5
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


def replica_peak(path, center_type, rcut):
    """Moda del CN_O de los cationes center_type en un archivo de datos."""
    pos, types, box = read_data(path)
    pos = np.mod(pos, box)
    o_tree = cKDTree(pos[types == TYPE_O], boxsize=box)
    cn = o_tree.query_ball_point(pos[types == center_type], rcut, return_length=True)
    return float(np.argmax(np.bincount(np.asarray(cn, dtype=int))))


def series_stats(dirpath, pat, x, center_type, rcut):
    pk = np.array([replica_peak(dirpath / pat.format(x=x, r=r), center_type, rcut) for r in REPLICAS])
    return pk.mean(), pk.std(ddof=1)


AM_PAT = "AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"
IEX_PAT = "IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"

mean = {s: {k: np.full(len(CHI), np.nan) for k in ("na", "ka", "ki")} for s in ("cao", "noca")}
sd = {s: {k: np.full(len(CHI), np.nan) for k in ("na", "ka", "ki")} for s in ("cao", "noca")}

for ix, (XX, (xna_c, xka_c, xt_c, xna_n, xka_n, xt_n)) in enumerate(sorted(XX_CONFIG.items())):
    print(f"XX={XX}%...")
    jobs = {
        ("cao", "na"):   (CAO_AM_DIR,   AM_PAT,  xna_c, TYPE_NA, RCUT_NA),
        ("cao", "ka"):   (CAO_AM_DIR,   AM_PAT,  xka_c, TYPE_K,  RCUT_K),
        ("cao", "ki"):   (CAO_IEX1_DIR, IEX_PAT, xt_c,  TYPE_K,  RCUT_IEX),
        ("noca", "na"):  (NOCA_AM_DIR,   AM_PAT,  xna_n, TYPE_NA, RCUT_NA),
        ("noca", "ka"):  (NOCA_AM_DIR,   AM_PAT,  xka_n, TYPE_K,  RCUT_K),
        ("noca", "ki"):  (NOCA_IEX1_DIR, IEX_PAT, xt_n,  TYPE_K,  RCUT_IEX),
    }
    for (sys_, key), (d, pat, x, ctype, rcut) in jobs.items():
        if key == "na" and XX == 100:      # en chi=100 no hay Na
            continue
        mean[sys_][key][ix], sd[sys_][key][ix] = series_stats(d, pat, x, ctype, rcut)

SERIES = [
    ("na", "#1a3a5c", "o", relabel("Na (As-Melted)")),
    ("ka", "#2ca0c4", "s", relabel("K (As-Melted)")),
    ("ki", "#c0392b", "^", relabel("K (Ion-Exchanged)")),
]

fig, axes = make_fig(1, 2)
for col, (letter, (sys_, title)) in enumerate(zip(("(a)", "(b)"), (("cao", "10CaO"), ("noca", "Ca-free")))):
    ax = axes[col]
    for key, color, marker, label in SERIES:
        y, e = mean[sys_][key], sd[sys_][key]
        ok = ~np.isnan(y)
        ax.errorbar(CHI[ok], y[ok], yerr=e[ok], color=color, marker=marker, ls="none", ms=6, mew=0.8,
                    capsize=3, elinewidth=1.0, label=label, zorder=3)
        m, b = np.polyfit(CHI[ok], y[ok], 1)
        xf = np.linspace(CHI[ok].min(), CHI[ok].max(), 200)
        ax.plot(xf, m * xf + b, color=color, lw=1.4, label="_nolegend_", zorder=2)
    ax.set_title(relabel(title), fontsize=FS_LABEL)
    panel_letter(ax, letter, x=0.97, y=0.97, ha="right")
    ax.set_xlabel(r"$\chi$ (%)", fontsize=FS_LABEL)
    if col == 0:
        ax.set_ylabel(r"CN$_\mathrm{O}$ peak", fontsize=FS_LABEL)
    ax.set_ylim(4, 11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=FS_TEXT, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=FS_TEXT,
           frameon=True, framealpha=0.9, edgecolor="#cccccc")
fig.subplots_adjust(top=0.85, wspace=0.18)

out = HERE / "fig_cn_av_peaks_vs_chi.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")

for sys_ in ("cao", "noca"):
    for key in ("na", "ka", "ki"):
        print(sys_, key, " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(mean[sys_][key], sd[sys_][key])))
