"""
fig_voronoi_av_peaks_vs_chi.py
Pico del volumen de Voronoi vs chi — Na(AM), K(AM), K(IEX1) — promediado sobre las 3 replicas.

Para cada chi y cada replica (r=1,2,3) se calcula el pico de la KDE de los volumenes de Voronoi
(Na o K); el punto es la media de los 3 picos y la barra de error es la desviacion estandar
muestral (n-1). Ajuste lineal sobre las medias. 2 paneles: 10CaO | Ca-free.
Misma estetica que fig_voronoi_peaks_vs_chi.png (en esa figura las 3 replicas se juntan en un
unico conjunto y se toma un solo pico, sin barra de error).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import gaussian_kde

HERE = Path(__file__).parent
VORO_DIR = HERE / "VORONOI"
TYPE_NA, TYPE_K = 4, 5
REPLICAS = (1, 2, 3)

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}
CHI = np.array(sorted(XX_CONFIG), dtype=float)


def read_vol(path):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            if s.startswith("ITEM:"):
                in_atoms = False
                continue
            if in_atoms and s:
                parts = s.split()
                data[int(parts[0])] = (int(parts[1]), float(parts[2]))
    return data


def vol_peak_replica(tag, center_type):
    vol_data = read_vol(VORO_DIR / f"{tag}_vol.dat")
    vols = np.array([v for (t, v) in vol_data.values() if t == center_type])
    kde = gaussian_kde(vols, bw_method="silverman")
    x_grid = np.linspace(vols.min(), vols.max(), 2000)
    return x_grid[np.argmax(kde(x_grid))]


def peaks_mean_sd(tag_pattern, center_type):
    pk = np.array([vol_peak_replica(tag_pattern.format(r=r), center_type) for r in REPLICAS])
    return pk.mean(), pk.std(ddof=1)


mean = {s: {k: [] for k in ("na", "ka", "ki")} for s in ("cao", "noca")}
sd = {s: {k: [] for k in ("na", "ka", "ki")} for s in ("cao", "noca")}

for XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca) in sorted(XX_CONFIG.items()):
    print(f"XX={XX}%...")
    specs = {
        ("cao", "na"):   (f"cao_am_x{xna_cao}_r{{r}}",       TYPE_NA),
        ("cao", "ka"):   (f"cao_am_x{xka_cao}_r{{r}}",       TYPE_K),
        ("cao", "ki"):   (f"cao_iex1_xt{xt_cao}_r{{r}}",     TYPE_K),
        ("noca", "na"):  (f"noca_am_x{xna_noca}_r{{r}}",     TYPE_NA),
        ("noca", "ka"):  (f"noca_am_x{xka_noca}_r{{r}}",     TYPE_K),
        ("noca", "ki"):  (f"noca_iex1_xt{xt_noca}_r{{r}}",   TYPE_K),
    }
    for (sys_, key), (pat, ctype) in specs.items():
        m, s = peaks_mean_sd(pat, ctype)
        mean[sys_][key].append(m)
        sd[sys_][key].append(s)

plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       14,
    "axes.linewidth":  0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

SERIES = [
    ("na", "#1a3a5c", "o", "Na (As-Melted)"),
    ("ka", "#2ca0c4", "s", "K (As-Melted)"),
    ("ki", "#c0392b", "^", "K (Ion-Exchanged)"),
]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for col, (sys_, title) in enumerate([("cao", "10CaO"), ("noca", "Ca-free")]):
    ax = axes[col]
    for key, color, marker, label in SERIES:
        y = np.array(mean[sys_][key])
        e = np.array(sd[sys_][key])
        ax.errorbar(CHI, y, yerr=e, color=color, marker=marker, ls="none", ms=6, mew=0.8,
                    capsize=3, elinewidth=1.0, label=label, zorder=3)
        m, b = np.polyfit(CHI, y, 1)
        xf = np.linspace(CHI.min(), CHI.max(), 200)
        ax.plot(xf, m * xf + b, color=color, lw=1.4, label="_nolegend_", zorder=2)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r"$\chi$ (%)", fontsize=13)
    ax.set_ylabel(r"$V_\mathrm{Vor}$ peak (Å$^3$)", fontsize=13)
    ax.set_ylim(16, 24)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=11, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, fontsize=11,
           frameon=True, framealpha=0.9, edgecolor="#cccccc")
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = HERE / "fig_voronoi_av_peaks_vs_chi.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")

for sys_ in ("cao", "noca"):
    for key in ("na", "ka", "ki"):
        print(sys_, key, " ".join(f"{m:.2f}±{s:.2f}" for m, s in zip(mean[sys_][key], sd[sys_][key])))
