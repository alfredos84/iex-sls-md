"""
fig_surface_profile.py
Perfil promedio de altura de superficie post-indentacion (tras la descarga),
siguiendo el metodo de la Fig. 8 de Pedone et al. (J. Non-Cryst. Solids 683, 2026, 124105):

  - Frame final (step=320000, despues de la descarga) de cada indent.lammpstrj.
  - Rebanada en x de +-6 A centrada en x_mid = xlo + Lx/2 (eje de la punta).
  - Bins de 1 A en y; en cada bin, posicion z mas externa (maxima) de los atomos
    de vidrio (se excluye el tipo 6 = punta).
  - Perfil promediado sobre las replicas de cada caso.
  - Altura relativa a la superficie sin indentar (mediana de z en bins lejos del centro).

Casos: CaO / Ca-free x IEX / sin IEX.
CaO sin IEX usa solo r2,r3 (r1 es la muestra piloto, caja 115 A, no comparable con las
demas simulaciones, caja 173-180 A). Los otros 3 casos usan r1,r2,r3.

Atom types: Si=1, O=2, Ca=3, Na=4, K=5, C(punta)=6
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE = Path(__file__).parent
FRAMES = HERE / "lastframes"

TYPE_TIP = 6
X_HALF_WIDTH = 6.0
BIN = 1.0

CASES = {
    "CaO":     {"AM": ("cao_x0_r{r}", (2, 3)),      "IOX": ("cao_iex_xt15_r{r}_300K", (1, 2, 3))},
    "Ca-free": {"AM": ("noca_x0_r{r}", (1, 2, 3)),  "IOX": ("noca_iex_xt25_r{r}_300K", (1, 2, 3))},
}
COLORS = {"AM": "#1a3a5c", "IOX": "#e08c2a"}


def read_last_frame(path):
    with open(path) as f:
        header = f.readline()
    lo_str = header.split("box_lo=")[1].split("box_len=")[0]
    len_str = header.split("box_len=")[1]
    lo = np.array([float(v) for v in lo_str.split()])
    length = np.array([float(v) for v in len_str.split()])
    data = np.loadtxt(path, skiprows=1)
    types = data[:, 1].astype(int)
    xyz = data[:, 2:5]
    return types, xyz, lo, length


def replica_profile(tag):
    types, xyz, lo, length = read_last_frame(FRAMES / f"{tag}_last.txt")
    xmid = lo[0] + 0.5 * length[0]
    glass = types != TYPE_TIP
    sl = glass & (np.abs(xyz[:, 0] - xmid) <= X_HALF_WIDTH)
    y = xyz[sl, 1]
    z = xyz[sl, 2]

    y0 = lo[1]
    n_bins = int(round(length[1] / BIN))
    edges = y0 + np.arange(n_bins + 1) * BIN
    centers = 0.5 * (edges[:-1] + edges[1:])
    zmax = np.full(n_bins, np.nan)
    idx = np.clip(((y - y0) / BIN).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            zmax[b] = z[sel].max()

    # y relativa al centro de la punta (y_mid = 0)
    y_rel = centers - (y0 + 0.5 * length[1])

    # linea de base: mediana de z en el 25% de bins mas alejados del centro a cada lado
    far = np.abs(y_rel) > 0.35 * length[1]
    baseline = np.nanmedian(zmax[far])
    return y_rel, zmax - baseline


Y_GRID = np.arange(-95.0, 95.0 + BIN, BIN)


def averaged_profile(tag_pat, reps):
    profiles = [replica_profile(tag_pat.format(r=r)) for r in reps]
    interp = []
    for y, z in profiles:
        ok = ~np.isnan(z)
        interp.append(np.interp(Y_GRID, y[ok], z[ok], left=np.nan, right=np.nan))
    stacked = np.array(interp)
    return Y_GRID, np.nanmean(stacked, axis=0)


plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       17,
    "axes.linewidth":  0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
letters = ("(a)", "(b)")
for ax, letter, (glass, conds) in zip(axes, letters, CASES.items()):
    for cond, (pat, reps) in conds.items():
        y, z = averaged_profile(pat, reps)
        ax.plot(y, z, color=COLORS[cond], lw=1.8, label=cond)
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(-90, 90)
    ax.set_xlabel("y (Å)", fontsize=16)
    ax.text(-60, -10, glass, ha="center", va="center", fontsize=17, zorder=5)
    ax.text(0.02, 0.97, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=17, fontweight="bold", zorder=5)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=13, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor="#cccccc")

axes[0].set_ylabel("Surface height (Å)", fontsize=16)

fig.tight_layout()
out = HERE / "fig_surface_profile.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")
