"""
fig_densification_bin_density.py
Mapa 2D de densificacion local replicando el metodo de la Fig. 7 de Pedone 2026
("Quantifying densification versus shear flow..."): chunk/atom bin/3d con bins de
2x2x2 A^3, rebanada centrada en x de 12 A de espesor, densidad local de cada bin
comparada contra la densidad bulk pre-indentacion (rho0) de esa composicion:

  densificacion(%) = (rho_local(bin) - rho0) / rho0 * 100

A diferencia de fig_densification_maps.py (gradiente de deformacion Falk-Langer,
que trackea los mismos atomos entre dos configuraciones), este metodo NO trackea
atomos: calcula la densidad de masa local en cada bin de la configuracion final
(post-descarga) de forma independiente, y la compara contra rho0 = masa total /
volumen de una subregion bulk (lejos de la superficie libre) en la configuracion
de referencia (step20000, antes de cargar).

rho0 se calcula con la subregion bulk z in [10,80] A (todo x,y), evitando la
superficie libre superior y el borde inferior fijo.

Suavizado gaussiano sigma=1.5 bins (igual que el paper) y rango de color fijo
[-30%, +30%] centrado en 0. Los bins vacios (vacio/hueco bajo la punta o por
encima de la superficie) dan densidad ~0 y por lo tanto densificacion muy
negativa: aparecen saturados en azul oscuro (region de "vacio", no densificacion
negativa real), igual que en la Fig. 7 del paper.

Promediado sobre replicas (r1,r2,r3; CaO sin IEX usa solo r2,r3 porque r1 es la
muestra piloto de caja mas chica, no comparable con las demas).

Atom types: Si=1, O=2, Ca=3, Na=4, K=5, C(punta)=6
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).parent
FRAMES = HERE / "surface_profile" / "lastframes"
sys.path.insert(0, str(HERE.parent))
from paper_style import FS_LABEL, FS_TEXT, apply_rcparams, make_fig, panel_letter, relabel
apply_rcparams()

TYPE_TIP = 6
TYPE_MASS = {1: 28.0855, 2: 15.9994, 3: 40.078, 4: 22.98977, 5: 39.0983}

BIN = 2.0
X_HALF_WIDTH = 6.0       # rebanada de 12 A de espesor en x, como en el paper
BULK_Z_LO, BULK_Z_HI = 10.0, 80.0
SIGMA_BINS = 1.5

CASES = {
    "CaO":     {"AM": ["cao_x0_r2", "cao_x0_r3"],
                "IOX": ["cao_iex_xt15_r1_300K", "cao_iex_xt15_r2_300K", "cao_iex_xt15_r3_300K"]},
    "Ca-free": {"AM": ["noca_x0_r1", "noca_x0_r2", "noca_x0_r3"],
                "IOX": ["noca_iex_xt25_r1_300K", "noca_iex_xt25_r2_300K", "noca_iex_xt25_r3_300K"]},
}


def read_frame(tag, step):
    path = FRAMES / f"{tag}_{step}.txt"
    with open(path) as f:
        header = f.readline()
    lo = np.array([float(v) for v in header.split("box_lo=")[1].split("box_len=")[0].split()])
    length = np.array([float(v) for v in header.split("box_len=")[1].split()])
    data = np.loadtxt(path, skiprows=1)
    types = data[:, 1].astype(int)
    xyz = data[:, 2:5]
    return types, xyz, lo, length


def masses_of(types):
    m = np.zeros(len(types))
    for t, mv in TYPE_MASS.items():
        m[types == t] = mv
    return m


def bulk_density(tag):
    types0, xyz0, lo0, len0 = read_frame(tag, "step20000")
    glass = types0 != TYPE_TIP
    types0, xyz0 = types0[glass], xyz0[glass]
    z = xyz0[:, 2]
    sel = (z >= BULK_Z_LO) & (z <= BULK_Z_HI)
    mass = masses_of(types0[sel]).sum()
    vol = len0[0] * len0[1] * (BULK_Z_HI - BULK_Z_LO)
    return mass / vol


def local_density_map(tag, y_edges, z_edges):
    types1, xyz1, lo1, len1 = read_frame(tag, "last")
    glass = types1 != TYPE_TIP
    types1, xyz1 = types1[glass], xyz1[glass]
    xmid = lo1[0] + 0.5 * len1[0]
    ymid = lo1[1] + 0.5 * len1[1]

    x = xyz1[:, 0] - xmid
    y = xyz1[:, 1] - ymid
    z = xyz1[:, 2]
    mass = masses_of(types1)

    x_edges = np.arange(-X_HALF_WIDTH, X_HALF_WIDTH + BIN, BIN)
    nx = len(x_edges) - 1

    slice_mask = (x >= x_edges[0]) & (x < x_edges[-1])
    x, y, z, mass = x[slice_mask], y[slice_mask], z[slice_mask], mass[slice_mask]

    ix = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)
    iz = np.clip(np.digitize(z, z_edges) - 1, 0, len(z_edges) - 2)

    grid_mass = np.zeros((nx, len(z_edges) - 1, len(y_edges) - 1))
    np.add.at(grid_mass, (ix, iz, iy), mass)

    bin_vol = BIN ** 3
    rho = grid_mass / bin_vol
    return rho.mean(axis=0)      # promedio sobre los bins de x de la rebanada


Y_EDGES = np.arange(-95, 95 + BIN, BIN)
Z_EDGES = np.arange(-10, 130 + BIN, BIN)

results = {}
for glass, conds in CASES.items():
    for cond, tags in conds.items():
        print(f"{glass} {cond} (n={len(tags)} replicas: {tags})...")
        pct_maps = []
        for tag in tags:
            rho0 = bulk_density(tag)
            rho_local = local_density_map(tag, Y_EDGES, Z_EDGES)
            pct = (rho_local - rho0) / rho0 * 100.0
            pct_maps.append(pct)
            print(f"  {tag}: rho0={rho0:.4f}  rango densif.: [{pct.min():.1f}%, {pct.max():.1f}%]")
        avg = np.mean(np.stack(pct_maps), axis=0)
        smoothed = gaussian_filter(avg, sigma=SIGMA_BINS)
        results[(glass, cond)] = smoothed

vmax = 30.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig, axes = make_fig(2, 2)
letters = {("CaO", "AM"): "(a)", ("CaO", "IOX"): "(b)", ("Ca-free", "AM"): "(c)", ("Ca-free", "IOX"): "(d)"}
im = None
for row, glass in enumerate(("CaO", "Ca-free")):
    for col, cond in enumerate(("AM", "IOX")):
        ax = axes[row, col]
        im = ax.pcolormesh(Y_EDGES, Z_EDGES, results[(glass, cond)], cmap="RdBu_r", norm=norm, shading="flat")
        ax.set_aspect("equal")
        ax.set_title(f"{relabel(glass)} — {cond}", fontsize=FS_LABEL)
        panel_letter(ax, letters[(glass, cond)], x=0.03, y=0.97, ha="left")
        ax.tick_params(labelsize=FS_TEXT, which="both")
        if row == 1:
            ax.set_xlabel("y (Å)", fontsize=FS_LABEL)
        if col == 0:
            ax.set_ylabel("z (Å)", fontsize=FS_LABEL)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
cbar.set_label("Local densification (%)", fontsize=FS_LABEL)
cbar.ax.tick_params(labelsize=FS_TEXT)

out = HERE / "fig_densification_bin_density.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")
