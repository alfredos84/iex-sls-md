"""
fig_densification_maps.py
Mapa 2D de densificacion residual Δρ/ρ tras la descarga (Pedone 2026, "Quantifying
densification versus shear flow...", Fig. 3, ec. 5-9 y 14), para los 4 vidrios de H.

Para cada atomo i: gradiente de deformacion local F_i (Falk-Langer) ajustado por
minimos cuadrados a partir de los vecinos j dentro de rc=4.0 A (misma referencia que
en el paper, valido para vidrios de oxido) en la configuracion de referencia (antes de
cargar, step=20000) y la configuracion final (despues de descargar, step=320000):

  X_i = sum_j  r_ij  (x)  r_ij0        Y_i = sum_j  r_ij0  (x)  r_ij0
  F_i = X_i Y_i^-1                      Delta_rho_i / rho_i = 1/det(F_i) - 1

Se calcula solo para los atomos dentro de una rebanada en x de +-6 A centrada en el eje
de la punta (mismo criterio que fig_surface_profile.py) y se promedia en bins 2D (y,z)
de 2x2 A, usando la posicion de cada atomo en la configuracion final.

Promediado sobre replicas (r1,r2,r3; CaO sin IEX usa solo r2,r3 porque r1 es la muestra
piloto de caja mas chica, no comparable con las demas).

Atom types: Si=1, O=2, Ca=3, Na=4, K=5, C(punta)=6
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
FRAMES = HERE / "surface_profile" / "lastframes"
sys.path.insert(0, str(HERE.parent))
from paper_style import FS_LABEL, FS_TEXT, apply_rcparams, make_fig, panel_letter, relabel
apply_rcparams()

TYPE_TIP = 6
RC = 4.0
X_HALF_WIDTH = 8.0
BIN = 3.0
MIN_NEIGH = 6
MAX_COND = 100.0   # descarta Y_i mal condicionado (vecinos casi coplanares)

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
    ids = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    xyz = data[:, 2:5]
    order = np.argsort(ids)
    return ids[order], types[order], xyz[order], lo, length


def min_image(d, box):
    return d - box * np.round(d / box)


def densification_map(tag):
    ids0, types0, xyz0, lo0, len0 = read_frame(tag, "step20000")
    ids1, types1, xyz1, lo1, len1 = read_frame(tag, "last")

    common = np.intersect1d(ids0, ids1)
    idx0 = np.searchsorted(ids0, common)
    idx1 = np.searchsorted(ids1, common)
    types = types0[idx0]
    r0 = xyz0[idx0]
    r1 = xyz1[idx1]
    glass = types != TYPE_TIP
    common, types, r0, r1 = common[glass], types[glass], r0[glass], r1[glass]

    box_ref = np.array([len0[0], len0[1], 1e6])     # z no periodica: caja "enorme" evita wraparound
    box_def = np.array([len1[0], len1[1], 1e6])
    xmid = lo1[0] + 0.5 * len1[0]
    ymid = lo1[1] + 0.5 * len1[1]

    tree_ref = cKDTree(np.mod(r0 - lo0 + [0, 0, 3e5], box_ref), boxsize=box_ref)
    r0w = np.mod(r0 - lo0 + [0, 0, 3e5], box_ref)

    slice_mask = np.abs(r1[:, 0] - xmid) <= X_HALF_WIDTH
    cand = np.where(slice_mask)[0]

    drho = np.full(len(common), np.nan)
    neigh_lists = tree_ref.query_ball_point(r0w[cand], RC)
    for k, i in enumerate(cand):
        nb = [j for j in neigh_lists[k] if j != i]
        if len(nb) < MIN_NEIGH:
            continue
        rij0 = min_image(r0[nb] - r0[i], box_ref)
        rij1 = min_image(r1[nb] - r1[i], box_def)
        X = rij1.T @ rij0
        Y = rij0.T @ rij0
        ev = np.linalg.eigvalsh(Y)
        if ev[0] <= 0 or ev[-1] / ev[0] > MAX_COND:
            continue
        F = X @ np.linalg.inv(Y)
        detF = np.linalg.det(F)
        if detF <= 0:
            continue
        val = 1.0 / detF - 1.0
        if abs(val) > 0.6:      # fuera de rango fisico plausible para vidrios de oxido -> ajuste espurio
            continue
        drho[i] = val

    y = r1[cand, 1] - ymid
    z = r1[cand, 2] - lo1[2]
    d = drho[cand]
    ok = ~np.isnan(d)
    return y[ok], z[ok], d[ok]


def bin_map(y, z, d, y_edges, z_edges):
    grid = np.full((len(z_edges) - 1, len(y_edges) - 1), np.nan)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, len(y_edges) - 2)
    iz = np.clip(np.digitize(z, z_edges) - 1, 0, len(z_edges) - 2)
    for by in range(len(y_edges) - 1):
        for bz in range(len(z_edges) - 1):
            sel = (iy == by) & (iz == bz)
            if sel.any():
                grid[bz, by] = d[sel].mean()
    return grid


Y_EDGES = np.arange(-95, 95 + BIN, BIN)
Z_EDGES = np.arange(0, 170 + BIN, BIN)

results = {}
for glass, conds in CASES.items():
    for cond, tags in conds.items():
        print(f"{glass} {cond} (n={len(tags)} replicas: {tags})...")
        grids = []
        for tag in tags:
            y, z, d = densification_map(tag)
            g = bin_map(y, z, d, Y_EDGES, Z_EDGES)
            grids.append(g)
            print(f"  {tag}: atomos activos={len(d)}  rango Δρ/ρ: [{d.min():.3f}, {d.max():.3f}]")
        results[(glass, cond)] = np.nanmean(np.stack(grids), axis=0)

vmax = 0.15   # ~percentil 99 de |Δρ/ρ| en los mapas; realza el contraste en la zona activa
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
cbar.set_label(r"$\Delta\rho/\rho$", fontsize=FS_LABEL)
cbar.ax.tick_params(labelsize=FS_TEXT)

out = HERE / "fig_densification_maps.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")
