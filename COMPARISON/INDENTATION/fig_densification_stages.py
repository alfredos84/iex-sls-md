"""
fig_densification_stages.py
Mapa 2D de densificacion local (metodo chunk/atom bin/3d, replica del enfoque de
Pedone 2026 Fig. 7) en las 3 etapas de la indentacion: Start (step=20000, antes de
cargar), Max load (step=120000, fin de la carga) y Unload (ultimo frame, despues de
descargar). Figura de 3 filas x 4 columnas: filas = etapa, columnas = vidrio
(CaO-AM, CaO-IOX, Ca-free-AM, Ca-free-IOX).

Densidad local por bin (2x2x2 A^3) en una rebanada de 12 A de espesor centrada en el
eje de la punta, comparada contra rho0 = densidad bulk (z in [10,80] A) de la
configuracion de referencia (step=20000) de esa misma replica:

  densificacion(%) = (rho_local(bin) - rho0) / rho0 * 100

Promediado sobre replicas (r1,r2,r3; CaO-AM usa solo r2,r3 porque r1 es la muestra
piloto, no comparable). Suavizado gaussiano sigma=1.5 bins, rango de color fijo
[-30%, +30%]. Fondo gris oscuro continuo (sin bins vacios visibles) para las
regiones sin atomos (vacio bajo la punta, por encima de la superficie, cavidad del
crater).

Atom types: Si=1, O=2, Ca=3, Na=4, K=5, C(punta)=6
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter, binary_opening

HERE = Path(__file__).parent
FRAMES = HERE / "surface_profile" / "lastframes"
sys.path.insert(0, str(HERE.parent))
from paper_style import FS_LABEL, FS_TEXT, apply_rcparams, relabel
import string
apply_rcparams()

TYPE_TIP = 6
TYPE_MASS = {1: 28.0855, 2: 15.9994, 3: 40.078, 4: 22.98977, 5: 39.0983}

BIN = 2.0
X_HALF_WIDTH = 6.0
BULK_Z_LO, BULK_Z_HI = 10.0, 80.0
SIGMA_BINS = 1.5
MIN_ATOMS_BIN = 1      # bins con >=1 atomo se consideran "materia"; el resto es vacio

STAGES = [("step20000", "Start"), ("step120000", "Max load"), ("last", "Unload")]

CASES = {
    "CaO":     {"AM": ["cao_x0_r2", "cao_x0_r3"],
                "IOX": ["cao_iex_xt15_r1_300K", "cao_iex_xt15_r2_300K", "cao_iex_xt15_r3_300K"]},
    "Ca-free": {"AM": ["noca_x0_r1", "noca_x0_r2", "noca_x0_r3"],
                "IOX": ["noca_iex_xt25_r1_300K", "noca_iex_xt25_r2_300K", "noca_iex_xt25_r3_300K"]},
}
COLUMNS = [("CaO", "AM"), ("CaO", "IOX"), ("Ca-free", "AM"), ("Ca-free", "IOX")]


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


def local_density_and_mask(tag, step, y_edges, z_edges):
    types1, xyz1, lo1, len1 = read_frame(tag, step)
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
    grid_n = np.zeros((nx, len(z_edges) - 1, len(y_edges) - 1))
    np.add.at(grid_mass, (ix, iz, iy), mass)
    np.add.at(grid_n, (ix, iz, iy), 1)

    bin_vol = BIN ** 3
    rho = (grid_mass / bin_vol).mean(axis=0)
    occupied = (grid_n >= MIN_ATOMS_BIN).mean(axis=0) > 0    # bin "con materia" si algun x-bin de la rebanada tiene atomos
    return rho, occupied


Y_EDGES = np.arange(-95, 95 + BIN, BIN)
Z_EDGES = np.arange(-10, 150 + BIN, BIN)

results = {}   # (glass, cond, stage_key) -> (pct_smoothed, mask_occupied)
for glass, conds in CASES.items():
    for cond, tags in conds.items():
        rho0s = {tag: bulk_density(tag) for tag in tags}
        for step, label in STAGES:
            print(f"{glass} {cond} {label}...")
            pct_maps, masks = [], []
            for tag in tags:
                rho_local, occ = local_density_and_mask(tag, step, Y_EDGES, Z_EDGES)
                pct = (rho_local - rho0s[tag]) / rho0s[tag] * 100.0
                pct_maps.append(pct)
                masks.append(occ)
            avg = np.mean(np.stack(pct_maps), axis=0)
            occ_any = np.any(np.stack(masks), axis=0)   # ocupado si >=1 replica tiene materia ahi
            occ_clean = binary_opening(occ_any, structure=np.ones((3, 3)))   # descarta bins aislados (1 px)
            smoothed = gaussian_filter(avg, sigma=SIGMA_BINS)
            masked = np.where(occ_clean, smoothed, np.nan)
            results[(glass, cond, step)] = masked

vmax = 30.0
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
BG = "#053061"   # azul mas oscuro del colormap RdBu_r (extremo de la escala), como en Pedone 2026
EXTENT = [Y_EDGES[0], Y_EDGES[-1], Z_EDGES[0], Z_EDGES[-1]]

letters = iter(string.ascii_lowercase)
fig, axes = plt.subplots(3, 4, figsize=(15.5, 8.6))
fig.subplots_adjust(left=0.075, right=0.93, top=0.94, bottom=0.075, wspace=0.04, hspace=0.015)
im = None
for row, (step, stage_label) in enumerate(STAGES):
    for col, (glass, cond) in enumerate(COLUMNS):
        ax = axes[row, col]
        grid = results[(glass, cond, step)]
        im = ax.imshow(grid, extent=EXTENT, origin="lower", cmap="RdBu_r", norm=norm,
                        interpolation="nearest", aspect="equal")
        ax.set_facecolor(BG)
        ax.tick_params(labelsize=FS_TEXT, which="both")
        ax.text(0.03, 0.97, f"({next(letters)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS_TEXT, fontweight="normal", color="white", zorder=6)
        if row == 0:
            ax.set_title(f"{relabel(glass)} — {cond}", fontsize=FS_LABEL)
        if row == 2:
            ax.set_xlabel("y (Å)", fontsize=FS_LABEL)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel("z (Å)", fontsize=FS_LABEL, labelpad=2)
        else:
            ax.set_yticklabels([])
    pos = axes[row, 0].get_position()
    fig.text(0.012, pos.y0 + 0.5 * pos.height, stage_label, fontsize=FS_LABEL,
              fontweight="normal", ha="left", va="center", rotation=90)

cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
cbar.set_label("Local densification (%)", fontsize=FS_LABEL)
cbar.ax.tick_params(labelsize=FS_TEXT)

out = HERE / "fig_densification_stages.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", facecolor="white")
print(f"Guardado: {out.name}")
