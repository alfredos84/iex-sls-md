"""
fig_indentation_summary.py
Figura combinada de 3 paneles cuadrados sobre nanoindentacion:
  (a) perfil de superficie post-indentacion, CaO (ver surface_profile/fig_surface_profile.py)
  (b) idem, Ca-free
  (c) dureza H (Oliver-Pharr), media +/- desviacion estandar entre replicas (ver fig_H_bars.py)

Colores iguales a fig_molar_volume_comparison.py: AM = negro; IOX = rojo claro (Ca) /
rojo oscuro (Ca-free).

CaO AM en el perfil de superficie y en H usa solo r2,r3 (r1 es la muestra piloto,
caja mas chica, no comparable con el resto).
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from paper_style import BOX, FS_LABEL, FS_TEXT, apply_rcparams, make_fig, panel_letter, relabel
apply_rcparams()

C_AM = "black"
C_AM_BAR = "#8c8c8c"   # gris en las barras de H para que se vea la barra de error
C_IOX_CAO = "#e74c3c"
C_IOX_NOCA = "#7b241c"
DISPLAY = {"CaO": "Ca", "Ca-free": "Ca-free"}

# ── (a),(b): perfil de superficie ────────────────────────────────────────────

FRAMES = HERE / "surface_profile" / "lastframes"
TYPE_TIP = 6
X_HALF_WIDTH = 6.0
SBIN = 1.0

PROFILE_CASES = {
    "CaO":     {"AM": ("cao_x0_r{r}", (2, 3)),      "IOX": ("cao_iex_xt15_r{r}_300K", (1, 2, 3))},
    "Ca-free": {"AM": ("noca_x0_r{r}", (1, 2, 3)),  "IOX": ("noca_iex_xt25_r{r}_300K", (1, 2, 3))},
}


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
    n_bins = int(round(length[1] / SBIN))
    edges = y0 + np.arange(n_bins + 1) * SBIN
    centers = 0.5 * (edges[:-1] + edges[1:])
    zmax = np.full(n_bins, np.nan)
    idx = np.clip(((y - y0) / SBIN).astype(int), 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            zmax[b] = z[sel].max()

    y_rel = centers - (y0 + 0.5 * length[1])
    far = np.abs(y_rel) > 0.35 * length[1]
    baseline = np.nanmedian(zmax[far])
    return y_rel, zmax - baseline


Y_GRID = np.arange(-95.0, 95.0 + SBIN, SBIN)


def averaged_profile(tag_pat, reps):
    profiles = [replica_profile(tag_pat.format(r=r)) for r in reps]
    interp = []
    for y, z in profiles:
        ok = ~np.isnan(z)
        interp.append(np.interp(Y_GRID, y[ok], z[ok], left=np.nan, right=np.nan))
    stacked = np.array(interp)
    return Y_GRID, np.nanmean(stacked, axis=0)


# ── (c): dureza H ────────────────────────────────────────────────────────────

H_CASES = {
    "CaO":  {"AM": "cao_x0_r{r}",  "IOX": "cao_iex_xt15_r{r}"},
    "Ca-free": {"AM": "noca_x0_r{r}", "IOX": "noca_iex_xt25_r{r}"},
}
H_EXCLUDE = {("Ca-free", "AM"): (2,), ("CaO", "AM"): (1,)}


def hardness(folder):
    out = subprocess.run(["python3", "compute_H_auto.py"], cwd=HERE / folder,
                         capture_output=True, text=True, check=True).stdout
    return float(re.search(r"H = ([\d.]+) GPa", out).group(1))


h_stats = {}
for glass, conds in H_CASES.items():
    for cond, pat in conds.items():
        v = np.array([hardness(pat.format(r=r)) for r in (1, 2, 3) if r not in H_EXCLUDE.get((glass, cond), ())])
        h_stats[(glass, cond)] = (v.mean(), v.std(ddof=1))

# ── Figura: 1x3 paneles cuadrados ────────────────────────────────────────────

fig, axes = make_fig(1, 3)

for ax, glass, c_iox, letter in zip(axes[:2], ("CaO", "Ca-free"), (C_IOX_CAO, C_IOX_NOCA), ("(a)", "(b)")):
    conds = PROFILE_CASES[glass]
    for cond, color in (("AM", C_AM), ("IOX", c_iox)):
        pat, reps = conds[cond]
        y, z = averaged_profile(pat, reps)
        ax.plot(y, z, color=color, lw=1.8, label=relabel(cond))
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-35, 5)
    ax.set_yticks(np.arange(-35, 6, 5))
    ax.set_xlabel("y (Å)", fontsize=FS_LABEL)
    ax.text(-60, -10, DISPLAY[glass], ha="center", va="center", fontsize=FS_LABEL, zorder=5)
    ax.xaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=FS_TEXT, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)
    ax.legend(fontsize=FS_TEXT, frameon=True, framealpha=0.9, edgecolor="#cccccc")
    panel_letter(ax, letter)

axes[0].set_ylabel("Surface height (Å)", fontsize=FS_LABEL)

# (c) barras de H
ax = axes[2]
glasses = list(H_CASES)
x0 = np.arange(len(glasses))
width = 0.32
bar_colors = {"CaO": C_IOX_CAO, "Ca-free": C_IOX_NOCA}
for k, cond in enumerate(("AM", "IOX")):
    means = [h_stats[(g, cond)][0] for g in glasses]
    sds = [h_stats[(g, cond)][1] for g in glasses]
    colors = [C_AM_BAR if cond == "AM" else bar_colors[g] for g in glasses]
    ax.bar(x0 + (k - 0.5) * width, means, width, yerr=sds, capsize=4, color=colors,
           edgecolor="none", error_kw=dict(lw=1.3, ecolor="#333333"), zorder=3)

ax.set_xticks(x0)
ax.set_xticklabels([DISPLAY[g] for g in glasses], fontsize=FS_TEXT)
ax.set_ylabel("Hardness H (GPa)", fontsize=FS_LABEL)
ax.set_ylim(0, 20)
ax.grid(axis="y", lw=0.35, color="#dddddd", zorder=0)
ax.tick_params(labelsize=FS_TEXT)
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(0.8)

from matplotlib.patches import Patch
handles = [Patch(color=C_AM_BAR, label="AM"),
           Patch(color=C_IOX_CAO, label="IOX — Ca"),
           Patch(color=C_IOX_NOCA, label="IOX — Ca-free")]
ax.legend(handles=handles, fontsize=14, frameon=True, framealpha=0.9,
          edgecolor="#cccccc", loc="upper right")
panel_letter(ax, "(c)")

fig.subplots_adjust(wspace=0.18, top=0.95)
pos_c = axes[2].get_position()
axes[2].set_position([pos_c.x0 + 0.02, pos_c.y0, pos_c.width, pos_c.height])
out = HERE / "fig_indentation_summary.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}")
