"""
fig_angles.py
Bond angle distributions O-Si-O and Si-O-Si at 300 K.
One figure per x value: 2 panels (O-Si-O | Si-O-Si), 3 curves (AM, IEX1, IEX2).
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from structure_analysis import avg_angles

BASE = HERE.parent.parent / "STAGE7_QUENCH_300K" / "data"

PROTOCOLS = {
    'AM':   {'dir': BASE / 'asmelted_300K',
             'fname': lambda x, r: f"AsMelted_300K_x{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [0, 1, 3, 6, 9, 12, 15],
             'color': '#1a6bb0', 'label': 'As-melted',  'ls': '-'},
    'IEX1': {'dir': BASE / 'iex1_300K',
             'fname': lambda x, r: f"IEX1_300K_xt{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [1, 3, 6, 9, 12, 15],
             'color': '#c0392b', 'label': 'IEX Proto1', 'ls': '--'},
    'IEX2': {'dir': BASE / 'iex2_300K',
             'fname': lambda x, r: f"IEX2_300K_xp{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [0, 1, 3, 6, 9, 12],
             'color': '#27ae60', 'label': 'IEX Proto2', 'ls': ':'},
}

REPS     = [1, 2, 3]
BINS_OSiO  = np.linspace(60, 180, 241)   # 0.5° bins
BINS_SiOSi = np.linspace(100, 180, 161)  # 0.5° bins
BW = 0.5   # bin width in degrees

ALL_X = sorted({0, 1, 3, 6, 9, 12, 15})


def get_paths(proto, x):
    info = PROTOCOLS[proto]
    if x not in info['x_list']:
        return []
    paths = [info['dir'] / info['fname'](x, r) for r in REPS]
    return [p for p in paths if p.exists()]


def kde_hist(angles, bins):
    """Normalized histogram (probability density, area=1)."""
    if len(angles) == 0:
        return bins[:-1] + BW/2, np.zeros(len(bins)-1)
    counts, edges = np.histogram(angles, bins=bins, density=True)
    centers = 0.5*(edges[:-1] + edges[1:])
    return centers, counts


# ── Generate one figure per x ────────────────────────────────────────────────

for x in ALL_X:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.91, bottom=0.14,
                        wspace=0.28)

    any_data = False

    for proto, info in PROTOCOLS.items():
        paths = get_paths(proto, x)
        if not paths:
            continue

        print(f"  x={x}  {proto}  ({len(paths)} replicas)...")

        a_osio  = avg_angles(paths, 'OSiO')
        a_siosi = avg_angles(paths, 'SiOSi')

        c  = info['color']
        ls = info['ls']
        lbl = info['label']

        cx, hy = kde_hist(a_osio,  BINS_OSiO)
        axes[0].plot(cx, hy, lw=1.6, color=c, ls=ls, label=lbl, zorder=3)

        cx, hy = kde_hist(a_siosi, BINS_SiOSi)
        axes[1].plot(cx, hy, lw=1.6, color=c, ls=ls, label=lbl, zorder=3)

        any_data = True

    for ax, title, xlims, xtks in [
        (axes[0], 'O–Si–O', (80, 160),
         [90, 100, 109.47, 120, 130, 140, 150, 160]),
        (axes[1], 'Si–O–Si', (120, 180),
         [120, 130, 140, 150, 160, 170, 180]),
    ]:
        ax.set_xlabel('Angle (°)', fontsize=11)
        ax.set_ylabel('Probability density', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(*xlims)
        ax.set_xticks(xtks)
        ax.set_xticklabels([f'{t:.0f}' if t != 109.47 else '109.47°'
                            for t in xtks], fontsize=8)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)
        if any_data:
            ax.legend(fontsize=9, frameon=True, framealpha=0.9,
                      edgecolor='#cccccc')

    fname = HERE / f"fig_angles_x{x}.pdf"
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    fig.savefig(HERE / f"fig_angles_x{x}.png", dpi=150, bbox_inches='tight')
    print(f"  Guardado: fig_angles_x{x}.pdf")
    plt.close(fig)

print("\nTodas las figuras de ángulos generadas.")
