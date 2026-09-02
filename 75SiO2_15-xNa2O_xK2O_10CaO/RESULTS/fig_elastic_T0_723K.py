"""
fig_elastic_T0_723K.py
Propiedades elásticas adiabáticas (T=0 K) — vidrios a 723 K.
PMMCS rc=8 Å — As-melted, IEX Proto1, IEX Proto2.
Paneles 2×2: E, K, G, ν.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE       = Path(__file__).parent
BASE       = HERE.parent
DIR_AM723  = BASE / "STAGE2_ELASTIC_T0" / "results"
DIR_IX1723 = BASE / "STAGE5_ELASTIC_T0_PROTO1" / "results"
DIR_IX2723 = BASE / "STAGE6_ELASTIC_T0_PROTO2" / "results"
OUT_DIR    = HERE

XA_VALS = [0, 1, 3, 6, 9, 12, 15]
XT_VALS = [1, 3, 6, 9, 12, 15]
XP_VALS = [0, 1, 3, 6, 9, 12]

C_AM  = '#1a6bb0'
C_IX1 = '#c0392b'
C_IX2 = '#27ae60'
KW = dict(lw=1.5, ms=6, capsize=3, capthick=1.0, elinewidth=0.8)


def load_vrh(path):
    lines = path.read_text().strip().split('\n')
    d = dict(zip(lines[0].split(), [float(v) for v in lines[1].split()]))
    return d['K'], d['G'], d['E'], d['nu']


def collect(paths_fn, x_list):
    means = {p: [] for p in ['K', 'G', 'E', 'nu']}
    stds  = {p: [] for p in ['K', 'G', 'E', 'nu']}
    for x in x_list:
        vals = {p: [] for p in ['K', 'G', 'E', 'nu']}
        for r in [1, 2, 3]:
            p = paths_fn(x, r)
            if p.exists():
                k, g, e, n = load_vrh(p)
                vals['K'].append(k); vals['G'].append(g)
                vals['E'].append(e); vals['nu'].append(n)
        for prop in ['K', 'G', 'E', 'nu']:
            v = vals[prop]
            means[prop].append(np.mean(v) if v else np.nan)
            stds[prop].append(np.std(v, ddof=1) if len(v) > 1 else 0.0)
    return means, stds


am723   = collect(lambda x, r: DIR_AM723  / f"elastic_T0_Cij_PMMCS_rc8p0_x{x}_r{r}.txt",         XA_VALS)
ix1_723 = collect(lambda x, r: DIR_IX1723 / f"elastic_T0_Cij_PMMCS_rc8p0_IEX1_xt{x}_r{r}.txt",   XT_VALS)
ix2_723 = collect(lambda x, r: DIR_IX2723 / f"elastic_T0_Cij_PMMCS_rc8p0_IEX2_xp{x}_r{r}.txt",   XP_VALS)

props   = ['E', 'K', 'G', 'nu']
ylabels = [r"$E$ (GPa)", r"$K$ (GPa)", r"$G$ (GPa)", r"$\nu$"]

fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=False)
axes = axes.flatten()

for ax, prop, ylabel in zip(axes, props, ylabels):
    ax.errorbar(XA_VALS, am723[0][prop],   yerr=am723[1][prop],
                color=C_AM,  marker='o', label='As-melted', **KW, zorder=4)
    ax.errorbar(XT_VALS, ix1_723[0][prop], yerr=ix1_723[1][prop],
                color=C_IX1, marker='s', label='IEX Proto1', **KW, zorder=4)
    ax.errorbar(XP_VALS, ix2_723[0][prop], yerr=ix2_723[1][prop],
                color=C_IX2, marker='D', label='IEX Proto2', **KW, zorder=4)

    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks([0, 1, 3, 6, 9, 12, 15])
    ax.set_xlim(-1, 16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#eeeeee', zorder=0)
    ax.tick_params(labelsize=8.5)

axes[0].legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#cccccc')

for ax in axes[2:]:
    ax.set_xlabel(r'$x$ / $x_\mathrm{target}$ / $x_\mathrm{parent}$ (mol% K$_2$O)', fontsize=9)

fig.suptitle(
    r'PMMCS $r_c=8$ Å — Propiedades elásticas adiabáticas ($T=0$ K)  |  723 K'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=10
)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_elastic_T0_723K.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_elastic_T0_723K.png', dpi=150, bbox_inches='tight')
print(f"Guardado: {OUT_DIR / 'fig_elastic_T0_723K.pdf'}")
plt.close(fig)
