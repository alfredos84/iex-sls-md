"""
fig_elastic_born_pmmcs.py
4 elastic properties from Born matrix (STAGE2) — PMMCS only, 4 cutoffs.
Voigt-Reuss-Hill average over full 6x6 Cij tensor.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import re

RESULTS_DIR = Path(__file__).parent.parent / "STAGE2_BORN_MATRIX" / "results"
OUT_DIR     = Path(__file__).parent

x_vals    = [0, 6, 12, 15]
rc_labels = ['rc5p5', 'rc6p5', 'rc8p0', 'rc10p0']


def parse_cij(filepath):
    vals = {}
    with open(filepath) as f:
        for line in f:
            for m in re.finditer(r'(C\d{2})\s+([-\d.eE+]+)', line):
                vals[m.group(1)] = float(m.group(2))
    C = np.zeros((6, 6))
    for k, v in vals.items():
        i, j = int(k[1]) - 1, int(k[2]) - 1
        C[i, j] = v
        C[j, i] = v
    return C


def vrh(C):
    KV = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9.0
    GV = (C[0,0]+C[1,1]+C[2,2] - C[0,1]-C[0,2]-C[1,2]
          + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, np.nan, np.nan
    KR = 1.0 / (S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))
    GR = 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2])
                 + 3*(S[3,3]+S[4,4]+S[5,5]))
    K = (KV + KR) / 2.0
    G = (GV + GR) / 2.0
    E = 9*K*G / (3*K + G)
    nu = (3*K - 2*G) / (2*(3*K + G))
    return E, K, G, nu


# ── Collect data ──────────────────────────────────────────────────────────────
data = {rc: {'E': [], 'K': [], 'G': [], 'nu': [],
             'Ee': [], 'Ke': [], 'Ge': [], 'nue': []} for rc in rc_labels}

for rc in rc_labels:
    for x in x_vals:
        Er, Kr, Gr, nur = [], [], [], []
        for r in [1, 2, 3]:
            fname = RESULTS_DIR / f"born_Cij_PMMCS_{rc}_x{x}_r{r}.dat"
            if not fname.exists():
                print(f"  MISSING: {fname.name}")
                continue
            e, k, g, n = vrh(parse_cij(fname))
            Er.append(e); Kr.append(k); Gr.append(g); nur.append(n)
        for key, lst in [('E', Er), ('K', Kr), ('G', Gr), ('nu', nur)]:
            data[rc][key].append(np.mean(lst) if lst else np.nan)
            data[rc][key+'e'].append(np.std(lst, ddof=1) if len(lst) > 1 else 0.0)

# ── Palette ───────────────────────────────────────────────────────────────────
colors  = {'rc5p5': '#2a78d6', 'rc6p5': '#1baf7a',
           'rc8p0': '#eda100', 'rc10p0': '#9b30d9'}
markers = {'rc5p5': 'o', 'rc6p5': '^', 'rc8p0': 'D', 'rc10p0': 'v'}
labels  = {'rc5p5': r'$r_c=5.5$ Å', 'rc6p5': r'$r_c=6.5$ Å',
           'rc8p0': r'$r_c=8.0$ Å', 'rc10p0': r'$r_c=10.0$ Å'}

props = [
    ('E',  r"Young's modulus $E$ (GPa)"),
    ('K',  r"Bulk modulus $K$ (GPa)"),
    ('G',  r"Shear modulus $G$ (GPa)"),
    ('nu', r"Poisson's ratio $\nu$"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), sharex=True)
axes = axes.flatten()
kw = dict(lw=1.5, ms=6, capsize=3, capthick=1, elinewidth=0.8, zorder=4)

for ax, (prop, ylabel) in zip(axes, props):
    for rc in rc_labels:
        ax.errorbar(x_vals, data[rc][prop], yerr=data[rc][prop+'e'],
                    color=colors[rc], marker=markers[rc],
                    label=labels[rc], **kw)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(x_vals)
    ax.set_xlim(-1, 16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
    ax.tick_params(labelsize=8.5)

for ax in axes[2:]:
    ax.set_xlabel(r'$x$ (mol% K$_2$O)', fontsize=9.5)

handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=4, fontsize=8.5,
           frameon=True, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('PMMCS — Elastic properties from Born matrix\n'
             r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO, 300 K',
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_elastic_born_pmmcs.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_elastic_born_pmmcs.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'fig_elastic_born_pmmcs.pdf'}")
plt.show()
