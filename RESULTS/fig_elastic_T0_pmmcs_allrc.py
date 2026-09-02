"""
fig_elastic_T0_pmmcs_allrc.py
4 elastic properties (E, K, G, nu) from athermal elastic tensor (T=0 K)
PMMCS — all short-range cutoffs rc=5.5, 6.5, 8.0, 10.0 A.
VRH averages (mean ± std over 3 replicas) from STAGE4_ELASTIC_T0.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "STAGE4_ELASTIC_T0" / "results"
OUT_DIR     = Path(__file__).parent

x_vals = [0, 6, 12, 15]

RC_CONFIG = [
    ("rc5p5",  r"$r_c=5.5$ Å",  "#e63946", "o"),
    ("rc6p5",  r"$r_c=6.5$ Å",  "#f4a261", "s"),
    ("rc8p0",  r"$r_c=8.0$ Å",  "#2a9d8f", "^"),
    ("rc10p0", r"$r_c=10.0$ Å", "#9b30d9", "v"),
]


def load_vrh(rc, x, r):
    fname = RESULTS_DIR / f"elastic_T0_Cij_PMMCS_{rc}_x{x}_r{r}.txt"
    if not fname.exists():
        return None
    lines = fname.read_text().strip().split('\n')
    header = lines[0].split()
    vals   = [float(v) for v in lines[1].split()]
    d = dict(zip(header, vals))
    return d['K'], d['G'], d['E'], d['nu']


# ── Collect data ──────────────────────────────────────────────────────────────
data = {}
for rc, label, color, marker in RC_CONFIG:
    E_m, K_m, G_m, nu_m = [], [], [], []
    E_s, K_s, G_s, nu_s = [], [], [], []
    for x in x_vals:
        Ev, Kv, Gv, nuv = [], [], [], []
        for r in [1, 2, 3]:
            res = load_vrh(rc, x, r)
            if res is None:
                continue
            k, g, e, n = res
            Kv.append(k); Gv.append(g); Ev.append(e); nuv.append(n)
        for lst, m, s in [(Ev, E_m, E_s), (Kv, K_m, K_s),
                          (Gv, G_m, G_s), (nuv, nu_m, nu_s)]:
            m.append(np.mean(lst) if lst else np.nan)
            s.append(np.std(lst, ddof=1) if len(lst) > 1 else 0.0)
    data[rc] = dict(E=(E_m, E_s), K=(K_m, K_s), G=(G_m, G_s), nu=(nu_m, nu_s))

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.5), sharex=True)
axes = axes.flatten()

prop_keys = [
    ('E',  r"Young's modulus $E$ (GPa)"),
    ('K',  r"Bulk modulus $K$ (GPa)"),
    ('G',  r"Shear modulus $G$ (GPa)"),
    ('nu', r"Poisson's ratio $\nu$"),
]

kw = dict(lw=1.5, ms=5.5, capsize=3, capthick=1, elinewidth=0.8, zorder=4)

for ax, (prop, ylabel) in zip(axes, prop_keys):
    for rc, label, color, marker in RC_CONFIG:
        m, s = data[rc][prop]
        ax.errorbar(x_vals, m, yerr=s, color=color, marker=marker,
                    label=label, **kw)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(x_vals)
    ax.set_xlim(-1, 16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
    ax.tick_params(labelsize=8.5)

# Legend only in first panel
axes[0].legend(fontsize=8.5, frameon=True, framealpha=0.9,
               edgecolor='#cccccc', loc='upper right')

for ax in axes[2:]:
    ax.set_xlabel(r'$x$ (mol% K$_2$O)', fontsize=9.5)

fig.suptitle(r'PMMCS — Elastic properties vs. short-range cutoff $r_c$, athermal ($T=0$ K)'
             '\n' r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO',
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_elastic_T0_pmmcs_allrc.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_elastic_T0_pmmcs_allrc.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'fig_elastic_T0_pmmcs_allrc.pdf'}")

# ── Print table ───────────────────────────────────────────────────────────────
for prop, ylabel in prop_keys:
    print(f"\n── {ylabel} ──")
    print(f"{'rc':>8}  " + "  ".join(f"x={x:>2}" for x in x_vals))
    for rc, label, color, marker in RC_CONFIG:
        m, s = data[rc][prop]
        vals = "  ".join(f"{v:>7.3f}" for v in m)
        print(f"{rc:>8}  {vals}")
