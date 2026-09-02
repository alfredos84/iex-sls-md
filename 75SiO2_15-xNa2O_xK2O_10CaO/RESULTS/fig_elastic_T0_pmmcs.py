"""
fig_elastic_T0_pmmcs.py
4 elastic properties (E, K, G, nu) from athermal elastic tensor (T=0 K)
PMMCS rc=10 A — VRH averages from STAGE4_ELASTIC_T0.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "STAGE4_ELASTIC_T0" / "results"
OUT_DIR     = Path(__file__).parent

x_vals = [0, 6, 12, 15]

# ── Parse Cij files ───────────────────────────────────────────────────────────
def load_vrh(x, r):
    fname = RESULTS_DIR / f"elastic_T0_Cij_PMMCS_rc10p0_x{x}_r{r}.txt"
    if not fname.exists():
        return None
    header = fname.read_text().strip().split('\n')[0].split()
    vals   = [float(v) for v in fname.read_text().strip().split('\n')[1].split()]
    d = dict(zip(header, vals))
    return d['K'], d['G'], d['E'], d['nu']

E_mean, K_mean, G_mean, nu_mean = [], [], [], []
E_std,  K_std,  G_std,  nu_std  = [], [], [], []

for x in x_vals:
    Ev, Kv, Gv, nuv = [], [], [], []
    for r in [1, 2, 3]:
        res = load_vrh(x, r)
        if res is None:
            continue
        k, g, e, n = res
        Kv.append(k); Gv.append(g); Ev.append(e); nuv.append(n)
    for lst, m, s in [(Ev, E_mean, E_std), (Kv, K_mean, K_std),
                      (Gv, G_mean, G_std), (nuv, nu_mean, nu_std)]:
        m.append(np.mean(lst) if lst else np.nan)
        s.append(np.std(lst, ddof=1) if len(lst) > 1 else 0.0)

# ── Figure ────────────────────────────────────────────────────────────────────
color  = '#9b30d9'
marker = 'v'
kw = dict(lw=1.5, ms=6, capsize=3, capthick=1, elinewidth=0.8, zorder=4)

props = [
    (E_mean,  E_std,  r"Young's modulus $E$ (GPa)"),
    (K_mean,  K_std,  r"Bulk modulus $K$ (GPa)"),
    (G_mean,  G_std,  r"Shear modulus $G$ (GPa)"),
    (nu_mean, nu_std, r"Poisson's ratio $\nu$"),
]

fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5), sharex=True)
axes = axes.flatten()

for ax, (m, s, ylabel) in zip(axes, props):
    ax.errorbar(x_vals, m, yerr=s, color=color, marker=marker,
                label=r'PMMCS $r_c=10$ Å', **kw)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_xticks(x_vals)
    ax.set_xlim(-1, 16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

for ax in axes[2:]:
    ax.set_xlabel(r'$x$ (mol% K$_2$O)', fontsize=9.5)

fig.suptitle(r'PMMCS $r_c=10$ Å — Elastic properties, athermal ($T=0$ K)'
             '\n' r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO',
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_elastic_T0_pmmcs.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_elastic_T0_pmmcs.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'fig_elastic_T0_pmmcs.pdf'}")

print(f"\n{'x':>4}  {'E (GPa)':>10}  {'K (GPa)':>10}  {'G (GPa)':>10}  {'nu':>8}")
for i, x in enumerate(x_vals):
    print(f"{x:>4}  {E_mean[i]:>10.2f}  {K_mean[i]:>10.2f}  {G_mean[i]:>10.2f}  {nu_mean[i]:>8.4f}")

plt.show()
