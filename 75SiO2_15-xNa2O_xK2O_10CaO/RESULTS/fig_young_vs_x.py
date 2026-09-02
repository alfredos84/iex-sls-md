"""
fig_young_vs_x.py
Young's modulus vs K2O content (x) for 3 pair_pedone cutoffs.
E averaged over 3 principal directions (1/S11 + 1/S22 + 1/S33)/3
using full 6x6 compliance matrix inversion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import re

RESULTS_DIR = Path(__file__).parent.parent / "STAGE2_BORN_MATRIX" / "results"
OUT_DIR     = Path(__file__).parent

x_vals   = [0, 6, 12, 15]
rc_labels = ['rc5p5', 'rc6p5', 'rc8p0', 'rc10p0']

# ── Parse Cij file → 6×6 numpy array ─────────────────────────────────────────
def parse_cij(filepath):
    vals = {}
    with open(filepath) as f:
        for line in f:
            for m in re.finditer(r'(C\d{2})\s+([-\d.eE+]+)', line):
                vals[m.group(1)] = float(m.group(2))

    C = np.zeros((6, 6))
    idx = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}
    for k, v in vals.items():
        i, j = int(k[1])-1, int(k[2])-1
        C[i, j] = v
        C[j, i] = v
    return C

# ── Young's modulus: average of E1, E2, E3 from compliance matrix ────────────
def young_modulus(C):
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return np.nan
    E1 = 1.0 / S[0, 0]
    E2 = 1.0 / S[1, 1]
    E3 = 1.0 / S[2, 2]
    return (E1 + E2 + E3) / 3.0

# ── Collect E values ──────────────────────────────────────────────────────────
E_mean = {rc: [] for rc in rc_labels}
E_std  = {rc: [] for rc in rc_labels}

for x in x_vals:
    for rc in rc_labels:
        E_reps = []
        for r in [1, 2, 3]:
            fname = RESULTS_DIR / f"born_Cij_PMMCS_{rc}_x{x}_r{r}.dat"
            if not fname.exists():
                print(f"  MISSING: {fname.name}")
                continue
            C = parse_cij(fname)
            E = young_modulus(C)
            E_reps.append(E)
        E_mean[rc].append(np.mean(E_reps))
        E_std[rc].append(np.std(E_reps, ddof=1) if len(E_reps) > 1 else 0.0)

# ── Palette (validated: blue, aqua, yellow) ───────────────────────────────────
colors  = {'rc5p5': '#2a78d6', 'rc6p5': '#1baf7a', 'rc8p0': '#eda100', 'rc10p0': '#9b30d9'}
markers = {'rc5p5': 'o',       'rc6p5': '^',        'rc8p0': 'D', 'rc10p0': 'v'}
labels  = {'rc5p5': r'$r_c=5.5$ Å', 'rc6p5': r'$r_c=6.5$ Å', 'rc8p0': r'$r_c=8.0$ Å', 'rc10p0': r'$r_c=10.0$ Å'}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

for rc in rc_labels:
    ax.errorbar(x_vals, E_mean[rc], yerr=E_std[rc],
                color=colors[rc], lw=1.5, marker=markers[rc], ms=6,
                capsize=3, capthick=1, elinewidth=0.8, zorder=4,
                label=labels[rc])

ax.set_xlabel(r'$x$ (mol% K$_2$O substituting Na$_2$O)', fontsize=10)
ax.set_ylabel(r"Young's modulus $E$ (GPa)", fontsize=10)
ax.set_title("PMMCS: Young's modulus vs composition\n"
             r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO',
             fontsize=9)

ax.set_xticks(x_vals)
ax.set_xlim(-1, 16)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
          fontsize=8.5, loc='best')

fig.tight_layout()

out_pdf = OUT_DIR / "fig_young_vs_x.pdf"
out_png = OUT_DIR / "fig_young_vs_x.png"
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'x':>4}  {'rc':>8}  {'E_mean (GPa)':>14}  {'E_std':>8}")
for x, i in zip(x_vals, range(4)):
    for rc in rc_labels:
        print(f"{x:>4}  {rc:>8}  {E_mean[rc][i]:>14.2f}  {E_std[rc][i]:>8.2f}")

plt.show()
