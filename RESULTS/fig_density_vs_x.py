"""
fig_density_vs_x.py
Density vs K2O content (x) for 3 pair_pedone cutoffs + reference values.
Glass: 75SiO2.(15-x)Na2O.xK2O.10CaO
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Data ─────────────────────────────────────────────────────────────────────
# Raw densities: {(x, rc_label): [r1, r2, r3]}
raw = {
    (0,  'rc5p5'): [2.4590194, 2.4608511, 2.4922244],
    (0,  'rc6p5'): [2.6204922, 2.5922368, 2.5987335],
    (0,  'rc8p0'): [2.659407,  2.60906,   2.6361684],
    (6,  'rc5p5'): [2.4119461, 2.4178269, 2.4242223],
    (6,  'rc6p5'): [2.5480197, 2.5227992, 2.5305508],
    (6,  'rc8p0'): [2.5997018, 2.6008943, 2.6201689],
    (12, 'rc5p5'): [2.3721013, 2.3689017, 2.3800979],
    (12, 'rc6p5'): [2.5247352, 2.5154169, 2.4906626],
    (12, 'rc8p0'): [2.519972,  2.5979792, 2.5561426],
    (15, 'rc5p5'): [2.3424742, 2.3248181, 2.3658536],
    (15, 'rc6p5'): [2.4696728, 2.4849314, 2.4731997],
    (15, 'rc8p0'): [2.504046,  2.5239867, 2.5286938],
    (0,  'rc10p0'): [2.6690772, 2.6611842, 2.6466582],
    (6,  'rc10p0'): [2.6087622, 2.618482,  2.6307989],
    (12, 'rc10p0'): [2.5796799, 2.5612491, 2.5757783],
    (15, 'rc10p0'): [2.5514201, 2.5293769, 2.5586352],
}

x_vals = [0, 6, 12, 15]
rc_labels = ['rc5p5', 'rc6p5', 'rc8p0', 'rc10p0']
rc_display = {'rc5p5': r'$r_c=5.5$ Å', 'rc6p5': r'$r_c=6.5$ Å', 'rc8p0': r'$r_c=8.0$ Å', 'rc10p0': r'$r_c=10.0$ Å'}

means = {rc: [] for rc in rc_labels}
stds  = {rc: [] for rc in rc_labels}
for x in x_vals:
    for rc in rc_labels:
        vals = [v for v in raw[(x, rc)] if v is not None]
        means[rc].append(np.mean(vals) if vals else np.nan)
        stds[rc].append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

# Experimental densities
x_ref   = [0,       6,       12,      15     ]
rho_ref  = [2.48211, 2.48284, 2.47525, 2.45893]

# ── Palette (validated: blue, aqua, yellow, red) ─────────────────────────────
colors = {
    'rc5p5': '#2a78d6',
    'rc6p5': '#1baf7a',
    'rc8p0': '#eda100', 'rc10p0': '#9b30d9',
}
color_ref = '#e34948'

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.2))

# Reference line
ax.plot(x_ref, rho_ref, color=color_ref, lw=1.5,
        linestyle='--', marker='s', ms=6, zorder=5,
        label='Reference')

# Simulated curves
markers = {'rc5p5': 'o', 'rc6p5': '^', 'rc8p0': 'D', 'rc10p0': 'v'}
for rc in rc_labels:
    ax.errorbar(x_vals, means[rc], yerr=stds[rc],
                color=colors[rc], lw=1.5, marker=markers[rc], ms=6,
                capsize=3, capthick=1, elinewidth=0.8, zorder=4,
                label=rc_display[rc])

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel(r'$x$ (mol% K$_2$O substituting Na$_2$O)', fontsize=10)
ax.set_ylabel(r'Density (g cm$^{-3}$)', fontsize=10)
ax.set_title('PMMCS: density vs composition\n'
             r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO, 5 K/ps quench',
             fontsize=9)

ax.set_xticks(x_vals)
ax.set_xlim(-1, 16)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)

ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc',
          fontsize=8.5, loc='upper right')

fig.tight_layout()

out = Path(__file__).parent / 'fig_density_vs_x.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight')
out_png = out.with_suffix('.png')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
print(f"Saved: {out_png}")
plt.show()
