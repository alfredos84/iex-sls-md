"""
fig_Qn.py
Qn speciation (n=2,3,4) at 300 K.
Panel (a): As-melted vs IEX Proto1
Panel (b): As-melted vs IEX Proto2
One curve per x value, style distinguishes protocol.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "BOND_ANGLE_DIST"))
from structure_analysis import avg_Qn

BASE = HERE.parent / "STAGE7_QUENCH_300K" / "data"

PROTOCOLS = {
    'AM':   {'dir': BASE / 'asmelted_300K',
             'fname': lambda x, r: f"AsMelted_300K_x{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [0, 1, 3, 6, 9, 12, 15]},
    'IEX1': {'dir': BASE / 'iex1_300K',
             'fname': lambda x, r: f"IEX1_300K_xt{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [1, 3, 6, 9, 12, 15]},
    'IEX2': {'dir': BASE / 'iex2_300K',
             'fname': lambda x, r: f"IEX2_300K_xp{x}_r{r}_PMMCS_rc8p0.data",
             'x_list': [0, 1, 3, 6, 9, 12]},
}

X_COLORS = {0: '#1a1a2e', 1: '#1a6bb0', 3: '#27ae60',
            6: '#e67e22', 9: '#8e44ad', 12: '#c0392b', 15: '#2c3e50'}

QN_LABELS = {2: 'Q²', 3: 'Q³', 4: 'Q⁴'}
QN_MARKERS = {2: 'D', 3: 's', 4: 'o'}

REPS = [1, 2, 3]


def get_paths(proto, x):
    info = PROTOCOLS[proto]
    paths = []
    for r in REPS:
        p = info['dir'] / info['fname'](x, r)
        if p.exists():
            paths.append(p)
    return paths


def load_qn(proto, x):
    paths = get_paths(proto, x)
    if not paths:
        return None
    print(f"  Computing Qn: {proto} x={x} ({len(paths)} replicas)...")
    return avg_Qn(paths)   # array [Q0..Q4 fractions]


# ── Compute all Qn ───────────────────────────────────────────────────────────

print("Computing Qn speciation (this may take a few minutes)...")

qn = {}
for proto, info in PROTOCOLS.items():
    qn[proto] = {}
    for x in info['x_list']:
        qn[proto][x] = load_qn(proto, x)

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.13,
                    wspace=0.08)

panel_configs = [
    ('AM', 'IEX1', 'As-melted vs IEX Proto1', axes[0]),
    ('AM', 'IEX2', 'As-melted vs IEX Proto2', axes[1]),
]

for proto_a, proto_b, title, ax in panel_configs:
    x_union = sorted(set(PROTOCOLS[proto_a]['x_list']) |
                     set(PROTOCOLS[proto_b]['x_list']))

    for n in [2, 3, 4]:
        xa_vals, ya_vals = [], []
        xb_vals, yb_vals = [], []

        for x in PROTOCOLS[proto_a]['x_list']:
            if qn[proto_a].get(x) is not None:
                xa_vals.append(x)
                ya_vals.append(qn[proto_a][x][n] * 100)

        for x in PROTOCOLS[proto_b]['x_list']:
            if qn[proto_b].get(x) is not None:
                xb_vals.append(x)
                yb_vals.append(qn[proto_b][x][n] * 100)

        col = {2: '#c0392b', 3: '#1a6bb0', 4: '#27ae60'}[n]
        mk  = QN_MARKERS[n]
        lbl = QN_LABELS[n]

        if xa_vals:
            ax.plot(xa_vals, ya_vals, marker=mk, ms=6.5, lw=1.8,
                    color=col, ls='-', label=f'{lbl} As-melted', zorder=4)
        if xb_vals:
            ax.plot(xb_vals, yb_vals, marker=mk, ms=6.5, lw=1.8,
                    color=col, ls='--', label=f'{lbl} {proto_b.replace("IEX","IEX Proto").replace("1"," 1").replace("2"," 2")}',
                    zorder=4)

    ax.set_xlabel('x (mol% K₂O)', fontsize=11)
    ax.set_ylabel('Fraction (%)', fontsize=11)
    ax.set_xticks([0, 1, 3, 6, 9, 12, 15])
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#cccccc',
              ncol=2)

axes[1].set_ylabel('')

out = HERE / "fig_Qn.pdf"
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(HERE / "fig_Qn.png", dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out}")
plt.close(fig)
