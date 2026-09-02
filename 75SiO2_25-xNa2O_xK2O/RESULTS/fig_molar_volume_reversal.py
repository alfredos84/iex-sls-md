"""
fig_molar_volume_reversal.py
Molar volume (cm³/mol) at 723 K vs x (mol% K₂O exchange parameter).
Three series: As-melted, IEX Proto1, Reverse IEX1.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE    = Path(__file__).parent.parent
AM_DIR  = BASE / "STAGE1_MELTQUENCH/data/asmelted_723K"
IEX1_DIR= BASE / "STAGE3_IEX_PROTO1/data/iox1_723K"
REV_DIR = BASE / "STAGE11_REVERSE_IEX1/data/rev_iex1_723K"

NA   = 6.02214076e23
N_OX = 5000           # oxide units per simulation box

# ── Helper ───────────────────────────────────────────────────────────────────

def box_volume(path):
    """Return simulation box volume (Å³) from a LAMMPS data file."""
    L = {}
    with open(path, errors='replace') as f:
        for line in f:
            s = line.strip()
            if 'xlo xhi' in s:
                p = s.split(); L['x'] = float(p[1]) - float(p[0])
            elif 'ylo yhi' in s:
                p = s.split(); L['y'] = float(p[1]) - float(p[0])
            elif 'zlo zhi' in s:
                p = s.split(); L['z'] = float(p[1]) - float(p[0])
            if len(L) == 3:
                break
    return L['x'] * L['y'] * L['z']

def molar_vol(path):
    return box_volume(path) * 1e-24 * NA / N_OX   # cm³/mol

def collect(directory, fname_fn, x_list, reps=(1,2,3)):
    xs, means, stds = [], [], []
    for x in x_list:
        vs = []
        for r in reps:
            p = directory / fname_fn(x, r)
            if p.exists():
                vs.append(molar_vol(p))
        if vs:
            xs.append(x)
            means.append(np.mean(vs))
            stds.append(np.std(vs))
    return np.array(xs), np.array(means), np.array(stds)

# ── Collect data ─────────────────────────────────────────────────────────────

x_am, vm_am, sd_am = collect(
    AM_DIR,
    lambda x, r: f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
    x_list=[0, 1, 3, 6, 9, 12, 15]
)

x_iex, vm_iex, sd_iex = collect(
    IEX1_DIR,
    lambda x, r: f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
    x_list=[1, 3, 6, 9, 12, 15]
)

x_rev, vm_rev, sd_rev = collect(
    REV_DIR,
    lambda x, r: f"REIEX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
    x_list=[1, 3, 6, 9, 12, 15]
)

# ── Validated categorical palette (slots 1–3) ─────────────────────────────────
# Light: #2a78d6 (blue), #eb6834 (orange), #1baf7a (aqua)
# Dark : #3987e5,         #d95926,           #199e70
# REIEX1 (aqua) is sub-3:1 on light surface → relief via distinct marker+linestyle.

C_AM   = '#2a78d6'   # blue   — slot 1
C_IEX1 = '#eb6834'   # orange — slot 2
C_REV  = '#1baf7a'   # aqua   — slot 3

# ── Figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

EKW = dict(elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)

ax.errorbar(x_am,  vm_am,  yerr=sd_am,
            color=C_AM,   marker='o', ms=7.5, lw=1.8, ls='-',
            label='As-melted', **EKW)
ax.errorbar(x_iex, vm_iex, yerr=sd_iex,
            color=C_IEX1, marker='s', ms=7.0, lw=1.8, ls='--',
            label='IEX Proto1', **EKW)
ax.errorbar(x_rev, vm_rev, yerr=sd_rev,
            color=C_REV,  marker='^', ms=7.5, lw=1.8, ls=':',
            label='Reverse IEX1', **EKW)

# Reference line: mean of AM x=0 replicas
am0_val = vm_am[x_am == 0][0]
am0_sd  = sd_am[x_am == 0][0]
ax.axhline(am0_val, color=C_AM, lw=0.8, ls='-', alpha=0.30, zorder=1)
ax.axhspan(am0_val - am0_sd, am0_val + am0_sd,
           color=C_AM, alpha=0.07, zorder=0)

# Direct label for REIEX1 (relief rule — aqua is sub-3:1 contrast on light)
ax.annotate('Reverse IEX1',
            xy=(x_rev[-1], vm_rev[-1]),
            xytext=(14.2, vm_rev[-1] - 0.18),
            fontsize=8.5, color='#52514e',
            ha='right', va='top')

# ── Axes & chrome ─────────────────────────────────────────────────────────────

ax.set_xlabel('x (mol% K₂O)', fontsize=11, color='#0b0b0b')
ax.set_ylabel('Molar volume (cm³ mol⁻¹)', fontsize=11, color='#0b0b0b')

ax.set_xlim(-0.8, 16.2)
ax.set_ylim(22.72, 26.40)
ax.set_xticks([0, 1, 3, 6, 9, 12, 15])
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

ax.tick_params(axis='both', labelsize=9.5, color='#c3c2b7', labelcolor='#0b0b0b')
ax.tick_params(axis='y', which='minor', left=True, length=3)

ax.spines[['top', 'right']].set_visible(False)
ax.spines['left'].set_color('#c3c2b7')
ax.spines['bottom'].set_color('#c3c2b7')
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

ax.grid(axis='y', color='#e1e0d9', lw=0.6, zorder=0)

# Note on temperature
ax.text(0.98, 0.02, '723 K', transform=ax.transAxes,
        fontsize=8.5, color='#898781', ha='right', va='bottom')

# Legend
leg = ax.legend(fontsize=9.5, frameon=True, loc='upper left',
                framealpha=0.92, edgecolor='#e1e0d9')
leg.get_frame().set_linewidth(0.6)

fig.tight_layout(pad=1.2)

out_base = Path(__file__).with_suffix('')
fig.savefig(out_base.with_suffix('.pdf'), dpi=300, bbox_inches='tight',
            facecolor='#fcfcfb')
fig.savefig(out_base.with_suffix('.png'), dpi=150, bbox_inches='tight',
            facecolor='#fcfcfb')
print(f"Saved: {out_base}.pdf / .png")
plt.close(fig)
