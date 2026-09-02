"""
fig_molar_volume_reversal2.py
Molar volume (cm³/mol) at 723 K vs x_parent (mol% K₂O in starting AM glass).
Three series: As-melted, IEX Proto2, Reverse IEX2.
Analogous to Tandia et al. J. Non-Cryst. Solids 358 (2012) Fig. 1.
If the process is elastic, REIEX2 must overlap the AM curve.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE     = Path(__file__).parent.parent
AM_DIR   = BASE / "STAGE1_MELTQUENCH/data/asmelted_723K"
IEX2_DIR = BASE / "STAGE4_IEX_PROTO2/data/iox2_723K"
REV2_DIR = BASE / "STAGE12_REVERSE_IEX2/data/rev_iex2_723K"

NA   = 6.02214076e23
N_OX = 5000

# ── Helper ───────────────────────────────────────────────────────────────────

def box_volume(path):
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
    return box_volume(path) * 1e-24 * NA / N_OX

def collect(directory, fname_fn, x_list, reps=(1, 2, 3)):
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

# ── Collect data ──────────────────────────────────────────────────────────────

X_LIST = [0, 1, 3, 6, 9, 12]

x_am, vm_am, sd_am = collect(
    AM_DIR,
    lambda x, r: f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
    x_list=X_LIST
)
x_iex2, vm_iex2, sd_iex2 = collect(
    IEX2_DIR,
    lambda x, r: f"IOX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
    x_list=X_LIST
)
x_rev2, vm_rev2, sd_rev2 = collect(
    REV2_DIR,
    lambda x, r: f"REIEX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
    x_list=X_LIST
)

# Print table
print(f"{'x_p':>4}  {'AM':>8}  {'IEX2':>8}  {'REIEX2':>8}  {'REIEX2-AM':>10}")
for i, x in enumerate(x_am):
    xi = np.where(x_iex2 == x)[0]
    xr = np.where(x_rev2 == x)[0]
    vm_i = vm_iex2[xi[0]] if len(xi) else float('nan')
    vm_r = vm_rev2[xr[0]] if len(xr) else float('nan')
    print(f"{x:>4}  {vm_am[i]:>8.4f}  {vm_i:>8.4f}  {vm_r:>8.4f}  {vm_r-vm_am[i]:>+10.4f}")

# ── Validated categorical palette (slots 1–3) ─────────────────────────────────
C_AM   = '#2a78d6'   # blue   — slot 1
C_IEX2 = '#eb6834'   # orange — slot 2
C_REV2 = '#1baf7a'   # aqua   — slot 3

# ── Figure ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('#fcfcfb')
ax.set_facecolor('#fcfcfb')

EKW = dict(elinewidth=1.1, capsize=3.5, capthick=1.1, zorder=3)

ax.errorbar(x_am,   vm_am,   yerr=sd_am,
            color=C_AM,   marker='o', ms=7.5, lw=1.8, ls='-',
            label='As-melted', **EKW)
ax.errorbar(x_iex2, vm_iex2, yerr=sd_iex2,
            color=C_IEX2, marker='s', ms=7.0, lw=1.8, ls='--',
            label='IEX Proto2', **EKW)
ax.errorbar(x_rev2, vm_rev2, yerr=sd_rev2,
            color=C_REV2, marker='^', ms=7.5, lw=1.8, ls=':',
            label='Reverse IEX2', **EKW)

# Direct label for REIEX2 (relief rule — aqua sub-3:1 on light surface)
ax.annotate('Reverse IEX2',
            xy=(x_rev2[-1], vm_rev2[-1]),
            xytext=(11.6, vm_rev2[-1] + 0.15),
            fontsize=8.5, color='#52514e', ha='right', va='bottom')

# ── Axes & chrome ─────────────────────────────────────────────────────────────

ax.set_xlabel('x (mol% K₂O in as-melted glass)', fontsize=11, color='#0b0b0b')
ax.set_ylabel('Molar volume (cm³ mol⁻¹)', fontsize=11, color='#0b0b0b')

all_vals = np.concatenate([vm_am, vm_iex2, vm_rev2])
ylo = all_vals.min() - 0.25
yhi = all_vals.max() + 0.25

ax.set_xlim(-0.6, 13.2)
ax.set_ylim(ylo, yhi)
ax.set_xticks(X_LIST)
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

ax.text(0.98, 0.02, '723 K', transform=ax.transAxes,
        fontsize=8.5, color='#898781', ha='right', va='bottom')

leg = ax.legend(fontsize=9.5, frameon=True, loc='upper left',
                framealpha=0.92, edgecolor='#e1e0d9')
leg.get_frame().set_linewidth(0.6)

fig.tight_layout(pad=1.2)

out_base = Path(__file__).with_suffix('')
fig.savefig(out_base.with_suffix('.pdf'), dpi=300, bbox_inches='tight',
            facecolor='#fcfcfb')
fig.savefig(out_base.with_suffix('.png'), dpi=150, bbox_inches='tight',
            facecolor='#fcfcfb')
print(f"\nSaved: {out_base}.pdf / .png")
plt.close(fig)
