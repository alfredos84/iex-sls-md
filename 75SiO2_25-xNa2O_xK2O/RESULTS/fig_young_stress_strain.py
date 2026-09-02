"""
fig_young_stress_strain.py — PMMCS
Young's modulus via stress-strain: cubic polynomial fit to sigma(eps).
E0 = linear coefficient at zero strain (Pedone et al.).
Averaged over 3 replicas x 3 directions = 9 values per composition.
LAMMPS metal units: pressure in bars -> GPa = bar * 1e-4.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
from pathlib import Path
import re

LOGS_DIR = Path(__file__).parent.parent / "STAGE3_STRESS_STRAIN" / "logs"
OUT_DIR  = Path(__file__).parent

x_vals    = [0, 6, 12, 15]
rc_labels = ['rc5p5', 'rc6p5', 'rc8p0', 'rc10p0']
dirs      = ['x', 'y', 'z']
BAR_TO_GPA = 1.0e-4

def parse_young_log(fpath, direction):
    with open(fpath) as f:
        lines = f.readlines()

    blocks = []
    current = []
    cols = {}
    in_data = False

    for line in lines:
        if line.strip().startswith('Step') and 'Pxx' in line:
            labels = line.split()
            cols = {k: labels.index(k) for k in ('Pxx','Pyy','Pzz','Lx','Ly','Lz')}
            in_data = True
            current = []
        elif in_data and re.match(r'^\s*\d', line):
            current.append(line.split())
        elif in_data and 'Loop time' in line:
            blocks.append((cols, current))
            in_data = False

    if len(blocks) < 2:
        return None, None

    cols, data_lines = blocks[-1]
    if not data_lines:
        return None, None

    data = np.array([[float(v) for v in row] for row in data_lines])
    lx = data[:, cols['Lx']];  ly = data[:, cols['Ly']];  lz = data[:, cols['Lz']]
    pxx = data[:, cols['Pxx']]; pyy = data[:, cols['Pyy']]; pzz = data[:, cols['Pzz']]

    if direction == 'x':
        elon = lx;  A = ly * lz;  P = pxx
    elif direction == 'y':
        elon = ly;  A = lx * lz;  P = pyy
    else:
        elon = lz;  A = lx * ly;  P = pzz

    strain = (elon - elon[0]) / elon[0]
    stress = (-P) * (A / A[0]) * BAR_TO_GPA
    return strain, stress

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

# ── Collect E values per rc ───────────────────────────────────────────────────
E_mean = {rc: [] for rc in rc_labels}
E_std  = {rc: [] for rc in rc_labels}

for x in x_vals:
    for rc in rc_labels:
        E_all = []
        for r in [1, 2, 3]:
            for d in dirs:
                fname = LOGS_DIR / f"young_x{x}_r{r}_dir_{d}_PMMCS_{rc}.lammps"
                if not fname.exists():
                    print(f"  MISSING: {fname.name}")
                    continue
                strain, stress = parse_young_log(fname, d)
                if strain is None or len(strain) < 10:
                    continue
                try:
                    popt, _ = curve_fit(cubic, strain, stress,
                                        p0=[1000, -100, 70, 0], maxfev=5000)
                    E_all.append(popt[2])
                except RuntimeError:
                    print(f"  FIT FAIL: {fname.name}")
        E_mean[rc].append(np.mean(E_all) if E_all else np.nan)
        E_std[rc].append(np.std(E_all, ddof=1) if len(E_all) > 1 else 0.0)

# ── Figure ────────────────────────────────────────────────────────────────────
colors  = {'rc5p5': '#2a78d6', 'rc6p5': '#1baf7a', 'rc8p0': '#eda100', 'rc10p0': '#9b30d9'}
markers = {'rc5p5': 'o',       'rc6p5': '^',        'rc8p0': 'D', 'rc10p0': 'v'}
labels  = {'rc5p5': r'$r_c=5.5$ Å', 'rc6p5': r'$r_c=6.5$ Å', 'rc8p0': r'$r_c=8.0$ Å', 'rc10p0': r'$r_c=10.0$ Å'}

fig, ax = plt.subplots(figsize=(5.5, 4.2))

for rc in rc_labels:
    ax.errorbar(x_vals, E_mean[rc], yerr=E_std[rc],
                color=colors[rc], lw=1.5, marker=markers[rc], ms=6,
                capsize=3, capthick=1, elinewidth=0.8, zorder=4,
                label=labels[rc])

ax.set_xlabel(r'$x$ (mol% K$_2$O substituting Na$_2$O)', fontsize=10)
ax.set_ylabel(r"Young's modulus $E_0$ (GPa)", fontsize=10)
ax.set_title("PMMCS: Young's modulus via stress-strain\n"
             r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO, 300 K',
             fontsize=9)
ax.set_xticks(x_vals)
ax.set_xlim(-1, 16)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
ax.legend(frameon=True, framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_young_stress_strain.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_young_stress_strain.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'fig_young_stress_strain.pdf'}")

print(f"\n{'x':>4}  {'rc':>8}  {'E0_mean (GPa)':>14}  {'E0_std':>8}")
for x, i in zip(x_vals, range(4)):
    for rc in rc_labels:
        print(f"{x:>4}  {rc:>8}  {E_mean[rc][i]:>14.2f}  {E_std[rc][i]:>8.2f}")

plt.show()
