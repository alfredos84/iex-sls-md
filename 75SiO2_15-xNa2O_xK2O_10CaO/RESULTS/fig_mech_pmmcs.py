"""
fig_mech_pmmcs.py
Young's modulus (stress-strain) and Bulk modulus (pressure ramp) — PMMCS, 4 cutoffs.
Side-by-side 1×2 figure.
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
    blocks, current, cols, in_data = [], [], {}, False
    with open(fpath) as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith('Step') and 'Pxx' in line:
            lbl = line.split()
            cols = {k: lbl.index(k) for k in ('Pxx','Pyy','Pzz','Lx','Ly','Lz')}
            in_data = True; current = []
        elif in_data and re.match(r'^\s*\d', line):
            current.append(line.split())
        elif in_data and 'Loop time' in line:
            blocks.append((cols, current)); in_data = False
    if len(blocks) < 2:
        return None, None
    cols, data_lines = blocks[-1]
    if not data_lines:
        return None, None
    data = np.array([[float(v) for v in row] for row in data_lines])
    lx, ly, lz = data[:, cols['Lx']], data[:, cols['Ly']], data[:, cols['Lz']]
    pxx, pyy, pzz = data[:, cols['Pxx']], data[:, cols['Pyy']], data[:, cols['Pzz']]
    if direction == 'x':
        elon, A, P = lx, ly * lz, pxx
    elif direction == 'y':
        elon, A, P = ly, lx * lz, pyy
    else:
        elon, A, P = lz, lx * ly, pzz
    strain = (elon - elon[0]) / elon[0]
    stress = (-P) * (A / A[0]) * BAR_TO_GPA
    return strain, stress


def parse_bulk_log(fpath):
    sections, cur, c_press, c_vol, in_data = [], [], None, None, False
    with open(fpath) as f:
        lines = f.readlines()
    for line in lines:
        if line.strip().startswith('Step') and 'Press' in line and 'Volume' in line:
            hdr = line.split()
            c_press, c_vol = hdr.index('Press'), hdr.index('Volume')
            in_data = True; cur = []
        elif in_data and re.match(r'^\s*\d', line):
            cur.append(line.split())
        elif in_data and 'Loop time' in line:
            sections.append(np.array([[float(v) for v in row] for row in cur]))
            in_data = False
    if len(sections) < 2:
        return None, None
    return sections[1][:, c_press], sections[1][:, c_vol]


def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d


# ── Collect E and B ───────────────────────────────────────────────────────────
E_mean = {rc: [] for rc in rc_labels}
E_std  = {rc: [] for rc in rc_labels}
B_mean = {rc: [] for rc in rc_labels}
B_std  = {rc: [] for rc in rc_labels}

for x in x_vals:
    for rc in rc_labels:
        E_all, B_all = [], []
        for r in [1, 2, 3]:
            # Young
            for d in dirs:
                fname = LOGS_DIR / f"young_x{x}_r{r}_dir_{d}_PMMCS_{rc}.lammps"
                if not fname.exists():
                    continue
                strain, stress = parse_young_log(fname, d)
                if strain is None or len(strain) < 10:
                    continue
                try:
                    popt, _ = curve_fit(cubic, strain, stress,
                                        p0=[1000, -100, 70, 0], maxfev=5000)
                    E_all.append(popt[2])
                except RuntimeError:
                    pass
            # Bulk
            fname = LOGS_DIR / f"bulk_x{x}_r{r}_PMMCS_{rc}.lammps"
            if fname.exists():
                P, V = parse_bulk_log(fname)
                if P is not None:
                    coeffs = np.polyfit(P, V, 1)
                    if coeffs[0] < 0:
                        B_all.append(-coeffs[1] / coeffs[0] * BAR_TO_GPA)
        E_mean[rc].append(np.mean(E_all) if E_all else np.nan)
        E_std[rc].append(np.std(E_all, ddof=1) if len(E_all) > 1 else 0.0)
        B_mean[rc].append(np.mean(B_all) if B_all else np.nan)
        B_std[rc].append(np.std(B_all, ddof=1) if len(B_all) > 1 else 0.0)

# ── Palette ───────────────────────────────────────────────────────────────────
colors  = {'rc5p5': '#2a78d6', 'rc6p5': '#1baf7a',
           'rc8p0': '#eda100', 'rc10p0': '#9b30d9'}
markers = {'rc5p5': 'o', 'rc6p5': '^', 'rc8p0': 'D', 'rc10p0': 'v'}
labels  = {'rc5p5': r'$r_c=5.5$ Å', 'rc6p5': r'$r_c=6.5$ Å',
           'rc8p0': r'$r_c=8.0$ Å', 'rc10p0': r'$r_c=10.0$ Å'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
kw = dict(lw=1.5, ms=6, capsize=3, capthick=1, elinewidth=0.8, zorder=4)

for rc in rc_labels:
    ax1.errorbar(x_vals, E_mean[rc], yerr=E_std[rc],
                 color=colors[rc], marker=markers[rc], label=labels[rc], **kw)
    ax2.errorbar(x_vals, B_mean[rc], yerr=B_std[rc],
                 color=colors[rc], marker=markers[rc], label=labels[rc], **kw)

for ax, ylabel in [(ax1, r"Young's modulus $E_0$ (GPa)"),
                   (ax2, r"Bulk modulus $B$ (GPa)")]:
    ax.set_xlabel(r'$x$ (mol% K$_2$O)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x_vals)
    ax.set_xlim(-1, 16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#e8e8e8', zorder=0)
    ax.tick_params(labelsize=8.5)

handles, labels_ = ax1.get_legend_handles_labels()
fig.legend(handles, labels_, loc='lower center', ncol=4, fontsize=8.5,
           frameon=True, framealpha=0.9, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.06))

fig.suptitle('PMMCS — Young\'s modulus (stress-strain) and Bulk modulus (pressure ramp)\n'
             r'75SiO$_2$·(15$-x$)Na$_2$O·$x$K$_2$O·10CaO, 300 K',
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_mech_pmmcs.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_mech_pmmcs.png', dpi=150, bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'fig_mech_pmmcs.pdf'}")

print(f"\n{'rc':>8}  {'x':>4}  {'E0 (GPa)':>10}  {'B (GPa)':>10}")
for rc in rc_labels:
    for i, x in enumerate(x_vals):
        print(f"{rc:>8}  {x:>4}  {E_mean[rc][i]:>10.2f}  {B_mean[rc][i]:>10.2f}")
    print()

plt.show()
