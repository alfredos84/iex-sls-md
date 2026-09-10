"""
fig_local_density_XX100.py
Local atomic number density and O-coordination number around alkali ions.
χ = 100%  →  IEX1 puro K (sin Na):  CaO xt=15,  Ca-free xt=25.

Na(AM)  : x=0  glass (puro Na)
K(AM)   : x=15 (CaO) / x=25 (Ca-free) glass (puro K as-melted)
K(IEX1) : xt=15 / xt=25 glass (ion-exchanged, puro K)

Methodology (Vargheese 2014):
  r_cut = 3.8 Å (first min K-O g(r))
  ρ_local = N_neighbors / (4/3 π r_cut³)   [Å⁻³]
  CN_O    = N_O neighbors within r_cut

4-panel figure: 2 systems × {density, CN_O} — paneles cuadrados.
Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree

HERE  = Path(__file__).parent
BASE  = HERE.parent

CAO_AM_DATA   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
CAO_IEX1_DATA = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
NOCA_AM_DATA  = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
NOCA_IEX1_DATA= BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"

RCUT_NA = 3.21
RCUT_K  = 3.77
V_NA    = (4.0 / 3.0) * np.pi * RCUT_NA**3
V_K     = (4.0 / 3.0) * np.pi * RCUT_K**3
TYPE_O, TYPE_NA, TYPE_K = 2, 4, 5


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_data(path):
    box, lo = np.zeros(3), np.zeros(3)
    atoms, section = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if 'xlo xhi' in s:
                lo[0], hi = float(s.split()[0]), float(s.split()[1])
                box[0] = hi - lo[0]; continue
            elif 'ylo yhi' in s:
                lo[1], hi = float(s.split()[0]), float(s.split()[1])
                box[1] = hi - lo[1]; continue
            elif 'zlo zhi' in s:
                lo[2], hi = float(s.split()[0]), float(s.split()[1])
                box[2] = hi - lo[2]; continue
            if s[0].isalpha():
                section = 'atoms' if s.startswith('Atoms') else s.split()[0].lower()
                continue
            if section == 'atoms':
                p = s.split()
                atoms.append((int(p[1]), float(p[3]), float(p[4]), float(p[5])))
    atoms = np.array(atoms)
    return atoms[:, 1:4] - lo, atoms[:, 0].astype(int), box


def local_quantities(pos, types, box, center_type):
    rcut = RCUT_NA if center_type == TYPE_NA else RCUT_K
    vsph = V_NA    if center_type == TYPE_NA else V_K
    tree = cKDTree(pos, boxsize=box)
    centers = pos[types == center_type]
    rho_list, cn_list = [], []
    for neighbors in tree.query_ball_point(centers, rcut):
        rho_list.append((len(neighbors) - 1) / vsph)
        cn_list.append(int(np.sum(types[neighbors] == TYPE_O)))
    return np.array(rho_list), np.array(cn_list)


def load_am(am_dir, x_val, center_type, tmpl="AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"):
    rho_all, cn_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(am_dir / tmpl.format(x=x_val, r=r))
        rho, cn = local_quantities(pos, types, box, center_type)
        rho_all.extend(rho); cn_all.extend(cn)
    return np.array(rho_all), np.array(cn_all)


def load_iex1(iex_dir, x_val, center_type, tmpl="IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"):
    rho_all, cn_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(iex_dir / tmpl.format(x=x_val, r=r))
        rho, cn = local_quantities(pos, types, box, center_type)
        rho_all.extend(rho); cn_all.extend(cn)
    return np.array(rho_all), np.array(cn_all)


# ── Compute ───────────────────────────────────────────────────────────────────

print("CaO — Na(AM) from x=0 ...")
cao_rho_na, cao_cn_na = load_am(CAO_AM_DATA, 0,  TYPE_NA)

print("CaO — K(AM) from x=15 ...")
cao_rho_ka, cao_cn_ka = load_am(CAO_AM_DATA, 15, TYPE_K)

print("CaO — K(IEX1) xt=15 ...")
cao_rho_ki, cao_cn_ki = load_iex1(CAO_IEX1_DATA, 15, TYPE_K)

print("Ca-free — Na(AM) from x=0 ...")
noca_rho_na, noca_cn_na = load_am(NOCA_AM_DATA, 0,  TYPE_NA)

print("Ca-free — K(AM) from x=25 ...")
noca_rho_ka, noca_cn_ka = load_am(NOCA_AM_DATA, 25, TYPE_K)

print("Ca-free — K(IEX1) xt=25 ...")
noca_rho_ki, noca_cn_ki = load_iex1(NOCA_IEX1_DATA, 25, TYPE_K)

print(f"\n{'':30s} {'ρ_mean':>8} {'CN_O':>8}")
for lbl, rho, cn in [
    ("CaO — Na(AM)",      cao_rho_na, cao_cn_na),
    ("CaO — K(AM)",       cao_rho_ka, cao_cn_ka),
    ("CaO — K(IEX1)",     cao_rho_ki, cao_cn_ki),
    ("Ca-free — Na(AM)",  noca_rho_na,noca_cn_na),
    ("Ca-free — K(AM)",   noca_rho_ka,noca_cn_ka),
    ("Ca-free — K(IEX1)", noca_rho_ki,noca_cn_ki),
]:
    print(f"  {lbl:28s}  {rho.mean():.4f}  {cn.mean():.2f}")


# ── Figure ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       17,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_NA_AM = '#1a3a5c'
C_K_AM  = '#2ca0c4'
C_K_IEX = '#c0392b'

def norm_hist(data, bins):
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = counts / counts.max() if counts.max() > 0 else counts.astype(float)
    return centers, norm

def plot_nonzero(ax, centers, norm, **kw):
    mask = norm > 0
    ax.plot(centers[mask], norm[mask], **kw)

N_BINS_RHO = 40
all_rho  = np.concatenate([cao_rho_na, cao_rho_ka, cao_rho_ki,
                            noca_rho_na, noca_rho_ka, noca_rho_ki])
rho_bins = np.linspace(0, all_rho.max() * 1.05, N_BINS_RHO + 1)
RHO_MAX  = rho_bins[-1]

CN_MAX  = int(max(cao_cn_na.max(), cao_cn_ka.max(),
                  noca_cn_na.max(), noca_cn_ka.max())) + 1
cn_bins = np.arange(-0.5, CN_MAX + 1.5, 1.0)

# Panels cuadrados: 2×2, cada uno ~6.2×6.2 cm equivalente en pulgadas
fig, axes = plt.subplots(2, 2, figsize=(12.5, 12.5))

SERIES = [
    ('Na (As-Melted)',   C_NA_AM, '-'),
    ('K (As-Melted)',    C_K_AM,  '--'),
    ('K (Ion-Exchanged)',C_K_IEX, ':'),
]

# ── Row 0: ρ_local ─────────────────────────────────────────────────────────────
for col, (rho_na, rho_ka, rho_ki, title) in enumerate([
    (cao_rho_na,  cao_rho_ka,  cao_rho_ki,  r'10CaO — $\chi$ = 100%'),
    (noca_rho_na, noca_rho_ka, noca_rho_ki, r'Ca-free — $\chi$ = 100%'),
]):
    ax = axes[0, col]
    for data, (lbl, color, ls) in zip([rho_na, rho_ka, rho_ki], SERIES):
        cx, ny = norm_hist(data, rho_bins)
        plot_nonzero(ax, cx, ny, color=color, ls=ls, lw=1.8,
                     marker='s', ms=4, label=lbl)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'Number Density (Å$^{-3}$)', fontsize=16)
    ax.set_ylabel('Normalized Amplitude', fontsize=16)
    ax.set_xlim(0, RHO_MAX)
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=14, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    if col == 0:
        ax.legend(fontsize=13, frameon=True, framealpha=0.9,
                  edgecolor='#cccccc', loc='upper left')

# ── Row 1: CN_O ───────────────────────────────────────────────────────────────
for col, (cn_na, cn_ka, cn_ki, title) in enumerate([
    (cao_cn_na,  cao_cn_ka,  cao_cn_ki,  r'10CaO — $\chi$ = 100%'),
    (noca_cn_na, noca_cn_ka, noca_cn_ki, r'Ca-free — $\chi$ = 100%'),
]):
    ax = axes[1, col]
    for data, (lbl, color, ls) in zip([cn_na, cn_ka, cn_ki], SERIES):
        cx, ny = norm_hist(data, cn_bins)
        plot_nonzero(ax, cx, ny, color=color, ls=ls, lw=1.8,
                     marker='s', ms=4, label=lbl)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(r'O Coordination Number', fontsize=16)
    ax.set_ylabel('Normalized Amplitude', fontsize=16)
    ax.set_xlim(-0.5, CN_MAX + 0.5)
    ax.set_ylim(0, 1.12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=14, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)

fig.tight_layout()
fig.savefig(HERE / 'fig_local_density_XX100.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_local_density_XX100.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {HERE / 'fig_local_density_XX100.pdf'}")
plt.close(fig)
