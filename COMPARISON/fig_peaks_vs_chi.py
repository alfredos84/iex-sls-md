"""
fig_peaks_vs_chi.py
Pico del CN_O y del volumen de Voronoi vs chi — Na(AM), K(AM), K(IEX1).
2 figuras, cada una con 2 paneles: 10CaO | Ca-free. Ajuste lineal. Sin leyenda.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

HERE  = Path(__file__).parent
BASE  = HERE.parent

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
VORO_DIR      = HERE / "VORONOI"

RCUT_NA  = 3.21
RCUT_K   = 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0   # 3.49 Å — promedio Na/K para sitio IEX
TYPE_O, TYPE_NA, TYPE_K = 2, 4, 5

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}

CHI = np.array(sorted(XX_CONFIG.keys()), dtype=float)


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
                section = 'atoms' if s.startswith('Atoms') else None
                continue
            if section == 'atoms':
                p = s.split()
                atoms.append((int(p[1]), float(p[3]), float(p[4]), float(p[5])))
    atoms = np.array(atoms)
    return atoms[:, 1:4] - lo, atoms[:, 0].astype(int), box


def cn_around(pos, types, box, center_type, rcut=None):
    if rcut is None:
        rcut = RCUT_NA if center_type == TYPE_NA else RCUT_K
    tree = cKDTree(pos, boxsize=box)
    centers = pos[types == center_type]
    cn_list = []
    for neighbors in tree.query_ball_point(centers, rcut):
        cn_list.append(int(np.sum(types[neighbors] == TYPE_O)))
    return np.array(cn_list)


def load_am_cn(am_dir, x, ctype):
    cn_all = []
    for r in (1, 2, 3):
        pos, types, box = read_data(am_dir / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data")
        cn_all.extend(cn_around(pos, types, box, ctype))
    return np.array(cn_all)


def load_iex1_cn(iex_dir, x, ctype, rcut=None):
    cn_all = []
    for r in (1, 2, 3):
        pos, types, box = read_data(iex_dir / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data")
        cn_all.extend(cn_around(pos, types, box, ctype, rcut=rcut))
    return np.array(cn_all)


def read_vol(path):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"):
                in_atoms = True; continue
            if s.startswith("ITEM:"):
                in_atoms = False; continue
            if in_atoms and s:
                parts = s.split()
                data[int(parts[0])] = (int(parts[1]), float(parts[2]))
    return data


def pool_vol(tag_list, center_type):
    vols = []
    for tag in tag_list:
        vol_data = read_vol(VORO_DIR / f"{tag}_vol.dat")
        vols.extend(v for (t, v) in vol_data.values() if t == center_type)
    return np.array(vols)


# ── Peak extraction ───────────────────────────────────────────────────────────

def cn_peak(cn_arr):
    counts = np.bincount(cn_arr)
    return float(np.argmax(counts))


def vol_peak(vol_arr):
    kde = gaussian_kde(vol_arr, bw_method='silverman')
    x_grid = np.linspace(vol_arr.min(), vol_arr.max(), 2000)
    return x_grid[np.argmax(kde(x_grid))]


# ── Recopilar picos ───────────────────────────────────────────────────────────

cn_pk    = {'cao': {'na': [], 'ka': [], 'ki': []}, 'noca': {'na': [], 'ka': [], 'ki': []}}
cn_pk_av = {'cao': {'na': [], 'ka': [], 'ki': []}, 'noca': {'na': [], 'ka': [], 'ki': []}}
vol_pk   = {'cao': {'na': [], 'ka': [], 'ki': []}, 'noca': {'na': [], 'ka': [], 'ki': []}}

for XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca) in sorted(XX_CONFIG.items()):
    print(f"XX={XX}%...")

    cn_pk['cao']['na'].append(cn_peak(load_am_cn(CAO_AM_DIR,    xna_cao,  TYPE_NA)))
    cn_pk['cao']['ka'].append(cn_peak(load_am_cn(CAO_AM_DIR,    xka_cao,  TYPE_K)))
    cn_pk['cao']['ki'].append(cn_peak(load_iex1_cn(CAO_IEX1_DIR, xt_cao,  TYPE_K)))
    cn_pk['noca']['na'].append(cn_peak(load_am_cn(NOCA_AM_DIR,   xna_noca, TYPE_NA)))
    cn_pk['noca']['ka'].append(cn_peak(load_am_cn(NOCA_AM_DIR,   xka_noca, TYPE_K)))
    cn_pk['noca']['ki'].append(cn_peak(load_iex1_cn(NOCA_IEX1_DIR, xt_noca, TYPE_K)))

    # IEX con r_cut promedio (RCUT_IEX) — Na y K(AM) sin cambio
    cn_pk_av['cao']['na'].append(cn_pk['cao']['na'][-1])
    cn_pk_av['cao']['ka'].append(cn_pk['cao']['ka'][-1])
    cn_pk_av['cao']['ki'].append(cn_peak(load_iex1_cn(CAO_IEX1_DIR, xt_cao,  TYPE_K, rcut=RCUT_IEX)))
    cn_pk_av['noca']['na'].append(cn_pk['noca']['na'][-1])
    cn_pk_av['noca']['ka'].append(cn_pk['noca']['ka'][-1])
    cn_pk_av['noca']['ki'].append(cn_peak(load_iex1_cn(NOCA_IEX1_DIR, xt_noca, TYPE_K, rcut=RCUT_IEX)))

    tags_na_cao  = [f"cao_am_x{xna_cao}_r{r}"    for r in (1, 2, 3)]
    tags_ka_cao  = [f"cao_am_x{xka_cao}_r{r}"    for r in (1, 2, 3)]
    tags_ki_cao  = [f"cao_iex1_xt{xt_cao}_r{r}"  for r in (1, 2, 3)]
    tags_na_noca = [f"noca_am_x{xna_noca}_r{r}"  for r in (1, 2, 3)]
    tags_ka_noca = [f"noca_am_x{xka_noca}_r{r}"  for r in (1, 2, 3)]
    tags_ki_noca = [f"noca_iex1_xt{xt_noca}_r{r}" for r in (1, 2, 3)]

    vol_pk['cao']['na'].append(vol_peak(pool_vol(tags_na_cao,  TYPE_NA)))
    vol_pk['cao']['ka'].append(vol_peak(pool_vol(tags_ka_cao,  TYPE_K)))
    vol_pk['cao']['ki'].append(vol_peak(pool_vol(tags_ki_cao,  TYPE_K)))
    vol_pk['noca']['na'].append(vol_peak(pool_vol(tags_na_noca, TYPE_NA)))
    vol_pk['noca']['ka'].append(vol_peak(pool_vol(tags_ka_noca, TYPE_K)))
    vol_pk['noca']['ki'].append(vol_peak(pool_vol(tags_ki_noca, TYPE_K)))

for d in (cn_pk, cn_pk_av, vol_pk):
    for sys in ('cao', 'noca'):
        for s in ('na', 'ka', 'ki'):
            d[sys][s] = np.array(d[sys][s], dtype=float)

# ── Plot helpers ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       14,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_NA = '#1a3a5c'
C_KA = '#2ca0c4'
C_KI = '#c0392b'
SERIES = [
    ('na', C_NA, 'o', 'Na (As-Melted)'),
    ('ka', C_KA, 's', 'K (As-Melted)'),
    ('ki', C_KI, '^', 'K (Ion-Exchanged)'),
]


def plot_with_fit(ax, chi, y, color, marker, label):
    ax.plot(chi, y, color=color, marker=marker, ls='none',
            ms=6, mew=0.8, label=label, zorder=3)
    m, b = np.polyfit(chi, y, 1)
    xf = np.linspace(chi.min(), chi.max(), 200)
    ax.plot(xf, m * xf + b, color=color, lw=1.4, label='_nolegend_', zorder=2)


def decorate(ax, ylabel, ylim=None):
    ax.set_xlabel(r'$\chi$ (%)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=11, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)


# ── Figura 1: CN_O peaks vs χ ─────────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(9, 4.5))
for col, (sys, title) in enumerate([('cao', r'10CaO'), ('noca', r'Ca-free')]):
    ax = axes1[col]
    for key, color, marker, label in SERIES:
        plot_with_fit(ax, CHI, cn_pk[sys][key], color, marker, label)
    ax.set_title(title, fontsize=13)
    decorate(ax, r'CN$_\mathrm{O}$ peak', ylim=(4, 11))

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
            ncol=3, fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
fig1.tight_layout(rect=[0, 0, 1, 0.94])
out1 = HERE / 'fig_cn_peaks_vs_chi.pdf'
fig1.savefig(out1, dpi=300, bbox_inches='tight')
fig1.savefig(out1.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out1.name}")
plt.close(fig1)

# ── Figura 2: V_Vor peaks vs χ ────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.5))
for col, (sys, title) in enumerate([('cao', r'10CaO'), ('noca', r'Ca-free')]):
    ax = axes2[col]
    for key, color, marker, label in SERIES:
        plot_with_fit(ax, CHI, vol_pk[sys][key], color, marker, label)
    ax.set_title(title, fontsize=13)
    decorate(ax, r'$V_\mathrm{Vor}$ peak (Å$^3$)', ylim=(16, 24))

handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
            ncol=3, fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
fig2.tight_layout(rect=[0, 0, 1, 0.94])
out2 = HERE / 'fig_voronoi_peaks_vs_chi.pdf'
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(out2.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Guardado: {out2.name}")
plt.close(fig2)

# ── Figura 3: CN_O peaks (r_cut promedio IEX) vs χ ───────────────────────────

fig3, axes3 = plt.subplots(1, 2, figsize=(9, 4.5))
for col, (sys, title) in enumerate([('cao', r'10CaO'), ('noca', r'Ca-free')]):
    ax = axes3[col]
    for key, color, marker, label in SERIES:
        plot_with_fit(ax, CHI, cn_pk_av[sys][key], color, marker, label)
    ax.set_title(title, fontsize=13)
    decorate(ax, r'CN$_\mathrm{O}$ peak', ylim=(4, 11))

handles, labels = axes3[0].get_legend_handles_labels()
fig3.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
            ncol=3, fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
fig3.tight_layout(rect=[0, 0, 1, 0.94])
out3 = HERE / 'fig_cn_av_peaks_vs_chi.pdf'
fig3.savefig(out3, dpi=300, bbox_inches='tight')
fig3.savefig(out3.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Guardado: {out3.name}")
plt.close(fig3)

print("\nListo.")
