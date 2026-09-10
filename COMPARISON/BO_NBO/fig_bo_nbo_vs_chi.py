"""
fig_bo_nbo_vs_chi.py
<CN_BO> y <CN_NBO> vs χ — Na(AM), K(AM), K(IEX1), Ca(AM) — con ajuste lineal.

Clasificación de oxígenos (r_cut Si–O = 2.0 Å):
  BO  : O con exactamente 2 vecinos Si
  NBO : O con exactamente 1 vecino Si

Radios de corte:
  Na–O : RCUT_NA  = 3.21 Å
  K–O  : RCUT_K   = 3.77 Å  (AM); RCUT_IEX = 3.49 Å (IEX1)
  Ca–O : RCUT_CA  = 3.20 Å

Genera:
  fig_bo_vs_chi.pdf   — <CN_BO>  vs χ (2 paneles: 10CaO | Ca-free)
  fig_nbo_vs_chi.pdf  — <CN_NBO> vs χ (2 paneles: 10CaO | Ca-free)

Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree

HERE  = Path(__file__).parent
BASE  = HERE.parent.parent

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"

RCUT_SIO = 2.00
RCUT_NA  = 3.21
RCUT_K   = 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0   # 3.49 Å
RCUT_CA  = 3.20

TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5

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


def classify_oxygens(pos, types, box):
    tree = cKDTree(pos, boxsize=box)
    o_mask = (types == TYPE_O)
    o_idx  = np.where(o_mask)[0]
    si_count = np.zeros(len(pos), dtype=int)
    for idx in o_idx:
        nbrs = tree.query_ball_point(pos[idx], RCUT_SIO)
        si_count[idx] = int(np.sum(types[nbrs] == TYPE_SI))
    is_bo  = o_mask & (si_count == 2)
    is_nbo = o_mask & (si_count == 1)
    return is_bo, is_nbo


def cn_bo_nbo_around(pos, types, box, center_type, rcut, is_bo, is_nbo):
    tree = cKDTree(pos, boxsize=box)
    centers = pos[types == center_type]
    cn_bo_list, cn_nbo_list = [], []
    for cp in centers:
        nbrs = np.array(tree.query_ball_point(cp, rcut), dtype=int)
        cn_bo_list.append(int(np.sum(is_bo[nbrs])))
        cn_nbo_list.append(int(np.sum(is_nbo[nbrs])))
    return np.array(cn_bo_list), np.array(cn_nbo_list)


def load_am(am_dir, x, ctype, rcut):
    bo_all, nbo_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(am_dir / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data")
        is_bo, is_nbo  = classify_oxygens(pos, types, box)
        cb, cn = cn_bo_nbo_around(pos, types, box, ctype, rcut, is_bo, is_nbo)
        bo_all.extend(cb); nbo_all.extend(cn)
    return np.array(bo_all), np.array(nbo_all)


def load_iex1(iex_dir, x, ctype, rcut):
    bo_all, nbo_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(iex_dir / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data")
        is_bo, is_nbo  = classify_oxygens(pos, types, box)
        cb, cn = cn_bo_nbo_around(pos, types, box, ctype, rcut, is_bo, is_nbo)
        bo_all.extend(cb); nbo_all.extend(cn)
    return np.array(bo_all), np.array(nbo_all)


# ── Recopilar medias ──────────────────────────────────────────────────────────

data = {
    'cao':  {'na_bo': [], 'na_nbo': [], 'ka_bo': [], 'ka_nbo': [],
             'ki_bo': [], 'ki_nbo': [], 'ca_bo': [], 'ca_nbo': []},
    'noca': {'na_bo': [], 'na_nbo': [], 'ka_bo': [], 'ka_nbo': [],
             'ki_bo': [], 'ki_nbo': []},
}

for XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca) in sorted(XX_CONFIG.items()):
    print(f"XX={XX}%...")

    bo, nbo = load_am(CAO_AM_DIR,    xna_cao,  TYPE_NA, RCUT_NA)
    data['cao']['na_bo'].append(bo.mean());   data['cao']['na_nbo'].append(nbo.mean())

    bo, nbo = load_am(CAO_AM_DIR,    xka_cao,  TYPE_K,  RCUT_K)
    data['cao']['ka_bo'].append(bo.mean());   data['cao']['ka_nbo'].append(nbo.mean())

    bo, nbo = load_iex1(CAO_IEX1_DIR, xt_cao,  TYPE_K,  RCUT_IEX)
    data['cao']['ki_bo'].append(bo.mean());   data['cao']['ki_nbo'].append(nbo.mean())

    if xka_cao > 0:
        bo, nbo = load_am(CAO_AM_DIR, xka_cao, TYPE_CA, RCUT_CA)
    else:
        bo, nbo = np.array([np.nan]), np.array([np.nan])
    data['cao']['ca_bo'].append(bo.mean());   data['cao']['ca_nbo'].append(nbo.mean())

    bo, nbo = load_am(NOCA_AM_DIR,   xna_noca, TYPE_NA, RCUT_NA)
    data['noca']['na_bo'].append(bo.mean());  data['noca']['na_nbo'].append(nbo.mean())

    bo, nbo = load_am(NOCA_AM_DIR,   xka_noca, TYPE_K,  RCUT_K)
    data['noca']['ka_bo'].append(bo.mean());  data['noca']['ka_nbo'].append(nbo.mean())

    bo, nbo = load_iex1(NOCA_IEX1_DIR, xt_noca, TYPE_K, RCUT_IEX)
    data['noca']['ki_bo'].append(bo.mean());  data['noca']['ki_nbo'].append(nbo.mean())

# Convertir a arrays
for sys in data:
    for k in data[sys]:
        data[sys][k] = np.array(data[sys][k], dtype=float)


# ── Plot helpers ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       14,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_NA_AM = '#1a3a5c'
C_K_AM  = '#2ca0c4'
C_K_IEX = '#c0392b'
C_CA_AM = '#7b3f00'


def plot_with_fit(ax, chi, y, color, marker, label):
    mask = ~np.isnan(y)
    ax.plot(chi[mask], y[mask], color=color, marker=marker, ls='none',
            ms=6, mew=0.8, label=label, zorder=3)
    if mask.sum() > 1:
        m, b = np.polyfit(chi[mask], y[mask], 1)
        xf = np.linspace(chi[mask].min(), chi[mask].max(), 200)
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


SERIES_CAO  = [('na_bo', 'na_nbo', C_NA_AM, 'o', 'Na (As-Melted)'),
               ('ka_bo', 'ka_nbo', C_K_AM,  's', 'K (As-Melted)'),
               ('ki_bo', 'ki_nbo', C_K_IEX, '^', 'K (Ion-Exchanged)'),
               ('ca_bo', 'ca_nbo', C_CA_AM, 'D', 'Ca (As-Melted)')]

SERIES_NOCA = [('na_bo', 'na_nbo', C_NA_AM, 'o', 'Na (As-Melted)'),
               ('ka_bo', 'ka_nbo', C_K_AM,  's', 'K (As-Melted)'),
               ('ki_bo', 'ki_nbo', C_K_IEX, '^', 'K (Ion-Exchanged)')]


# ── Figura CN_BO vs χ ─────────────────────────────────────────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(9, 4.5))

for key_bo, key_nbo, color, marker, label in SERIES_CAO:
    plot_with_fit(axes1[0], CHI, data['cao'][key_bo], color, marker, label)
axes1[0].set_title(r'10CaO', fontsize=13)
decorate(axes1[0], r'$\langle$CN$_\mathrm{BO}\rangle$')

for key_bo, key_nbo, color, marker, label in SERIES_NOCA:
    plot_with_fit(axes1[1], CHI, data['noca'][key_bo], color, marker, label)
axes1[1].set_title(r'Ca-free', fontsize=13)
decorate(axes1[1], r'$\langle$CN$_\mathrm{BO}\rangle$')

handles, labels = axes1[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
            ncol=4, fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
fig1.tight_layout(rect=[0, 0, 1, 0.94])
out1 = HERE / 'fig_bo_vs_chi.pdf'
fig1.savefig(out1, dpi=300, bbox_inches='tight')
fig1.savefig(out1.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out1.name}")
plt.close(fig1)


# ── Figura CN_NBO vs χ ────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.5))

for key_bo, key_nbo, color, marker, label in SERIES_CAO:
    plot_with_fit(axes2[0], CHI, data['cao'][key_nbo], color, marker, label)
axes2[0].set_title(r'10CaO', fontsize=13)
decorate(axes2[0], r'$\langle$CN$_\mathrm{NBO}\rangle$')

for key_bo, key_nbo, color, marker, label in SERIES_NOCA:
    plot_with_fit(axes2[1], CHI, data['noca'][key_nbo], color, marker, label)
axes2[1].set_title(r'Ca-free', fontsize=13)
decorate(axes2[1], r'$\langle$CN$_\mathrm{NBO}\rangle$')

handles, labels = axes2[0].get_legend_handles_labels()
fig2.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04),
            ncol=4, fontsize=11, frameon=True, framealpha=0.9, edgecolor='#cccccc')
fig2.tight_layout(rect=[0, 0, 1, 0.94])
out2 = HERE / 'fig_nbo_vs_chi.pdf'
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(out2.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"Guardado: {out2.name}")
plt.close(fig2)

print("\nListo.")
